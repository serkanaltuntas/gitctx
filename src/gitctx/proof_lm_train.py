"""PyTorch proof language-model trainer for GCTX-1 trainer jobs.

This module is the first real decoder-only language-model trainer entrypoint in
gitctx. It consumes the proof trainer job manifest, rebuilds deterministic
trainer sequences from the lineage artifacts, trains only on DEV records, and
writes resumable checkpoints plus a trainer report. PyTorch is intentionally an
optional runtime dependency: importing gitctx must stay dependency-free, while
this command clearly blocks when the backend is unavailable.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
from typing import Any

from gitctx.proof_sequences import materialize_training_sequence
from gitctx.proof_train import DEFAULT_RUN_ID, SCHEMA_VERSION
from gitctx.proof_train_job import proof_trainer_job_path

TRAIN_RUN_DIR = Path("artifacts/train-runs")
LM_TRAINER_ID = "gctx1-proof-lm-trainer-v0"
DEFAULT_DEVICE = "cpu"
DEFAULT_TRAIN_SPLIT = "DEV"
DEFAULT_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 3e-4


def proof_lm_train_report_path(run_id: str) -> Path:
    """Return the proof LM trainer report path for a run id."""

    _validate_identifier(run_id, "run_id")
    return TRAIN_RUN_DIR / f"{run_id}.trainer.report.json"


def proof_lm_checkpoint_dir(run_id: str) -> Path:
    """Return the proof LM checkpoint directory for a run id."""

    _validate_identifier(run_id, "run_id")
    return TRAIN_RUN_DIR / run_id / "checkpoints"


def proof_lm_latest_checkpoint_path(run_id: str) -> Path:
    """Return the proof LM latest checkpoint manifest path for a run id."""

    return proof_lm_checkpoint_dir(run_id) / "latest.json"


def proof_lm_final_checkpoint_path(run_id: str) -> Path:
    """Return the proof LM final checkpoint manifest path for a run id."""

    return proof_lm_checkpoint_dir(run_id) / "final.json"


def run_proof_lm_training(
    data_dir: str | Path,
    *,
    run_id: str = DEFAULT_RUN_ID,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_records: int | None = None,
    max_steps: int | None = None,
    resume: bool = False,
    write: bool = False,
    override_layers: int | None = None,
    override_hidden_size: int | None = None,
    override_attention_heads: int | None = None,
    override_kv_heads: int | None = None,
    override_intermediate_size: int | None = None,
    override_context_tokens: int | None = None,
) -> dict[str, Any]:
    """Train a decoder-only proof LM from a ready proof trainer job manifest."""

    _validate_identifier(run_id, "run_id")
    _validate_positive_int(batch_size, "batch_size")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_records is not None:
        _validate_positive_int(max_records, "max_records")
    if max_steps is not None:
        _validate_positive_int(max_steps, "max_steps")

    data_dir = Path(data_dir)
    job_path = data_dir / proof_trainer_job_path(run_id)
    job = _load_json(job_path)
    torch = _load_torch()
    blockers = _job_blockers(data_dir, job_path=job_path, job=job)
    model_contract = _model_contract_with_overrides(
        job,
        override_layers=override_layers,
        override_hidden_size=override_hidden_size,
        override_attention_heads=override_attention_heads,
        override_kv_heads=override_kv_heads,
        override_intermediate_size=override_intermediate_size,
        override_context_tokens=override_context_tokens,
    )
    blockers.extend(_model_contract_blockers(model_contract))
    if torch is None:
        blockers.append("torch is not importable")
    elif device != "cpu":
        blockers.extend(_device_blockers(torch, device))

    selected_sequences: list[dict[str, Any]] = []
    if not blockers:
        selected_sequences, sequence_blockers = _load_train_sequences(
            data_dir,
            job=job,
            train_split=DEFAULT_TRAIN_SPLIT,
            max_records=max_records,
            context_tokens=model_contract["context_tokens"],
        )
        blockers.extend(sequence_blockers)
    if not selected_sequences and not blockers:
        blockers.append("no DEV trainer sequences selected")

    config = _training_config(
        run_id=run_id,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_records=max_records,
        max_steps=max_steps,
        model_contract=model_contract,
    )
    config_sha = _stable_sha256(config)
    if blockers:
        report = _blocked_report(
            run_id=run_id,
            blockers=blockers,
            config=config,
            config_sha256=config_sha,
            job_path=job_path,
            data_dir=data_dir,
            job=job,
        )
        if write:
            _write_report(data_dir, report)
        _print_report(report)
        return report

    assert torch is not None
    _seed_torch(torch, job.get("seed", 0))
    model = _build_model(torch, model_contract).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    state = _initial_state()
    resumed_from_checkpoint = False
    if resume:
        checkpoint_path = data_dir / proof_lm_latest_checkpoint_path(run_id)
        if not checkpoint_path.exists():
            raise ValueError("resume requested but latest checkpoint is missing")
        checkpoint = _load_json(checkpoint_path)
        _load_checkpoint_state(
            torch,
            checkpoint,
            data_dir=data_dir,
            run_id=run_id,
            config_sha256=config_sha,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        state = _state_from_checkpoint(checkpoint)
        resumed_from_checkpoint = True

    if state["record_cursor"] > len(selected_sequences):
        raise ValueError("checkpoint cursor is beyond selected training records")

    model.train()
    processed_this_run = 0
    while state["record_cursor"] < len(selected_sequences):
        if max_steps is not None and state["optimizer_steps"] >= max_steps:
            break
        batch_sequences = selected_sequences[
            state["record_cursor"]:state["record_cursor"] + batch_size
        ]
        batch = _batch_for_torch(torch, batch_sequences, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model_contract["tokenizer_vocab_size"]),
            batch["labels"].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        optimizer.step()

        batch_records = len(batch_sequences)
        batch_loss_tokens = int(batch["loss_tokens"])
        state["record_cursor"] += batch_records
        state["completed_records"] += batch_records
        state["optimizer_steps"] += 1
        state["input_tokens"] += int(batch["input_tokens"])
        state["loss_tokens"] += batch_loss_tokens
        state["loss_sum"] += float(loss.detach().cpu()) * batch_loss_tokens
        state["last_record_id"] = batch_sequences[-1]["record_id"]
        processed_this_run += batch_records

    status = "trained" if state["record_cursor"] >= len(selected_sequences) else "partial"
    report = _trained_report(
        run_id=run_id,
        status=status,
        config=config,
        config_sha256=config_sha,
        job_path=job_path,
        data_dir=data_dir,
        job=job,
        state=state,
        selected_train_records=len(selected_sequences),
        processed_this_run=processed_this_run,
        resumed_from_checkpoint=resumed_from_checkpoint,
    )
    if write:
        _write_training_artifacts(
            torch,
            data_dir=data_dir,
            run_id=run_id,
            status=status,
            config=config,
            config_sha256=config_sha,
            report=report,
            state=state,
            model=model,
            optimizer=optimizer,
        )
    _print_report(report)
    return report


def validate_proof_lm_training(data_dir: str | Path, *, run_id: str) -> dict[str, Any]:
    """Validate a proof LM trainer report and checkpoint manifest."""

    _validate_identifier(run_id, "run_id")
    data_dir = Path(data_dir)
    report_path = data_dir / proof_lm_train_report_path(run_id)
    report = _load_json(report_path)
    errors: list[str] = []
    if report.get("trainer_id") != LM_TRAINER_ID:
        errors.append("unexpected trainer_id")
    if report.get("run_id") != run_id:
        errors.append("report run_id mismatch")
    if report.get("status") not in {"trained", "partial"}:
        errors.append("report status is not trained or partial")
    if report.get("blockers") != []:
        errors.append("report blockers must be empty")
    checkpoint_manifest_path = data_dir / proof_lm_latest_checkpoint_path(run_id)
    if report.get("status") == "trained":
        checkpoint_manifest_path = data_dir / proof_lm_final_checkpoint_path(run_id)
    if not checkpoint_manifest_path.exists():
        errors.append("checkpoint manifest is missing")
        checkpoint: dict[str, Any] = {}
    else:
        checkpoint = _load_json(checkpoint_manifest_path)
    if checkpoint:
        if checkpoint.get("trainer_id") != LM_TRAINER_ID:
            errors.append("checkpoint trainer_id mismatch")
        if checkpoint.get("run_id") != run_id:
            errors.append("checkpoint run_id mismatch")
        if checkpoint.get("config_sha256") != report.get("config_sha256"):
            errors.append("checkpoint config_sha256 mismatch")
        if checkpoint.get("report_sha256") != _sha256(report_path):
            errors.append("checkpoint report_sha256 does not match report")
        state_path = _data_path(data_dir, Path(str(checkpoint.get("state_path", ""))))
        if not state_path.exists():
            errors.append("checkpoint state file is missing")
        elif checkpoint.get("state_sha256") != _sha256(state_path):
            errors.append("checkpoint state sha256 mismatch")
    validation = {
        "run_id": run_id,
        "valid": not errors,
        "errors": errors,
        "report_path": str(proof_lm_train_report_path(run_id)),
        "checkpoint_path": str(checkpoint_manifest_path.relative_to(data_dir))
        if checkpoint_manifest_path.is_absolute()
        else str(checkpoint_manifest_path),
    }
    for key, value in validation.items():
        if key != "errors":
            print(key, value)
    for error in errors:
        print("error", error)
    if errors:
        raise SystemExit(1)
    return validation


def _job_blockers(data_dir: Path, *, job_path: Path, job: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if job.get("status") != "ready_for_trainer":
        blockers.append("trainer job is not ready_for_trainer")
    if job.get("blockers") != []:
        blockers.append("trainer job blockers must be empty")
    inputs = job.get("inputs")
    if not isinstance(inputs, dict):
        blockers.append("trainer job inputs must be an object")
        return blockers
    for name, entry in sorted(inputs.items()):
        if not isinstance(entry, dict):
            blockers.append(f"input {name}: entry must be an object")
            continue
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            blockers.append(f"input {name}: path is missing")
            continue
        path = _data_path(data_dir, Path(path_value))
        if not path.exists():
            blockers.append(f"input {name}: file is missing")
            continue
        actual_sha = _sha256(path)
        if entry.get("ok") is False:
            blockers.append(f"input {name}: manifest says input is not ok")
        present_hashes = 0
        for hash_name in ("sha256", "actual_sha256", "expected_sha256"):
            expected = entry.get(hash_name)
            if expected is None:
                continue
            if not isinstance(expected, str):
                blockers.append(f"input {name}: {hash_name} must be a string")
                continue
            present_hashes += 1
            if actual_sha != expected:
                blockers.append(f"input {name}: {hash_name} mismatch")
        if present_hashes == 0:
            blockers.append(f"input {name}: sha256 is missing")
    handoff_entry = inputs.get("handoff")
    if (
        isinstance(handoff_entry, dict)
        and isinstance(handoff_entry.get("path"), str)
        and isinstance(handoff_entry.get("sha256"), str)
    ):
        if handoff_entry["sha256"] != _sha256(_data_path(data_dir, Path(handoff_entry["path"]))):
            blockers.append("handoff sha256 mismatch")
    if not job_path.exists():
        blockers.append("trainer job manifest is missing")
    return blockers


def _model_contract_with_overrides(
    job: dict[str, Any],
    *,
    override_layers: int | None,
    override_hidden_size: int | None,
    override_attention_heads: int | None,
    override_kv_heads: int | None,
    override_intermediate_size: int | None,
    override_context_tokens: int | None,
) -> dict[str, Any]:
    contract = dict(job.get("model_contract") if isinstance(job.get("model_contract"), dict) else {})
    overrides = {
        "layers": override_layers,
        "hidden_size": override_hidden_size,
        "attention_heads": override_attention_heads,
        "kv_heads": override_kv_heads,
        "intermediate_size": override_intermediate_size,
        "context_tokens": override_context_tokens,
    }
    for key, value in overrides.items():
        if value is not None:
            _validate_positive_int(value, key)
            contract[key] = value
    return contract


def _model_contract_blockers(contract: dict[str, Any]) -> list[str]:
    blockers = []
    required_ints = (
        "tokenizer_vocab_size",
        "context_tokens",
        "layers",
        "hidden_size",
        "attention_heads",
        "kv_heads",
        "intermediate_size",
    )
    for name in required_ints:
        if not isinstance(contract.get(name), int) or contract[name] < 1:
            blockers.append(f"model_contract.{name} must be a positive integer")
    if blockers:
        return blockers
    if contract["hidden_size"] % contract["attention_heads"] != 0:
        blockers.append("model_contract.hidden_size must be divisible by attention_heads")
    if contract["attention_heads"] % contract["kv_heads"] != 0:
        blockers.append("model_contract.attention_heads must be divisible by kv_heads")
    return blockers


def _load_train_sequences(
    data_dir: Path,
    *,
    job: dict[str, Any],
    train_split: str,
    max_records: int | None,
    context_tokens: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    inputs = job.get("inputs", {})
    handoff = _load_json(_input_path(data_dir, inputs, "handoff", blockers))
    training_path = _input_path(data_dir, inputs, "training_artifact", blockers)
    tokenizer_path = _input_path(data_dir, inputs, "tokenizer", blockers)
    sequence_plan_path = _input_path(data_dir, inputs, "sequence_plan", blockers)
    sequence_metadata_path = _input_path(data_dir, inputs, "sequence_metadata", blockers)
    if blockers:
        return [], blockers
    tokenizer = _load_json(tokenizer_path)
    sequence_plan = list(_iter_jsonl(sequence_plan_path))
    plan_by_record_id = {
        record["record_id"]: record
        for record in sequence_plan
        if isinstance(record.get("record_id"), str)
    }
    metadata_by_record_id = {
        record["record_id"]: record
        for record in _iter_jsonl(sequence_metadata_path)
        if isinstance(record.get("record_id"), str)
    }
    sequences: list[dict[str, Any]] = []
    for record in _iter_jsonl(training_path):
        if record.get("data_split") != train_split:
            continue
        record_id = record.get("id")
        plan_record = plan_by_record_id.get(record_id)
        if plan_record is None:
            blockers.append(f"{record_id}: missing sequence plan record")
            continue
        if plan_record.get("decision") == "exclude_oversize":
            continue
        sequence = materialize_training_sequence(
            record,
            plan_record,
            tokenizer,
            context_tokens=context_tokens,
        )
        metadata = metadata_by_record_id.get(record_id)
        if metadata is None:
            blockers.append(f"{record_id}: missing sequence metadata")
            continue
        if not _sequence_matches_metadata(sequence, metadata):
            blockers.append(f"{record_id}: materialized sequence does not match metadata")
            continue
        sequences.append(sequence)
        if max_records is not None and len(sequences) >= max_records:
            break
    expected_split = handoff.get("training_contract", {}).get("train_split")
    if isinstance(expected_split, str) and expected_split != train_split:
        blockers.append("handoff train split does not match trainer split")
    return sequences, blockers


def _build_model(torch: Any, contract: dict[str, Any]) -> Any:
    nn = torch.nn
    functional = torch.nn.functional

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x: Any) -> Any:
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            return self.weight * x * torch.rsqrt(variance + self.eps)

    class GQACausalSelfAttention(nn.Module):
        def __init__(self, hidden_size: int, attention_heads: int, kv_heads: int) -> None:
            super().__init__()
            self.attention_heads = attention_heads
            self.kv_heads = kv_heads
            self.head_dim = hidden_size // attention_heads
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, x: Any, attention_mask: Any) -> Any:
            batch, seq_len, _ = x.shape
            q = self.q_proj(x).view(batch, seq_len, self.attention_heads, self.head_dim)
            k = self.k_proj(x).view(batch, seq_len, self.kv_heads, self.head_dim)
            v = self.v_proj(x).view(batch, seq_len, self.kv_heads, self.head_dim)
            repeat = self.attention_heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
            key_padding = attention_mask[:, None, None, :] == 0
            scores = scores.masked_fill(key_padding, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn)
            out = torch.matmul(attn, v).transpose(1, 2).contiguous()
            return self.o_proj(out.view(batch, seq_len, -1))

    class DecoderBlock(nn.Module):
        def __init__(self, hidden_size: int, attention_heads: int, kv_heads: int, intermediate: int) -> None:
            super().__init__()
            self.attn_norm = RMSNorm(hidden_size)
            self.attn = GQACausalSelfAttention(hidden_size, attention_heads, kv_heads)
            self.mlp_norm = RMSNorm(hidden_size)
            self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)

        def forward(self, x: Any, attention_mask: Any) -> Any:
            x = x + self.attn(self.attn_norm(x), attention_mask)
            hidden = self.mlp_norm(x)
            x = x + self.down_proj(functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))
            return x

    class ProofDecoderLM(nn.Module):
        def __init__(self, selected_contract: dict[str, Any]) -> None:
            super().__init__()
            self.vocab_size = selected_contract["tokenizer_vocab_size"]
            hidden_size = selected_contract["hidden_size"]
            self.token_embedding = nn.Embedding(self.vocab_size, hidden_size)
            self.position_embedding = nn.Embedding(selected_contract["context_tokens"], hidden_size)
            self.blocks = nn.ModuleList(
                [
                    DecoderBlock(
                        hidden_size,
                        selected_contract["attention_heads"],
                        selected_contract["kv_heads"],
                        selected_contract["intermediate_size"],
                    )
                    for _ in range(selected_contract["layers"])
                ]
            )
            self.norm = RMSNorm(hidden_size)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            for block in self.blocks:
                x = block(x, attention_mask)
            x = self.norm(x)
            return torch.matmul(x, self.token_embedding.weight.transpose(0, 1))

    return ProofDecoderLM(contract)


def _batch_for_torch(torch: Any, sequences: list[dict[str, Any]], *, device: str) -> dict[str, Any]:
    shifted = []
    for sequence in sequences:
        input_ids = sequence["input_ids"]
        loss_mask = sequence["loss_mask"]
        if len(input_ids) < 2:
            continue
        labels = list(input_ids[1:])
        active_loss = list(loss_mask[1:])
        for index, active in enumerate(active_loss):
            if not active:
                labels[index] = -100
        shifted.append(
            {
                "input_ids": list(input_ids[:-1]),
                "labels": labels,
                "loss_tokens": sum(1 for value in labels if value != -100),
            }
        )
    if not shifted:
        raise ValueError("batch has no trainable loss tokens")
    max_len = max(len(item["input_ids"]) for item in shifted)
    input_rows = []
    label_rows = []
    attention_rows = []
    input_tokens = 0
    loss_tokens = 0
    for item in shifted:
        pad = max_len - len(item["input_ids"])
        input_rows.append(item["input_ids"] + [0] * pad)
        label_rows.append(item["labels"] + [-100] * pad)
        attention_rows.append([1] * len(item["input_ids"]) + [0] * pad)
        input_tokens += len(item["input_ids"])
        loss_tokens += item["loss_tokens"]
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "labels": torch.tensor(label_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long, device=device),
        "input_tokens": input_tokens,
        "loss_tokens": loss_tokens,
    }


def _write_training_artifacts(
    torch: Any,
    *,
    data_dir: Path,
    run_id: str,
    status: str,
    config: dict[str, Any],
    config_sha256: str,
    report: dict[str, Any],
    state: dict[str, Any],
    model: Any,
    optimizer: Any,
) -> None:
    report_path = data_dir / proof_lm_train_report_path(run_id)
    checkpoint_dir = data_dir / proof_lm_checkpoint_dir(run_id)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / f"step-{state['optimizer_steps']:06d}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "trainer_state": state,
            "config": config,
            "torch_rng_state": torch.get_rng_state(),
            "python_random_state": random.getstate(),
        },
        state_path,
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint = _checkpoint_manifest(
        run_id=run_id,
        status=status,
        config_sha256=config_sha256,
        report_path=report_path,
        state_path=state_path,
        data_dir=data_dir,
        state=state,
    )
    latest_path = data_dir / proof_lm_latest_checkpoint_path(run_id)
    latest_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status == "trained":
        final_path = data_dir / proof_lm_final_checkpoint_path(run_id)
        final_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(data_dir: Path, report: dict[str, Any]) -> None:
    path = data_dir / proof_lm_train_report_path(report["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_checkpoint_state(
    torch: Any,
    checkpoint: dict[str, Any],
    *,
    data_dir: Path,
    run_id: str,
    config_sha256: str,
    model: Any,
    optimizer: Any,
    device: str,
) -> None:
    if checkpoint.get("run_id") != run_id:
        raise ValueError("resume checkpoint run_id mismatch")
    if checkpoint.get("trainer_id") != LM_TRAINER_ID:
        raise ValueError("resume checkpoint trainer_id mismatch")
    if checkpoint.get("config_sha256") != config_sha256:
        raise ValueError("resume checkpoint config mismatch")
    state_path = _data_path(data_dir, Path(str(checkpoint.get("state_path"))))
    if checkpoint.get("state_sha256") != _sha256(state_path):
        raise ValueError("resume checkpoint state sha256 mismatch")
    try:
        state = torch.load(state_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch fallback
        state = torch.load(state_path, map_location=device)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])


def _checkpoint_manifest(
    *,
    run_id: str,
    status: str,
    config_sha256: str,
    report_path: Path,
    state_path: Path,
    data_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": "proof_lm_checkpoint",
        "trainer_id": LM_TRAINER_ID,
        "run_id": run_id,
        "status": status,
        "contains_model_weights": True,
        "config_sha256": config_sha256,
        "report_sha256": _sha256(report_path),
        "state_path": _display_path(state_path, base=data_dir),
        "state_sha256": _sha256(state_path),
        "record_cursor": state["record_cursor"],
        "optimizer_steps": state["optimizer_steps"],
        "accounting": _state_accounting(state),
        "resume_policy": (
            "Resume by loading this checkpoint state, verifying config and input hashes, "
            "then continuing from record_cursor over the same deterministic DEV sequence order."
        ),
    }


def _training_config(
    *,
    run_id: str,
    device: str,
    batch_size: int,
    learning_rate: float,
    max_records: int | None,
    max_steps: int | None,
    model_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trainer_id": LM_TRAINER_ID,
        "run_id": run_id,
        "device": device,
        "train_split": DEFAULT_TRAIN_SPLIT,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_records": max_records,
        "max_steps": max_steps,
        "model_contract": model_contract,
        "loss": "causal_cross_entropy_on_assistant_tokens",
        "optimizer": "adamw",
    }


def _blocked_report(
    *,
    run_id: str,
    blockers: list[str],
    config: dict[str, Any],
    config_sha256: str,
    job_path: Path,
    data_dir: Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "trainer_id": LM_TRAINER_ID,
        "run_id": run_id,
        "status": "blocked",
        "blockers": blockers,
        "config": config,
        "config_sha256": config_sha256,
        "trainer_job": _job_reference(job_path, data_dir=data_dir, job=job),
        "training": _empty_training_summary(),
        "outputs": _output_paths(run_id),
        "claim_policy": _claim_policy(),
    }


def _trained_report(
    *,
    run_id: str,
    status: str,
    config: dict[str, Any],
    config_sha256: str,
    job_path: Path,
    data_dir: Path,
    job: dict[str, Any],
    state: dict[str, Any],
    selected_train_records: int,
    processed_this_run: int,
    resumed_from_checkpoint: bool,
) -> dict[str, Any]:
    training = {
        "selected_train_records": selected_train_records,
        "processed_this_run": processed_this_run,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        **_state_accounting(state),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "trainer_id": LM_TRAINER_ID,
        "run_id": run_id,
        "status": status,
        "blockers": [],
        "config": config,
        "config_sha256": config_sha256,
        "trainer_job": _job_reference(job_path, data_dir=data_dir, job=job),
        "training": training,
        "outputs": _output_paths(run_id),
        "claim_policy": _claim_policy(),
    }


def _job_reference(job_path: Path, *, data_dir: Path, job: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": _display_path(job_path, base=data_dir),
        "sha256": _sha256(job_path),
        "trainer_job_id": job.get("trainer_job_id"),
        "status": job.get("status"),
    }


def _output_paths(run_id: str) -> dict[str, Any]:
    return {
        "training_report_path": str(proof_lm_train_report_path(run_id)),
        "checkpoint_dir": str(proof_lm_checkpoint_dir(run_id)),
        "latest_checkpoint": str(proof_lm_latest_checkpoint_path(run_id)),
        "final_checkpoint": str(proof_lm_final_checkpoint_path(run_id)),
    }


def _claim_policy() -> str:
    return (
        "This report proves that a decoder-only proof LM training job ran over reviewed DEV "
        "records. It is not a public quality claim until locked REPORT evaluation and model "
        "cards are produced."
    )


def _empty_training_summary() -> dict[str, Any]:
    return {
        "selected_train_records": 0,
        "processed_this_run": 0,
        "completed_records": 0,
        "optimizer_steps": 0,
        "input_tokens": 0,
        "loss_tokens": 0,
        "loss_sum": 0.0,
        "average_loss": 0.0,
        "last_record_id": None,
    }


def _initial_state() -> dict[str, Any]:
    return {
        "record_cursor": 0,
        "completed_records": 0,
        "optimizer_steps": 0,
        "input_tokens": 0,
        "loss_tokens": 0,
        "loss_sum": 0.0,
        "last_record_id": None,
    }


def _state_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    accounting = checkpoint.get("accounting", {})
    return {
        "record_cursor": _int_value(checkpoint.get("record_cursor")),
        "completed_records": _int_value(accounting.get("completed_records")),
        "optimizer_steps": _int_value(checkpoint.get("optimizer_steps")),
        "input_tokens": _int_value(accounting.get("input_tokens")),
        "loss_tokens": _int_value(accounting.get("loss_tokens")),
        "loss_sum": float(accounting.get("loss_sum", 0.0)),
        "last_record_id": accounting.get("last_record_id"),
    }


def _state_accounting(state: dict[str, Any]) -> dict[str, Any]:
    loss_tokens = state["loss_tokens"]
    loss_sum = state["loss_sum"]
    return {
        "completed_records": state["completed_records"],
        "optimizer_steps": state["optimizer_steps"],
        "input_tokens": state["input_tokens"],
        "loss_tokens": loss_tokens,
        "loss_sum": round(loss_sum, 8),
        "average_loss": round(loss_sum / loss_tokens, 8) if loss_tokens else 0.0,
        "last_record_id": state["last_record_id"],
    }


def _sequence_matches_metadata(sequence: dict[str, Any], metadata: dict[str, Any]) -> bool:
    return (
        sequence.get("input_length") == metadata.get("input_length")
        and sequence.get("loss_tokens") == metadata.get("loss_tokens")
        and _hash_ints(sequence["input_ids"]) == metadata.get("input_ids_sha256")
        and _hash_ints(sequence["loss_mask"]) == metadata.get("loss_mask_sha256")
    )


def _input_path(
    data_dir: Path,
    inputs: Any,
    input_name: str,
    blockers: list[str],
) -> Path:
    entry = inputs.get(input_name) if isinstance(inputs, dict) else None
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        blockers.append(f"input {input_name}: missing from trainer job")
        return data_dir / "__missing__"
    return _data_path(data_dir, Path(entry["path"]))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _device_blockers(torch: Any, device: str) -> list[str]:
    if device == "cuda" and not torch.cuda.is_available():
        return ["requested cuda device is not available"]
    if device == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        return ["requested mps device is not available"]
    if device not in {"cpu", "cuda", "mps"}:
        return ["device must be cpu, cuda, or mps"]
    return []


def _seed_torch(torch: Any, seed: Any) -> None:
    selected_seed = seed if isinstance(seed, int) and seed >= 0 else 0
    random.seed(selected_seed)
    torch.manual_seed(selected_seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover - older torch fallback
        torch.use_deterministic_algorithms(True)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_ints(values: list[int]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(int(value).to_bytes(4, byteorder="little", signed=True))
    return h.hexdigest()


def _data_path(data_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else data_dir / path


def _display_path(path: Path, *, base: Path | None = None) -> str:
    selected = path.resolve(strict=False)
    if base is not None:
        try:
            return selected.relative_to(base.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _validate_identifier(value: str, name: str) -> None:
    if not value.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"{name} must be a stable alphanumeric identifier")


def _validate_positive_int(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _print_report(report: dict[str, Any]) -> None:
    print("run_id", report["run_id"])
    print("status", report["status"])
    training = report["training"]
    for key in (
        "selected_train_records",
        "processed_this_run",
        "completed_records",
        "optimizer_steps",
        "input_tokens",
        "loss_tokens",
        "average_loss",
    ):
        print(key, training[key])
    for blocker in report["blockers"]:
        print("blocker", blocker)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train or validate a GCTX proof language model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--run-id", default=DEFAULT_RUN_ID)
    train.add_argument("--device", default=DEFAULT_DEVICE)
    train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    train.add_argument("--max-records", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--write", action="store_true")
    train.add_argument("--fail-on-blocked", action="store_true")
    train.add_argument("--override-layers", type=int)
    train.add_argument("--override-hidden-size", type=int)
    train.add_argument("--override-attention-heads", type=int)
    train.add_argument("--override-kv-heads", type=int)
    train.add_argument("--override-intermediate-size", type=int)
    train.add_argument("--override-context-tokens", type=int)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--run-id", default=DEFAULT_RUN_ID)

    args = parser.parse_args(argv)
    if args.command == "train":
        report = run_proof_lm_training(
            args.data_dir,
            run_id=args.run_id,
            device=args.device,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_records=args.max_records,
            max_steps=args.max_steps,
            resume=args.resume,
            write=args.write,
            override_layers=args.override_layers,
            override_hidden_size=args.override_hidden_size,
            override_attention_heads=args.override_attention_heads,
            override_kv_heads=args.override_kv_heads,
            override_intermediate_size=args.override_intermediate_size,
            override_context_tokens=args.override_context_tokens,
        )
        if args.fail_on_blocked and report["status"] == "blocked":
            return 1
    elif args.command == "validate":
        validate_proof_lm_training(args.data_dir, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

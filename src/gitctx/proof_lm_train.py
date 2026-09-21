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
import time
import platform
import subprocess
from typing import Any

from gitctx.proof_sequences import materialize_training_sequence
from gitctx.proof_train import DEFAULT_RUN_ID, SCHEMA_VERSION
from gitctx.proof_train_job import proof_trainer_job_path

TRAIN_RUN_DIR = Path("artifacts/train-runs")
LM_TRAINER_ID = "gctx1-proof-lm-trainer-v1"
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
    record_ids: list[str] | None = None,
    attention_chunk_size: int = 256,
    activation_checkpointing: bool = True,
    checkpoint_every: int = 100,
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

    _validate_positive_int(attention_chunk_size, "attention_chunk_size")
    _validate_positive_int(checkpoint_every, "checkpoint_every")
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
            max_records=None if record_ids else max_records,
            context_tokens=model_contract["context_tokens"],
        )
        blockers.extend(sequence_blockers)
        if record_ids:
            by_id = {sequence["record_id"]: sequence for sequence in selected_sequences}
            if len(set(record_ids)) != len(record_ids) or any(i not in by_id for i in record_ids):
                blockers.append("record_ids must be distinct selected DEV records")
            else:
                selected_sequences = [by_id[i] for i in record_ids][:max_records]
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
    config["record_ids"] = record_ids
    config["execution"] = {
        "attention_chunk_size": attention_chunk_size,
        "activation_checkpointing": activation_checkpointing,
        "precision": "float32",
        "checkpoint_every": checkpoint_every,
    }
    config["seed"] = job.get("seed", 0)
    config["input_hashes"] = {
        name: entry.get("sha256", entry.get("actual_sha256"))
        for name, entry in job.get("inputs", {}).items()
    }
    config["implementation_hashes"] = {
        name: _sha256(Path(__file__).with_name(name))
        for name in ("proof_lm_train.py", "proof_sequences.py", "proof_tokenizer.py")
    }
    config_sha = _resume_config_sha256(config)
    if write and not resume and (data_dir / proof_lm_latest_checkpoint_path(run_id)).exists():
        raise ValueError("run already has a checkpoint; use resume or a new run_id")
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
    _configure_device_runtime(torch, device)
    _seed_torch(torch, job.get("seed", 0))
    model = _build_model(torch, model_contract,
                         attention_chunk_size=attention_chunk_size,
                         activation_checkpointing=activation_checkpointing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, foreach=False)
    state = _initial_state()
    resumed_from_checkpoint = False
    if resume:
        checkpoint_path = data_dir / proof_lm_latest_checkpoint_path(run_id)
        if not checkpoint_path.exists():
            raise ValueError("resume requested but latest checkpoint is missing")
        checkpoint = _load_json(checkpoint_path)
        state = _load_checkpoint_state(
            torch,
            checkpoint,
            data_dir=data_dir,
            run_id=run_id,
            config_sha256=config_sha,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        resumed_from_checkpoint = True

    if state["record_cursor"] > len(selected_sequences):
        raise ValueError("checkpoint cursor is beyond selected training records")

    model.train()
    processed_this_run = 0
    started = time.monotonic()
    runtime = _runtime_info(torch, device, model)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    while state["record_cursor"] < len(selected_sequences):
        if max_steps is not None and state["optimizer_steps"] >= max_steps:
            break
        step_started = time.monotonic()
        batch_sequences = selected_sequences[
            state["record_cursor"]:state["record_cursor"] + batch_size
        ]
        batch = _batch_for_torch(torch, batch_sequences, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
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
        elapsed = time.monotonic() - started
        progress = {"step": state["optimizer_steps"], "record_cursor": state["record_cursor"],
                    "records": len(selected_sequences), "input_length": batch["input_ids"].shape[1],
                    "loss": float(loss.detach().cpu()), "elapsed_seconds": round(elapsed, 3),
                    "step_seconds": round(time.monotonic() - step_started, 6)}
        if device == "cuda":
            progress["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            progress["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        print(json.dumps(progress, sort_keys=True), flush=True)
        if write and state["optimizer_steps"] % checkpoint_every == 0:
            interim = _trained_report(
                run_id=run_id, status="partial", config=config, config_sha256=config_sha,
                job_path=job_path, data_dir=data_dir, job=job, state=state,
                selected_train_records=len(selected_sequences), processed_this_run=processed_this_run,
                resumed_from_checkpoint=resumed_from_checkpoint)
            interim["runtime"] = {**runtime, **progress}
            _write_training_artifacts(torch, data_dir=data_dir, run_id=run_id, status="partial",
                                      config=config, config_sha256=config_sha, report=interim,
                                      state=state, model=model, optimizer=optimizer)

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
    report["runtime"] = {**runtime, "elapsed_seconds": time.monotonic() - started}
    if device == "cuda":
        report["runtime"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
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
    if contract.get("position_encoding") != "rope":
        blockers.append("model_contract.position_encoding must be rope")
    if (contract["hidden_size"] // contract["attention_heads"]) % 2:
        blockers.append("RoPE requires an even attention head dimension")
    if contract.get("tie_input_output_embeddings") is not True:
        blockers.append("model_contract must tie input/output embeddings")
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
        sequence.pop("tokens", None)
        sequences.append(sequence)
        if max_records is not None and len(sequences) >= max_records:
            break
    expected_split = handoff.get("training_contract", {}).get("train_split")
    if isinstance(expected_split, str) and expected_split != train_split:
        blockers.append("handoff train split does not match trainer split")
    return sequences, blockers


def _apply_rope(torch: Any, x: Any, positions: Any) -> Any:
    """Rotate adjacent feature pairs; x has shape [batch, sequence, heads, dim]."""
    head_dim = x.shape[-1]
    frequencies = 10000.0 ** (-torch.arange(0, head_dim, 2, device=x.device).float() / head_dim)
    angles = positions.float()[:, None] * frequencies[None, :]
    cos = angles.cos().to(x.dtype)[None, :, None, :]
    sin = angles.sin().to(x.dtype)[None, :, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def _build_model(torch: Any, contract: dict[str, Any], *,
                 attention_chunk_size: int | None = None,
                 activation_checkpointing: bool = False) -> Any:
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

        def forward(self, x: Any, attention_mask: Any, past: Any = None, use_cache: bool = False) -> Any:
            batch, seq_len, _ = x.shape
            q = self.q_proj(x).view(batch, seq_len, self.attention_heads, self.head_dim)
            k = self.k_proj(x).view(batch, seq_len, self.kv_heads, self.head_dim)
            v = self.v_proj(x).view(batch, seq_len, self.kv_heads, self.head_dim)
            past_length = past[0].shape[1] if past is not None else 0
            positions = torch.arange(past_length, past_length + seq_len, device=x.device)
            q = _apply_rope(torch, q, positions)
            k = _apply_rope(torch, k, positions)
            if past is not None:
                k = torch.cat((past[0], k), dim=1)
                v = torch.cat((past[1], v), dim=1)
            present = (k, v) if use_cache else None
            repeat = self.attention_heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            chunk_size = attention_chunk_size or seq_len
            chunks = []
            for start in range(0, seq_len, chunk_size):
                stop = min(start + chunk_size, seq_len)
                # All causally visible keys are retained. Chunking only splits queries.
                def attend(q_chunk, keys, values, key_mask, offset=past_length + start):
                    scores = (q_chunk @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    query_pos = torch.arange(offset, offset + q_chunk.shape[-2], device=x.device)
                    key_pos = torch.arange(keys.shape[-2], device=x.device)
                    allowed = (key_pos[None, :] <= query_pos[:, None])[None, None, :, :]
                    allowed = allowed & key_mask[:, None, None, :].bool()
                    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
                    return torch.softmax(scores, dim=-1) @ values
                args = (q[:, :, start:stop], k[:, :, :past_length + stop], v[:, :, :past_length + stop], attention_mask[:, :past_length + stop])
                if self.training and torch.is_grad_enabled() and seq_len > chunk_size:
                    from torch.utils.checkpoint import checkpoint
                    chunks.append(checkpoint(attend, *args, use_reentrant=False))
                else:
                    chunks.append(attend(*args))
            out = torch.cat(chunks, dim=2).transpose(1, 2).contiguous()
            result = self.o_proj(out.view(batch, seq_len, -1))
            return (result, present) if use_cache else result

    class DecoderBlock(nn.Module):
        def __init__(self, hidden_size: int, attention_heads: int, kv_heads: int, intermediate: int) -> None:
            super().__init__()
            self.attn_norm = RMSNorm(hidden_size)
            self.attn = GQACausalSelfAttention(hidden_size, attention_heads, kv_heads)
            self.mlp_norm = RMSNorm(hidden_size)
            self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)

        def forward(self, x: Any, attention_mask: Any, past: Any = None, use_cache: bool = False) -> Any:
            attended = self.attn(self.attn_norm(x), attention_mask, past, use_cache)
            if use_cache:
                attended, present = attended
            x = x + attended
            hidden = self.mlp_norm(x)
            x = x + self.down_proj(functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))
            return (x, present) if use_cache else x

    class ProofDecoderLM(nn.Module):
        def __init__(self, selected_contract: dict[str, Any]) -> None:
            super().__init__()
            self.vocab_size = selected_contract["tokenizer_vocab_size"]
            hidden_size = selected_contract["hidden_size"]
            self.token_embedding = nn.Embedding(self.vocab_size, hidden_size)
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
            self.apply(self._initialize)

        @staticmethod
        def _initialize(module: Any) -> None:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any = None,
                    past_key_values: Any = None, use_cache: bool = False,
                    last_token_only: bool = False) -> Any:
            past_length = past_key_values[0][0].shape[1] if past_key_values else 0
            if input_ids.shape[1] + past_length > contract["context_tokens"]:
                raise ValueError("input exceeds model context")
            x = self.token_embedding(input_ids)
            presents = []
            for index, block in enumerate(self.blocks):
                if use_cache:
                    past = past_key_values[index] if past_key_values else None
                    x, present = block(x, attention_mask, past, True)
                    presents.append(present)
                elif activation_checkpointing and self.training and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    x = checkpoint(block, x, attention_mask, use_reentrant=False)
                else:
                    x = block(x, attention_mask)
            x = self.norm(x)
            if labels is not None:
                active = labels != -100
                # Ignored prompt/padding positions contribute zero gradient to the head.
                logits = x[active] @ self.token_embedding.weight.transpose(0, 1)
                return functional.cross_entropy(logits, labels[active])
            if last_token_only:
                x = x[:, -1:]
            logits = torch.matmul(x, self.token_embedding.weight.transpose(0, 1))
            return (logits, presents) if use_cache else logits

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
    temporary_state = state_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None,
            "trainer_state": state,
            "config": config,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_random_state": random.getstate(),
        },
        temporary_state,
    )
    temporary_state.replace(state_path)
    _atomic_json(report_path, report)
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
    _atomic_json(latest_path, checkpoint)
    if status == "trained":
        final_path = data_dir / proof_lm_final_checkpoint_path(run_id)
        _atomic_json(final_path, checkpoint)
    elif (data_dir / proof_lm_final_checkpoint_path(run_id)).exists():
        (data_dir / proof_lm_final_checkpoint_path(run_id)).unlink()
    # Keep the current and previous durable states, not an unbounded weight archive.
    for old in sorted(checkpoint_dir.glob("step-*.pt"))[:-2]:
        old.unlink()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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
) -> dict[str, Any]:
    if checkpoint.get("run_id") != run_id:
        raise ValueError("resume checkpoint run_id mismatch")
    if checkpoint.get("trainer_id") != LM_TRAINER_ID:
        raise ValueError("resume checkpoint trainer_id mismatch")
    if checkpoint.get("config_sha256") != config_sha256:
        raise ValueError("resume checkpoint config mismatch")
    state_path = _data_path(data_dir, Path(str(checkpoint.get("state_path"))))
    if checkpoint.get("state_sha256") != _sha256(state_path):
        raise ValueError("resume checkpoint state sha256 mismatch")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if _resume_config_sha256(state["config"]) != config_sha256:
        raise ValueError("resume checkpoint embedded config mismatch")
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if device == "cuda" and state.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return state["trainer_state"]


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


def _resume_config_sha256(config: dict[str, Any]) -> str:
    # Limits cap a deterministic prefix; increasing them must preserve prior work.
    return _stable_sha256({
        key: value for key, value in config.items()
        if key not in {"device", "max_steps", "max_records", "execution"}
    })


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
        "learning_rate_schedule": "constant",
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


def _configure_device_runtime(torch: Any, device: str) -> None:
    if device == "cuda" and torch.cuda.get_device_capability()[0] < 8:
        # Recent PyTorch eager bmm dispatch can select Triton even on older GPUs.
        # Keep the numerically equivalent compiled CUDA path on these devices.
        native = getattr(torch.backends, "python_native", None)
        if native is not None and hasattr(native, "triton"):
            native.triton.enabled = False


def _runtime_info(torch: Any, device: str, model: Any) -> dict[str, Any]:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "device": device,
            "device_name": torch.cuda.get_device_name() if device == "cuda" else device,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "training_code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "implementation_hashes_are_authoritative": True,
            "python_native_triton_enabled": getattr(
                getattr(getattr(torch.backends, "python_native", None), "triton", None), "enabled", None)}


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
    train.add_argument("--record-id", action="append")
    train.add_argument("--attention-chunk-size", type=int, default=256)
    train.add_argument("--no-activation-checkpointing", action="store_true")
    train.add_argument("--checkpoint-every", type=int, default=100)
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
            record_ids=args.record_id,
            attention_chunk_size=args.attention_chunk_size,
            activation_checkpointing=not args.no_activation_checkpointing,
            checkpoint_every=args.checkpoint_every,
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

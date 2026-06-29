"""Dependency-free SFT trainer smoke for proof-model sequence artifacts.

This module is intentionally tiny. It consumes the same sequence materializer
that the real proof trainer must use, updates a small deterministic hashed
weight vector, and writes a resumable checkpoint. It is not a language model and
does not establish model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from gitctx.proof_sequences import (
    SEQUENCE_MATERIALIZATION_ID,
    materialize_training_sequence,
    proof_sequence_metadata_path,
    proof_sequence_report_path,
)
from gitctx.proof_train import DEFAULT_RUN_ID, SCHEMA_VERSION, proof_train_sequence_plan_path

TRAIN_RUN_DIR = Path("artifacts/train-runs")
DEFAULT_TRAIN_SPLIT = "DEV"
DEFAULT_MAX_RECORDS = 64
DEFAULT_MODEL_BUCKETS = 256
DEFAULT_LEARNING_RATE_UNITS = 3
LOSS_SCALE_UNITS = 1_000
SFT_SMOKE_TRAINER_ID = "gctx1-proof-sft-smoke-v0"


def proof_sft_smoke_report_path(run_id: str) -> Path:
    """Return the proof SFT smoke report path for a run id."""

    _validate_identifier(run_id, "run_id")
    return TRAIN_RUN_DIR / f"{run_id}.sft-smoke.report.json"


def proof_sft_smoke_checkpoint_path(run_id: str) -> Path:
    """Return the proof SFT smoke checkpoint path for a run id."""

    _validate_identifier(run_id, "run_id")
    return TRAIN_RUN_DIR / f"{run_id}.sft-smoke.checkpoint.json"


def run_proof_sft_smoke(
    data_dir: str | Path,
    *,
    handoff_path: str | Path,
    run_id: str = DEFAULT_RUN_ID,
    train_split: str = DEFAULT_TRAIN_SPLIT,
    max_records: int = DEFAULT_MAX_RECORDS,
    model_buckets: int = DEFAULT_MODEL_BUCKETS,
    learning_rate_units: int = DEFAULT_LEARNING_RATE_UNITS,
    stop_after_records: int | None = None,
    resume: bool = False,
    write: bool = False,
) -> dict[str, Any]:
    """Run a deterministic checkpoint/resume smoke over materialized SFT records."""

    _validate_identifier(run_id, "run_id")
    _validate_train_split(train_split)
    if max_records < 1:
        raise ValueError("max_records must be positive")
    if model_buckets < 1:
        raise ValueError("model_buckets must be positive")
    if learning_rate_units < 1:
        raise ValueError("learning_rate_units must be positive")
    if stop_after_records is not None and stop_after_records < 1:
        raise ValueError("stop_after_records must be positive")

    data_dir = Path(data_dir)
    selected_handoff_path = _data_path(data_dir, Path(handoff_path))
    handoff = _load_json(selected_handoff_path)
    config = {
        "trainer_id": SFT_SMOKE_TRAINER_ID,
        "run_id": run_id,
        "train_split": train_split,
        "max_records": max_records,
        "model_buckets": model_buckets,
        "learning_rate_units": learning_rate_units,
        "sequence_materialization_id": SEQUENCE_MATERIALIZATION_ID,
    }
    config_sha = _stable_sha256(config)
    blockers = _input_blockers(data_dir, handoff, selected_handoff_path, run_id)
    state = _initial_state(model_buckets=model_buckets)
    resumed_from_checkpoint = False
    checkpoint_path = data_dir / proof_sft_smoke_checkpoint_path(run_id)
    if resume and not checkpoint_path.exists():
        raise ValueError("resume requested but checkpoint is missing")
    if resume:
        checkpoint = _load_json(checkpoint_path)
        _validate_resume_checkpoint(
            checkpoint,
            run_id=run_id,
            config_sha256=config_sha,
            model_buckets=model_buckets,
        )
        state = _state_from_checkpoint(checkpoint)
        resumed_from_checkpoint = True

    selected_sequences: list[dict[str, Any]] = []
    if not blockers:
        selected_sequences, blockers = _load_train_sequences(
            data_dir,
            handoff=handoff,
            run_id=run_id,
            train_split=train_split,
            max_records=max_records,
        )
    if state["record_cursor"] > len(selected_sequences):
        blockers.append("checkpoint cursor is beyond selected training records")

    processed_this_run = 0
    if not blockers:
        target_cursor = len(selected_sequences)
        if stop_after_records is not None:
            target_cursor = min(target_cursor, state["record_cursor"] + stop_after_records)
        for sequence in selected_sequences[state["record_cursor"]:target_cursor]:
            update = _train_sequence(
                state,
                sequence,
                learning_rate_units=learning_rate_units,
            )
            state["record_cursor"] += 1
            state["completed_records"] += 1
            state["optimizer_steps"] += update["optimizer_steps"]
            state["input_tokens"] += sequence["input_length"]
            state["loss_tokens"] += update["loss_tokens"]
            state["loss_units"] += update["loss_units"]
            state["last_record_id"] = sequence["record_id"]
            processed_this_run += 1

    status = _status_for(blockers, state["record_cursor"], len(selected_sequences))
    checkpoint = _checkpoint(
        run_id=run_id,
        status=status,
        config=config,
        config_sha256=config_sha,
        report_sha256=None,
        state=state,
    )
    report = _report(
        run_id=run_id,
        status=status,
        blockers=blockers,
        config=config,
        config_sha256=config_sha,
        handoff_path=selected_handoff_path,
        data_dir=data_dir,
        handoff=handoff,
        state=state,
        selected_train_records=len(selected_sequences),
        processed_this_run=processed_this_run,
        resumed_from_checkpoint=resumed_from_checkpoint,
    )
    if write:
        _write_artifacts(data_dir, run_id=run_id, report=report, checkpoint=checkpoint)
    _print_report(report)
    return report


def validate_proof_sft_smoke(data_dir: str | Path, *, run_id: str) -> dict[str, Any]:
    """Validate a proof SFT smoke report and checkpoint."""

    _validate_identifier(run_id, "run_id")
    data_dir = Path(data_dir)
    report_path = data_dir / proof_sft_smoke_report_path(run_id)
    checkpoint_path = data_dir / proof_sft_smoke_checkpoint_path(run_id)
    report = _load_json(report_path)
    checkpoint = _load_json(checkpoint_path)
    errors: list[str] = []
    if report.get("trainer_id") != SFT_SMOKE_TRAINER_ID:
        errors.append("unexpected report trainer_id")
    if checkpoint.get("trainer_id") != SFT_SMOKE_TRAINER_ID:
        errors.append("unexpected checkpoint trainer_id")
    if report.get("run_id") != run_id:
        errors.append("report run_id mismatch")
    if checkpoint.get("run_id") != run_id:
        errors.append("checkpoint run_id mismatch")
    if report.get("status") not in {"trained", "partial"}:
        errors.append("report status is not trained or partial")
    if checkpoint.get("status") != report.get("status"):
        errors.append("checkpoint status mismatch")
    if checkpoint.get("contains_model_weights") is not True:
        errors.append("checkpoint must contain smoke model weights")
    if checkpoint.get("report_sha256") != _sha256(report_path):
        errors.append("checkpoint report_sha256 does not match report")
    model = checkpoint.get("model", {})
    weights = model.get("weight_units") if isinstance(model, dict) else None
    if not isinstance(weights, list) or not all(isinstance(value, int) for value in weights):
        errors.append("checkpoint model weights are malformed")
    elif model.get("weight_sha256") != _hash_ints(weights):
        errors.append("checkpoint model weight hash mismatch")
    if checkpoint.get("config_sha256") != report.get("config_sha256"):
        errors.append("checkpoint config_sha256 mismatch")

    validation = {
        "run_id": run_id,
        "valid": not errors,
        "errors": errors,
        "report_path": str(proof_sft_smoke_report_path(run_id)),
        "checkpoint_path": str(proof_sft_smoke_checkpoint_path(run_id)),
    }
    for key, value in validation.items():
        if key != "errors":
            print(key, value)
    for error in errors:
        print("error", error)
    if errors:
        raise SystemExit(1)
    return validation


def _input_blockers(
    data_dir: Path,
    handoff: dict[str, Any],
    handoff_path: Path,
    run_id: str,
) -> list[str]:
    blockers: list[str] = []
    if handoff.get("status") != "ready_for_training":
        blockers.append("handoff status is not ready_for_training")
    for name in ("training_artifact", "tokenizer"):
        path = _input_path(data_dir, handoff, name, blockers)
        if path is None:
            continue
        expected_sha = _input_sha(handoff, name)
        actual_sha = _sha256(path)
        if not path.exists():
            blockers.append(f"input {name}: missing {path.as_posix()}")
        elif isinstance(expected_sha, str) and actual_sha != expected_sha:
            blockers.append(f"input {name}: sha256 mismatch")
    sequence_report_path = data_dir / proof_sequence_report_path(run_id)
    sequence_metadata_path = data_dir / proof_sequence_metadata_path(run_id)
    sequence_plan_path = data_dir / proof_train_sequence_plan_path(run_id)
    if not sequence_report_path.exists():
        blockers.append("sequence materialization report is missing")
    else:
        sequence_report = _load_json(sequence_report_path)
        if sequence_report.get("status") != "materialized":
            blockers.append("sequence materialization report is not materialized")
        if sequence_report.get("outputs", {}).get("metadata_sha256") != _sha256(sequence_metadata_path):
            blockers.append("sequence metadata sha256 does not match report")
    if not sequence_metadata_path.exists():
        blockers.append("sequence metadata is missing")
    if not sequence_plan_path.exists():
        blockers.append("sequence plan is missing")
    if not handoff_path.exists():
        blockers.append("handoff is missing")
    return blockers


def _load_train_sequences(
    data_dir: Path,
    *,
    handoff: dict[str, Any],
    run_id: str,
    train_split: str,
    max_records: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    training_path = _input_path(data_dir, handoff, "training_artifact", blockers)
    tokenizer_path = _input_path(data_dir, handoff, "tokenizer", blockers)
    if training_path is None or tokenizer_path is None:
        return [], blockers
    tokenizer = _load_json(tokenizer_path)
    sequence_plan = list(_iter_jsonl(data_dir / proof_train_sequence_plan_path(run_id)))
    plan_by_record_id = {
        record["record_id"]: record
        for record in sequence_plan
        if isinstance(record.get("record_id"), str)
    }
    metadata_by_record_id = {
        record["record_id"]: record
        for record in _iter_jsonl(data_dir / proof_sequence_metadata_path(run_id))
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
        try:
            sequence = materialize_training_sequence(record, plan_record, tokenizer)
        except ValueError as exc:
            blockers.append(f"{record_id}: {exc}")
            continue
        metadata = metadata_by_record_id.get(record_id)
        if metadata is None:
            blockers.append(f"{record_id}: missing sequence metadata")
            continue
        if not _sequence_matches_metadata(sequence, metadata):
            blockers.append(f"{record_id}: materialized sequence does not match metadata")
            continue
        sequences.append(sequence)
        if len(sequences) >= max_records:
            break
    if not sequences and not blockers:
        blockers.append(f"no kept training records found for split {train_split}")
    return sequences, blockers


def _train_sequence(
    state: dict[str, Any],
    sequence: dict[str, Any],
    *,
    learning_rate_units: int,
) -> dict[str, int]:
    weights = state["weights"]
    loss_units = 0
    loss_tokens = 0
    optimizer_steps = 0
    input_ids = sequence["input_ids"]
    loss_mask = sequence["loss_mask"]
    for position, active in enumerate(loss_mask):
        if not active:
            continue
        previous_id = input_ids[position - 1] if position > 0 else 0
        target_id = input_ids[position]
        bucket = _feature_bucket(previous_id, target_id, position, len(weights))
        target_sign = 1 if target_id % 2 else -1
        margin_units = target_sign * weights[bucket]
        token_loss = max(0, LOSS_SCALE_UNITS - margin_units)
        loss_units += token_loss
        loss_tokens += 1
        if token_loss > 0:
            weights[bucket] += learning_rate_units * target_sign
            optimizer_steps += 1
    return {
        "loss_units": loss_units,
        "loss_tokens": loss_tokens,
        "optimizer_steps": optimizer_steps,
    }


def _checkpoint(
    *,
    run_id: str,
    status: str,
    config: dict[str, Any],
    config_sha256: str,
    report_sha256: str | None,
    state: dict[str, Any],
) -> dict[str, Any]:
    weights = state["weights"]
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": "proof_sft_smoke_checkpoint",
        "trainer_id": SFT_SMOKE_TRAINER_ID,
        "run_id": run_id,
        "status": status,
        "contains_model_weights": True,
        "config": config,
        "config_sha256": config_sha256,
        "report_sha256": report_sha256,
        "record_cursor": state["record_cursor"],
        "optimizer": {
            "kind": "deterministic_integer_margin_updates",
            "learning_rate_units": config["learning_rate_units"],
            "optimizer_steps": state["optimizer_steps"],
        },
        "accounting": _state_accounting(state),
        "model": {
            "model_kind": "dependency_free_hashed_token_margin_smoke_model",
            "model_buckets": len(weights),
            "weight_units": weights,
            "weight_sha256": _hash_ints(weights),
        },
        "resume_policy": (
            "Resume by loading record_cursor, optimizer accounting, and weight_units, "
            "then continue over the same deterministic selected DEV sequence order."
        ),
    }


def _report(
    *,
    run_id: str,
    status: str,
    blockers: list[str],
    config: dict[str, Any],
    config_sha256: str,
    handoff_path: Path,
    data_dir: Path,
    handoff: dict[str, Any],
    state: dict[str, Any],
    selected_train_records: int,
    processed_this_run: int,
    resumed_from_checkpoint: bool,
) -> dict[str, Any]:
    accounting = _state_accounting(state)
    return {
        "schema_version": SCHEMA_VERSION,
        "trainer_id": SFT_SMOKE_TRAINER_ID,
        "run_id": run_id,
        "status": status,
        "blockers": blockers,
        "config": config,
        "config_sha256": config_sha256,
        "handoff": {
            "path": _display_path(handoff_path, base=data_dir),
            "id": handoff.get("id"),
            "sha256": _sha256(handoff_path),
        },
        "inputs": {
            "sequence_plan": {
                "path": str(proof_train_sequence_plan_path(run_id)),
                "sha256": _sha256(data_dir / proof_train_sequence_plan_path(run_id)),
            },
            "sequence_metadata": {
                "path": str(proof_sequence_metadata_path(run_id)),
                "sha256": _sha256(data_dir / proof_sequence_metadata_path(run_id)),
            },
            "sequence_materialization_report": {
                "path": str(proof_sequence_report_path(run_id)),
                "sha256": _sha256(data_dir / proof_sequence_report_path(run_id)),
            },
        },
        "runtime": {
            "python": sys.version.split()[0],
        },
        "training": {
            "train_split": config["train_split"],
            "selected_train_records": selected_train_records,
            "processed_this_run": processed_this_run,
            "completed_train_records": state["record_cursor"],
            "max_records": config["max_records"],
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "last_record_id": state["last_record_id"],
            **accounting,
        },
        "outputs": {
            "report_path": str(proof_sft_smoke_report_path(run_id)),
            "checkpoint_path": str(proof_sft_smoke_checkpoint_path(run_id)),
        },
        "claim_policy": (
            "This dependency-free smoke updates weights and proves checkpoint/resume "
            "plumbing only. It is not the 60M-100M proof language model and not a "
            "model-quality claim."
        ),
    }


def _write_artifacts(
    data_dir: Path,
    *,
    run_id: str,
    report: dict[str, Any],
    checkpoint: dict[str, Any],
) -> None:
    report_path = data_dir / proof_sft_smoke_report_path(run_id)
    checkpoint_path = data_dir / proof_sft_smoke_checkpoint_path(run_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint["report_sha256"] = _sha256(report_path)
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _initial_state(*, model_buckets: int) -> dict[str, Any]:
    return {
        "record_cursor": 0,
        "completed_records": 0,
        "optimizer_steps": 0,
        "input_tokens": 0,
        "loss_tokens": 0,
        "loss_units": 0,
        "last_record_id": None,
        "weights": [0 for _ in range(model_buckets)],
    }


def _state_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    model = checkpoint.get("model", {})
    accounting = checkpoint.get("accounting", {})
    optimizer = checkpoint.get("optimizer", {})
    return {
        "record_cursor": _int_value(checkpoint.get("record_cursor")),
        "completed_records": _int_value(accounting.get("completed_records")),
        "optimizer_steps": _int_value(optimizer.get("optimizer_steps")),
        "input_tokens": _int_value(accounting.get("input_tokens")),
        "loss_tokens": _int_value(accounting.get("loss_tokens")),
        "loss_units": _int_value(accounting.get("loss_units")),
        "last_record_id": accounting.get("last_record_id"),
        "weights": list(model.get("weight_units", [])),
    }


def _state_accounting(state: dict[str, Any]) -> dict[str, Any]:
    loss_tokens = state["loss_tokens"]
    return {
        "completed_records": state["completed_records"],
        "optimizer_steps": state["optimizer_steps"],
        "input_tokens": state["input_tokens"],
        "loss_tokens": loss_tokens,
        "loss_units": state["loss_units"],
        "average_loss_units": round(state["loss_units"] / loss_tokens, 6) if loss_tokens else 0.0,
        "last_record_id": state["last_record_id"],
    }


def _validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    run_id: str,
    config_sha256: str,
    model_buckets: int,
) -> None:
    if checkpoint.get("run_id") != run_id:
        raise ValueError("resume checkpoint run_id mismatch")
    if checkpoint.get("trainer_id") != SFT_SMOKE_TRAINER_ID:
        raise ValueError("resume checkpoint trainer_id mismatch")
    if checkpoint.get("status") not in {"partial", "trained"}:
        raise ValueError("resume checkpoint status must be partial or trained")
    if checkpoint.get("config_sha256") != config_sha256:
        raise ValueError("resume checkpoint config mismatch")
    model = checkpoint.get("model", {})
    weights = model.get("weight_units") if isinstance(model, dict) else None
    if (
        not isinstance(weights, list)
        or len(weights) != model_buckets
        or not all(isinstance(value, int) for value in weights)
    ):
        raise ValueError("resume checkpoint weight shape mismatch")
    if model.get("weight_sha256") != _hash_ints(weights):
        raise ValueError("resume checkpoint weight hash mismatch")


def _sequence_matches_metadata(sequence: dict[str, Any], metadata: dict[str, Any]) -> bool:
    return (
        sequence.get("input_length") == metadata.get("input_length")
        and sequence.get("loss_tokens") == metadata.get("loss_tokens")
        and _hash_ints(sequence["input_ids"]) == metadata.get("input_ids_sha256")
        and _hash_ints(sequence["loss_mask"]) == metadata.get("loss_mask_sha256")
    )


def _status_for(blockers: list[str], cursor: int, selected_records: int) -> str:
    if blockers:
        return "blocked"
    return "trained" if cursor >= selected_records else "partial"


def _feature_bucket(previous_id: int, target_id: int, position: int, model_buckets: int) -> int:
    payload = f"{previous_id}:{target_id}:{position}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % model_buckets


def _input_path(
    data_dir: Path,
    handoff: dict[str, Any],
    input_name: str,
    blockers: list[str],
) -> Path | None:
    inputs = handoff.get("inputs", {})
    entry = inputs.get(input_name) if isinstance(inputs, dict) else None
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        blockers.append(f"input {input_name}: missing from handoff")
        return None
    return _data_path(data_dir, Path(entry["path"]))


def _input_sha(handoff: dict[str, Any], input_name: str) -> str | None:
    inputs = handoff.get("inputs", {})
    entry = inputs.get(input_name) if isinstance(inputs, dict) else None
    value = entry.get("sha256") if isinstance(entry, dict) else None
    return value if isinstance(value, str) else None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _validate_split(value: str) -> None:
    if value not in {"DEV", "REPORT", "HELD_OUT"}:
        raise ValueError("split must be DEV, REPORT, or HELD_OUT")


def _validate_train_split(value: str) -> None:
    _validate_split(value)
    if value != DEFAULT_TRAIN_SPLIT:
        raise ValueError("proof SFT smoke may train only on DEV")


def _print_report(report: dict[str, Any]) -> None:
    print("run_id", report["run_id"])
    print("status", report["status"])
    training = report["training"]
    for key in (
        "train_split",
        "selected_train_records",
        "processed_this_run",
        "completed_train_records",
        "input_tokens",
        "loss_tokens",
        "average_loss_units",
    ):
        print(key, training[key])
    for blocker in report["blockers"]:
        print("blocker", blocker)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a dependency-free proof SFT smoke trainer.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--handoff", required=True)
    train.add_argument("--run-id", default=DEFAULT_RUN_ID)
    train.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    train.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    train.add_argument("--model-buckets", type=int, default=DEFAULT_MODEL_BUCKETS)
    train.add_argument("--learning-rate-units", type=int, default=DEFAULT_LEARNING_RATE_UNITS)
    train.add_argument("--stop-after-records", type=int)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--write", action="store_true")
    train.add_argument("--fail-on-blocked", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--run-id", default=DEFAULT_RUN_ID)

    args = parser.parse_args(argv)
    if args.command == "train":
        report = run_proof_sft_smoke(
            args.data_dir,
            handoff_path=args.handoff,
            run_id=args.run_id,
            train_split=args.train_split,
            max_records=args.max_records,
            model_buckets=args.model_buckets,
            learning_rate_units=args.learning_rate_units,
            stop_after_records=args.stop_after_records,
            resume=args.resume,
            write=args.write,
        )
        if args.fail_on_blocked and report["status"] == "blocked":
            return 1
    elif args.command == "validate":
        validate_proof_sft_smoke(args.data_dir, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

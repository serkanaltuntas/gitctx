"""Proof-model trainer entrypoint and dry-run artifact contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

from gitctx.proof_tokenizer import record_tokens

TRAIN_RUN_DIR = Path("artifacts/train-runs")
DEFAULT_RUN_ID = "gctx1-proof-model.v0.dry-run"
SCHEMA_VERSION = "v0"
SEQUENCE_POLICY_ID = "gctx1-proof-sequence-policy-v0"
DEFAULT_MAX_RAW_RECORD_TOKENS = 65_536
DEFAULT_LONG_RECORD_SAMPLE_LIMIT = 20


def proof_train_report_path(run_id: str) -> Path:
    """Return the proof-trainer report path for a run id."""

    _validate_identifier(run_id, "run_id")
    return TRAIN_RUN_DIR / f"{run_id}.report.json"


def proof_train_checkpoint_path(run_id: str) -> Path:
    """Return the proof-trainer checkpoint path for a run id."""

    _validate_identifier(run_id, "run_id")
    return TRAIN_RUN_DIR / f"{run_id}.checkpoint.json"


def proof_train_sequence_plan_path(run_id: str) -> Path:
    """Return the proof-trainer sequence-plan path for a run id."""

    _validate_identifier(run_id, "run_id")
    return TRAIN_RUN_DIR / f"{run_id}.sequence-plan.jsonl"


def dry_run_proof_training(
    data_dir: str | Path,
    *,
    handoff_path: str | Path,
    run_id: str = DEFAULT_RUN_ID,
    max_records_per_split: int | None = None,
    max_raw_record_tokens: int = DEFAULT_MAX_RAW_RECORD_TOKENS,
    long_record_sample_limit: int = DEFAULT_LONG_RECORD_SAMPLE_LIMIT,
    require_torch: bool = False,
    write: bool = False,
) -> dict[str, Any]:
    """Validate proof-training inputs and write a no-weights checkpoint skeleton."""

    _validate_identifier(run_id, "run_id")
    if max_records_per_split is not None and max_records_per_split < 1:
        raise ValueError("max_records_per_split must be positive")
    if max_raw_record_tokens < 1:
        raise ValueError("max_raw_record_tokens must be positive")
    if long_record_sample_limit < 0:
        raise ValueError("long_record_sample_limit must not be negative")

    data_dir = Path(data_dir)
    selected_handoff_path = _data_path(data_dir, Path(handoff_path))
    handoff = json.loads(selected_handoff_path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    if handoff.get("status") != "ready_for_training":
        blockers.append("handoff status is not ready_for_training")

    input_checks = _verify_inputs(data_dir, handoff.get("inputs", {}))
    blockers.extend(input_checks["errors"])
    torch_status = _torch_status()
    if require_torch and not torch_status["available"]:
        blockers.append("torch is required but not importable")

    tokenizer = _load_input_json(data_dir, handoff, "tokenizer", blockers)
    training_path = _input_path(data_dir, handoff, "training_artifact", blockers)
    tokenization = _empty_tokenization_summary()
    sequence_plan_records: list[dict[str, Any]] = []
    if tokenizer is not None and training_path is not None and training_path.exists():
        tokenization, sequence_plan_records = _tokenization_summary(
            training_path,
            tokenizer=tokenizer,
            context_tokens=_context_tokens(handoff),
            max_records_per_split=max_records_per_split,
            max_raw_record_tokens=max_raw_record_tokens,
            long_record_sample_limit=long_record_sample_limit,
        )
        blockers.extend(_sequence_policy_blockers(tokenization, handoff=handoff))

    status = "dry_run_passed" if not blockers else "blocked"
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_kind": "proof_training_dry_run",
        "status": status,
        "blockers": blockers,
        "handoff": {
            "path": _display_path(selected_handoff_path, base=data_dir),
            "id": handoff.get("id"),
            "status": handoff.get("status"),
            "sha256": _sha256(selected_handoff_path),
        },
        "code": handoff.get("code", {}),
        "training_contract": handoff.get("training_contract", {}),
        "input_checks": input_checks["files"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch_status,
        },
        "tokenization": tokenization,
        "outputs": {
            "checkpoint_path": str(proof_train_checkpoint_path(run_id)),
            "report_path": str(proof_train_report_path(run_id)),
            "sequence_plan_path": str(proof_train_sequence_plan_path(run_id)),
            "sequence_plan_sha256": None,
        },
        "claim_policy": (
            "Dry-run artifacts validate trainer inputs and accounting only; "
            "they are not model-quality evidence."
        ),
    }
    checkpoint = _checkpoint_skeleton(report, handoff=handoff)
    if write:
        _write_run_artifacts(
            data_dir,
            run_id=run_id,
            report=report,
            checkpoint=checkpoint,
            sequence_plan_records=sequence_plan_records,
        )
    _print_report(report)
    return report


def validate_proof_training_run(data_dir: str | Path, *, run_id: str) -> dict[str, Any]:
    """Validate dry-run trainer report and checkpoint artifacts."""

    _validate_identifier(run_id, "run_id")
    data_dir = Path(data_dir)
    report_path = data_dir / proof_train_report_path(run_id)
    checkpoint_path = data_dir / proof_train_checkpoint_path(run_id)
    sequence_plan_path = data_dir / proof_train_sequence_plan_path(run_id)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if report.get("run_id") != run_id:
        errors.append("report run_id mismatch")
    if report.get("status") != "dry_run_passed":
        errors.append("report status is not dry_run_passed")
    if checkpoint.get("run_id") != run_id:
        errors.append("checkpoint run_id mismatch")
    if checkpoint.get("contains_model_weights") is not False:
        errors.append("dry-run checkpoint must not contain model weights")
    if checkpoint.get("report_sha256") != _sha256(report_path):
        errors.append("checkpoint report_sha256 does not match report")
    if not sequence_plan_path.exists():
        errors.append("sequence plan is missing")
        sequence_plan_records = []
    else:
        sequence_plan_records = list(_iter_jsonl(sequence_plan_path))
        expected_plan_sha = report.get("outputs", {}).get("sequence_plan_sha256")
        if isinstance(expected_plan_sha, str) and expected_plan_sha != _sha256(sequence_plan_path):
            errors.append("sequence plan sha256 does not match report")
        processed_records = report.get("tokenization", {}).get("processed_records")
        if processed_records != len(sequence_plan_records):
            errors.append("sequence plan record count does not match tokenization report")
    validation = {
        "run_id": run_id,
        "valid": not errors,
        "errors": errors,
        "report_path": str(proof_train_report_path(run_id)),
        "checkpoint_path": str(proof_train_checkpoint_path(run_id)),
        "sequence_plan_path": str(proof_train_sequence_plan_path(run_id)),
    }
    for key, value in validation.items():
        if key != "errors":
            print(key, value)
    for error in errors:
        print("error", error)
    if errors:
        raise SystemExit(1)
    return validation


def _verify_inputs(data_dir: Path, inputs: Any) -> dict[str, Any]:
    if not isinstance(inputs, dict):
        return {"files": {}, "errors": ["handoff inputs must be an object"]}
    files: dict[str, Any] = {}
    errors: list[str] = []
    for name, entry in sorted(inputs.items()):
        if not isinstance(entry, dict):
            errors.append(f"input {name}: entry must be an object")
            continue
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            errors.append(f"input {name}: path is missing")
            continue
        path = _data_path(data_dir, Path(path_value))
        expected_sha = entry.get("sha256")
        actual_sha = _sha256(path)
        ok = path.exists() and (not isinstance(expected_sha, str) or actual_sha == expected_sha)
        files[name] = {
            "path": _display_path(path, base=data_dir),
            "exists": path.exists(),
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "ok": ok,
        }
        if not path.exists():
            errors.append(f"input {name}: missing {files[name]['path']}")
        elif isinstance(expected_sha, str) and actual_sha != expected_sha:
            errors.append(f"input {name}: sha256 mismatch")
    return {"files": files, "errors": errors}


def _load_input_json(
    data_dir: Path,
    handoff: dict[str, Any],
    input_name: str,
    blockers: list[str],
) -> dict[str, Any] | None:
    path = _input_path(data_dir, handoff, input_name, blockers)
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def _tokenization_summary(
    training_path: Path,
    *,
    tokenizer: dict[str, Any],
    context_tokens: int,
    max_records_per_split: int | None,
    max_raw_record_tokens: int,
    long_record_sample_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    vocab = tokenizer.get("vocab")
    if not isinstance(vocab, list):
        raise ValueError("tokenizer vocab must be a list")
    token_to_id = {
        entry["token"]: entry["id"]
        for entry in vocab
        if isinstance(entry, dict) and isinstance(entry.get("token"), str)
    }
    split_counts = _split_counter()
    split_tokens = _split_counter()
    split_unknown = _split_counter()
    split_sequences = _split_counter()
    split_max_record_tokens = _split_counter()
    split_full_records = _split_counter()
    split_truncated_records = _split_counter()
    split_excluded_records = _split_counter()
    split_kept_trainer_tokens = _split_counter()
    split_truncated_raw_tokens = _split_counter()
    split_excluded_raw_tokens = _split_counter()
    total_records = 0
    processed_records = 0
    sequence_plan_records: list[dict[str, Any]] = []
    long_records: list[dict[str, Any]] = []

    for record in _iter_jsonl(training_path):
        split = record["data_split"]
        total_records += 1
        if split not in split_counts:
            split_counts[split] = 0
            split_tokens[split] = 0
            split_unknown[split] = 0
            split_sequences[split] = 0
            split_max_record_tokens[split] = 0
            split_full_records[split] = 0
            split_truncated_records[split] = 0
            split_excluded_records[split] = 0
            split_kept_trainer_tokens[split] = 0
            split_truncated_raw_tokens[split] = 0
            split_excluded_raw_tokens[split] = 0
        if max_records_per_split is not None and split_counts[split] >= max_records_per_split:
            continue
        tokens = record_tokens(record)
        unknown = sum(1 for token in tokens if token not in token_to_id)
        token_count = len(tokens)
        sequence_count = math.ceil(token_count / context_tokens)
        decision, reason = _sequence_decision(
            token_count,
            context_tokens=context_tokens,
            max_raw_record_tokens=max_raw_record_tokens,
        )
        trainer_sequence_tokens = 0 if decision == "exclude_oversize" else min(
            token_count,
            context_tokens,
        )
        split_counts[split] += 1
        split_tokens[split] += token_count
        split_unknown[split] += unknown
        split_sequences[split] += sequence_count
        split_max_record_tokens[split] = max(split_max_record_tokens[split], token_count)
        if decision == "exclude_oversize":
            split_excluded_records[split] += 1
            split_excluded_raw_tokens[split] += token_count
        elif decision == "use_truncated":
            split_truncated_records[split] += 1
            split_truncated_raw_tokens[split] += token_count - context_tokens
            split_kept_trainer_tokens[split] += trainer_sequence_tokens
        else:
            split_full_records[split] += 1
            split_kept_trainer_tokens[split] += trainer_sequence_tokens
        processed_records += 1
        plan_record = _sequence_plan_record(
            record,
            raw_token_count=token_count,
            unknown_token_count=unknown,
            raw_sequences_at_context=sequence_count,
            decision=decision,
            reason=reason,
            trainer_context_tokens=context_tokens,
            trainer_sequence_tokens=trainer_sequence_tokens,
        )
        sequence_plan_records.append(plan_record)
        if token_count > context_tokens:
            long_records.append(plan_record)

    by_split = {}
    for split in sorted(split_counts):
        tokens = split_tokens[split]
        unknown = split_unknown[split]
        by_split[split] = {
            "records": split_counts[split],
            "tokens": tokens,
            "unknown_tokens": unknown,
            "known_token_fraction": round((tokens - unknown) / tokens, 6) if tokens else 0.0,
            "sequences_at_context": split_sequences[split],
            "max_record_tokens": split_max_record_tokens[split],
            "full_records": split_full_records[split],
            "kept_records": split_full_records[split] + split_truncated_records[split],
            "truncated_records": split_truncated_records[split],
            "excluded_records": split_excluded_records[split],
            "kept_trainer_tokens": split_kept_trainer_tokens[split],
            "truncated_raw_tokens": split_truncated_raw_tokens[split],
            "excluded_raw_tokens": split_excluded_raw_tokens[split],
        }
    long_records.sort(key=lambda record: record["raw_token_count"], reverse=True)
    sequence_policy = _sequence_policy_summary(
        by_split,
        context_tokens=context_tokens,
        max_raw_record_tokens=max_raw_record_tokens,
        long_records=long_records[:long_record_sample_limit],
    )
    return {
        "tokenizer_version": tokenizer.get("tokenizer_version"),
        "vocab_size": tokenizer.get("vocab_size"),
        "context_tokens": context_tokens,
        "total_records_seen": total_records,
        "processed_records": processed_records,
        "max_records_per_split": max_records_per_split,
        "sequence_policy": sequence_policy,
        "by_split": by_split,
    }, sequence_plan_records


def _checkpoint_skeleton(report: dict[str, Any], *, handoff: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": "proof_training_dry_run_checkpoint",
        "run_id": report["run_id"],
        "status": report["status"],
        "contains_model_weights": False,
        "report_sha256": None,
        "handoff_id": handoff.get("id"),
        "training_code_revision": report["code"].get("training_code_revision"),
        "tokenization": report["tokenization"],
        "resume_policy": "real trainer checkpoints must include optimizer, scheduler, and RNG state",
    }


def _write_run_artifacts(
    data_dir: Path,
    *,
    run_id: str,
    report: dict[str, Any],
    checkpoint: dict[str, Any],
    sequence_plan_records: list[dict[str, Any]],
) -> None:
    report_path = data_dir / proof_train_report_path(run_id)
    checkpoint_path = data_dir / proof_train_checkpoint_path(run_id)
    sequence_plan_path = data_dir / proof_train_sequence_plan_path(run_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sequence_plan_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in sequence_plan_records),
        encoding="utf-8",
    )
    report["outputs"]["sequence_plan_sha256"] = _sha256(sequence_plan_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint["report_sha256"] = _sha256(report_path)
    checkpoint["sequence_plan_sha256"] = report["outputs"]["sequence_plan_sha256"]
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _empty_tokenization_summary() -> dict[str, Any]:
    return {
        "context_tokens": None,
        "total_records_seen": 0,
        "processed_records": 0,
        "sequence_policy": {
            "id": SEQUENCE_POLICY_ID,
            "status": "not_evaluated",
            "max_raw_record_tokens": None,
            "longest_records": [],
        },
        "by_split": {},
    }


def _context_tokens(handoff: dict[str, Any]) -> int:
    contract = handoff.get("training_contract", {})
    value = contract.get("context_tokens") if isinstance(contract, dict) else None
    return int(value) if isinstance(value, int) and value > 0 else 8192


def _torch_status() -> dict[str, Any]:
    spec = importlib.util.find_spec("torch")
    status: dict[str, Any] = {"available": spec is not None}
    if spec is not None:
        try:
            import torch  # type: ignore[import-not-found]

            status["version"] = torch.__version__
            status["cuda_available"] = bool(torch.cuda.is_available())
            status["mps_available"] = bool(
                getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            )
        except Exception as exc:  # pragma: no cover - defensive environment report
            status["import_error"] = str(exc)
    return status


def _split_counter() -> dict[str, int]:
    return {"DEV": 0, "REPORT": 0, "HELD_OUT": 0}


def _sequence_decision(
    raw_token_count: int,
    *,
    context_tokens: int,
    max_raw_record_tokens: int,
) -> tuple[str, str]:
    if raw_token_count > max_raw_record_tokens:
        return "exclude_oversize", "raw_token_count exceeds max_raw_record_tokens"
    if raw_token_count > context_tokens:
        return "use_truncated", "raw_token_count exceeds context_tokens"
    return "use_full", "raw_token_count fits context_tokens"


def _sequence_plan_record(
    record: dict[str, Any],
    *,
    raw_token_count: int,
    unknown_token_count: int,
    raw_sequences_at_context: int,
    decision: str,
    reason: str,
    trainer_context_tokens: int,
    trainer_sequence_tokens: int,
) -> dict[str, Any]:
    changed_paths = record.get("changed_paths")
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_id": SEQUENCE_POLICY_ID,
        "record_id": record.get("id"),
        "source_diff_id": record.get("source_diff_id"),
        "data_split": record.get("data_split"),
        "source_repo_url": record.get("source_repo_url"),
        "source_commit": record.get("source_commit"),
        "parent_commit": record.get("parent_commit"),
        "diff_sha256": record.get("diff_sha256"),
        "changed_path_count": len(changed_paths) if isinstance(changed_paths, list) else None,
        "raw_token_count": raw_token_count,
        "unknown_token_count": unknown_token_count,
        "raw_sequences_at_context": raw_sequences_at_context,
        "decision": decision,
        "reason": reason,
        "trainer_context_tokens": trainer_context_tokens,
        "trainer_sequence_tokens": trainer_sequence_tokens,
    }


def _sequence_policy_summary(
    by_split: dict[str, dict[str, Any]],
    *,
    context_tokens: int,
    max_raw_record_tokens: int,
    long_records: list[dict[str, Any]],
) -> dict[str, Any]:
    total_excluded = sum(split["excluded_records"] for split in by_split.values())
    report_excluded = by_split.get("REPORT", {}).get("excluded_records", 0)
    return {
        "id": SEQUENCE_POLICY_ID,
        "status": "blocked" if report_excluded else "ready",
        "context_tokens": context_tokens,
        "max_raw_record_tokens": max_raw_record_tokens,
        "trainer_record_policy": (
            "Use one supervised trainer sequence per kept record. Records that fit the "
            "context are used whole; records over context but under the raw-token cap "
            "must be deterministically cropped by the real trainer; records over the "
            "raw-token cap are excluded from the first proof run."
        ),
        "crop_contract": (
            "The real trainer must preserve system instructions and the assistant target, "
            "then allocate remaining context to deterministic user/diff prefix and suffix "
            "tokens."
        ),
        "exclusion_contract": (
            "REPORT exclusions block quality claims. DEV exclusions are allowed only when "
            "the kept DEV count remains above the proof-run minimum."
        ),
        "excluded_records": total_excluded,
        "report_excluded_records": report_excluded,
        "longest_records": long_records,
    }


def _sequence_policy_blockers(tokenization: dict[str, Any], *, handoff: dict[str, Any]) -> list[str]:
    blockers = []
    policy = tokenization.get("sequence_policy", {})
    report_excluded = policy.get("report_excluded_records") if isinstance(policy, dict) else None
    if isinstance(report_excluded, int) and report_excluded > 0:
        blockers.append("sequence policy: REPORT contains excluded oversize records")
    dev_minimum = _minimum_dev_records(handoff)
    dev_summary = tokenization.get("by_split", {}).get("DEV", {})
    kept_dev_records = dev_summary.get("kept_records") if isinstance(dev_summary, dict) else None
    if (
        isinstance(dev_minimum, int)
        and isinstance(kept_dev_records, int)
        and kept_dev_records < dev_minimum
    ):
        blockers.append(
            "sequence policy: kept DEV records fall below minimum "
            f"({kept_dev_records} < {dev_minimum})"
        )
    return blockers


def _minimum_dev_records(handoff: dict[str, Any]) -> int | None:
    readiness = handoff.get("readiness", {})
    gates = readiness.get("gates", {}) if isinstance(readiness, dict) else {}
    dev_gate = gates.get("dev_training_records") if isinstance(gates, dict) else None
    minimum = dev_gate.get("minimum") if isinstance(dev_gate, dict) else None
    return minimum if isinstance(minimum, int) else None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
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


def _validate_identifier(value: str, name: str) -> None:
    if not value.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"{name} must be a stable alphanumeric identifier")


def _print_report(report: dict[str, Any]) -> None:
    print("run_id", report["run_id"])
    print("status", report["status"])
    for split, summary in report["tokenization"]["by_split"].items():
        print(f"{split}_records", summary["records"])
        print(f"{split}_tokens", summary["tokens"])
        print(f"{split}_known_token_fraction", summary["known_token_fraction"])
    for blocker in report["blockers"]:
        print("blocker", blocker)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GCTX proof-trainer setup checks.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("--handoff", required=True)
    dry_run.add_argument("--run-id", default=DEFAULT_RUN_ID)
    dry_run.add_argument("--max-records-per-split", type=int)
    dry_run.add_argument("--max-raw-record-tokens", type=int, default=DEFAULT_MAX_RAW_RECORD_TOKENS)
    dry_run.add_argument(
        "--long-record-sample-limit",
        type=int,
        default=DEFAULT_LONG_RECORD_SAMPLE_LIMIT,
    )
    dry_run.add_argument("--require-torch", action="store_true")
    dry_run.add_argument("--write", action="store_true")
    dry_run.add_argument("--fail-on-blocked", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--run-id", default=DEFAULT_RUN_ID)

    args = parser.parse_args(argv)
    if args.command == "dry-run":
        report = dry_run_proof_training(
            args.data_dir,
            handoff_path=args.handoff,
            run_id=args.run_id,
            max_records_per_split=args.max_records_per_split,
            max_raw_record_tokens=args.max_raw_record_tokens,
            long_record_sample_limit=args.long_record_sample_limit,
            require_torch=args.require_torch,
            write=args.write,
        )
        if args.fail_on_blocked and report["status"] != "dry_run_passed":
            return 1
    elif args.command == "validate":
        validate_proof_training_run(args.data_dir, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

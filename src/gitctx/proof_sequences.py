"""Materialize deterministic proof-trainer sequences from a sequence plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from gitctx.proof_tokenizer import record_tokens, tokenize_text
from gitctx.proof_train import (
    DEFAULT_RUN_ID,
    SCHEMA_VERSION,
    proof_train_report_path,
    proof_train_sequence_plan_path,
)

TRAIN_RUN_DIR = Path("artifacts/train-runs")
SEQUENCE_SCHEMA_VERSION = "v0"
SEQUENCE_MATERIALIZATION_ID = "gctx1-proof-sequence-materialization-v0"


def proof_sequence_metadata_path(run_id: str) -> Path:
    """Return the proof-trainer sequence metadata path for a run id."""

    _validate_identifier(run_id, "run_id")
    return TRAIN_RUN_DIR / f"{run_id}.sequence-metadata.jsonl"


def proof_sequence_report_path(run_id: str) -> Path:
    """Return the proof-trainer sequence materialization report path for a run id."""

    _validate_identifier(run_id, "run_id")
    return TRAIN_RUN_DIR / f"{run_id}.sequence-materialization.report.json"


def materialize_training_sequence(
    record: dict[str, Any],
    plan_record: dict[str, Any],
    tokenizer: dict[str, Any],
    *,
    context_tokens: int | None = None,
) -> dict[str, Any]:
    """Materialize one deterministic trainer sequence from a reviewed SFT record."""

    decision = plan_record.get("decision")
    if decision == "exclude_oversize":
        raise ValueError("excluded records are not materialized")
    if decision not in {"use_full", "use_truncated"}:
        raise ValueError(f"unknown sequence decision: {decision}")

    selected_context_tokens = _context_tokens(plan_record, context_tokens=context_tokens)
    token_to_id = _token_to_id(tokenizer)
    unknown_id = _unknown_id(tokenizer)
    sequence = _render_sequence(
        record,
        decision=decision,
        context_tokens=selected_context_tokens,
    )
    input_ids = [token_to_id.get(token, unknown_id) for token in sequence["tokens"]]
    unknown_tokens = sum(1 for token in sequence["tokens"] if token not in token_to_id)
    expected_tokens = plan_record.get("trainer_sequence_tokens")
    if isinstance(expected_tokens, int) and len(input_ids) != expected_tokens:
        raise ValueError(
            "materialized token count does not match sequence plan "
            f"({len(input_ids)} != {expected_tokens})"
        )
    if len(input_ids) > selected_context_tokens:
        raise ValueError("materialized sequence exceeds context_tokens")
    if not any(sequence["loss_mask"]):
        raise ValueError("materialized sequence has no loss tokens")
    return {
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "materialization_id": SEQUENCE_MATERIALIZATION_ID,
        "record_id": record.get("id"),
        "source_diff_id": record.get("source_diff_id"),
        "data_split": record.get("data_split"),
        "decision": decision,
        "context_tokens": selected_context_tokens,
        "tokens": sequence["tokens"],
        "input_ids": input_ids,
        "loss_mask": sequence["loss_mask"],
        "input_length": len(input_ids),
        "loss_tokens": sum(sequence["loss_mask"]),
        "unknown_tokens": unknown_tokens,
        "crop": sequence["crop"],
    }


def build_proof_sequence_metadata(
    data_dir: str | Path,
    *,
    handoff_path: str | Path,
    run_id: str = DEFAULT_RUN_ID,
    write: bool = False,
) -> dict[str, Any]:
    """Materialize proof-trainer sequence metadata without storing token payloads."""

    _validate_identifier(run_id, "run_id")
    data_dir = Path(data_dir)
    handoff = _load_json(_data_path(data_dir, Path(handoff_path)))
    dry_run_report_path = data_dir / proof_train_report_path(run_id)
    dry_run_report = _load_json(dry_run_report_path)
    sequence_plan_path = data_dir / proof_train_sequence_plan_path(run_id)
    sequence_plan = list(_iter_jsonl(sequence_plan_path))
    blockers: list[str] = []
    if dry_run_report.get("status") != "dry_run_passed":
        blockers.append("dry-run report is not dry_run_passed")
    expected_plan_sha = dry_run_report.get("outputs", {}).get("sequence_plan_sha256")
    if isinstance(expected_plan_sha, str) and expected_plan_sha != _sha256(sequence_plan_path):
        blockers.append("sequence plan sha256 does not match dry-run report")
    if dry_run_report.get("handoff", {}).get("sha256") != _sha256(_data_path(data_dir, Path(handoff_path))):
        blockers.append("handoff sha256 does not match dry-run report")

    training_path = _input_path(data_dir, handoff, "training_artifact", blockers)
    tokenizer_path = _input_path(data_dir, handoff, "tokenizer", blockers)
    tokenizer = _load_json(tokenizer_path) if tokenizer_path is not None and tokenizer_path.exists() else {}
    plan_by_record_id = {
        record["record_id"]: record
        for record in sequence_plan
        if isinstance(record.get("record_id"), str)
    }
    context_tokens = _context_tokens_from_report(dry_run_report)
    metadata_records: list[dict[str, Any]] = []
    report = _empty_report(run_id, context_tokens=context_tokens)

    if training_path is not None and training_path.exists() and tokenizer:
        for record in _iter_jsonl(training_path):
            record_id = record.get("id")
            plan_record = plan_by_record_id.get(record_id)
            if plan_record is None:
                blockers.append(f"missing sequence plan record for {record_id}")
                continue
            _add_plan_count(report, plan_record)
            if plan_record.get("decision") == "exclude_oversize":
                continue
            try:
                sequence = materialize_training_sequence(
                    record,
                    plan_record,
                    tokenizer,
                    context_tokens=context_tokens,
                )
            except ValueError as exc:
                blockers.append(f"{record_id}: {exc}")
                continue
            metadata = _sequence_metadata(sequence, plan_record=plan_record)
            metadata_records.append(metadata)
            _add_sequence_count(report, metadata)

    expected_kept = _expected_kept_records(sequence_plan)
    if expected_kept != len(metadata_records):
        blockers.append(
            "kept sequence count does not match sequence plan "
            f"({len(metadata_records)} != {expected_kept})"
        )
    report["status"] = "materialized" if not blockers else "blocked"
    report["blockers"] = blockers
    report["input_checks"] = {
        "handoff": {
            "path": _display_path(_data_path(data_dir, Path(handoff_path)), base=data_dir),
            "sha256": _sha256(_data_path(data_dir, Path(handoff_path))),
        },
        "dry_run_report": {
            "path": str(proof_train_report_path(run_id)),
            "sha256": _sha256(dry_run_report_path),
        },
        "sequence_plan": {
            "path": str(proof_train_sequence_plan_path(run_id)),
            "sha256": _sha256(sequence_plan_path),
        },
        "training_artifact": _input_check(data_dir, handoff, "training_artifact"),
        "tokenizer": _input_check(data_dir, handoff, "tokenizer"),
    }
    if write:
        _write_sequence_artifacts(data_dir, run_id=run_id, report=report, records=metadata_records)
    _print_report(report)
    return report


def validate_proof_sequence_metadata(data_dir: str | Path, *, run_id: str) -> dict[str, Any]:
    """Validate proof sequence metadata and its report."""

    _validate_identifier(run_id, "run_id")
    data_dir = Path(data_dir)
    report_path = data_dir / proof_sequence_report_path(run_id)
    metadata_path = data_dir / proof_sequence_metadata_path(run_id)
    report = _load_json(report_path)
    metadata_records = list(_iter_jsonl(metadata_path))
    errors: list[str] = []
    if report.get("status") != "materialized":
        errors.append("sequence materialization report is not materialized")
    if report.get("outputs", {}).get("metadata_sha256") != _sha256(metadata_path):
        errors.append("metadata sha256 does not match report")
    if report.get("materialization", {}).get("kept_records") != len(metadata_records):
        errors.append("metadata record count does not match report")
    context_tokens = report.get("context_tokens")
    for record in metadata_records:
        if isinstance(context_tokens, int) and record.get("input_length", 0) > context_tokens:
            errors.append(f"{record.get('record_id')}: input_length exceeds context_tokens")
        if record.get("loss_tokens", 0) <= 0:
            errors.append(f"{record.get('record_id')}: missing loss tokens")
    validation = {
        "run_id": run_id,
        "valid": not errors,
        "errors": errors,
        "report_path": str(proof_sequence_report_path(run_id)),
        "metadata_path": str(proof_sequence_metadata_path(run_id)),
    }
    for key, value in validation.items():
        if key != "errors":
            print(key, value)
    for error in errors:
        print("error", error)
    if errors:
        raise SystemExit(1)
    return validation


def _render_sequence(
    record: dict[str, Any],
    *,
    decision: str,
    context_tokens: int,
) -> dict[str, Any]:
    blocks = _message_blocks(record)
    if decision == "use_full":
        return _render_blocks(blocks, context_tokens=context_tokens, crop_user=False)
    return _render_blocks(blocks, context_tokens=context_tokens, crop_user=True)


def _render_blocks(
    blocks: list[dict[str, Any]],
    *,
    context_tokens: int,
    crop_user: bool,
) -> dict[str, Any]:
    user_blocks = [block for block in blocks if block["role"] == "user"]
    if crop_user and len(user_blocks) != 1:
        raise ValueError("truncated records must contain exactly one user message")
    user_block = user_blocks[0] if user_blocks else None
    user_tokens = list(user_block["content_tokens"]) if user_block is not None else []
    fixed_tokens = 2  # <bos>, <eos>
    for block in blocks:
        fixed_tokens += 2  # role marker and <sep>
        if not (crop_user and block["role"] == "user"):
            fixed_tokens += len(block["content_tokens"])
    user_budget = len(user_tokens)
    if crop_user:
        user_budget = context_tokens - fixed_tokens
        if user_budget < 0:
            raise ValueError("preserved system and assistant tokens exceed context_tokens")
    cropped_user_tokens, crop = _crop_user_tokens(
        user_tokens,
        budget=user_budget,
        crop_user=crop_user,
    )

    tokens: list[str] = ["<bos>"]
    loss_mask: list[int] = [0]
    for block in blocks:
        role = block["role"]
        tokens.append(f"<{role}>")
        loss_mask.append(0)
        content = cropped_user_tokens if crop_user and role == "user" else block["content_tokens"]
        for token in content:
            tokens.append(token)
            loss_mask.append(1 if role == "assistant" else 0)
        tokens.append("<sep>")
        loss_mask.append(1 if role == "assistant" else 0)
    tokens.append("<eos>")
    loss_mask.append(1 if any(block["role"] == "assistant" for block in blocks) else 0)
    if len(tokens) > context_tokens:
        raise ValueError("rendered sequence exceeds context_tokens")
    crop["output_tokens"] = len(tokens)
    return {
        "tokens": tokens,
        "loss_mask": loss_mask,
        "crop": crop,
    }


def _message_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = []
    for message in record.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        content = message.get("content")
        blocks.append(
            {
                "role": role,
                "content_tokens": tokenize_text(content) if isinstance(content, str) else [],
            }
        )
    if not any(block["role"] == "assistant" for block in blocks):
        raise ValueError("record has no assistant message")
    rendered = ["<bos>"]
    for block in blocks:
        rendered.append(f"<{block['role']}>")
        rendered.extend(block["content_tokens"])
        rendered.append("<sep>")
    rendered.append("<eos>")
    if rendered != record_tokens(record):
        raise ValueError("message block tokenization does not match proof tokenizer")
    return blocks


def _crop_user_tokens(
    user_tokens: list[str],
    *,
    budget: int,
    crop_user: bool,
) -> tuple[list[str], dict[str, Any]]:
    if budget < 0:
        raise ValueError("user token budget must not be negative")
    if not crop_user or len(user_tokens) <= budget:
        return list(user_tokens), {
            "policy": "full",
            "user_raw_tokens": len(user_tokens),
            "user_kept_tokens": len(user_tokens),
            "user_prefix_tokens": len(user_tokens),
            "user_suffix_tokens": 0,
        }
    prefix = (budget + 1) // 2
    suffix = budget - prefix
    if suffix:
        cropped = user_tokens[:prefix] + user_tokens[-suffix:]
    else:
        cropped = user_tokens[:prefix]
    return cropped, {
        "policy": "deterministic_prefix_suffix",
        "user_raw_tokens": len(user_tokens),
        "user_kept_tokens": len(cropped),
        "user_prefix_tokens": prefix,
        "user_suffix_tokens": suffix,
    }


def _sequence_metadata(
    sequence: dict[str, Any],
    *,
    plan_record: dict[str, Any],
) -> dict[str, Any]:
    input_ids = sequence["input_ids"]
    loss_mask = sequence["loss_mask"]
    return {
        "schema_version": SCHEMA_VERSION,
        "materialization_id": SEQUENCE_MATERIALIZATION_ID,
        "record_id": sequence["record_id"],
        "source_diff_id": sequence["source_diff_id"],
        "data_split": sequence["data_split"],
        "decision": sequence["decision"],
        "context_tokens": sequence["context_tokens"],
        "input_length": sequence["input_length"],
        "loss_tokens": sequence["loss_tokens"],
        "unknown_tokens": sequence["unknown_tokens"],
        "input_ids_sha256": _hash_ints(input_ids),
        "loss_mask_sha256": _hash_ints(loss_mask),
        "raw_token_count": plan_record.get("raw_token_count"),
        "raw_sequences_at_context": plan_record.get("raw_sequences_at_context"),
        "crop": sequence["crop"],
    }


def _empty_report(run_id: str, *, context_tokens: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "materialization_id": SEQUENCE_MATERIALIZATION_ID,
        "run_id": run_id,
        "status": "blocked",
        "blockers": [],
        "context_tokens": context_tokens,
        "materialization": {
            "kept_records": 0,
            "excluded_records": 0,
            "full_records": 0,
            "truncated_records": 0,
            "input_tokens": 0,
            "loss_tokens": 0,
            "unknown_tokens": 0,
            "max_input_length": 0,
            "by_split": {},
        },
        "outputs": {
            "metadata_path": str(proof_sequence_metadata_path(run_id)),
            "report_path": str(proof_sequence_report_path(run_id)),
            "metadata_sha256": None,
        },
        "claim_policy": (
            "Sequence materialization proves deterministic trainer inputs only; "
            "it is not model-quality evidence."
        ),
    }


def _add_plan_count(report: dict[str, Any], plan_record: dict[str, Any]) -> None:
    split = str(plan_record.get("data_split"))
    split_report = _split_report(report, split)
    decision = plan_record.get("decision")
    if decision == "exclude_oversize":
        split_report["excluded_records"] += 1
        report["materialization"]["excluded_records"] += 1
    elif decision == "use_truncated":
        split_report["truncated_records"] += 1
        report["materialization"]["truncated_records"] += 1
    elif decision == "use_full":
        split_report["full_records"] += 1
        report["materialization"]["full_records"] += 1


def _add_sequence_count(report: dict[str, Any], metadata: dict[str, Any]) -> None:
    split = str(metadata.get("data_split"))
    split_report = _split_report(report, split)
    materialization = report["materialization"]
    split_report["kept_records"] += 1
    split_report["input_tokens"] += metadata["input_length"]
    split_report["loss_tokens"] += metadata["loss_tokens"]
    split_report["unknown_tokens"] += metadata["unknown_tokens"]
    split_report["max_input_length"] = max(split_report["max_input_length"], metadata["input_length"])
    materialization["kept_records"] += 1
    materialization["input_tokens"] += metadata["input_length"]
    materialization["loss_tokens"] += metadata["loss_tokens"]
    materialization["unknown_tokens"] += metadata["unknown_tokens"]
    materialization["max_input_length"] = max(
        materialization["max_input_length"],
        metadata["input_length"],
    )


def _split_report(report: dict[str, Any], split: str) -> dict[str, int]:
    by_split = report["materialization"]["by_split"]
    if split not in by_split:
        by_split[split] = {
            "kept_records": 0,
            "excluded_records": 0,
            "full_records": 0,
            "truncated_records": 0,
            "input_tokens": 0,
            "loss_tokens": 0,
            "unknown_tokens": 0,
            "max_input_length": 0,
        }
    return by_split[split]


def _expected_kept_records(sequence_plan: list[dict[str, Any]]) -> int:
    return sum(1 for record in sequence_plan if record.get("decision") != "exclude_oversize")


def _write_sequence_artifacts(
    data_dir: Path,
    *,
    run_id: str,
    report: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    metadata_path = data_dir / proof_sequence_metadata_path(run_id)
    report_path = data_dir / proof_sequence_report_path(run_id)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    report["outputs"]["metadata_sha256"] = _sha256(metadata_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _token_to_id(tokenizer: dict[str, Any]) -> dict[str, int]:
    vocab = tokenizer.get("vocab")
    if not isinstance(vocab, list):
        raise ValueError("tokenizer vocab must be a list")
    return {
        entry["token"]: entry["id"]
        for entry in vocab
        if isinstance(entry, dict) and isinstance(entry.get("token"), str) and isinstance(entry.get("id"), int)
    }


def _unknown_id(tokenizer: dict[str, Any]) -> int:
    special_tokens = tokenizer.get("special_tokens")
    if isinstance(special_tokens, dict) and isinstance(special_tokens.get("<unk>"), int):
        return special_tokens["<unk>"]
    for entry in tokenizer.get("vocab", []):
        if isinstance(entry, dict) and entry.get("token") == "<unk>" and isinstance(entry.get("id"), int):
            return entry["id"]
    raise ValueError("tokenizer does not define <unk>")


def _context_tokens(plan_record: dict[str, Any], *, context_tokens: int | None) -> int:
    if context_tokens is not None:
        if context_tokens < 1:
            raise ValueError("context_tokens must be positive")
        return context_tokens
    value = plan_record.get("trainer_context_tokens")
    return int(value) if isinstance(value, int) and value > 0 else 8192


def _context_tokens_from_report(report: dict[str, Any]) -> int:
    value = report.get("tokenization", {}).get("context_tokens")
    return int(value) if isinstance(value, int) and value > 0 else 8192


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


def _input_check(data_dir: Path, handoff: dict[str, Any], input_name: str) -> dict[str, Any]:
    inputs = handoff.get("inputs", {})
    entry = inputs.get(input_name) if isinstance(inputs, dict) else None
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        return {"ok": False, "error": "missing from handoff"}
    path = _data_path(data_dir, Path(entry["path"]))
    expected_sha = entry.get("sha256")
    actual_sha = _sha256(path)
    return {
        "path": _display_path(path, base=data_dir),
        "exists": path.exists(),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "ok": path.exists() and (not isinstance(expected_sha, str) or actual_sha == expected_sha),
    }


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


def _validate_identifier(value: str, name: str) -> None:
    if not value.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"{name} must be a stable alphanumeric identifier")


def _print_report(report: dict[str, Any]) -> None:
    print("run_id", report["run_id"])
    print("status", report["status"])
    materialization = report["materialization"]
    for key in (
        "kept_records",
        "excluded_records",
        "full_records",
        "truncated_records",
        "input_tokens",
        "loss_tokens",
        "unknown_tokens",
        "max_input_length",
    ):
        print(key, materialization[key])
    for blocker in report["blockers"]:
        print("blocker", blocker)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize GCTX proof-trainer sequences.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--handoff", required=True)
    build.add_argument("--run-id", default=DEFAULT_RUN_ID)
    build.add_argument("--write", action="store_true")
    build.add_argument("--fail-on-blocked", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--run-id", default=DEFAULT_RUN_ID)

    args = parser.parse_args(argv)
    if args.command == "build":
        report = build_proof_sequence_metadata(
            args.data_dir,
            handoff_path=args.handoff,
            run_id=args.run_id,
            write=args.write,
        )
        if args.fail_on_blocked and report["status"] != "materialized":
            return 1
    elif args.command == "validate":
        validate_proof_sequence_metadata(args.data_dir, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

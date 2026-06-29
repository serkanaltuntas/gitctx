"""Create and validate proof-model trainer job manifests.

The handoff says the data is ready. The sequence materializer says the trainer
inputs are deterministic. This module writes the next contract: the exact
minimal trainer job shape, checkpoint/eval output paths, and preconditions that
a real decoder-only proof trainer must satisfy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from gitctx.proof_sequences import proof_sequence_metadata_path, proof_sequence_report_path
from gitctx.proof_train import DEFAULT_RUN_ID, proof_train_sequence_plan_path
from gitctx.proof_tokenizer import SPECIAL_TOKENS

TRAIN_RUN_DIR = Path("artifacts/train-runs")
TRAINER_JOB_SCHEMA_VERSION = "v0"
TRAINER_JOB_ID = "gctx1-proof-trainer-job-v0"
DEFAULT_SEED = 17
DEFAULT_LAYERS = 12
DEFAULT_HIDDEN_SIZE = 768
DEFAULT_ATTENTION_HEADS = 12
DEFAULT_KV_HEADS = 4
DEFAULT_INTERMEDIATE_SIZE = 1792
PARAMETER_RANGE = re.compile(
    r"^(?P<low>\d+(?:\.\d+)?)(?P<low_unit>[MB])-"
    r"(?P<high>\d+(?:\.\d+)?)(?P<high_unit>[MB])$"
)


def proof_trainer_job_path(run_id: str) -> Path:
    """Return the proof trainer job manifest path for a run id."""

    _validate_identifier(run_id, "run_id")
    return TRAIN_RUN_DIR / f"{run_id}.trainer-job.json"


def create_proof_trainer_job(
    data_dir: str | Path,
    *,
    handoff_path: str | Path,
    run_id: str = DEFAULT_RUN_ID,
    seed: int = DEFAULT_SEED,
    write: bool = False,
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Create a ready-or-blocked manifest for a real proof-model trainer job."""

    _validate_identifier(run_id, "run_id")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    data_dir = Path(data_dir)
    selected_handoff_path = _data_path(data_dir, Path(handoff_path))
    handoff = _load_json(selected_handoff_path)
    blockers = _handoff_blockers(handoff)
    _check_handoff_input(data_dir, handoff, "training_artifact", blockers)
    tokenizer = _load_tokenizer(data_dir, handoff, blockers)
    sequence_report = _load_sequence_report(data_dir, run_id, blockers)
    sequence_metadata_path = data_dir / proof_sequence_metadata_path(run_id)
    sequence_plan_path = data_dir / proof_train_sequence_plan_path(run_id)
    _check_sequence_inputs(
        sequence_report,
        sequence_metadata_path,
        sequence_plan_path,
        selected_handoff_path,
        blockers,
    )

    target_parameter_range = handoff.get("training_contract", {}).get("target_parameter_range")
    vocab_size = _tokenizer_vocab_size(tokenizer, blockers)
    context_tokens = _context_tokens(handoff, sequence_report, blockers)
    model_contract = _model_contract(
        vocab_size=vocab_size,
        context_tokens=context_tokens,
        target_parameter_range=target_parameter_range,
        blockers=blockers,
    )
    data_contract = _data_contract(handoff, sequence_report, run_id=run_id)
    _check_data_contract(data_contract, handoff, blockers)

    status = "ready_for_trainer" if not blockers else "blocked"
    job = {
        "schema_version": TRAINER_JOB_SCHEMA_VERSION,
        "trainer_job_id": TRAINER_JOB_ID,
        "run_id": run_id,
        "status": status,
        "blockers": blockers,
        "seed": seed,
        "code": {
            "job_manifest_code_revision": code_revision
            if code_revision is not None
            else _current_git_revision(Path.cwd()),
            "handoff_training_code_revision": handoff.get("code", {}).get("training_code_revision"),
        },
        "inputs": {
            "handoff": {
                "path": _display_path(selected_handoff_path, base=data_dir),
                "sha256": _sha256(selected_handoff_path),
            },
            "sequence_plan": {
                "path": str(proof_train_sequence_plan_path(run_id)),
                "sha256": _sha256(sequence_plan_path),
            },
            "sequence_metadata": {
                "path": str(proof_sequence_metadata_path(run_id)),
                "sha256": _sha256(sequence_metadata_path),
            },
            "sequence_materialization_report": {
                "path": str(proof_sequence_report_path(run_id)),
                "sha256": _sha256(data_dir / proof_sequence_report_path(run_id)),
            },
            "tokenizer": _input_file_entry(data_dir, handoff, "tokenizer"),
            "training_artifact": _input_file_entry(data_dir, handoff, "training_artifact"),
        },
        "model_contract": model_contract,
        "data_contract": data_contract,
        "checkpoint_contract": _checkpoint_contract(run_id),
        "eval_contract": _eval_contract(run_id, handoff),
        "claim_policy": (
            "This manifest defines the trainer contract for the future proof run. "
            "It is not a completed training run and not a model-quality claim."
        ),
    }
    if write:
        _write_job(data_dir, run_id=run_id, job=job)
    _print_job(job)
    return job


def validate_proof_trainer_job(data_dir: str | Path, *, run_id: str) -> dict[str, Any]:
    """Validate a proof trainer job manifest."""

    _validate_identifier(run_id, "run_id")
    data_dir = Path(data_dir)
    job_path = data_dir / proof_trainer_job_path(run_id)
    job = _load_json(job_path)
    errors: list[str] = []
    if job.get("schema_version") != TRAINER_JOB_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if job.get("trainer_job_id") != TRAINER_JOB_ID:
        errors.append("trainer_job_id mismatch")
    if job.get("run_id") != run_id:
        errors.append("run_id mismatch")
    if job.get("status") != "ready_for_trainer":
        errors.append("job status is not ready_for_trainer")
    if job.get("blockers") != []:
        errors.append("job blockers must be empty")
    _validate_job_input_hashes(data_dir, job, errors)
    model_contract = job.get("model_contract", {})
    if not isinstance(model_contract, dict):
        errors.append("model_contract must be an object")
    else:
        estimate = model_contract.get("estimated_parameters")
        target_range = model_contract.get("target_parameter_range")
        if not isinstance(estimate, int):
            errors.append("model_contract.estimated_parameters is missing")
        elif not isinstance(target_range, str) or not _parameter_count_in_range(estimate, target_range):
            errors.append("model_contract.estimated_parameters is outside target range")
        if model_contract.get("architecture") != "decoder-only transformer":
            errors.append("model_contract.architecture mismatch")
        if model_contract.get("tie_input_output_embeddings") is not True:
            errors.append("model_contract must tie input/output embeddings")
    data_contract = job.get("data_contract", {})
    if not isinstance(data_contract, dict):
        errors.append("data_contract must be an object")
    else:
        if data_contract.get("train_split") != "DEV":
            errors.append("data_contract.train_split must be DEV")
        if data_contract.get("eval_split") != "REPORT":
            errors.append("data_contract.eval_split must be REPORT")
        if data_contract.get("train_report_records") not in (0, None):
            errors.append("data_contract must not train on REPORT")
        if data_contract.get("report_excluded_records") != 0:
            errors.append("data_contract must not exclude REPORT records")
    for section_name in ("checkpoint_contract", "eval_contract"):
        section = job.get(section_name)
        if not isinstance(section, dict):
            errors.append(f"{section_name} must be an object")
    validation = {
        "run_id": run_id,
        "valid": not errors,
        "errors": errors,
        "job_path": str(proof_trainer_job_path(run_id)),
    }
    for key, value in validation.items():
        if key != "errors":
            print(key, value)
    for error in errors:
        print("error", error)
    if errors:
        raise SystemExit(1)
    return validation


def _handoff_blockers(handoff: dict[str, Any]) -> list[str]:
    blockers = []
    if handoff.get("status") != "ready_for_training":
        blockers.append("handoff status is not ready_for_training")
    contract = handoff.get("training_contract", {})
    if not isinstance(contract, dict):
        blockers.append("handoff training_contract is missing")
        return blockers
    if contract.get("architecture") != "decoder-only transformer":
        blockers.append("training contract architecture must be decoder-only transformer")
    if contract.get("train_split") != "DEV":
        blockers.append("training contract train_split must be DEV")
    if contract.get("eval_split") != "REPORT":
        blockers.append("training contract eval_split must be REPORT")
    if contract.get("reserved_split") != "HELD_OUT":
        blockers.append("training contract reserved_split must be HELD_OUT")
    if contract.get("optimizer") != "adamw":
        blockers.append("training contract optimizer must be adamw")
    return blockers


def _load_tokenizer(
    data_dir: Path,
    handoff: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    path = _input_path(data_dir, handoff, "tokenizer", blockers)
    if path is None or not path.exists():
        blockers.append("tokenizer input is missing")
        return {}
    expected_sha = _input_sha(handoff, "tokenizer")
    actual_sha = _sha256(path)
    if expected_sha is None:
        blockers.append("tokenizer sha256 is missing from handoff")
    if isinstance(expected_sha, str) and actual_sha != expected_sha:
        blockers.append("tokenizer sha256 mismatch")
    return _load_json(path)


def _load_sequence_report(data_dir: Path, run_id: str, blockers: list[str]) -> dict[str, Any]:
    path = data_dir / proof_sequence_report_path(run_id)
    if not path.exists():
        blockers.append("sequence materialization report is missing")
        return {}
    report = _load_json(path)
    if report.get("status") != "materialized":
        blockers.append("sequence materialization report is not materialized")
    return report


def _check_sequence_inputs(
    sequence_report: dict[str, Any],
    sequence_metadata_path: Path,
    sequence_plan_path: Path,
    handoff_path: Path,
    blockers: list[str],
) -> None:
    outputs = _as_mapping(sequence_report.get("outputs"))
    input_checks = _as_mapping(sequence_report.get("input_checks"))
    sequence_plan_check = _as_mapping(input_checks.get("sequence_plan"))
    handoff_check = _as_mapping(input_checks.get("handoff"))
    if not sequence_metadata_path.exists():
        blockers.append("sequence metadata is missing")
    else:
        expected_sha = outputs.get("metadata_sha256")
        if not isinstance(expected_sha, str):
            blockers.append("sequence report metadata sha256 is missing")
        elif expected_sha != _sha256(sequence_metadata_path):
            blockers.append("sequence metadata sha256 mismatch")
    if not sequence_plan_path.exists():
        blockers.append("sequence plan is missing")
    else:
        expected_plan_sha = sequence_plan_check.get("sha256")
        if not isinstance(expected_plan_sha, str):
            blockers.append("sequence report sequence-plan sha256 is missing")
        elif expected_plan_sha != _sha256(sequence_plan_path):
            blockers.append("sequence plan sha256 mismatch")
    expected_handoff_sha = handoff_check.get("sha256")
    if not isinstance(expected_handoff_sha, str):
        blockers.append("sequence report handoff sha256 is missing")
    elif expected_handoff_sha != _sha256(handoff_path):
        blockers.append("sequence report handoff sha256 mismatch")


def _model_contract(
    *,
    vocab_size: int | None,
    context_tokens: int | None,
    target_parameter_range: Any,
    blockers: list[str],
) -> dict[str, Any]:
    if not isinstance(target_parameter_range, str):
        blockers.append("target parameter range is missing")
        target_parameter_range = "60M-100M"
    selected_vocab_size = vocab_size or 0
    selected_context_tokens = context_tokens or 0
    shape = {
        "architecture": "decoder-only transformer",
        "target_parameter_range": target_parameter_range,
        "tokenizer_vocab_size": selected_vocab_size,
        "context_tokens": selected_context_tokens,
        "layers": DEFAULT_LAYERS,
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "attention_heads": DEFAULT_ATTENTION_HEADS,
        "kv_heads": DEFAULT_KV_HEADS,
        "intermediate_size": DEFAULT_INTERMEDIATE_SIZE,
        "activation": "silu",
        "normalization": "rmsnorm",
        "position_encoding": "rope",
        "tie_input_output_embeddings": True,
    }
    estimated = _estimate_decoder_parameters(shape)
    shape["estimated_parameters"] = estimated
    shape["estimated_parameter_millions"] = round(estimated / 1_000_000, 3)
    if not _parameter_count_in_range(estimated, target_parameter_range):
        blockers.append("default decoder shape is outside target parameter range")
    if selected_vocab_size < len(SPECIAL_TOKENS):
        blockers.append("tokenizer vocab size is invalid")
    if selected_context_tokens < 8192:
        blockers.append("context_tokens must be at least 8192")
    return shape


def _data_contract(
    handoff: dict[str, Any],
    sequence_report: dict[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    contract = handoff.get("training_contract", {})
    materialization = sequence_report.get("materialization", {})
    by_split = materialization.get("by_split", {}) if isinstance(materialization, dict) else {}
    dev = by_split.get("DEV", {}) if isinstance(by_split, dict) else {}
    report = by_split.get("REPORT", {}) if isinstance(by_split, dict) else {}
    return {
        "run_id": run_id,
        "train_split": contract.get("train_split"),
        "eval_split": contract.get("eval_split"),
        "reserved_split": contract.get("reserved_split"),
        "sequence_materialization_status": sequence_report.get("status"),
        "sequence_metadata_path": str(proof_sequence_metadata_path(run_id)),
        "train_dev_records": dev.get("kept_records", 0),
        "train_report_records": 0,
        "report_eval_records": report.get("kept_records", 0),
        "dev_excluded_records": dev.get("excluded_records", 0),
        "report_excluded_records": report.get("excluded_records", 0),
        "train_input_tokens": dev.get("input_tokens", 0),
        "train_loss_tokens": dev.get("loss_tokens", 0),
        "report_input_tokens": report.get("input_tokens", 0),
        "report_loss_tokens": report.get("loss_tokens", 0),
        "max_input_length": materialization.get("max_input_length", 0)
        if isinstance(materialization, dict)
        else 0,
    }


def _check_data_contract(
    data_contract: dict[str, Any],
    handoff: dict[str, Any],
    blockers: list[str],
) -> None:
    if data_contract["train_split"] != "DEV":
        blockers.append("data contract train_split must be DEV")
    if data_contract["eval_split"] != "REPORT":
        blockers.append("data contract eval_split must be REPORT")
    if data_contract["train_report_records"] != 0:
        blockers.append("data contract must not train on REPORT")
    if data_contract["report_excluded_records"] != 0:
        blockers.append("REPORT contains excluded records")
    minimum = (
        handoff.get("readiness", {})
        .get("gates", {})
        .get("dev_training_records", {})
        .get("minimum")
    )
    if isinstance(minimum, int) and data_contract["train_dev_records"] < minimum:
        blockers.append("DEV kept records are below proof-run minimum")
    if data_contract["report_eval_records"] < 1:
        blockers.append("REPORT eval records are missing")


def _check_handoff_input(
    data_dir: Path,
    handoff: dict[str, Any],
    input_name: str,
    blockers: list[str],
) -> None:
    path = _input_path(data_dir, handoff, input_name, blockers)
    if path is None:
        return
    expected_sha = _input_sha(handoff, input_name)
    if expected_sha is None:
        blockers.append(f"{input_name} sha256 is missing from handoff")
    if not path.exists():
        blockers.append(f"{input_name} input is missing")
        return
    actual_sha = _sha256(path)
    if isinstance(expected_sha, str) and actual_sha != expected_sha:
        blockers.append(f"{input_name} sha256 mismatch")


def _validate_job_input_hashes(
    data_dir: Path,
    job: dict[str, Any],
    errors: list[str],
) -> None:
    inputs = job.get("inputs")
    if not isinstance(inputs, dict):
        errors.append("inputs must be an object")
        return
    for name, entry in sorted(inputs.items()):
        if not isinstance(entry, dict):
            errors.append(f"inputs.{name}: entry must be an object")
            continue
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            errors.append(f"inputs.{name}: path is missing")
            continue
        path = _data_path(data_dir, Path(path_value))
        if entry.get("ok") is False:
            errors.append(f"inputs.{name}: manifest says input is not ok")
        if not path.exists():
            errors.append(f"inputs.{name}: file is missing")
            continue
        actual_sha = _sha256(path)
        hash_fields = {
            "sha256": "manifest",
            "actual_sha256": "recorded actual sha256",
            "expected_sha256": "expected sha256",
        }
        present_hashes = 0
        for field_name, label in hash_fields.items():
            recorded_sha = entry.get(field_name)
            if recorded_sha is None:
                continue
            if not isinstance(recorded_sha, str):
                errors.append(f"inputs.{name}.{field_name}: must be a string")
                continue
            present_hashes += 1
            if recorded_sha != actual_sha:
                errors.append(f"inputs.{name}: current sha256 does not match {label}")
        if present_hashes == 0:
            errors.append(f"inputs.{name}: sha256 is missing")


def _checkpoint_contract(run_id: str) -> dict[str, Any]:
    checkpoint_dir = TRAIN_RUN_DIR / run_id / "checkpoints"
    return {
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "latest_checkpoint": (checkpoint_dir / "latest.json").as_posix(),
        "final_checkpoint": (checkpoint_dir / "final.json").as_posix(),
        "final_model_dir": (TRAIN_RUN_DIR / run_id / "model").as_posix(),
        "training_report_path": (TRAIN_RUN_DIR / f"{run_id}.trainer.report.json").as_posix(),
        "required_checkpoint_state": [
            "model weights",
            "optimizer state",
            "scheduler state",
            "rng state",
            "sequence cursor",
            "epoch/step counters",
            "input artifact checksums",
            "training code revision",
        ],
        "resume_requirement": (
            "A resumed trainer job must reproduce the uninterrupted run for the same "
            "seed, input hashes, sequence cursor, and training code revision."
        ),
    }


def _eval_contract(run_id: str, handoff: dict[str, Any]) -> dict[str, Any]:
    metrics = [
        "format_validity",
        "type_match",
        "scope_quality",
        "specificity",
        "brevity",
        "exact_message_match",
    ]
    return {
        "eval_split": handoff.get("training_contract", {}).get("eval_split"),
        "locked_report_required": True,
        "report_prediction_path": (
            TRAIN_RUN_DIR / f"{run_id}.locked-report.predictions.jsonl"
        ).as_posix(),
        "report_eval_path": (TRAIN_RUN_DIR / f"{run_id}.locked-report.eval.json").as_posix(),
        "metrics": metrics,
        "precondition": "Do not run quality claims until the final checkpoint exists.",
        "claim_requirement": "Model card and eval card are required before public quality claims.",
    }


def _tokenizer_vocab_size(tokenizer: dict[str, Any], blockers: list[str]) -> int | None:
    special_tokens = tokenizer.get("special_tokens")
    if not isinstance(special_tokens, dict):
        blockers.append("tokenizer special_tokens is missing")
    else:
        for token in SPECIAL_TOKENS:
            if not isinstance(special_tokens.get(token), int):
                blockers.append(f"tokenizer special token {token} is missing")
    value = tokenizer.get("vocab_size")
    vocab = tokenizer.get("vocab")
    if isinstance(vocab, list):
        if isinstance(value, int) and len(vocab) != value:
            blockers.append("tokenizer vocab_size does not match vocab length")
            return value
        return len(vocab)
    if isinstance(value, int):
        return value
    blockers.append("tokenizer vocab_size is missing")
    return None


def _context_tokens(
    handoff: dict[str, Any],
    sequence_report: dict[str, Any],
    blockers: list[str],
) -> int | None:
    handoff_value = handoff.get("training_contract", {}).get("context_tokens")
    report_value = sequence_report.get("context_tokens")
    if not isinstance(handoff_value, int):
        blockers.append("handoff context_tokens is missing")
        return report_value if isinstance(report_value, int) else None
    if isinstance(report_value, int) and report_value != handoff_value:
        blockers.append("sequence report context_tokens does not match handoff")
    return handoff_value


def _estimate_decoder_parameters(shape: dict[str, Any]) -> int:
    vocab_size = int(shape["tokenizer_vocab_size"])
    hidden = int(shape["hidden_size"])
    layers = int(shape["layers"])
    heads = int(shape["attention_heads"])
    kv_heads = int(shape["kv_heads"])
    intermediate = int(shape["intermediate_size"])
    head_dim = hidden // heads
    embeddings = vocab_size * hidden
    q_proj = hidden * hidden
    k_proj = hidden * (kv_heads * head_dim)
    v_proj = hidden * (kv_heads * head_dim)
    o_proj = hidden * hidden
    mlp = hidden * intermediate * 3
    norms = hidden * 2
    per_layer = q_proj + k_proj + v_proj + o_proj + mlp + norms
    final_norm = hidden
    return embeddings + layers * per_layer + final_norm


def _parameter_count_in_range(value: int, parameter_range: str) -> bool:
    match = PARAMETER_RANGE.fullmatch(parameter_range)
    if match is None:
        return False
    low = _to_parameters(float(match.group("low")), match.group("low_unit"))
    high = _to_parameters(float(match.group("high")), match.group("high_unit"))
    return int(low) <= value <= int(high)


def _to_parameters(value: float, unit: str) -> float:
    multiplier = 1_000_000_000 if unit == "B" else 1_000_000
    return value * multiplier


def _input_file_entry(data_dir: Path, handoff: dict[str, Any], input_name: str) -> dict[str, Any]:
    blockers: list[str] = []
    path = _input_path(data_dir, handoff, input_name, blockers)
    if path is None:
        return {"ok": False, "error": "; ".join(blockers)}
    expected_sha = _input_sha(handoff, input_name)
    actual_sha = _sha256(path)
    return {
        "path": _display_path(path, base=data_dir),
        "exists": path.exists(),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "ok": path.exists() and (not isinstance(expected_sha, str) or actual_sha == expected_sha),
    }


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


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _write_job(data_dir: Path, *, run_id: str, job: dict[str, Any]) -> None:
    path = data_dir / proof_trainer_job_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _current_git_revision(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_identifier(value: str, name: str) -> None:
    if not value.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"{name} must be a stable alphanumeric identifier")


def _print_job(job: dict[str, Any]) -> None:
    print("run_id", job["run_id"])
    print("status", job["status"])
    print("estimated_parameters", job["model_contract"]["estimated_parameters"])
    print("train_dev_records", job["data_contract"]["train_dev_records"])
    print("report_eval_records", job["data_contract"]["report_eval_records"])
    for blocker in job["blockers"]:
        print("blocker", blocker)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or validate a GCTX proof trainer job.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--handoff", required=True)
    create.add_argument("--run-id", default=DEFAULT_RUN_ID)
    create.add_argument("--seed", type=int, default=DEFAULT_SEED)
    create.add_argument("--code-revision")
    create.add_argument("--write", action="store_true")
    create.add_argument("--fail-on-blocked", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--run-id", default=DEFAULT_RUN_ID)

    args = parser.parse_args(argv)
    if args.command == "create":
        job = create_proof_trainer_job(
            args.data_dir,
            handoff_path=args.handoff,
            run_id=args.run_id,
            seed=args.seed,
            write=args.write,
            code_revision=args.code_revision,
        )
        if args.fail_on_blocked and job["status"] != "ready_for_trainer":
            return 1
    elif args.command == "validate":
        validate_proof_trainer_job(args.data_dir, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

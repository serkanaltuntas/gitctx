"""Validate proof-model training run configuration."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any

EXPECTED_SPLITS = {
    "train_split": "DEV",
    "eval_split": "REPORT",
    "reserved_split": "HELD_OUT",
}
MINIMUM_COUNTS = {
    "minimum_dev_records": 10_000,
    "minimum_report_records": 1_000,
    "minimum_reserved_held_out_records": 1_000,
}
REQUIRED_EVAL_METRICS = {
    "format_validity",
    "type_match",
    "scope_quality",
    "specificity",
    "brevity",
    "exact_message_match",
}
REQUIRED_REPRODUCIBILITY_TERMS = {
    "seed",
    "git revision",
    "training code revision",
    "dataset checksum",
    "runtime",
}
STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PARAMETER_RANGE = re.compile(
    r"^(?P<low>\d+(?:\.\d+)?)(?P<low_unit>[MB])-"
    r"(?P<high>\d+(?:\.\d+)?)(?P<high_unit>[MB])$"
)


@dataclass(frozen=True)
class ProofModelConfigReport:
    """Validation result for a proof-model config."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    summary: dict[str, Any]


def load_proof_model_config(path: str | Path) -> dict[str, Any]:
    """Load a proof-model config JSON document."""

    selected = Path(path)
    try:
        document = json.loads(selected.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{selected}: invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{selected}: proof-model config must be a JSON object")
    return document


def validate_proof_model_config(config: dict[str, Any]) -> ProofModelConfigReport:
    """Validate a proof-model config contract without external dependencies."""

    errors: list[str] = []
    warnings: list[str] = []

    _validate_top_level(config, errors, warnings)
    data = _mapping(config, "data", errors)
    model = _mapping(config, "model", errors)
    training = _mapping(config, "training", errors)
    evaluation = _mapping(config, "evaluation", errors)
    release = _mapping(config, "release", errors)

    if data:
        _validate_data(data, errors)
    if model:
        _validate_model(model, errors)
    if training:
        _validate_training(training, errors)
    if evaluation:
        _validate_evaluation(evaluation, errors)
    if release:
        _validate_release(release, errors)

    summary = _summary(config)
    return ProofModelConfigReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        summary=summary,
    )


def _validate_top_level(
    config: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    required = {
        "id",
        "version",
        "status",
        "purpose",
        "data",
        "model",
        "training",
        "evaluation",
        "release",
    }
    for key in sorted(required - set(config)):
        errors.append(f"{key}: required field is missing")
    for key in sorted(set(config) - required):
        warnings.append(f"{key}: unknown top-level field is ignored by this validator")

    _stable_string(config, "id", errors)
    _stable_string(config, "version", errors)
    status = _string(config, "status", errors)
    if status and status not in {"planned", "active", "completed", "superseded"}:
        errors.append("status: expected one of planned, active, completed, superseded")
    _non_empty_string(config, "purpose", errors)


def _validate_data(data: dict[str, Any], errors: list[str]) -> None:
    _stable_string(data, "artifact_name", errors, path_prefix="data")
    _stable_string(data, "artifact_version", errors, path_prefix="data")
    for key, expected in EXPECTED_SPLITS.items():
        value = _string(data, key, errors, path_prefix="data")
        if value and value != expected:
            errors.append(f"data.{key}: expected {expected}, got {value}")
    for key, minimum in MINIMUM_COUNTS.items():
        _int_at_least(data, key, minimum, errors, path_prefix="data")


def _validate_model(model: dict[str, Any], errors: list[str]) -> None:
    target_rung = _string(model, "target_rung", errors, path_prefix="model")
    if target_rung and target_rung != "GCTX-1":
        errors.append("model.target_rung: expected GCTX-1 for this proof config")

    parameter_range = _string(model, "target_parameter_range", errors, path_prefix="model")
    if parameter_range:
        _validate_parameter_range(parameter_range, errors)

    architecture = _string(model, "architecture", errors, path_prefix="model")
    if architecture and architecture != "decoder-only transformer":
        errors.append("model.architecture: expected decoder-only transformer")

    _int_at_least(model, "context_tokens", 8_192, errors, path_prefix="model")

    tokenizer = _non_empty_string(model, "tokenizer", errors, path_prefix="model")
    if tokenizer and "diff" not in tokenizer.lower():
        errors.append("model.tokenizer: should explicitly be code/diff-aware")

    objective = _non_empty_string(model, "objective", errors, path_prefix="model")
    if objective and "conventional commit" not in objective.lower():
        errors.append("model.objective: should explicitly target Conventional Commit examples")


def _validate_training(training: dict[str, Any], errors: list[str]) -> None:
    precision = _non_empty_string(training, "precision", errors, path_prefix="training")
    if precision and "bf16" not in precision.lower() and "fp16" not in precision.lower():
        errors.append("training.precision: should name bf16 or fp16")

    optimizer = _string(training, "optimizer", errors, path_prefix="training")
    if optimizer and optimizer.lower() != "adamw":
        errors.append("training.optimizer: expected adamw")

    _non_empty_string(training, "checkpoint_policy", errors, path_prefix="training")
    reproducibility = _non_empty_string(training, "reproducibility", errors, path_prefix="training")
    if reproducibility:
        normalized = reproducibility.lower()
        for term in sorted(REQUIRED_REPRODUCIBILITY_TERMS):
            if term not in normalized:
                errors.append(f"training.reproducibility: missing {term}")


def _validate_evaluation(evaluation: dict[str, Any], errors: list[str]) -> None:
    _bool_value(
        evaluation,
        "locked_report_required",
        expected=True,
        errors=errors,
        path_prefix="evaluation",
    )
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, list) or not metrics or not all(isinstance(item, str) for item in metrics):
        errors.append("evaluation.metrics: expected a non-empty list of strings")
        metric_set: set[str] = set()
    else:
        metric_set = set(metrics)
    for metric in sorted(REQUIRED_EVAL_METRICS - metric_set):
        errors.append(f"evaluation.metrics: missing {metric}")

    policy = _non_empty_string(
        evaluation,
        "quality_claim_policy",
        errors,
        path_prefix="evaluation",
    )
    if policy:
        normalized = policy.lower()
        for term in ("report", "model card", "eval card"):
            if term not in normalized:
                errors.append(f"evaluation.quality_claim_policy: missing {term}")


def _validate_release(release: dict[str, Any], errors: list[str]) -> None:
    _bool_value(
        release,
        "public_model_release",
        expected=False,
        errors=errors,
        path_prefix="release",
    )
    for key in (
        "requires_model_card",
        "requires_eval_card",
        "requires_dataset_redistribution_review",
    ):
        _bool_value(release, key, expected=True, errors=errors, path_prefix="release")


def _validate_parameter_range(value: str, errors: list[str]) -> None:
    match = PARAMETER_RANGE.fullmatch(value)
    if not match:
        errors.append("model.target_parameter_range: expected a compact range such as 60M-100M")
        return
    low = _to_millions(float(match.group("low")), match.group("low_unit"))
    high = _to_millions(float(match.group("high")), match.group("high_unit"))
    if low >= high:
        errors.append("model.target_parameter_range: lower bound must be smaller than upper bound")
    if low < 50 or high > 150:
        errors.append("model.target_parameter_range: expected to stay within the GCTX-1 proof band")


def _to_millions(value: float, unit: str) -> float:
    return value * 1_000 if unit == "B" else value


def _mapping(config: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key}: expected object")
        return {}
    return value


def _stable_string(
    config: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    path_prefix: str | None = None,
) -> str | None:
    value = _string(config, key, errors, path_prefix=path_prefix)
    path = _path(key, path_prefix)
    if value and not STABLE_ID.fullmatch(value):
        errors.append(f"{path}: expected a stable identifier")
    return value


def _non_empty_string(
    config: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    path_prefix: str | None = None,
) -> str | None:
    value = _string(config, key, errors, path_prefix=path_prefix)
    if value is not None and not value.strip():
        errors.append(f"{_path(key, path_prefix)}: must not be empty")
    return value


def _string(
    config: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    path_prefix: str | None = None,
) -> str | None:
    value = config.get(key)
    path = _path(key, path_prefix)
    if not isinstance(value, str):
        errors.append(f"{path}: expected string")
        return None
    return value


def _int_at_least(
    config: dict[str, Any],
    key: str,
    minimum: int,
    errors: list[str],
    *,
    path_prefix: str,
) -> int | None:
    value = config.get(key)
    path = _path(key, path_prefix)
    if not isinstance(value, int):
        errors.append(f"{path}: expected integer")
        return None
    if value < minimum:
        errors.append(f"{path}: expected at least {minimum}")
    return value


def _bool_value(
    config: dict[str, Any],
    key: str,
    *,
    expected: bool,
    errors: list[str],
    path_prefix: str,
) -> bool | None:
    value = config.get(key)
    path = _path(key, path_prefix)
    if not isinstance(value, bool):
        errors.append(f"{path}: expected boolean")
        return None
    if value is not expected:
        errors.append(f"{path}: expected {expected}")
    return value


def _path(key: str, path_prefix: str | None) -> str:
    return f"{path_prefix}.{key}" if path_prefix else key


def _summary(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    return {
        "id": config.get("id"),
        "version": config.get("version"),
        "status": config.get("status"),
        "artifact_name": data.get("artifact_name"),
        "artifact_version": data.get("artifact_version"),
        "train_split": data.get("train_split"),
        "eval_split": data.get("eval_split"),
        "reserved_split": data.get("reserved_split"),
        "target_rung": model.get("target_rung"),
        "target_parameter_range": model.get("target_parameter_range"),
        "context_tokens": model.get("context_tokens"),
    }


def _print_report(report: ProofModelConfigReport) -> None:
    print("valid", str(report.valid).lower())
    for key, value in report.summary.items():
        print(key, value)
    for warning in report.warnings:
        print("warning", warning)
    for error in report.errors:
        print("error", error)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a GCTX proof-model config.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("config", type=Path)
    validate.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "validate":
        try:
            config = load_proof_model_config(args.config)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        report = validate_proof_model_config(config)
        if args.json:
            print(json.dumps(asdict(report), indent=2, sort_keys=True))
        else:
            _print_report(report)
        return 0 if report.valid else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

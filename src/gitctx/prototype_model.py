"""Dependency-free prototype model for training-pipeline smoke tests."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
from typing import Any

from gitctx.conventional import CommitContext, DEFAULT_TYPES, parse_commit_message, score_commit_message
from gitctx.provenance import load_jsonl
from gitctx.train_artifacts import training_examples_path

MODEL_DIR = Path("artifacts/models")
EVAL_DIR = Path("artifacts/eval")
DEFAULT_MODEL_VERSION = "path-type-v0"
DEFAULT_TRAIN_SPLIT = "DEV"
DEFAULT_EVAL_SPLIT = "REPORT"

GENERIC_TOPIC_TOKENS = frozenset(
    {
        "a",
        "app",
        "apps",
        "bin",
        "core",
        "file",
        "files",
        "lib",
        "libs",
        "main",
        "module",
        "modules",
        "package",
        "packages",
        "py",
        "python",
        "src",
        "test",
        "tests",
        "util",
        "utils",
    }
)


def prototype_model_path(
    artifact_name: str,
    *,
    artifact_version: str = "v0",
    model_version: str = DEFAULT_MODEL_VERSION,
) -> Path:
    """Return the prototype model artifact path."""

    _validate_identifier(artifact_name, "artifact_name")
    _validate_identifier(artifact_version, "artifact_version")
    _validate_identifier(model_version, "model_version")
    return MODEL_DIR / f"{model_version}.{artifact_name}.{artifact_version}.json"


def prototype_prediction_path(
    artifact_name: str,
    split: str,
    *,
    artifact_version: str = "v0",
    model_version: str = DEFAULT_MODEL_VERSION,
) -> Path:
    """Return the prototype prediction JSONL path."""

    _validate_identifier(artifact_name, "artifact_name")
    _validate_identifier(artifact_version, "artifact_version")
    _validate_identifier(model_version, "model_version")
    _validate_split(split)
    return EVAL_DIR / f"{model_version}.{artifact_name}.{artifact_version}.{split.lower()}.predictions.jsonl"


def prototype_eval_report_path(
    artifact_name: str,
    split: str,
    *,
    artifact_version: str = "v0",
    model_version: str = DEFAULT_MODEL_VERSION,
) -> Path:
    """Return the prototype eval report path."""

    _validate_identifier(artifact_name, "artifact_name")
    _validate_identifier(artifact_version, "artifact_version")
    _validate_identifier(model_version, "model_version")
    _validate_split(split)
    return EVAL_DIR / f"{model_version}.{artifact_name}.{artifact_version}.{split.lower()}.report.json"


def train_prototype_model(
    data_dir: str | Path,
    *,
    artifact_name: str,
    artifact_version: str = "v0",
    model_version: str = DEFAULT_MODEL_VERSION,
    train_split: str = DEFAULT_TRAIN_SPLIT,
) -> dict[str, Any]:
    """Create a small aggregate prototype model from reviewed training records."""

    _validate_split(train_split)
    data_dir = Path(data_dir)
    records = [
        record
        for record in load_jsonl(data_dir / training_examples_path(artifact_name, version=artifact_version))
        if record["data_split"] == train_split
    ]
    if not records:
        raise ValueError(f"no training records found for split {train_split}")

    type_counts: Counter[str] = Counter()
    token_type_counts: dict[str, Counter[str]] = {}
    review_decision_counts: Counter[str] = Counter()
    label_source_counts: Counter[str] = Counter()
    parse_failures: list[str] = []

    for record in records:
        try:
            parsed = parse_commit_message(record["target_message"])
        except ValueError:
            parse_failures.append(record["id"])
            continue
        if parsed.type not in DEFAULT_TYPES:
            parse_failures.append(record["id"])
            continue
        type_counts[parsed.type] += 1
        review_decision_counts[record["review_decision"]] += 1
        label_source_counts[record["label_source"]] += 1
        for token in _path_tokens(record["changed_paths"]):
            token_type_counts.setdefault(token, Counter())[parsed.type] += 1

    if parse_failures:
        raise ValueError(f"training records with invalid targets: {parse_failures}")

    model = {
        "model_kind": "dependency_free_path_type_prototype",
        "model_version": model_version,
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "train_split": train_split,
        "training_records": len(records),
        "intended_use": "training pipeline smoke only; not a model-quality claim",
        "type_counts": dict(sorted(type_counts.items())),
        "token_type_counts": {
            token: dict(sorted(counter.items())) for token, counter in sorted(token_type_counts.items())
        },
        "review_decision_counts": dict(sorted(review_decision_counts.items())),
        "label_source_counts": dict(sorted(label_source_counts.items())),
    }
    output_path = data_dir / prototype_model_path(
        artifact_name,
        artifact_version=artifact_version,
        model_version=model_version,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for key in ("model_kind", "model_version", "artifact_name", "training_records"):
        print(key, model[key])
    return model


def evaluate_prototype_model(
    data_dir: str | Path,
    *,
    artifact_name: str,
    artifact_version: str = "v0",
    model_version: str = DEFAULT_MODEL_VERSION,
    split: str = DEFAULT_EVAL_SPLIT,
) -> dict[str, Any]:
    """Evaluate the prototype model on a held-out split."""

    _validate_split(split)
    data_dir = Path(data_dir)
    model_path = data_dir / prototype_model_path(
        artifact_name,
        artifact_version=artifact_version,
        model_version=model_version,
    )
    model = json.loads(model_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in load_jsonl(data_dir / training_examples_path(artifact_name, version=artifact_version))
        if record["data_split"] == split
    ]
    if not records:
        raise ValueError(f"no eval records found for split {split}")

    prediction_records = [_prediction_record(model, record) for record in records]
    prediction_path = data_dir / prototype_prediction_path(
        artifact_name,
        split,
        artifact_version=artifact_version,
        model_version=model_version,
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in prediction_records),
        encoding="utf-8",
    )

    report = _eval_report(
        model=model,
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        model_version=model_version,
        split=split,
        records=prediction_records,
    )
    report_path = data_dir / prototype_eval_report_path(
        artifact_name,
        split,
        artifact_version=artifact_version,
        model_version=model_version,
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for key in ("model_version", "artifact_name", "split", "eval_records", "exact_message_match"):
        print(key, report[key])
    print("prediction_format_valid", report["prediction"]["format_validity"]["true"])
    return report


def _prediction_record(model: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    prediction = _predict_message(model, record["changed_paths"])
    target_score = score_commit_message(
        record["target_message"],
        CommitContext(changed_paths=tuple(record["changed_paths"])),
    )
    prediction_score = score_commit_message(
        prediction,
        CommitContext(changed_paths=tuple(record["changed_paths"])),
    )
    target_type = target_score.parsed.type if target_score.parsed is not None else None
    prediction_type = prediction_score.parsed.type if prediction_score.parsed is not None else None
    return {
        "id": record["id"],
        "source_diff_id": record["source_diff_id"],
        "data_split": record["data_split"],
        "changed_paths": record["changed_paths"],
        "target_message": record["target_message"],
        "prediction_message": prediction,
        "exact_message_match": prediction == record["target_message"],
        "type_match": prediction_type == target_type,
        "target_score": _score_report(target_score),
        "prediction_score": _score_report(prediction_score),
    }


def _predict_message(model: dict[str, Any], changed_paths: list[str]) -> str:
    predicted_type = _predict_type(model, changed_paths)
    topic = _topic_from_paths(changed_paths)
    subject = _subject_for_type(predicted_type, topic)
    return f"{predicted_type}: {subject}"


def _predict_type(model: dict[str, Any], changed_paths: list[str]) -> str:
    token_type_counts = model.get("token_type_counts", {})
    scores: Counter[str] = Counter()
    for token in _path_tokens(changed_paths):
        scores.update(token_type_counts.get(token, {}))

    if not scores:
        scores.update(model.get("type_counts", {}))

    if any(_path_looks_like_docs(path) for path in changed_paths):
        scores["docs"] += 3
    if any(_path_looks_like_tests(path) for path in changed_paths):
        scores["test"] += 2

    predicted, _ = scores.most_common(1)[0]
    if predicted not in DEFAULT_TYPES:
        return "chore"
    return predicted


def _subject_for_type(commit_type: str, topic: str) -> str:
    if commit_type == "docs":
        return f"update {topic} documentation"
    if commit_type == "test":
        return f"update {topic} tests"
    if commit_type == "fix":
        return f"handle {topic} behavior"
    if commit_type == "feat":
        return f"add {topic} support"
    if commit_type == "perf":
        return f"improve {topic} performance"
    if commit_type == "refactor":
        return f"update {topic} implementation"
    return f"update {topic} maintenance"


def _topic_from_paths(changed_paths: list[str]) -> str:
    for token in _path_tokens(changed_paths):
        if token not in GENERIC_TOPIC_TOKENS:
            return token
    return "project"


def _path_tokens(changed_paths: list[str]) -> list[str]:
    tokens: list[str] = []
    for path in changed_paths:
        normalized = path.replace("\\", "/").strip("/").lower()
        for part in normalized.split("/"):
            stem = part.rsplit(".", 1)[0].lstrip("_")
            tokens.extend(token for token in re.split(r"[^a-z0-9]+", stem) if token)
    return tokens


def _path_looks_like_docs(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return normalized.startswith("docs/") or normalized.endswith(".md") or "/docs/" in normalized


def _path_looks_like_tests(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return (
        normalized.startswith("tests/")
        or "/tests/" in normalized
        or normalized.startswith("test/")
        or "/test/" in normalized
        or "test_" in normalized
        or "_test." in normalized
    )


def _eval_report(
    *,
    model: dict[str, Any],
    artifact_name: str,
    artifact_version: str,
    model_version: str,
    split: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_message_match = sum(1 for record in records if record["exact_message_match"])
    type_match = sum(1 for record in records if record["type_match"])
    return {
        "model_kind": model["model_kind"],
        "model_version": model_version,
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "train_split": model["train_split"],
        "split": split,
        "training_records": model["training_records"],
        "eval_records": len(records),
        "exact_message_match": exact_message_match,
        "type_match": type_match,
        "output_path": str(
            prototype_prediction_path(
                artifact_name,
                split,
                artifact_version=artifact_version,
                model_version=model_version,
            )
        ),
        "report_path": str(
            prototype_eval_report_path(
                artifact_name,
                split,
                artifact_version=artifact_version,
                model_version=model_version,
            )
        ),
        "prediction": _score_messages(
            [(record, record["prediction_message"]) for record in records],
            key="prediction_score",
        ),
        "target": _score_messages(
            [(record, record["target_message"]) for record in records],
            key="target_score",
        ),
    }


def _score_messages(records: list[tuple[dict[str, Any], str]], *, key: str) -> dict[str, Any]:
    counters = {
        "format_validity": Counter(),
        "scope_quality": Counter(),
        "specificity": Counter(),
        "brevity": Counter(),
    }
    type_counts: Counter[str] = Counter()
    top_errors: Counter[str] = Counter()
    for record, _ in records:
        score = record[key]
        for metric in counters:
            counters[metric][_bucket(score[metric])] += 1
        if score["parsed_type"] is not None:
            type_counts[score["parsed_type"]] += 1
        top_errors.update(score["errors"])
    return {
        **{name: _counter_to_report(counter) for name, counter in counters.items()},
        "type_counts": dict(sorted(type_counts.items())),
        "top_errors": dict(top_errors.most_common(10)),
    }


def _score_report(score: Any) -> dict[str, Any]:
    return {
        "format_validity": score.format_validity,
        "scope_quality": score.scope_quality,
        "specificity": score.specificity,
        "brevity": score.brevity,
        "errors": list(score.errors),
        "parsed_type": score.parsed.type if score.parsed is not None else None,
        "parsed_scope": score.parsed.scope if score.parsed is not None else None,
        "parsed_subject": score.parsed.subject if score.parsed is not None else None,
    }


def _bucket(value: bool | None) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "none"


def _counter_to_report(counter: Counter[str]) -> dict[str, int]:
    return {
        "true": counter["true"],
        "false": counter["false"],
        "none": counter["none"],
    }


def _validate_split(split: str) -> None:
    if split not in {"DEV", "REPORT"}:
        raise ValueError("split must be DEV or REPORT for prototype model artifacts")


def _validate_identifier(value: str, name: str) -> None:
    if not value.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"{name} must be a stable identifier")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate the gitctx prototype model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--artifact-name", required=True)
    train.add_argument("--artifact-version", default="v0")
    train.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    train.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--artifact-name", required=True)
    evaluate.add_argument("--artifact-version", default="v0")
    evaluate.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    evaluate.add_argument("--split", default=DEFAULT_EVAL_SPLIT)

    args = parser.parse_args(argv)
    if args.command == "train":
        train_prototype_model(
            args.data_dir,
            artifact_name=args.artifact_name,
            artifact_version=args.artifact_version,
            model_version=args.model_version,
            train_split=args.train_split,
        )
    elif args.command == "evaluate":
        evaluate_prototype_model(
            args.data_dir,
            artifact_name=args.artifact_name,
            artifact_version=args.artifact_version,
            model_version=args.model_version,
            split=args.split,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tiny dependency-free neural smoke model for gitctx artifacts.

This module intentionally implements only a single-layer softmax classifier.
It is a real gradient-descent training loop, but it is not a language model and
does not establish model quality. Its purpose is to validate neural-style
checkpoint and eval artifact plumbing before larger training code exists.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any

from gitctx.conventional import CommitContext, DEFAULT_TYPES, parse_commit_message, score_commit_message
from gitctx.provenance import load_jsonl
from gitctx.train_artifacts import training_examples_path

MODEL_DIR = Path("artifacts/models")
EVAL_DIR = Path("artifacts/eval")
DEFAULT_MODEL_VERSION = "tiny-softmax-v0"
DEFAULT_TRAIN_SPLIT = "DEV"
DEFAULT_EVAL_SPLIT = "REPORT"
DEFAULT_EPOCHS = 25
DEFAULT_LEARNING_RATE = 0.35
DEFAULT_L2 = 0.0001

CLASS_LABELS = tuple(sorted(DEFAULT_TYPES))
GENERIC_SCOPE_TOKENS = frozenset(
    {
        "a",
        "app",
        "apps",
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


def neural_model_path(
    artifact_name: str,
    *,
    artifact_version: str = "v0",
    model_version: str = DEFAULT_MODEL_VERSION,
) -> Path:
    """Return the tiny neural model artifact path."""

    _validate_identifier(artifact_name, "artifact_name")
    _validate_identifier(artifact_version, "artifact_version")
    _validate_identifier(model_version, "model_version")
    return MODEL_DIR / f"{model_version}.{artifact_name}.{artifact_version}.json"


def neural_prediction_path(
    artifact_name: str,
    split: str,
    *,
    artifact_version: str = "v0",
    model_version: str = DEFAULT_MODEL_VERSION,
) -> Path:
    """Return the tiny neural prediction JSONL path."""

    _validate_identifier(artifact_name, "artifact_name")
    _validate_identifier(artifact_version, "artifact_version")
    _validate_identifier(model_version, "model_version")
    _validate_split(split)
    return EVAL_DIR / f"{model_version}.{artifact_name}.{artifact_version}.{split.lower()}.predictions.jsonl"


def neural_eval_report_path(
    artifact_name: str,
    split: str,
    *,
    artifact_version: str = "v0",
    model_version: str = DEFAULT_MODEL_VERSION,
) -> Path:
    """Return the tiny neural eval report path."""

    _validate_identifier(artifact_name, "artifact_name")
    _validate_identifier(artifact_version, "artifact_version")
    _validate_identifier(model_version, "model_version")
    _validate_split(split)
    return EVAL_DIR / f"{model_version}.{artifact_name}.{artifact_version}.{split.lower()}.report.json"


def train_tiny_neural_model(
    data_dir: str | Path,
    *,
    artifact_name: str,
    artifact_version: str = "v0",
    model_version: str = DEFAULT_MODEL_VERSION,
    train_split: str = DEFAULT_TRAIN_SPLIT,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    l2: float = DEFAULT_L2,
) -> dict[str, Any]:
    """Train a tiny softmax classifier over path/diff-stat features."""

    _validate_split(train_split)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")

    data_dir = Path(data_dir)
    records = [
        record
        for record in load_jsonl(data_dir / training_examples_path(artifact_name, version=artifact_version))
        if record["data_split"] == train_split
    ]
    if not records:
        raise ValueError(f"no training records found for split {train_split}")
    examples = [_training_example(record) for record in sorted(records, key=lambda item: item["id"])]

    class_to_index = {label: index for index, label in enumerate(CLASS_LABELS)}
    weights: dict[str, list[float]] = {}
    bias = [0.0 for _ in CLASS_LABELS]
    loss_history: list[float] = []

    for _epoch in range(epochs):
        total_loss = 0.0
        for features, label, _record in examples:
            target = class_to_index[label]
            logits = _logits(features, weights, bias)
            probs = _softmax(logits)
            total_loss += -math.log(max(probs[target], 1e-12))
            for class_index, prob in enumerate(probs):
                grad = prob - (1.0 if class_index == target else 0.0)
                bias[class_index] -= learning_rate * grad
                for feature, value in features.items():
                    row = weights.setdefault(feature, [0.0 for _ in CLASS_LABELS])
                    row[class_index] -= learning_rate * (grad * value + l2 * row[class_index])
        loss_history.append(round(total_loss / len(examples), 6))

    type_counts = Counter(label for _features, label, _record in examples)
    model = {
        "model_kind": "dependency_free_single_layer_softmax_classifier",
        "model_version": model_version,
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "train_split": train_split,
        "training_records": len(examples),
        "intended_use": "neural training pipeline smoke only; not a model-quality claim",
        "classes": list(CLASS_LABELS),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "loss_history": loss_history,
        "target_type_counts": dict(sorted(type_counts.items())),
        "bias": [round(value, 8) for value in bias],
        "weights": {
            feature: [round(value, 8) for value in values]
            for feature, values in sorted(weights.items())
        },
    }
    output_path = data_dir / neural_model_path(
        artifact_name,
        artifact_version=artifact_version,
        model_version=model_version,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for key in (
        "model_kind",
        "model_version",
        "artifact_name",
        "training_records",
    ):
        print(key, model[key])
    print("loss_first", loss_history[0])
    print("loss_last", loss_history[-1])
    return model


def evaluate_tiny_neural_model(
    data_dir: str | Path,
    *,
    artifact_name: str,
    artifact_version: str = "v0",
    model_version: str = DEFAULT_MODEL_VERSION,
    split: str = DEFAULT_EVAL_SPLIT,
) -> dict[str, Any]:
    """Evaluate the tiny neural model on one artifact split."""

    _validate_split(split)
    data_dir = Path(data_dir)
    model_path = data_dir / neural_model_path(
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
    prediction_path = data_dir / neural_prediction_path(
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
    report_path = data_dir / neural_eval_report_path(
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


def _training_example(record: dict[str, Any]) -> tuple[dict[str, float], str, dict[str, Any]]:
    try:
        parsed = parse_commit_message(record["target_message"])
    except ValueError as exc:
        raise ValueError(f"invalid training target for {record['id']}") from exc
    if parsed.type not in CLASS_LABELS:
        raise ValueError(f"unsupported target type for {record['id']}: {parsed.type}")
    return _features(record), parsed.type, record


def _prediction_record(model: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    prediction = _predict_message(model, record)
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


def _predict_message(model: dict[str, Any], record: dict[str, Any]) -> str:
    features = _features(record)
    logits = _logits(
        features,
        {key: list(value) for key, value in model["weights"].items()},
        list(model["bias"]),
    )
    predicted_type = model["classes"][_argmax(logits)]
    topic = _topic_from_paths(record["changed_paths"])
    subject = _subject_for_type(predicted_type, topic)
    return f"{predicted_type}: {subject}"


def _features(record: dict[str, Any]) -> dict[str, float]:
    features: Counter[str] = Counter()
    features["bias"] = 1
    changed_paths = record.get("changed_paths", [])
    for path in changed_paths:
        normalized = path.replace("\\", "/").strip("/").lower()
        parts = normalized.split("/")
        if parts:
            features[f"topdir:{parts[0]}"] += 1
        if "." in parts[-1]:
            features[f"ext:{parts[-1].rsplit('.', 1)[1]}"] += 1
        for token in _path_tokens([path]):
            features[f"path:{token}"] += 1
    if any(_path_looks_like_docs(path) for path in changed_paths):
        features["flag:docs"] = 1
    if any(_path_looks_like_tests(path) for path in changed_paths):
        features["flag:tests"] = 1
    diff_stat = record.get("diff_stat", {})
    if isinstance(diff_stat, dict):
        for key in ("files_changed", "insertions", "deletions"):
            value = diff_stat.get(key)
            if isinstance(value, int):
                features[f"stat:{key}"] = min(value, 50) / 50.0
    total = sum(abs(value) for value in features.values()) or 1.0
    return {key: value / total for key, value in sorted(features.items())}


def _logits(
    features: dict[str, float],
    weights: dict[str, list[float]],
    bias: list[float],
) -> list[float]:
    logits = list(bias)
    for feature, value in features.items():
        row = weights.get(feature)
        if row is None:
            continue
        for index, weight in enumerate(row):
            logits[index] += weight * value
    return logits


def _softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exps)
    return [value / total for value in exps]


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _subject_for_type(commit_type: str, topic: str) -> str:
    if commit_type == "docs":
        return f"update {topic} documentation"
    if commit_type == "test":
        return f"cover {topic} behavior"
    if commit_type == "fix":
        return f"handle {topic} behavior"
    if commit_type == "feat":
        return f"add {topic} support"
    if commit_type == "perf":
        return f"improve {topic} performance"
    if commit_type == "refactor":
        return f"update {topic} implementation"
    if commit_type == "build":
        return f"update {topic} build"
    if commit_type == "ci":
        return f"update {topic} automation"
    if commit_type == "style":
        return f"format {topic} code"
    if commit_type == "revert":
        return f"revert {topic} change"
    return f"update {topic} maintenance"


def _topic_from_paths(changed_paths: list[str]) -> str:
    for token in _path_tokens(changed_paths):
        if token not in GENERIC_SCOPE_TOKENS:
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
            neural_prediction_path(
                artifact_name,
                split,
                artifact_version=artifact_version,
                model_version=model_version,
            )
        ),
        "report_path": str(
            neural_eval_report_path(
                artifact_name,
                split,
                artifact_version=artifact_version,
                model_version=model_version,
            )
        ),
        "loss_first": model["loss_history"][0],
        "loss_last": model["loss_history"][-1],
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
        raise ValueError("split must be DEV or REPORT for tiny neural model artifacts")


def _validate_identifier(value: str, name: str) -> None:
    if not value.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"{name} must be a stable identifier")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate the gitctx tiny neural smoke model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--artifact-name", required=True)
    train.add_argument("--artifact-version", default="v0")
    train.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    train.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    train.add_argument("--l2", type=float, default=DEFAULT_L2)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--artifact-name", required=True)
    evaluate.add_argument("--artifact-version", default="v0")
    evaluate.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    evaluate.add_argument("--split", default=DEFAULT_EVAL_SPLIT)

    args = parser.parse_args(argv)
    if args.command == "train":
        train_tiny_neural_model(
            args.data_dir,
            artifact_name=args.artifact_name,
            artifact_version=args.artifact_version,
            model_version=args.model_version,
            train_split=args.train_split,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
    elif args.command == "evaluate":
        evaluate_tiny_neural_model(
            args.data_dir,
            artifact_name=args.artifact_name,
            artifact_version=args.artifact_version,
            model_version=args.model_version,
            split=args.split,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate reviewed SFT artifacts against deterministic baselines."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from gitctx.conventional import CommitContext, score_commit_message
from gitctx.ollama_generate import generated_labels_path
from gitctx.provenance import load_jsonl
from gitctx.train_artifacts import format_commit_message, training_examples_path

EVAL_DIR = Path("artifacts/eval")


def eval_report_path(artifact_name: str, *, version: str = "v0") -> Path:
    """Return the baseline eval report path for a named training artifact."""

    _validate_artifact_name(artifact_name)
    _validate_version(version)
    return EVAL_DIR / f"sft.{artifact_name}.{version}.baseline.report.json"


def evaluate_training_artifact(
    data_dir: str | Path,
    *,
    artifact_name: str,
    version: str = "v0",
) -> dict[str, Any]:
    """Score reviewed targets, raw teacher labels, and historical subjects."""

    _validate_artifact_name(artifact_name)
    _validate_version(version)
    data_dir = Path(data_dir)
    training_records = load_jsonl(data_dir / training_examples_path(artifact_name, version=version))
    generated_labels = load_jsonl(data_dir / generated_labels_path(artifact_name))
    generated_by_id = {record["id"]: record for record in generated_labels}
    failures: list[dict[str, str]] = []

    target_messages: list[tuple[dict[str, Any], str]] = []
    teacher_messages: list[tuple[dict[str, Any], str]] = []
    historical_messages: list[tuple[dict[str, Any], str]] = []
    changed_from_teacher = 0

    for record in training_records:
        generated_label = generated_by_id.get(record["generated_label_id"])
        if generated_label is None:
            failures.append(
                {
                    "id": record["id"],
                    "error": f"missing generated label: {record['generated_label_id']}",
                }
            )
            continue
        teacher_message = format_commit_message(
            generated_label["header"],
            generated_label["body"],
            generated_label["footers"],
        )
        if teacher_message != record["target_message"]:
            changed_from_teacher += 1
        target_messages.append((record, record["target_message"]))
        teacher_messages.append((record, teacher_message))
        historical_messages.append((record, record["historical_subject"]))

    if failures:
        report = {
            "artifact_name": artifact_name,
            "artifact_version": version,
            "failed_records": len(failures),
            "failures": failures,
        }
        _write_report(data_dir, artifact_name, version, report)
        raise SystemExit(1)

    report = {
        "artifact_name": artifact_name,
        "artifact_version": version,
        "training_records": len(training_records),
        "output_path": str(eval_report_path(artifact_name, version=version)),
        "label_source_counts": dict(sorted(Counter(r["label_source"] for r in training_records).items())),
        "review_decision_counts": dict(
            sorted(Counter(r["review_decision"] for r in training_records).items())
        ),
        "source_license_counts": dict(
            sorted(Counter(r["source_license"] for r in training_records).items())
        ),
        "data_split_counts": dict(sorted(Counter(r["data_split"] for r in training_records).items())),
        "changed_from_teacher_records": changed_from_teacher,
        "target": _score_messages(target_messages),
        "teacher": _score_messages(teacher_messages),
        "historical": _score_messages(historical_messages),
        "by_data_split": _score_by_data_split(
            target_messages=target_messages,
            teacher_messages=teacher_messages,
            historical_messages=historical_messages,
        ),
    }
    _write_report(data_dir, artifact_name, version, report)
    for key in (
        "artifact_name",
        "artifact_version",
        "training_records",
        "changed_from_teacher_records",
    ):
        print(key, report[key])
    for label in ("target", "teacher", "historical"):
        print(f"{label}_format_validity", report[label]["format_validity"]["true"])
        print(f"{label}_scope_quality", report[label]["scope_quality"]["true"])
    return report


def _score_by_data_split(
    *,
    target_messages: list[tuple[dict[str, Any], str]],
    teacher_messages: list[tuple[dict[str, Any], str]],
    historical_messages: list[tuple[dict[str, Any], str]],
) -> dict[str, Any]:
    splits = sorted({record["data_split"] for record, _ in target_messages})
    report: dict[str, Any] = {}
    for split in splits:
        split_target = [(record, message) for record, message in target_messages if record["data_split"] == split]
        split_teacher = [
            (record, message) for record, message in teacher_messages if record["data_split"] == split
        ]
        split_historical = [
            (record, message)
            for record, message in historical_messages
            if record["data_split"] == split
        ]
        report[split] = {
            "training_records": len(split_target),
            "target": _score_messages(split_target),
            "teacher": _score_messages(split_teacher),
            "historical": _score_messages(split_historical),
        }
    return report


def _score_messages(records: list[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    counters = {
        "format_validity": Counter(),
        "scope_quality": Counter(),
        "specificity": Counter(),
        "brevity": Counter(),
    }
    type_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()

    for record, message in records:
        score = score_commit_message(
            message,
            CommitContext(changed_paths=tuple(record["changed_paths"])),
        )
        counters["format_validity"][_bucket(score.format_validity)] += 1
        counters["scope_quality"][_bucket(score.scope_quality)] += 1
        counters["specificity"][_bucket(score.specificity)] += 1
        counters["brevity"][_bucket(score.brevity)] += 1
        if score.parsed is not None:
            type_counts[score.parsed.type] += 1
        for error in score.errors:
            error_counts[error] += 1

    return {
        **{name: _counter_to_report(counter) for name, counter in counters.items()},
        "type_counts": dict(sorted(type_counts.items())),
        "top_errors": dict(error_counts.most_common(10)),
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


def _write_report(
    data_dir: Path,
    artifact_name: str,
    version: str,
    report: dict[str, Any],
) -> Path:
    output_path = data_dir / eval_report_path(artifact_name, version=version)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _validate_artifact_name(artifact_name: str) -> None:
    if not artifact_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("artifact_name must be a stable alphanumeric identifier")


def _validate_version(version: str) -> None:
    if not version.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("version must be a stable alphanumeric identifier")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate gitctx training artifacts.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--artifact-name", required=True)
    evaluate.add_argument("--version", default="v0")
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        evaluate_training_artifact(
            args.data_dir,
            artifact_name=args.artifact_name,
            version=args.version,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

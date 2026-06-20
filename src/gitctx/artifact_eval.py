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


def split_inspection_path(artifact_name: str, split: str, *, version: str = "v0") -> Path:
    """Return the per-record inspection JSONL path for a training-artifact split."""

    _validate_artifact_name(artifact_name)
    _validate_version(version)
    _validate_split(split)
    return EVAL_DIR / f"sft.{artifact_name}.{version}.{split.lower()}.inspection.jsonl"


def split_inspection_report_path(artifact_name: str, split: str, *, version: str = "v0") -> Path:
    """Return the per-record inspection report path for a training-artifact split."""

    _validate_artifact_name(artifact_name)
    _validate_version(version)
    _validate_split(split)
    return EVAL_DIR / f"sft.{artifact_name}.{version}.{split.lower()}.inspection.report.json"


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


def inspect_training_artifact_split(
    data_dir: str | Path,
    *,
    artifact_name: str,
    split: str,
    version: str = "v0",
) -> dict[str, Any]:
    """Write per-record scoring material for a single training-artifact split."""

    _validate_artifact_name(artifact_name)
    _validate_version(version)
    _validate_split(split)
    data_dir = Path(data_dir)
    training_records = [
        record
        for record in load_jsonl(data_dir / training_examples_path(artifact_name, version=version))
        if record["data_split"] == split
    ]
    generated_labels = load_jsonl(data_dir / generated_labels_path(artifact_name))
    generated_by_id = {record["id"]: record for record in generated_labels}
    output_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

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
        output_records.append(
            _inspection_record(
                record=record,
                teacher_message=teacher_message,
                historical_message=record["historical_subject"],
            )
        )

    output_path = data_dir / split_inspection_path(artifact_name, split, version=version)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in output_records),
        encoding="utf-8",
    )

    report = {
        "artifact_name": artifact_name,
        "artifact_version": version,
        "split": split,
        "training_records": len(training_records),
        "inspection_records": len(output_records),
        "failed_records": len(failures),
        "failures": failures,
        "output_path": str(split_inspection_path(artifact_name, split, version=version)),
        "report_path": str(split_inspection_report_path(artifact_name, split, version=version)),
        "review_decision_counts": dict(
            sorted(Counter(record["review_decision"] for record in training_records).items())
        ),
        "target": _score_messages([(record, record["target_message"]) for record in training_records]),
        "teacher": _score_messages(
            [
                (
                    record,
                    format_commit_message(
                        generated_by_id[record["generated_label_id"]]["header"],
                        generated_by_id[record["generated_label_id"]]["body"],
                        generated_by_id[record["generated_label_id"]]["footers"],
                    ),
                )
                for record in training_records
                if record["generated_label_id"] in generated_by_id
            ]
        ),
        "historical": _score_messages(
            [(record, record["historical_subject"]) for record in training_records]
        ),
    }
    report_path = data_dir / split_inspection_report_path(artifact_name, split, version=version)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for key in ("artifact_name", "artifact_version", "split", "training_records", "inspection_records"):
        print(key, report[key])
    if failures:
        return report
    return report


def _inspection_record(
    *,
    record: dict[str, Any],
    teacher_message: str,
    historical_message: str,
) -> dict[str, Any]:
    target_score = _score_one(record, record["target_message"])
    teacher_score = _score_one(record, teacher_message)
    historical_score = _score_one(record, historical_message)
    return {
        "id": record["id"],
        "source_diff_id": record["source_diff_id"],
        "source_repo_url": record["source_repo_url"],
        "source_commit": record["source_commit"],
        "data_split": record["data_split"],
        "changed_paths": record["changed_paths"],
        "diff_stat": record["diff_stat"],
        "review_decision": record["review_decision"],
        "review_issues": record["review_issues"],
        "review_notes": record["review_notes"],
        "label_source": record["label_source"],
        "target_message": record["target_message"],
        "teacher_message": teacher_message,
        "historical_subject": historical_message,
        "target_score": target_score,
        "teacher_score": teacher_score,
        "historical_score": historical_score,
    }


def _score_one(record: dict[str, Any], message: str) -> dict[str, Any]:
    score = score_commit_message(
        message,
        CommitContext(changed_paths=tuple(record["changed_paths"])),
    )
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


def _validate_split(split: str) -> None:
    if split not in {"DEV", "REPORT", "HELD_OUT"}:
        raise ValueError("split must be DEV, REPORT, or HELD_OUT")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate gitctx training artifacts.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--artifact-name", required=True)
    evaluate.add_argument("--version", default="v0")
    inspect_split = subparsers.add_parser("inspect-split")
    inspect_split.add_argument("--artifact-name", required=True)
    inspect_split.add_argument("--split", required=True)
    inspect_split.add_argument("--version", default="v0")
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        evaluate_training_artifact(
            args.data_dir,
            artifact_name=args.artifact_name,
            version=args.version,
        )
    elif args.command == "inspect-split":
        inspect_training_artifact_split(
            args.data_dir,
            artifact_name=args.artifact_name,
            split=args.split,
            version=args.version,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

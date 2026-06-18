"""Utilities for private gitctx-data smoke artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from gitctx.provenance import (
    load_jsonl,
    validate_generated_label_review_decision,
    validate_source_diff_record,
    validate_source_diff_review_decision,
    validate_source_manifest_entry,
)

SMOKE_JSONL = Path("artifacts/smoke/source-diffs.smoke.jsonl")
SMOKE_REPORT = Path("artifacts/smoke/source-diffs.smoke.report.json")
SMOKE_REVIEW = Path("reviews/source-diffs.smoke.review.jsonl")
SMOKE_TEACHER_INPUTS = Path("artifacts/teacher/teacher-inputs.smoke.jsonl")
SMOKE_GENERATED_LABELS = Path("artifacts/teacher/generated-labels.smoke.jsonl")
SMOKE_GENERATED_REPORT = Path("artifacts/teacher/generated-labels.smoke.report.json")
SMOKE_GENERATED_REVIEW = Path("reviews/generated-labels.smoke.review.jsonl")
SMOKE_MANIFEST = Path("manifests/source-manifest.audit.jsonl")
GITCTX_COMMIT = Path("lineage/gitctx-public-commit.txt")
CHECKSUMS = Path("checksums/sha256.txt")
REVIEW_PROTOCOL = "source-diff-smoke-review-v0.1"
GENERATED_LABEL_REVIEW_PROTOCOL = "generated-label-smoke-review-v0.1"


def normalize_smoke_report(data_dir: str | Path) -> dict[str, Any]:
    """Normalize machine-local paths in the smoke report."""

    data_dir = Path(data_dir)
    report_path = data_dir / SMOKE_REPORT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["data_dir"] = "$GITCTX_DATA_DIR"
    report["manifest_path"] = str(SMOKE_MANIFEST)
    report["output_path"] = str(SMOKE_JSONL)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_smoke_artifact(data_dir: str | Path) -> dict[str, int]:
    """Validate source-diff smoke records and their manifest."""

    data_dir = Path(data_dir)
    source_records = load_jsonl(data_dir / SMOKE_JSONL)
    manifest_records = load_jsonl(data_dir / SMOKE_MANIFEST)

    source_errors = [
        (record.get("id", "<missing-id>"), validate_source_diff_record(record))
        for record in source_records
    ]
    manifest_errors = [
        (record.get("repo_url", "<missing-repo>"), validate_source_manifest_entry(record))
        for record in manifest_records
    ]

    failed_sources = [(name, errors) for name, errors in source_errors if errors]
    failed_manifests = [(name, errors) for name, errors in manifest_errors if errors]
    if failed_sources or failed_manifests:
        for name, errors in failed_sources + failed_manifests:
            print(f"FAIL {name}: {errors}")
        raise SystemExit(1)

    summary = {
        "source_records": len(source_records),
        "manifest_records": len(manifest_records),
        "source_errors": 0,
        "manifest_errors": 0,
    }
    for key, value in summary.items():
        print(key, value)
    return summary


def create_smoke_review_template(
    data_dir: str | Path,
    *,
    reviewer: str,
    overwrite: bool = False,
) -> Path:
    """Create a review-decision JSONL template for the smoke source diffs."""

    data_dir = Path(data_dir)
    output_path = data_dir / SMOKE_REVIEW
    if output_path.exists() and not overwrite:
        raise SystemExit(f"{output_path} already exists; pass --overwrite to replace it")

    source_records = load_jsonl(data_dir / SMOKE_JSONL)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for record in source_records:
        decision = {
            "id": f"review-{record['id']}",
            "source_diff_id": record["id"],
            "source_repo_url": record["source_repo_url"],
            "source_commit": record["source_commit"],
            "parent_commit": record["parent_commit"],
            "data_split": record["data_split"],
            "changed_paths": record["changed_paths"],
            "decision": "needs_review",
            "reasons": [],
            "notes": "",
            "reviewer": reviewer,
            "review_timestamp": "TBD",
            "review_protocol": REVIEW_PROTOCOL,
        }
        lines.append(json.dumps(decision, sort_keys=True))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def validate_smoke_review(data_dir: str | Path) -> dict[str, int]:
    """Validate smoke review decisions against the source-diff artifact."""

    data_dir = Path(data_dir)
    source_records = load_jsonl(data_dir / SMOKE_JSONL)
    review_records = load_jsonl(data_dir / SMOKE_REVIEW)
    source_by_id = {record["id"]: record for record in source_records}
    review_ids: set[str] = set()
    failures: list[tuple[str, tuple[str, ...]]] = []
    decision_counts = {
        "needs_review": 0,
        "accepted_for_teacher_labeling": 0,
        "rejected": 0,
    }

    for record in review_records:
        record_id = record.get("id", "<missing-id>")
        errors = list(validate_source_diff_review_decision(record))
        source_diff_id = record.get("source_diff_id")
        source_record = source_by_id.get(source_diff_id)
        if source_diff_id in review_ids:
            errors.append(f"duplicate review for source_diff_id: {source_diff_id}")
        if isinstance(source_diff_id, str):
            review_ids.add(source_diff_id)
        if source_record is None:
            errors.append(f"unknown source_diff_id: {source_diff_id}")
        else:
            for key in (
                "source_repo_url",
                "source_commit",
                "parent_commit",
                "data_split",
                "changed_paths",
            ):
                if record.get(key) != source_record.get(key):
                    errors.append(f"{key} does not match source diff record")
        decision = record.get("decision")
        if isinstance(decision, str) and decision in decision_counts:
            decision_counts[decision] += 1
        if errors:
            failures.append((str(record_id), tuple(errors)))

    missing_reviews = sorted(set(source_by_id) - review_ids)
    if missing_reviews:
        failures.append(("<missing-reviews>", tuple(missing_reviews)))

    if failures:
        for name, errors in failures:
            print(f"FAIL {name}: {errors}")
        raise SystemExit(1)

    summary = {
        "source_records": len(source_records),
        "review_records": len(review_records),
        **decision_counts,
    }
    for key, value in summary.items():
        print(key, value)
    return summary


def create_generated_label_review_template(
    data_dir: str | Path,
    *,
    reviewer: str,
    overwrite: bool = False,
) -> Path:
    """Create a human-review JSONL template for generated smoke labels."""

    data_dir = Path(data_dir)
    output_path = data_dir / SMOKE_GENERATED_REVIEW
    if output_path.exists() and not overwrite:
        raise SystemExit(f"{output_path} already exists; pass --overwrite to replace it")

    generated_labels = load_jsonl(data_dir / SMOKE_GENERATED_LABELS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for label in generated_labels:
        decision = {
            "id": f"review-{label['id']}",
            "generated_label_id": label["id"],
            "source_repo_url": label["source_repo_url"],
            "source_commit": label["source_commit"],
            "teacher_model_id": label["teacher_model_id"],
            "prompt_version": label["prompt_version"],
            "header": label["header"],
            "verifier_score": label["verifier_score"],
            "decision": "needs_review",
            "issues": [],
            "edited_header": None,
            "edited_body": None,
            "notes": "",
            "reviewer": reviewer,
            "review_timestamp": "TBD",
            "review_protocol": GENERATED_LABEL_REVIEW_PROTOCOL,
        }
        lines.append(json.dumps(decision, sort_keys=True))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def validate_generated_label_review(data_dir: str | Path) -> dict[str, int]:
    """Validate generated-label human-review decisions."""

    data_dir = Path(data_dir)
    generated_labels = load_jsonl(data_dir / SMOKE_GENERATED_LABELS)
    review_records = load_jsonl(data_dir / SMOKE_GENERATED_REVIEW)
    labels_by_id = {record["id"]: record for record in generated_labels}
    review_ids: set[str] = set()
    failures: list[tuple[str, tuple[str, ...]]] = []
    decision_counts = {
        "needs_review": 0,
        "accept": 0,
        "edit": 0,
        "reject": 0,
    }

    for record in review_records:
        record_id = record.get("id", "<missing-id>")
        errors = list(validate_generated_label_review_decision(record))
        generated_label_id = record.get("generated_label_id")
        label = labels_by_id.get(generated_label_id)
        if generated_label_id in review_ids:
            errors.append(f"duplicate review for generated_label_id: {generated_label_id}")
        if isinstance(generated_label_id, str):
            review_ids.add(generated_label_id)
        if label is None:
            errors.append(f"unknown generated_label_id: {generated_label_id}")
        else:
            for key in (
                "source_repo_url",
                "source_commit",
                "teacher_model_id",
                "prompt_version",
                "header",
            ):
                if record.get(key) != label.get(key):
                    errors.append(f"{key} does not match generated label")
        decision = record.get("decision")
        if isinstance(decision, str) and decision in decision_counts:
            decision_counts[decision] += 1
        if errors:
            failures.append((str(record_id), tuple(errors)))

    missing_reviews = sorted(set(labels_by_id) - review_ids)
    if missing_reviews:
        failures.append(("<missing-generated-label-reviews>", tuple(missing_reviews)))

    if failures:
        for name, errors in failures:
            print(f"FAIL {name}: {errors}")
        raise SystemExit(1)

    summary = {
        "generated_label_records": len(generated_labels),
        "review_records": len(review_records),
        **decision_counts,
    }
    for key, value in summary.items():
        print(key, value)
    return summary


def write_checksums(data_dir: str | Path) -> Path:
    """Write sha256 checksums for tracked smoke lineage files."""

    data_dir = Path(data_dir)
    checksum_path = data_dir / CHECKSUMS
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [SMOKE_JSONL, SMOKE_REPORT, SMOKE_MANIFEST, GITCTX_COMMIT]
    if (data_dir / SMOKE_REVIEW).exists():
        paths.append(SMOKE_REVIEW)
    if (data_dir / SMOKE_TEACHER_INPUTS).exists():
        paths.append(SMOKE_TEACHER_INPUTS)
    if (data_dir / SMOKE_GENERATED_LABELS).exists():
        paths.append(SMOKE_GENERATED_LABELS)
    if (data_dir / SMOKE_GENERATED_REPORT).exists():
        paths.append(SMOKE_GENERATED_REPORT)
    if (data_dir / SMOKE_GENERATED_REVIEW).exists():
        paths.append(SMOKE_GENERATED_REVIEW)
    lines = []
    for relative_path in paths:
        digest = _sha256(data_dir / relative_path)
        lines.append(f"{digest}  {relative_path.as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage private gitctx-data artifacts.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("normalize-smoke")
    subparsers.add_parser("validate-smoke")
    review_template = subparsers.add_parser("create-smoke-review-template")
    review_template.add_argument("--reviewer", required=True)
    review_template.add_argument("--overwrite", action="store_true")
    subparsers.add_parser("validate-smoke-review")
    generated_review_template = subparsers.add_parser("create-generated-label-review-template")
    generated_review_template.add_argument("--reviewer", required=True)
    generated_review_template.add_argument("--overwrite", action="store_true")
    subparsers.add_parser("validate-generated-label-review")
    subparsers.add_parser("write-checksums")
    args = parser.parse_args(argv)

    if args.command == "normalize-smoke":
        normalize_smoke_report(args.data_dir)
    elif args.command == "validate-smoke":
        validate_smoke_artifact(args.data_dir)
    elif args.command == "create-smoke-review-template":
        print(
            create_smoke_review_template(
                args.data_dir,
                reviewer=args.reviewer,
                overwrite=args.overwrite,
            )
        )
    elif args.command == "validate-smoke-review":
        validate_smoke_review(args.data_dir)
    elif args.command == "create-generated-label-review-template":
        print(
            create_generated_label_review_template(
                args.data_dir,
                reviewer=args.reviewer,
                overwrite=args.overwrite,
            )
        )
    elif args.command == "validate-generated-label-review":
        validate_generated_label_review(args.data_dir)
    elif args.command == "write-checksums":
        print(write_checksums(args.data_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

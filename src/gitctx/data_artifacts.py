"""Utilities for private gitctx-data smoke artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
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
SOURCE_REVIEW_POLICY_VERSION = "source-review-policy-v0.1"
GENERATED_LABEL_REVIEW_POLICY_VERSION = "generated-label-review-policy-v0.1"
_NOISE_SUBJECT_RE = re.compile(
    r"\b(merge|revert|release|bump|deps?|dependenc|changelog|change log|"
    r"pre-commit|version)\b",
    re.IGNORECASE,
)
_DOC_EXTENSIONS = frozenset({".md", ".rst", ".txt", ".adoc", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico"})
_CONFIG_FILENAMES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "cargo.lock",
        "go.sum",
        "package.json",
        ".pre-commit-config.yaml",
    }
)
_NOISE_PATH_PARTS = frozenset(
    {
        ".github",
        "docs",
        "doc",
        "documentation",
        "changelog",
        "changes",
        "news",
        "release",
        "releases",
        "vendor",
        "generated",
        "dist",
        "build",
        "coverage",
    }
)
_GENERATED_EXPECTED_OUTPUT_PARTS = frozenset({"baselines", "snapshots", "snapshot"})
_CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".vue",
        ".svelte",
    }
)
_TEST_PATH_PARTS = frozenset({"test", "tests", "testing", "spec", "specs", "cases"})


def source_jsonl_path(artifact_name: str) -> Path:
    """Return the source-diff JSONL path for a named artifact."""

    _validate_artifact_name(artifact_name)
    return Path("artifacts") / artifact_name / f"source-diffs.{artifact_name}.jsonl"


def source_report_path(artifact_name: str) -> Path:
    """Return the source-diff report path for a named artifact."""

    _validate_artifact_name(artifact_name)
    return Path("artifacts") / artifact_name / f"source-diffs.{artifact_name}.report.json"


def source_review_path(artifact_name: str) -> Path:
    """Return the source-diff review path for a named artifact."""

    _validate_artifact_name(artifact_name)
    return Path("reviews") / f"source-diffs.{artifact_name}.review.jsonl"


def generated_labels_path(artifact_name: str) -> Path:
    """Return the generated-label JSONL path for a named artifact."""

    _validate_artifact_name(artifact_name)
    return Path("artifacts/teacher") / f"generated-labels.{artifact_name}.jsonl"


def generated_label_review_path(artifact_name: str) -> Path:
    """Return the generated-label review path for a named artifact."""

    _validate_artifact_name(artifact_name)
    return Path("reviews") / f"generated-labels.{artifact_name}.review.jsonl"


def normalize_smoke_report(data_dir: str | Path) -> dict[str, Any]:
    """Normalize machine-local paths in the smoke report."""

    return normalize_source_report(data_dir, artifact_name="smoke")


def normalize_source_report(
    data_dir: str | Path,
    *,
    artifact_name: str,
    manifest_path: str | Path = SMOKE_MANIFEST,
    split_plan_path: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize machine-local paths in a named source-diff report."""

    data_dir = Path(data_dir)
    manifest_path = Path(manifest_path)
    jsonl_path = source_jsonl_path(artifact_name)
    report_path = data_dir / source_report_path(artifact_name)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["data_dir"] = "$GITCTX_DATA_DIR"
    report["manifest_path"] = manifest_path.as_posix()
    if split_plan_path is not None:
        report["split_plan_path"] = Path(split_plan_path).as_posix()
    elif isinstance(report.get("split_plan_path"), str):
        report["split_plan_path"] = _normalize_under_data_dir(
            report["split_plan_path"],
            data_dir=data_dir,
        )
    report["output_path"] = str(jsonl_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_smoke_artifact(data_dir: str | Path) -> dict[str, int]:
    """Validate source-diff smoke records and their manifest."""

    return validate_source_artifact(data_dir, artifact_name="smoke")


def validate_source_artifact(
    data_dir: str | Path,
    *,
    artifact_name: str,
    manifest_path: str | Path = SMOKE_MANIFEST,
) -> dict[str, int]:
    """Validate named source-diff records and their manifest."""

    data_dir = Path(data_dir)
    manifest_path = Path(manifest_path)
    source_records = load_jsonl(data_dir / source_jsonl_path(artifact_name))
    manifest_records = load_jsonl(data_dir / manifest_path)
    manifest_repos = {record.get("repo_url") for record in manifest_records}

    source_errors = []
    for record in source_records:
        record_errors = list(validate_source_diff_record(record))
        if record.get("source_repo_url") not in manifest_repos:
            record_errors.append(
                "source_repo_url is not present in the source manifest: "
                f"{record.get('source_repo_url')}"
            )
        source_errors.append((record.get("id", "<missing-id>"), tuple(record_errors)))
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
        "artifact_name": artifact_name,
        "manifest_path": manifest_path.as_posix(),
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

    return create_source_review_template(
        data_dir,
        artifact_name="smoke",
        reviewer=reviewer,
        overwrite=overwrite,
    )


def create_source_review_template(
    data_dir: str | Path,
    *,
    artifact_name: str,
    reviewer: str,
    overwrite: bool = False,
) -> Path:
    """Create a review-decision JSONL template for a named source artifact."""

    data_dir = Path(data_dir)
    output_path = data_dir / source_review_path(artifact_name)
    if output_path.exists() and not overwrite:
        raise SystemExit(f"{output_path} already exists; pass --overwrite to replace it")

    source_records = load_jsonl(data_dir / source_jsonl_path(artifact_name))
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
            "review_protocol": f"source-diff-{artifact_name}-review-v0.1",
        }
        lines.append(json.dumps(decision, sort_keys=True))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def validate_smoke_review(data_dir: str | Path) -> dict[str, int]:
    """Validate smoke review decisions against the source-diff artifact."""

    return validate_source_review(data_dir, artifact_name="smoke")


def validate_source_review(data_dir: str | Path, *, artifact_name: str) -> dict[str, int]:
    """Validate review decisions against a named source-diff artifact."""

    data_dir = Path(data_dir)
    source_records = load_jsonl(data_dir / source_jsonl_path(artifact_name))
    review_records = load_jsonl(data_dir / source_review_path(artifact_name))
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
        "artifact_name": artifact_name,
        "source_records": len(source_records),
        "review_records": len(review_records),
        **decision_counts,
    }
    for key, value in summary.items():
        print(key, value)
    return summary


def apply_source_review_policy(
    data_dir: str | Path,
    *,
    artifact_name: str,
    reviewer: str,
    review_timestamp: str = "TBD",
    overwrite_reviewed: bool = False,
    write: bool = False,
) -> dict[str, int]:
    """Fill source-diff review decisions with deterministic eligibility policy."""

    data_dir = Path(data_dir)
    source_records = load_jsonl(data_dir / source_jsonl_path(artifact_name))
    review_path = data_dir / source_review_path(artifact_name)
    review_records = load_jsonl(review_path)
    source_by_id = {record["id"]: record for record in source_records}
    updated_records = []
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    split_decision_counts: Counter[tuple[str, str]] = Counter()
    changed_records = 0
    preserved_records = 0

    for review in review_records:
        source = source_by_id.get(review.get("source_diff_id"))
        if source is None:
            updated_records.append(review)
            decision = str(review.get("decision"))
            decision_counts[decision] += 1
            split_decision_counts[(str(review.get("data_split")), decision)] += 1
            preserved_records += 1
            continue
        if review.get("decision") != "needs_review" and not overwrite_reviewed:
            updated_records.append(review)
            decision = str(review.get("decision"))
            decision_counts[decision] += 1
            split_decision_counts[(str(review.get("data_split")), decision)] += 1
            preserved_records += 1
            continue

        decision, reasons, notes = _source_review_policy_decision(source)
        updated = dict(review)
        updated["decision"] = decision
        updated["reasons"] = reasons
        updated["notes"] = notes
        updated["reviewer"] = reviewer
        updated["review_timestamp"] = review_timestamp
        updated_records.append(updated)
        decision_counts[decision] += 1
        split_decision_counts[(str(source.get("data_split")), decision)] += 1
        reason_counts.update(reasons)
        changed_records += 1

    if write:
        review_path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in updated_records) + "\n",
            encoding="utf-8",
        )

    summary = {
        "artifact_name": artifact_name,
        "source_records": len(source_records),
        "review_records": len(review_records),
        "changed_records": changed_records,
        "preserved_records": preserved_records,
        "needs_review": decision_counts["needs_review"],
        "accepted_for_teacher_labeling": decision_counts["accepted_for_teacher_labeling"],
        "rejected": decision_counts["rejected"],
    }
    for key, value in summary.items():
        print(key, value)
    for (split, decision), count in sorted(split_decision_counts.items()):
        print(f"split_{split}_{decision}", count)
    for reason, count in sorted(reason_counts.items()):
        print(f"reason_{reason}", count)
    if not write:
        print("dry_run", True)
    return summary


def create_generated_label_review_template(
    data_dir: str | Path,
    *,
    reviewer: str,
    overwrite: bool = False,
) -> Path:
    """Create a human-review JSONL template for generated smoke labels."""

    return create_named_generated_label_review_template(
        data_dir,
        artifact_name="smoke",
        reviewer=reviewer,
        overwrite=overwrite,
    )


def create_named_generated_label_review_template(
    data_dir: str | Path,
    *,
    artifact_name: str,
    reviewer: str,
    overwrite: bool = False,
) -> Path:
    """Create a human-review JSONL template for generated labels in a named artifact."""

    data_dir = Path(data_dir)
    output_path = data_dir / generated_label_review_path(artifact_name)
    if output_path.exists() and not overwrite:
        raise SystemExit(f"{output_path} already exists; pass --overwrite to replace it")

    generated_labels = load_jsonl(data_dir / generated_labels_path(artifact_name))
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
            "review_protocol": f"generated-label-{artifact_name}-review-v0.1",
        }
        lines.append(json.dumps(decision, sort_keys=True))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def validate_generated_label_review(data_dir: str | Path) -> dict[str, int]:
    """Validate generated-label human-review decisions."""

    return validate_named_generated_label_review(data_dir, artifact_name="smoke")


def validate_named_generated_label_review(
    data_dir: str | Path,
    *,
    artifact_name: str,
) -> dict[str, int]:
    """Validate generated-label human-review decisions for a named artifact."""

    data_dir = Path(data_dir)
    generated_labels = load_jsonl(data_dir / generated_labels_path(artifact_name))
    review_records = load_jsonl(data_dir / generated_label_review_path(artifact_name))
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
        "artifact_name": artifact_name,
        "generated_label_records": len(generated_labels),
        "review_records": len(review_records),
        **decision_counts,
    }
    for key, value in summary.items():
        print(key, value)
    return summary


def apply_generated_label_review_policy(
    data_dir: str | Path,
    *,
    artifact_name: str,
    reviewer: str,
    review_timestamp: str = "TBD",
    overwrite_reviewed: bool = False,
    write: bool = False,
) -> dict[str, int]:
    """Fill generated-label review decisions with a deterministic conservative policy."""

    data_dir = Path(data_dir)
    generated_labels = load_jsonl(data_dir / generated_labels_path(artifact_name))
    review_path = data_dir / generated_label_review_path(artifact_name)
    review_records = load_jsonl(review_path)
    labels_by_id = {record["id"]: record for record in generated_labels}
    updated_records = []
    decision_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    split_decision_counts: Counter[tuple[str, str]] = Counter()
    changed_records = 0
    preserved_records = 0

    for review in review_records:
        label = labels_by_id.get(review.get("generated_label_id"))
        if label is None:
            updated_records.append(review)
            decision = str(review.get("decision"))
            decision_counts[decision] += 1
            preserved_records += 1
            continue
        if review.get("decision") != "needs_review" and not overwrite_reviewed:
            updated_records.append(review)
            decision = str(review.get("decision"))
            decision_counts[decision] += 1
            split_decision_counts[(str(label.get("data_split")), decision)] += 1
            preserved_records += 1
            continue

        decision, issues, notes = _generated_label_review_policy_decision(label)
        updated = dict(review)
        updated["decision"] = decision
        updated["issues"] = issues
        updated["notes"] = notes
        updated["reviewer"] = reviewer
        updated["review_timestamp"] = review_timestamp
        updated_records.append(updated)
        decision_counts[decision] += 1
        split_decision_counts[(str(label.get("data_split")), decision)] += 1
        issue_counts.update(issues)
        changed_records += 1

    if write:
        review_path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in updated_records) + "\n",
            encoding="utf-8",
        )

    summary = {
        "artifact_name": artifact_name,
        "generated_label_records": len(generated_labels),
        "review_records": len(review_records),
        "changed_records": changed_records,
        "preserved_records": preserved_records,
        "needs_review": decision_counts["needs_review"],
        "accept": decision_counts["accept"],
        "edit": decision_counts["edit"],
        "reject": decision_counts["reject"],
    }
    for key, value in summary.items():
        print(key, value)
    for (split, decision), count in sorted(split_decision_counts.items()):
        print(f"split_{split}_{decision}", count)
    for issue, count in sorted(issue_counts.items()):
        print(f"issue_{issue}", count)
    if not write:
        print("dry_run", True)
    return summary


def write_checksums(data_dir: str | Path) -> Path:
    """Write sha256 checksums for tracked smoke lineage files."""

    data_dir = Path(data_dir)
    checksum_path = data_dir / CHECKSUMS
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [GITCTX_COMMIT]
    manifests_dir = data_dir / "manifests"
    if manifests_dir.exists():
        for manifest_path in sorted(manifests_dir.glob("*.json*")):
            paths.append(manifest_path.relative_to(data_dir))
    artifacts_dir = data_dir / "artifacts"
    if artifacts_dir.exists():
        for artifact_path in sorted(artifacts_dir.glob("*/*.json*")):
            paths.append(artifact_path.relative_to(data_dir))
    reviews_dir = data_dir / "reviews"
    if reviews_dir.exists():
        for review_path in sorted(reviews_dir.glob("*.jsonl")):
            paths.append(review_path.relative_to(data_dir))
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


def _source_review_policy_decision(record: dict[str, Any]) -> tuple[str, list[str], str]:
    changed_paths = list(record["changed_paths"])
    file_count, insertions, deletions = _diff_stat_counts(record.get("diff_stat", ""), changed_paths)
    churn = insertions + deletions
    lowered_paths = [path.lower() for path in changed_paths]

    if record.get("data_split") == "HELD_OUT":
        return _reject("held_out_reserved", "HELD_OUT source diffs stay reserved.")
    if _NOISE_SUBJECT_RE.search(record.get("historical_subject", "")):
        return _reject("noise_subject", "Historical subject indicates release, dependency, or merge noise.")
    if file_count > 30:
        return _reject("too_many_files", f"Touches {file_count} files.")
    if churn > 3000:
        return _reject("too_large_churn", f"Diffstat churn is {churn} changed lines.")
    if _has_generated_expected_output_bulk(lowered_paths, file_count=file_count, churn=churn):
        return _reject(
            "generated_expected_output_bulk",
            "Large generated expected-output update is not useful teacher input.",
        )
    if _all_paths_are_noise(lowered_paths):
        return _reject("noise_paths_only", "Only docs, release, config, lockfile, or generated paths.")

    if _has_code_or_test_signal(lowered_paths):
        return (
            "accepted_for_teacher_labeling",
            ["source_or_test_change"],
            f"{SOURCE_REVIEW_POLICY_VERSION}; deterministic source/test eligibility.",
        )
    if file_count <= 3 and churn <= 200:
        return (
            "accepted_for_teacher_labeling",
            ["small_other_change"],
            f"{SOURCE_REVIEW_POLICY_VERSION}; small non-noise change.",
        )
    return _reject("no_code_signal", "No source or test signal detected.")


def _reject(reason: str, note: str) -> tuple[str, list[str], str]:
    return "rejected", [reason], f"{SOURCE_REVIEW_POLICY_VERSION}; {note}"


def _generated_label_review_policy_decision(label: dict[str, Any]) -> tuple[str, list[str], str]:
    parser_result = label.get("parser_result")
    errors = []
    if isinstance(parser_result, dict):
        errors = [error for error in parser_result.get("errors", []) if isinstance(error, str)]
    verifier_score = label.get("verifier_score")
    if verifier_score == 1.0 and not errors:
        return (
            "accept",
            [],
            f"{GENERATED_LABEL_REVIEW_POLICY_VERSION}; verifier_score=1.0 with no parser errors.",
        )

    issues = _generated_label_review_policy_issues(errors, verifier_score)
    return (
        "reject",
        issues,
        f"{GENERATED_LABEL_REVIEW_POLICY_VERSION}; conservative reject for non-perfect verifier signal.",
    )


def _generated_label_review_policy_issues(
    errors: list[str],
    verifier_score: Any,
) -> list[str]:
    issues: set[str] = set()
    for error in errors:
        if "scope" in error:
            issues.add("scope_issue")
        if "type" in error:
            issues.add("type_issue")
        if "subject" in error or "length" in error or "vague" in error:
            issues.add("subject_issue")
        if "invalid Conventional Commit" in error or "unknown type" in error:
            issues.add("invalid_format")
    if not isinstance(verifier_score, (int, float)) or verifier_score < 1.0:
        issues.add("evidence_issue")
    return sorted(issues or {"factual_issue"})


def _diff_stat_counts(diff_stat: str, changed_paths: list[str]) -> tuple[int, int, int]:
    files_match = re.search(r"(\d+) files? changed", diff_stat)
    file_count = int(files_match.group(1)) if files_match else len(changed_paths)
    insertions = sum(
        int(match.replace(",", "")) for match in re.findall(r"(\d[\d,]*) insertion", diff_stat)
    )
    deletions = sum(
        int(match.replace(",", "")) for match in re.findall(r"(\d[\d,]*) deletion", diff_stat)
    )
    return file_count, insertions, deletions


def _has_generated_expected_output_bulk(
    lowered_paths: list[str],
    *,
    file_count: int,
    churn: int,
) -> bool:
    has_generated_path = any(_path_has_part(path, _GENERATED_EXPECTED_OUTPUT_PARTS) for path in lowered_paths)
    return has_generated_path and (file_count > 8 or churn > 800)


def _all_paths_are_noise(lowered_paths: list[str]) -> bool:
    return all(_path_is_noise(path) for path in lowered_paths)


def _path_is_noise(path: str) -> bool:
    filename = path.rsplit("/", 1)[-1]
    return (
        _path_has_part(path, _NOISE_PATH_PARTS)
        or filename in _CONFIG_FILENAMES
        or _path_extension(path) in _DOC_EXTENSIONS
    )


def _has_code_or_test_signal(lowered_paths: list[str]) -> bool:
    return any(_path_extension(path) in _CODE_EXTENSIONS for path in lowered_paths) or any(
        _path_has_part(path, _TEST_PATH_PARTS) for path in lowered_paths
    )


def _path_has_part(path: str, candidates: frozenset[str]) -> bool:
    parts = set(path.split("/")[:-1])
    parts.add(path.rsplit("/", 1)[-1])
    return bool(parts & candidates)


def _path_extension(path: str) -> str:
    filename = path.rsplit("/", 1)[-1]
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[1]


def _normalize_under_data_dir(path: str, *, data_dir: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.relative_to(data_dir).as_posix()
    except ValueError:
        return candidate.as_posix()


def _validate_artifact_name(artifact_name: str) -> None:
    if not artifact_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("artifact_name must be a stable alphanumeric identifier")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage private gitctx-data artifacts.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("normalize-smoke")
    subparsers.add_parser("validate-smoke")
    normalize_source = subparsers.add_parser("normalize-source")
    normalize_source.add_argument("--artifact-name", required=True)
    normalize_source.add_argument("--manifest", default=str(SMOKE_MANIFEST))
    normalize_source.add_argument("--split-plan")
    validate_source = subparsers.add_parser("validate-source")
    validate_source.add_argument("--artifact-name", required=True)
    validate_source.add_argument("--manifest", default=str(SMOKE_MANIFEST))
    review_template = subparsers.add_parser("create-smoke-review-template")
    review_template.add_argument("--reviewer", required=True)
    review_template.add_argument("--overwrite", action="store_true")
    subparsers.add_parser("validate-smoke-review")
    source_review_template = subparsers.add_parser("create-source-review-template")
    source_review_template.add_argument("--artifact-name", required=True)
    source_review_template.add_argument("--reviewer", required=True)
    source_review_template.add_argument("--overwrite", action="store_true")
    validate_source_review_parser = subparsers.add_parser("validate-source-review")
    validate_source_review_parser.add_argument("--artifact-name", required=True)
    source_review_policy = subparsers.add_parser("apply-source-review-policy")
    source_review_policy.add_argument("--artifact-name", required=True)
    source_review_policy.add_argument("--reviewer", required=True)
    source_review_policy.add_argument("--review-timestamp", default="TBD")
    source_review_policy.add_argument("--overwrite-reviewed", action="store_true")
    source_review_policy.add_argument("--write", action="store_true")
    generated_review_template = subparsers.add_parser("create-generated-label-review-template")
    generated_review_template.add_argument("--reviewer", required=True)
    generated_review_template.add_argument("--overwrite", action="store_true")
    subparsers.add_parser("validate-generated-label-review")
    named_generated_review_template = subparsers.add_parser(
        "create-named-generated-label-review-template"
    )
    named_generated_review_template.add_argument("--artifact-name", required=True)
    named_generated_review_template.add_argument("--reviewer", required=True)
    named_generated_review_template.add_argument("--overwrite", action="store_true")
    named_generated_review_check = subparsers.add_parser("validate-named-generated-label-review")
    named_generated_review_check.add_argument("--artifact-name", required=True)
    generated_review_policy = subparsers.add_parser("apply-generated-label-review-policy")
    generated_review_policy.add_argument("--artifact-name", required=True)
    generated_review_policy.add_argument("--reviewer", required=True)
    generated_review_policy.add_argument("--review-timestamp", default="TBD")
    generated_review_policy.add_argument("--overwrite-reviewed", action="store_true")
    generated_review_policy.add_argument("--write", action="store_true")
    subparsers.add_parser("write-checksums")
    args = parser.parse_args(argv)

    if args.command == "normalize-smoke":
        normalize_smoke_report(args.data_dir)
    elif args.command == "validate-smoke":
        validate_smoke_artifact(args.data_dir)
    elif args.command == "normalize-source":
        normalize_source_report(
            args.data_dir,
            artifact_name=args.artifact_name,
            manifest_path=args.manifest,
            split_plan_path=args.split_plan,
        )
    elif args.command == "validate-source":
        validate_source_artifact(
            args.data_dir,
            artifact_name=args.artifact_name,
            manifest_path=args.manifest,
        )
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
    elif args.command == "create-source-review-template":
        print(
            create_source_review_template(
                args.data_dir,
                artifact_name=args.artifact_name,
                reviewer=args.reviewer,
                overwrite=args.overwrite,
            )
        )
    elif args.command == "validate-source-review":
        validate_source_review(args.data_dir, artifact_name=args.artifact_name)
    elif args.command == "apply-source-review-policy":
        apply_source_review_policy(
            args.data_dir,
            artifact_name=args.artifact_name,
            reviewer=args.reviewer,
            review_timestamp=args.review_timestamp,
            overwrite_reviewed=args.overwrite_reviewed,
            write=args.write,
        )
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
    elif args.command == "create-named-generated-label-review-template":
        print(
            create_named_generated_label_review_template(
                args.data_dir,
                artifact_name=args.artifact_name,
                reviewer=args.reviewer,
                overwrite=args.overwrite,
            )
        )
    elif args.command == "validate-named-generated-label-review":
        validate_named_generated_label_review(args.data_dir, artifact_name=args.artifact_name)
    elif args.command == "apply-generated-label-review-policy":
        apply_generated_label_review_policy(
            args.data_dir,
            artifact_name=args.artifact_name,
            reviewer=args.reviewer,
            review_timestamp=args.review_timestamp,
            overwrite_reviewed=args.overwrite_reviewed,
            write=args.write,
        )
    elif args.command == "write-checksums":
        print(write_checksums(args.data_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

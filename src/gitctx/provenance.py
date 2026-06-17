"""Source-manifest and generated-label provenance validation."""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from pathlib import Path
from typing import Any

APPROVED_SOURCE_LICENSES = frozenset(
    {
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "MIT",
    }
)

REVIEW_REQUIRED_SOURCE_LICENSES = frozenset({"MPL-2.0"})
SOURCE_REVIEW_STATUSES = frozenset(
    {
        "approved_for_audit",
        "approved_for_training",
        "rejected",
        "pending",
    }
)
DATA_SPLITS = frozenset({"DEV", "REPORT", "HELD_OUT"})
SOURCE_DIFF_REVIEW_STATUSES = frozenset(
    {
        "not_reviewed",
        "accepted_for_smoke",
        "rejected",
    }
)
HUMAN_REVIEW_STATUSES = frozenset(
    {
        "not_reviewed",
        "accept_as_is",
        "accept_with_light_edit",
        "reject",
    }
)

_HEX_REVISION_RE = re.compile(r"^[0-9a-f]{7,40}$")
_PROMPT_VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def validate_source_manifest_entry(entry: Mapping[str, Any]) -> tuple[str, ...]:
    """Return validation errors for one source manifest entry."""

    errors: list[str] = []

    _require_str(entry, "repo_url", errors, prefix=("https://", "git@"))
    _require_str(entry, "default_branch", errors)
    _require_str(entry, "source_license", errors)
    _require_str(entry, "license_url", errors, prefix=("https://",))
    _require_str(entry, "license_review_date", errors)
    _require_str(entry, "reviewer", errors)
    _require_str(entry, "review_status", errors)
    _require_str(entry, "source_revision", errors)

    license_id = entry.get("source_license")
    if isinstance(license_id, str) and license_id not in (
        APPROVED_SOURCE_LICENSES | REVIEW_REQUIRED_SOURCE_LICENSES
    ):
        errors.append(f"source_license is not in the first-audit allow/review list: {license_id}")

    review_status = entry.get("review_status")
    if isinstance(review_status, str) and review_status not in SOURCE_REVIEW_STATUSES:
        errors.append(f"invalid review_status: {review_status}")

    source_revision = entry.get("source_revision")
    if isinstance(source_revision, str) and not _HEX_REVISION_RE.match(source_revision):
        errors.append("source_revision must be a 7-40 character lowercase hex revision")

    allowed_splits = entry.get("allowed_splits")
    if not isinstance(allowed_splits, list) or not allowed_splits:
        errors.append("allowed_splits must be a non-empty list")
    else:
        invalid = [split for split in allowed_splits if split not in DATA_SPLITS]
        if invalid:
            errors.append(f"invalid allowed_splits: {invalid}")

    if review_status == "approved_for_audit" and "HELD_OUT" in (allowed_splits or []):
        errors.append("approved_for_audit sources must not include HELD_OUT by default")

    return tuple(errors)


def validate_generated_label_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return validation errors for one generated teacher-label record."""

    errors: list[str] = []

    for key in (
        "id",
        "source_repo_url",
        "source_license",
        "source_commit",
        "parent_commit",
        "data_split",
        "teacher_model_id",
        "teacher_revision",
        "teacher_license",
        "prompt_version",
        "generation_timestamp",
        "header",
        "human_review_status",
    ):
        _require_str(record, key, errors)

    data_split = record.get("data_split")
    if isinstance(data_split, str) and data_split not in DATA_SPLITS:
        errors.append(f"invalid data_split: {data_split}")
    if data_split == "HELD_OUT":
        errors.append("teacher-generated labels must not be stored as HELD_OUT labels")

    for key in ("source_commit", "parent_commit", "teacher_revision"):
        value = record.get(key)
        if isinstance(value, str) and not _HEX_REVISION_RE.match(value):
            errors.append(f"{key} must be a 7-40 character lowercase hex revision")

    prompt_version = record.get("prompt_version")
    if isinstance(prompt_version, str) and not _PROMPT_VERSION_RE.match(prompt_version):
        errors.append("prompt_version must be a stable lowercase identifier")

    human_status = record.get("human_review_status")
    if isinstance(human_status, str) and human_status not in HUMAN_REVIEW_STATUSES:
        errors.append(f"invalid human_review_status: {human_status}")

    _require_list(record, "changed_paths", errors, min_items=1)
    _require_list(record, "body", errors)
    _require_list(record, "footers", errors)
    _require_list(record, "warnings", errors)
    _require_list(record, "evidence_paths", errors)

    confidence = record.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        errors.append("confidence must be a number from 0.0 to 1.0")

    verifier_score = record.get("verifier_score")
    if not isinstance(verifier_score, (int, float)) or not 0.0 <= verifier_score <= 1.0:
        errors.append("verifier_score must be a number from 0.0 to 1.0")

    decoding_config = record.get("decoding_config")
    if not isinstance(decoding_config, dict):
        errors.append("decoding_config must be an object")

    parser_result = record.get("parser_result")
    if not isinstance(parser_result, dict):
        errors.append("parser_result must be an object")

    return tuple(errors)


def validate_source_diff_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return validation errors for one extracted source-diff record."""

    errors: list[str] = []

    for key in (
        "id",
        "source_repo_url",
        "source_license",
        "manifest_revision",
        "source_commit",
        "parent_commit",
        "data_split",
        "historical_subject",
        "extraction_command",
        "review_status",
    ):
        _require_str(record, key, errors)

    data_split = record.get("data_split")
    if isinstance(data_split, str) and data_split not in DATA_SPLITS:
        errors.append(f"invalid data_split: {data_split}")

    for key in ("manifest_revision", "source_commit", "parent_commit"):
        value = record.get(key)
        if isinstance(value, str) and not _HEX_REVISION_RE.match(value):
            errors.append(f"{key} must be a 7-40 character lowercase hex revision")

    review_status = record.get("review_status")
    if isinstance(review_status, str) and review_status not in SOURCE_DIFF_REVIEW_STATUSES:
        errors.append(f"invalid review_status: {review_status}")

    _require_list(record, "changed_paths", errors, min_items=1)
    _require_list(record, "excluded_paths", errors)

    diff_stat = record.get("diff_stat")
    if not isinstance(diff_stat, str):
        errors.append("diff_stat must be a string")

    return tuple(errors)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into dictionaries."""

    records: list[dict[str, Any]] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        records.append(value)
    return records


def _require_str(
    record: Mapping[str, Any],
    key: str,
    errors: list[str],
    *,
    prefix: tuple[str, ...] | None = None,
) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"{key} must be a non-empty string")
        return
    if prefix is not None and not value.startswith(prefix):
        errors.append(f"{key} must start with one of {prefix}")


def _require_list(
    record: Mapping[str, Any],
    key: str,
    errors: list[str],
    *,
    min_items: int = 0,
) -> None:
    value = record.get(key)
    if not isinstance(value, list):
        errors.append(f"{key} must be a list")
    elif len(value) < min_items:
        errors.append(f"{key} must contain at least {min_items} item(s)")

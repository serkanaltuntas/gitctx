"""Split-plan validation and assignment for source-diff extraction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from gitctx.provenance import DATA_SPLITS


def load_split_plan(path: str | Path) -> dict[str, Any]:
    """Load and validate a split plan JSON document."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("split plan must be a JSON object")
    errors = validate_split_plan(value)
    if errors:
        raise ValueError("; ".join(errors))
    return value


def validate_split_plan(plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Return validation errors for a split plan."""

    errors: list[str] = []
    for key in ("id", "version", "created_at"):
        _require_str(plan, key, errors)

    windows = plan.get("windows")
    if not isinstance(windows, list) or not windows:
        errors.append("windows must be a non-empty list")
        return tuple(errors)

    parsed_windows: list[tuple[str, datetime, datetime, str]] = []
    seen_window_ids: set[str] = set()
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            errors.append(f"windows[{index}] must be an object")
            continue

        window_errors = _validate_window(window, index, seen_window_ids)
        errors.extend(window_errors)
        if window_errors:
            continue

        parsed_windows.append(
            (
                _normalize_repo_url(window["repo_url"]),
                _parse_rfc3339(window["start"]),
                _parse_rfc3339(window["end"]),
                window["id"],
            )
        )

    by_repo: dict[str, list[tuple[datetime, datetime, str]]] = defaultdict(list)
    for repo_url, start, end, window_id in parsed_windows:
        by_repo[repo_url].append((start, end, window_id))

    for repo_url, repo_windows in sorted(by_repo.items()):
        repo_windows.sort(key=lambda item: item[0])
        previous_end: datetime | None = None
        previous_id: str | None = None
        for start, end, window_id in repo_windows:
            if previous_end is not None and start < previous_end:
                errors.append(
                    "overlapping split windows for "
                    f"{repo_url}: {previous_id} overlaps {window_id}"
                )
            previous_end = end
            previous_id = window_id

    return tuple(errors)


def select_split_for_commit(
    split_plan: Mapping[str, Any],
    repo_url: str,
    commit_timestamp: str,
) -> str | None:
    """Return the split assigned to a repo commit timestamp, if any."""

    timestamp = _parse_rfc3339(commit_timestamp)
    normalized_repo = _normalize_repo_url(repo_url)
    for window in split_plan.get("windows", []):
        if not isinstance(window, dict):
            continue
        if _normalize_repo_url(str(window.get("repo_url", ""))) != normalized_repo:
            continue
        start = _parse_rfc3339(window["start"])
        end = _parse_rfc3339(window["end"])
        if start <= timestamp < end:
            return str(window["split"])
    return None


def _validate_window(
    window: Mapping[str, Any],
    index: int,
    seen_window_ids: set[str],
) -> tuple[str, ...]:
    errors: list[str] = []
    for key in ("id", "repo_url", "split", "start", "end", "reason"):
        _require_str(window, key, errors, prefix=f"windows[{index}].")

    window_id = window.get("id")
    if isinstance(window_id, str):
        if window_id in seen_window_ids:
            errors.append(f"windows[{index}].id is duplicated: {window_id}")
        seen_window_ids.add(window_id)

    split = window.get("split")
    if isinstance(split, str) and split not in DATA_SPLITS:
        errors.append(f"windows[{index}].split is invalid: {split}")

    start = window.get("start")
    end = window.get("end")
    if isinstance(start, str) and isinstance(end, str):
        try:
            parsed_start = _parse_rfc3339(start)
            parsed_end = _parse_rfc3339(end)
        except ValueError as exc:
            errors.append(f"windows[{index}] has invalid timestamp: {exc}")
        else:
            if parsed_start >= parsed_end:
                errors.append(f"windows[{index}].start must be earlier than end")

    return tuple(errors)


def _parse_rfc3339(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be a non-empty string")
    normalized = value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include timezone: {value}")
    return parsed


def _normalize_repo_url(repo_url: str) -> str:
    return repo_url.rstrip("/").removesuffix(".git")


def _require_str(
    record: Mapping[str, Any],
    key: str,
    errors: list[str],
    *,
    prefix: str = "",
) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"{prefix}{key} must be a non-empty string")

"""Readiness checks for larger gitctx split plans."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gitctx.provenance import load_jsonl, validate_source_manifest_entry
from gitctx.split_plan import load_split_plan

GCTX1_MIN_DEV_RECORDS = 10_000
GCTX1_MIN_REPORT_RECORDS = 1_000
GCTX1_MIN_HELD_OUT_RECORDS = 1_000
GCTX1_MIN_DEV_REPOS = 25
GCTX1_MIN_REPORT_REPOS = 5
GCTX1_MIN_HELD_OUT_REPOS = 5
GCTX1_MIN_ECOSYSTEMS = 2
GCTX1_MAX_REPO_FRACTION = 0.25

SPLITS = ("DEV", "REPORT", "HELD_OUT")


def evaluate_split_readiness(
    *,
    source_manifest_path: str | Path,
    split_plan_path: str | Path,
) -> dict[str, Any]:
    """Return a readiness report for the GCTX-1 split-plan gate."""

    manifest_records = load_jsonl(source_manifest_path)
    split_plan = load_split_plan(split_plan_path)
    manifest_by_repo = {_normalize_repo_url(record["repo_url"]): record for record in manifest_records}
    manifest_errors = _manifest_errors(manifest_records)
    cross_errors = _cross_check_windows(split_plan, manifest_by_repo)
    split_summary = _split_summary(split_plan, manifest_by_repo)
    gates = _gctx1_gates(split_summary)
    return {
        "source_manifest_path": str(source_manifest_path),
        "split_plan_path": str(split_plan_path),
        "manifest_records": len(manifest_records),
        "split_windows": len(split_plan["windows"]),
        "manifest_errors": manifest_errors,
        "cross_errors": cross_errors,
        "licenses": dict(sorted(Counter(record["source_license"] for record in manifest_records).items())),
        "review_statuses": dict(
            sorted(Counter(record["review_status"] for record in manifest_records).items())
        ),
        "ecosystems": dict(
            sorted(
                Counter(record.get("ecosystem", "unknown") for record in manifest_records).items()
            )
        ),
        "splits": split_summary,
        "gates": gates,
        "ready_for_gctx1_planning": (
            not manifest_errors
            and not cross_errors
            and all(gate["status"] == "pass" for gate in gates.values())
        ),
    }


def _manifest_errors(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    seen_repos: set[str] = set()
    for index, record in enumerate(records):
        repo_url = _normalize_repo_url(str(record.get("repo_url", "")))
        record_errors = list(validate_source_manifest_entry(record))
        if repo_url in seen_repos:
            record_errors.append(f"duplicate repo_url: {record.get('repo_url')}")
        seen_repos.add(repo_url)
        if record_errors:
            errors.append(
                {
                    "index": index,
                    "repo_url": record.get("repo_url"),
                    "errors": record_errors,
                }
            )
    return errors


def _cross_check_windows(
    split_plan: dict[str, Any],
    manifest_by_repo: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for window in split_plan["windows"]:
        repo_url = _normalize_repo_url(window["repo_url"])
        manifest_record = manifest_by_repo.get(repo_url)
        window_errors: list[str] = []
        if manifest_record is None:
            window_errors.append("repo_url is not present in source manifest")
        else:
            allowed_splits = manifest_record.get("allowed_splits", [])
            if window["split"] not in allowed_splits:
                window_errors.append(
                    f"split {window['split']} is not allowed by source manifest"
                )
            if window["split"] == "HELD_OUT" and manifest_record.get("review_status") != "approved_for_training":
                window_errors.append("HELD_OUT windows require approved_for_training source review")
        if window_errors:
            errors.append(
                {
                    "window_id": window["id"],
                    "repo_url": window["repo_url"],
                    "split": window["split"],
                    "errors": window_errors,
                }
            )
    return errors


def _split_summary(
    split_plan: dict[str, Any],
    manifest_by_repo: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    repos_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    windows_by_split: Counter[str] = Counter()
    target_records_by_split: Counter[str] = Counter()
    target_records_by_repo: dict[str, Counter[str]] = defaultdict(Counter)
    ecosystems_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    missing_target_records = False

    for window in split_plan["windows"]:
        split = window["split"]
        repo_url = _normalize_repo_url(window["repo_url"])
        repos_by_split[split].add(repo_url)
        windows_by_split[split] += 1
        manifest_record = manifest_by_repo.get(repo_url, {})
        ecosystem = manifest_record.get("ecosystem")
        if isinstance(ecosystem, str) and ecosystem:
            ecosystems_by_split[split].add(ecosystem)
        target_records = window.get("target_records")
        if isinstance(target_records, int):
            target_records_by_split[split] += target_records
            target_records_by_repo[split][repo_url] += target_records
        else:
            missing_target_records = True

    return {
        split: {
            "repos": len(repos_by_split[split]),
            "windows": windows_by_split[split],
            "target_records": target_records_by_split[split],
            "ecosystems": sorted(ecosystems_by_split[split]),
            "repo_target_records": dict(sorted(target_records_by_repo[split].items())),
        }
        for split in SPLITS
    } | {"missing_target_records": missing_target_records}


def _gctx1_gates(split_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dev = split_summary["DEV"]
    report = split_summary["REPORT"]
    held_out = split_summary["HELD_OUT"]
    all_ecosystems = sorted(
        set(dev["ecosystems"]) | set(report["ecosystems"]) | set(held_out["ecosystems"])
    )
    gates = {
        "dev_records": _min_gate(dev["target_records"], GCTX1_MIN_DEV_RECORDS),
        "report_records": _min_gate(report["target_records"], GCTX1_MIN_REPORT_RECORDS),
        "held_out_records": _min_gate(
            held_out["target_records"],
            GCTX1_MIN_HELD_OUT_RECORDS,
        ),
        "dev_repos": _min_gate(dev["repos"], GCTX1_MIN_DEV_REPOS),
        "report_repos": _min_gate(report["repos"], GCTX1_MIN_REPORT_REPOS),
        "held_out_repos": _min_gate(held_out["repos"], GCTX1_MIN_HELD_OUT_REPOS),
        "ecosystems": _min_gate(len(all_ecosystems), GCTX1_MIN_ECOSYSTEMS),
        "max_repo_train_fraction": _max_repo_fraction_gate(dev["repo_target_records"]),
    }
    if split_summary["missing_target_records"]:
        for name in ("dev_records", "report_records", "held_out_records", "max_repo_train_fraction"):
            gates[name] = {
                **gates[name],
                "status": "unknown",
                "reason": "one or more split windows are missing target_records",
            }
    return gates


def _min_gate(actual: int | float, minimum: int) -> dict[str, Any]:
    return {
        "actual": actual,
        "minimum": minimum,
        "status": "pass" if actual >= minimum else "fail",
    }


def _max_repo_fraction_gate(repo_records: dict[str, int]) -> dict[str, Any]:
    total = sum(repo_records.values())
    if total <= 0:
        return {
            "actual": None,
            "maximum": GCTX1_MAX_REPO_FRACTION,
            "status": "unknown",
            "reason": "no DEV target_records available",
        }
    largest = max(repo_records.values()) / total
    return {
        "actual": round(largest, 6),
        "maximum": GCTX1_MAX_REPO_FRACTION,
        "status": "pass" if largest <= GCTX1_MAX_REPO_FRACTION else "fail",
    }


def _normalize_repo_url(repo_url: str) -> str:
    return repo_url.rstrip("/").removesuffix(".git")


def _print_report(report: dict[str, Any]) -> None:
    print("source_manifest_path", report["source_manifest_path"])
    print("split_plan_path", report["split_plan_path"])
    print("manifest_records", report["manifest_records"])
    print("split_windows", report["split_windows"])
    print("manifest_errors", len(report["manifest_errors"]))
    print("cross_errors", len(report["cross_errors"]))
    for split in SPLITS:
        summary = report["splits"][split]
        print(f"{split.lower()}_repos", summary["repos"])
        print(f"{split.lower()}_windows", summary["windows"])
        print(f"{split.lower()}_target_records", summary["target_records"])
    for name, gate in report["gates"].items():
        print(f"gate_{name}", gate["status"])
    print("ready_for_gctx1_planning", report["ready_for_gctx1_planning"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate gitctx split readiness.")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--split-plan", required=True)
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    args = parser.parse_args(argv)
    report = evaluate_split_readiness(
        source_manifest_path=args.source_manifest,
        split_plan_path=args.split_plan,
    )
    if args.json:
        import json

        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0 if not report["manifest_errors"] and not report["cross_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

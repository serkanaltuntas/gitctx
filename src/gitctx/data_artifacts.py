"""Utilities for private gitctx-data smoke artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from gitctx.provenance import (
    load_jsonl,
    validate_source_diff_record,
    validate_source_manifest_entry,
)

SMOKE_JSONL = Path("artifacts/smoke/source-diffs.smoke.jsonl")
SMOKE_REPORT = Path("artifacts/smoke/source-diffs.smoke.report.json")
SMOKE_MANIFEST = Path("manifests/source-manifest.audit.jsonl")
GITCTX_COMMIT = Path("lineage/gitctx-public-commit.txt")
CHECKSUMS = Path("checksums/sha256.txt")


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


def write_checksums(data_dir: str | Path) -> Path:
    """Write sha256 checksums for tracked smoke lineage files."""

    data_dir = Path(data_dir)
    checksum_path = data_dir / CHECKSUMS
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [SMOKE_JSONL, SMOKE_REPORT, SMOKE_MANIFEST, GITCTX_COMMIT]
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
    subparsers.add_parser("write-checksums")
    args = parser.parse_args(argv)

    if args.command == "normalize-smoke":
        normalize_smoke_report(args.data_dir)
    elif args.command == "validate-smoke":
        validate_smoke_artifact(args.data_dir)
    elif args.command == "write-checksums":
        print(write_checksums(args.data_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

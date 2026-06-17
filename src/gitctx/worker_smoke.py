"""Local worker smoke extraction for approved source manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from gitctx.provenance import (
    load_jsonl,
    validate_source_diff_record,
    validate_source_manifest_entry,
)
from gitctx.source_extract import extract_source_diff_record, iter_candidate_commits


def run_smoke(
    manifest_path: str | Path,
    data_dir: str | Path,
    *,
    records: int = 50,
    per_repo_limit: int = 20,
    split: str = "DEV",
) -> dict[str, Any]:
    """Clone approved sources and emit a source-diff smoke artifact."""

    manifest_path = Path(manifest_path)
    data_dir = Path(data_dir)
    sources_dir = data_dir / "sources" / "github.com"
    artifact_dir = data_dir / "artifacts" / "smoke"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    output_path = artifact_dir / "source-diffs.smoke.jsonl"
    report_path = artifact_dir / "source-diffs.smoke.report.json"

    manifest_entries = load_jsonl(manifest_path)
    output_records: list[dict[str, Any]] = []
    repo_reports: list[dict[str, Any]] = []
    started = time.time()

    for entry in manifest_entries:
        manifest_errors = validate_source_manifest_entry(entry)
        repo_report: dict[str, Any] = {
            "repo_url": entry.get("repo_url"),
            "source_revision": entry.get("source_revision"),
            "manifest_errors": list(manifest_errors),
            "records": 0,
            "errors": [],
        }
        repo_reports.append(repo_report)
        if manifest_errors:
            continue

        repo_path = sources_dir / _repo_path_from_url(entry["repo_url"])
        try:
            ensure_clone(repo_path, entry.get("clone_url", entry["repo_url"]))
            checkout_revision(repo_path, entry["source_revision"])
            commits = iter_candidate_commits(
                repo_path,
                entry["source_revision"],
                limit=per_repo_limit,
            )
            for commit in commits:
                if len(output_records) >= records:
                    break
                record = extract_source_diff_record(repo_path, entry, commit, data_split=split)
                if record is None:
                    continue
                validation_errors = validate_source_diff_record(record)
                if validation_errors:
                    repo_report["errors"].append(
                        {"commit": commit, "validation_errors": list(validation_errors)}
                    )
                    continue
                output_records.append(record)
                repo_report["records"] += 1
        except (subprocess.CalledProcessError, OSError) as exc:
            repo_report["errors"].append({"error": str(exc)})

        if len(output_records) >= records:
            break

    _write_jsonl(output_path, output_records)
    report = {
        "manifest_path": str(manifest_path),
        "data_dir": str(data_dir),
        "output_path": str(output_path),
        "requested_records": records,
        "written_records": len(output_records),
        "split": split,
        "repo_reports": repo_reports,
        "duration_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def ensure_clone(repo_path: Path, repo_url: str) -> None:
    """Clone ``repo_url`` if missing, otherwise fetch updates."""

    if (repo_path / ".git").exists():
        _git(repo_path, "fetch", "--quiet", "origin")
        return

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", repo_url, str(repo_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def checkout_revision(repo_path: Path, revision: str) -> None:
    """Checkout a pinned manifest revision in a detached state."""

    _git(repo_path, "checkout", "--quiet", "--detach", revision)


def _repo_path_from_url(repo_url: str) -> Path:
    slug = repo_url.rstrip("/").removesuffix(".git").split("github.com/")[-1]
    return Path(slug)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _git(repo_path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local source-diff smoke extraction.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--records", type=int, default=50)
    parser.add_argument("--per-repo-limit", type=int, default=20)
    parser.add_argument("--split", default="DEV")
    args = parser.parse_args(argv)

    report = run_smoke(
        args.manifest,
        args.data_dir,
        records=args.records,
        per_repo_limit=args.per_repo_limit,
        split=args.split,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["written_records"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

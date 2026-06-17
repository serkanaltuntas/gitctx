"""Extract source-diff records from local Git repositories."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import subprocess
from typing import Any


def iter_candidate_commits(repo_path: str | Path, revision: str, *, limit: int) -> list[str]:
    """Return recent non-merge commit ids at or below ``revision``."""

    output = _git(repo_path, "rev-list", "--no-merges", f"--max-count={limit}", revision)
    return [line for line in output.splitlines() if line]


def extract_source_diff_record(
    repo_path: str | Path,
    source_entry: dict[str, Any],
    commit: str,
    *,
    data_split: str = "DEV",
) -> dict[str, Any] | None:
    """Extract one source-diff record from a local repo.

    The record intentionally does not include full diff text. Full diffs can be
    generated later as local artifacts after redistribution review.
    """

    parents = _git(repo_path, "rev-list", "--parents", "-n", "1", commit).split()
    if len(parents) != 2:
        return None

    parent_commit = parents[1]
    changed_paths = _git(
        repo_path,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    ).splitlines()
    include_paths, excluded_paths = _partition_paths(
        changed_paths,
        source_entry.get("exclude_globs", []),
    )
    if not include_paths:
        return None

    diff_stat = _git(
        repo_path,
        "diff",
        "--stat",
        "--find-renames",
        parent_commit,
        commit,
        "--",
        *include_paths,
    )
    historical_subject = _git(repo_path, "show", "-s", "--format=%s", commit)

    short_commit = commit[:12]
    return {
        "id": f"{_repo_slug(source_entry['repo_url'])}-{short_commit}",
        "source_repo_url": source_entry["repo_url"],
        "source_license": source_entry["source_license"],
        "manifest_revision": source_entry["source_revision"],
        "source_commit": commit,
        "parent_commit": parent_commit,
        "data_split": data_split,
        "changed_paths": include_paths,
        "excluded_paths": excluded_paths,
        "diff_stat": diff_stat,
        "historical_subject": historical_subject,
        "extraction_command": (
            "git diff --stat --find-renames <parent> <commit> -- <included_paths>"
        ),
        "review_status": "not_reviewed",
    }


def _partition_paths(paths: list[str], exclude_globs: list[str]) -> tuple[list[str], list[str]]:
    include_paths: list[str] = []
    excluded_paths: list[str] = []
    for path in paths:
        if any(fnmatch.fnmatch(path, pattern) for pattern in exclude_globs):
            excluded_paths.append(path)
        else:
            include_paths.append(path)
    return include_paths, excluded_paths


def _repo_slug(repo_url: str) -> str:
    return repo_url.rstrip("/").removesuffix(".git").split("github.com/")[-1].replace("/", "-")


def _git(repo_path: str | Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract source-diff records from a local clone.")
    parser.add_argument("repo", type=Path, help="Path to a local Git clone.")
    parser.add_argument("source_entry", type=Path, help="Path to a JSON source-manifest entry.")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--split", default="DEV")
    args = parser.parse_args(argv)

    source_entry = json.loads(args.source_entry.read_text(encoding="utf-8"))
    commits = iter_candidate_commits(args.repo, source_entry["source_revision"], limit=args.limit)
    for commit in commits:
        record = extract_source_diff_record(args.repo, source_entry, commit, data_split=args.split)
        if record is not None:
            print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gitctx.provenance import validate_source_diff_record
from gitctx.source_extract import _diff_stat, extract_source_diff_record, iter_candidate_commits


class SourceExtractTests(unittest.TestCase):
    def test_extracts_source_diff_record_from_local_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _git(repo, "init")
            _git(repo, "config", "user.email", "test@example.com")
            _git(repo, "config", "user.name", "Test User")

            (repo / "parser.py").write_text(
                "def parse(value):\n    return value\n",
                encoding="utf-8",
            )
            _git(repo, "add", "parser.py")
            _git(
                repo,
                "commit",
                "-m",
                "chore: initial parser",
                env=_git_date_env("2025-01-01T00:00:00+00:00"),
            )

            (repo / "parser.py").write_text(
                "def parse(value):\n"
                "    if not value:\n"
                "        raise ValueError('empty')\n"
                "    return value\n",
                encoding="utf-8",
            )
            _git(repo, "add", "parser.py")
            _git(
                repo,
                "commit",
                "-m",
                "fix(parser): reject empty values",
                env=_git_date_env("2025-08-01T00:00:00+00:00"),
            )

            revision = _git(repo, "rev-parse", "HEAD")
            source_entry = {
                "repo_url": "https://github.com/example/repo",
                "source_license": "MIT",
                "source_revision": revision,
                "exclude_globs": ["vendor/**"],
            }

            commits = iter_candidate_commits(repo, revision, limit=5)
            record = extract_source_diff_record(repo, source_entry, commits[0])

            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record["historical_subject"], "fix(parser): reject empty values")
            self.assertEqual(record["source_commit_timestamp"], "2025-08-01T00:00:00Z")
            self.assertEqual(record["changed_paths"], ["parser.py"])
            self.assertEqual(validate_source_diff_record(record), ())

    def test_split_plan_assigns_split_and_skips_uncovered_commits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _git(repo, "init")
            _git(repo, "config", "user.email", "test@example.com")
            _git(repo, "config", "user.name", "Test User")

            (repo / "parser.py").write_text("value = 1\n", encoding="utf-8")
            _git(repo, "add", "parser.py")
            _git(
                repo,
                "commit",
                "-m",
                "chore: initial parser",
                env=_git_date_env("2025-01-01T00:00:00+00:00"),
            )
            (repo / "parser.py").write_text("value = 2\n", encoding="utf-8")
            _git(
                repo,
                "commit",
                "-am",
                "fix(parser): reject empty values",
                env=_git_date_env("2025-03-01T00:00:00+00:00"),
            )
            uncovered = _git(repo, "rev-parse", "HEAD")
            (repo / "parser.py").write_text("value = 3\n", encoding="utf-8")
            _git(
                repo,
                "commit",
                "-a",
                "-m",
                "test(parser): add validation coverage",
                env=_git_date_env("2025-08-01T00:00:00+00:00"),
            )
            covered = _git(repo, "rev-parse", "HEAD")

            source_entry = {
                "repo_url": "https://github.com/example/repo",
                "source_license": "MIT",
                "source_revision": covered,
                "exclude_globs": [],
            }
            split_plan = {
                "id": "test-plan",
                "version": "v0",
                "created_at": "2026-06-19",
                "windows": [
                    {
                        "id": "report",
                        "repo_url": "https://github.com/example/repo",
                        "split": "REPORT",
                        "start": "2025-07-01T00:00:00Z",
                        "end": "2026-01-01T00:00:00Z",
                        "reason": "test window",
                    }
                ],
            }

            record = extract_source_diff_record(repo, source_entry, covered, split_plan=split_plan)
            skipped = extract_source_diff_record(repo, source_entry, uncovered, split_plan=split_plan)

            self.assertIsNone(skipped)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record["data_split"], "REPORT")

    def test_diff_stat_falls_back_when_pathspec_is_too_large(self) -> None:
        calls: list[tuple[str, ...]] = []

        def fake_git(repo: Path, *args: str) -> str:
            calls.append(args)
            if "--" in args:
                raise OSError(7, "Argument list too long")
            return "fallback stat"

        with patch("gitctx.source_extract._git", side_effect=fake_git):
            diff_stat, scope = _diff_stat(
                Path("/tmp/repo"),
                "parent",
                "commit",
                ["src/file.py"] * 1000,
            )

        self.assertEqual(diff_stat, "fallback stat")
        self.assertEqual(scope, "all_paths")
        self.assertEqual(calls[-1], ("diff", "--stat", "--find-renames", "parent", "commit"))


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **(env or {})},
    )
    return completed.stdout.strip()


def _git_date_env(date: str) -> dict[str, str]:
    return {
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
    }


if __name__ == "__main__":
    unittest.main()

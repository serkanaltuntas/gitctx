import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from gitctx.provenance import load_jsonl
from gitctx.worker_smoke import run_smoke, run_source_extract


class WorkerSmokeTests(unittest.TestCase):
    def test_run_smoke_clones_local_remote_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote = root / "remote"
            work = root / "work"
            data_dir = root / "data"
            remote.mkdir()
            work.mkdir()
            _git(remote, "init", "--bare")
            _git(work, "init")
            _git(work, "config", "user.email", "test@example.com")
            _git(work, "config", "user.name", "Test User")
            _git(work, "remote", "add", "origin", str(remote))

            (work / "parser.py").write_text(
                "def parse(value):\n    return value\n",
                encoding="utf-8",
            )
            _git(work, "add", "parser.py")
            _git(work, "commit", "-m", "chore: initial parser")

            (work / "parser.py").write_text(
                "def parse(value):\n"
                "    if not value:\n"
                "        raise ValueError('empty')\n"
                "    return value\n",
                encoding="utf-8",
            )
            _git(work, "add", "parser.py")
            _git(work, "commit", "-m", "fix(parser): reject empty values")
            revision = _git(work, "rev-parse", "HEAD")
            _git(work, "push", "origin", "HEAD:main")

            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "{"
                "\"repo_url\":\"https://github.com/example/repo\","
                f"\"clone_url\":\"{remote}\","
                "\"default_branch\":\"main\","
                "\"source_license\":\"MIT\","
                f"\"license_url\":\"https://example.com/license\","
                "\"license_review_date\":\"2026-06-17\","
                "\"reviewer\":\"Test User\","
                "\"review_status\":\"approved_for_audit\","
                f"\"source_revision\":\"{revision}\","
                "\"allowed_splits\":[\"DEV\"],"
                "\"exclude_globs\":[\"vendor/**\"],"
                "\"notes\":\"test fixture\""
                "}\n",
                encoding="utf-8",
            )

            report = run_smoke(manifest, data_dir, records=5, per_repo_limit=5)
            output_path = Path(report["output_path"])

            self.assertEqual(report["written_records"], 1)
            self.assertTrue(output_path.exists())
            records = load_jsonl(output_path)
            self.assertEqual(records[0]["historical_subject"], "fix(parser): reject empty values")

    def test_run_smoke_can_write_named_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote = root / "remote"
            work = root / "work"
            data_dir = root / "data"
            remote.mkdir()
            work.mkdir()
            _git(remote, "init", "--bare")
            _git(work, "init")
            _git(work, "config", "user.email", "test@example.com")
            _git(work, "config", "user.name", "Test User")
            _git(work, "remote", "add", "origin", str(remote))

            (work / "parser.py").write_text("old = True\n", encoding="utf-8")
            _git(work, "add", "parser.py")
            _git(work, "commit", "-m", "chore: initial parser")
            (work / "parser.py").write_text("old = False\n", encoding="utf-8")
            _git(work, "commit", "-am", "fix(parser): reject empty values")
            revision = _git(work, "rev-parse", "HEAD")
            _git(work, "push", "origin", "HEAD:main")

            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "{"
                "\"repo_url\":\"https://github.com/example/repo\","
                f"\"clone_url\":\"{remote}\","
                "\"default_branch\":\"main\","
                "\"source_license\":\"MIT\","
                f"\"license_url\":\"https://example.com/license\","
                "\"license_review_date\":\"2026-06-17\","
                "\"reviewer\":\"Test User\","
                "\"review_status\":\"approved_for_audit\","
                f"\"source_revision\":\"{revision}\","
                "\"allowed_splits\":[\"DEV\"],"
                "\"exclude_globs\":[\"vendor/**\"],"
                "\"notes\":\"test fixture\""
                "}\n",
                encoding="utf-8",
            )

            report = run_source_extract(
                manifest,
                data_dir,
                artifact_name="pilot",
                records=5,
                per_repo_limit=5,
            )

            self.assertEqual(report["artifact_name"], "pilot")
            self.assertEqual(Path(report["output_path"]).name, "source-diffs.pilot.jsonl")
            self.assertTrue((data_dir / "artifacts/pilot/source-diffs.pilot.jsonl").exists())

    def test_run_source_extract_applies_split_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote = root / "remote"
            work = root / "work"
            data_dir = root / "data"
            remote.mkdir()
            work.mkdir()
            _git(remote, "init", "--bare")
            _git(work, "init")
            _git(work, "config", "user.email", "test@example.com")
            _git(work, "config", "user.name", "Test User")
            _git(work, "remote", "add", "origin", str(remote))

            (work / "parser.py").write_text("old = True\n", encoding="utf-8")
            _git(work, "add", "parser.py")
            _git(work, "commit", "-m", "chore: initial parser")
            (work / "parser.py").write_text("old = False\n", encoding="utf-8")
            _git(work, "commit", "-am", "fix(parser): reject empty values")
            revision = _git(work, "rev-parse", "HEAD")
            _git(work, "push", "origin", "HEAD:main")

            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "{"
                "\"repo_url\":\"https://github.com/example/repo\","
                f"\"clone_url\":\"{remote}\","
                "\"default_branch\":\"main\","
                "\"source_license\":\"MIT\","
                f"\"license_url\":\"https://example.com/license\","
                "\"license_review_date\":\"2026-06-17\","
                "\"reviewer\":\"Test User\","
                "\"review_status\":\"approved_for_audit\","
                f"\"source_revision\":\"{revision}\","
                "\"allowed_splits\":[\"DEV\",\"REPORT\"],"
                "\"exclude_globs\":[\"vendor/**\"],"
                "\"notes\":\"test fixture\""
                "}\n",
                encoding="utf-8",
            )
            split_plan = root / "split-plan.json"
            split_plan.write_text(
                json.dumps(
                    {
                        "id": "test-plan",
                        "version": "v0",
                        "created_at": "2026-06-19",
                        "windows": [
                            {
                                "id": "report-window",
                                "repo_url": "https://github.com/example/repo",
                                "split": "REPORT",
                                "start": "2000-01-01T00:00:00Z",
                                "end": "2100-01-01T00:00:00Z",
                                "reason": "test window",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = run_source_extract(
                manifest,
                data_dir,
                artifact_name="reportfixture",
                records=5,
                per_repo_limit=5,
                split_plan_path=split_plan,
            )

            self.assertEqual(report["split_plan_path"], str(split_plan))
            records = load_jsonl(report["output_path"])
            self.assertEqual(records[0]["data_split"], "REPORT")

    def test_run_source_extract_can_filter_allowed_data_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote = root / "remote"
            work = root / "work"
            data_dir = root / "data"
            remote.mkdir()
            work.mkdir()
            _git(remote, "init", "--bare")
            _git(work, "init")
            _git(work, "config", "user.email", "test@example.com")
            _git(work, "config", "user.name", "Test User")
            _git(work, "remote", "add", "origin", str(remote))

            (work / "parser.py").write_text("value = 1\n", encoding="utf-8")
            _git(work, "add", "parser.py")
            _git_with_env(
                work,
                {
                    "GIT_AUTHOR_DATE": "2026-06-01T12:00:00+00:00",
                    "GIT_COMMITTER_DATE": "2026-06-01T12:00:00+00:00",
                },
                "commit",
                "-m",
                "chore: initial parser",
            )
            (work / "parser.py").write_text("value = 2\n", encoding="utf-8")
            _git_with_env(
                work,
                {
                    "GIT_AUTHOR_DATE": "2026-06-02T12:00:00+00:00",
                    "GIT_COMMITTER_DATE": "2026-06-02T12:00:00+00:00",
                },
                "commit",
                "-am",
                "fix(parser): report-window change",
            )
            (work / "parser.py").write_text("value = 3\n", encoding="utf-8")
            _git_with_env(
                work,
                {
                    "GIT_AUTHOR_DATE": "2026-06-03T12:00:00+00:00",
                    "GIT_COMMITTER_DATE": "2026-06-03T12:00:00+00:00",
                },
                "commit",
                "-am",
                "fix(parser): dev-window change",
            )
            revision = _git(work, "rev-parse", "HEAD")
            dev_change = revision
            _git(work, "push", "origin", "HEAD:main")

            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "{"
                "\"repo_url\":\"https://github.com/example/repo\","
                f"\"clone_url\":\"{remote}\","
                "\"default_branch\":\"main\","
                "\"source_license\":\"MIT\","
                f"\"license_url\":\"https://example.com/license\","
                "\"license_review_date\":\"2026-06-17\","
                "\"reviewer\":\"Test User\","
                "\"review_status\":\"approved_for_audit\","
                f"\"source_revision\":\"{revision}\","
                "\"allowed_splits\":[\"DEV\",\"REPORT\"],"
                "\"exclude_globs\":[\"vendor/**\"],"
                "\"notes\":\"test fixture\""
                "}\n",
                encoding="utf-8",
            )
            split_plan = root / "split-plan.json"
            split_plan.write_text(
                json.dumps(
                    {
                        "id": "test-plan",
                        "version": "v0",
                        "created_at": "2026-06-19",
                        "windows": [
                            {
                                "id": "dev-window-early",
                                "repo_url": "https://github.com/example/repo",
                                "split": "DEV",
                                "start": "2026-06-01T00:00:00Z",
                                "end": "2026-06-02T00:00:00Z",
                                "reason": "test window",
                            },
                            {
                                "id": "report-window",
                                "repo_url": "https://github.com/example/repo",
                                "split": "REPORT",
                                "start": "2026-06-02T00:00:00Z",
                                "end": "2026-06-03T00:00:00Z",
                                "reason": "test window",
                            },
                            {
                                "id": "dev-window-late",
                                "repo_url": "https://github.com/example/repo",
                                "split": "DEV",
                                "start": "2026-06-03T00:00:00Z",
                                "end": "2026-06-04T00:00:00Z",
                                "reason": "test window",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = run_source_extract(
                manifest,
                data_dir,
                artifact_name="devonly",
                records=5,
                per_repo_limit=5,
                split_plan_path=split_plan,
                allowed_data_splits={"DEV"},
            )

            records = load_jsonl(report["output_path"])
            self.assertEqual(report["allowed_data_splits"], ["DEV"])
            self.assertEqual(report["skipped_split_records"], 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["source_commit"], dev_change)
            self.assertEqual(records[0]["data_split"], "DEV")

    def test_run_source_extract_can_skip_existing_source_artifact_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote = root / "remote"
            work = root / "work"
            data_dir = root / "data"
            remote.mkdir()
            work.mkdir()
            _git(remote, "init", "--bare")
            _git(work, "init")
            _git(work, "config", "user.email", "test@example.com")
            _git(work, "config", "user.name", "Test User")
            _git(work, "remote", "add", "origin", str(remote))

            (work / "parser.py").write_text("value = 1\n", encoding="utf-8")
            _git(work, "add", "parser.py")
            _git(work, "commit", "-m", "chore: initial parser")
            (work / "parser.py").write_text("value = 2\n", encoding="utf-8")
            _git(work, "commit", "-am", "fix(parser): update first value")
            first_change = _git(work, "rev-parse", "HEAD")
            (work / "parser.py").write_text("value = 3\n", encoding="utf-8")
            _git(work, "commit", "-am", "fix(parser): update second value")
            revision = _git(work, "rev-parse", "HEAD")
            _git(work, "push", "origin", "HEAD:main")

            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "{"
                "\"repo_url\":\"https://github.com/example/repo\","
                f"\"clone_url\":\"{remote}\","
                "\"default_branch\":\"main\","
                "\"source_license\":\"MIT\","
                f"\"license_url\":\"https://example.com/license\","
                "\"license_review_date\":\"2026-06-17\","
                "\"reviewer\":\"Test User\","
                "\"review_status\":\"approved_for_audit\","
                f"\"source_revision\":\"{revision}\","
                "\"allowed_splits\":[\"DEV\"],"
                "\"exclude_globs\":[\"vendor/**\"],"
                "\"notes\":\"test fixture\""
                "}\n",
                encoding="utf-8",
            )
            existing_artifact = root / "existing.jsonl"
            existing_artifact.write_text(
                json.dumps(
                    {
                        "id": f"example-repo-{revision[:12]}",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = run_source_extract(
                manifest,
                data_dir,
                artifact_name="expansion",
                records=5,
                per_repo_limit=5,
                exclude_source_artifact_paths=[existing_artifact],
            )

            records = load_jsonl(report["output_path"])
            self.assertEqual(report["skipped_existing_records"], 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["source_commit"], first_change)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _git_with_env(repo: Path, env: dict[str, str], *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **env},
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    unittest.main()

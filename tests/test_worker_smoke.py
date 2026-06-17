import subprocess
import tempfile
import unittest
from pathlib import Path

from gitctx.provenance import load_jsonl
from gitctx.worker_smoke import run_smoke


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


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    unittest.main()

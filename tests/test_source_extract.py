import subprocess
import tempfile
import unittest
from pathlib import Path

from gitctx.provenance import validate_source_diff_record
from gitctx.source_extract import extract_source_diff_record, iter_candidate_commits


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
            _git(repo, "commit", "-m", "chore: initial parser")

            (repo / "parser.py").write_text(
                "def parse(value):\n"
                "    if not value:\n"
                "        raise ValueError('empty')\n"
                "    return value\n",
                encoding="utf-8",
            )
            _git(repo, "add", "parser.py")
            _git(repo, "commit", "-m", "fix(parser): reject empty values")

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
            self.assertEqual(record["changed_paths"], ["parser.py"])
            self.assertEqual(validate_source_diff_record(record), ())


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

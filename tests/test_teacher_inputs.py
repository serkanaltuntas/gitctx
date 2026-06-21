import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from gitctx.teacher_inputs import (
    create_smoke_teacher_inputs,
    create_teacher_inputs,
    validate_source_cache,
    validate_smoke_teacher_inputs,
    validate_teacher_inputs,
)


class TeacherInputTests(unittest.TestCase):
    def test_creates_and_validates_smoke_teacher_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "sources/github.com/example/repo"
            repo.mkdir(parents=True)
            self._run(["git", "init"], cwd=repo)
            self._run(["git", "config", "user.email", "test@example.com"], cwd=repo)
            self._run(["git", "config", "user.name", "Test User"], cwd=repo)
            (repo / "parser.py").write_text("old = True\n", encoding="utf-8")
            self._run(["git", "add", "parser.py"], cwd=repo)
            self._run(["git", "commit", "-m", "initial"], cwd=repo)
            parent = self._run(["git", "rev-parse", "HEAD"], cwd=repo).strip()
            (repo / "parser.py").write_text("old = False\n", encoding="utf-8")
            self._run(["git", "commit", "-am", "fix parser"], cwd=repo)
            commit = self._run(["git", "rev-parse", "HEAD"], cwd=repo).strip()

            (root / "artifacts/smoke").mkdir(parents=True)
            (root / "reviews").mkdir()
            source = {
                "id": "example-repo-111111111111",
                "source_repo_url": "https://github.com/example/repo",
                "source_license": "MIT",
                "manifest_revision": commit,
                "source_commit": commit,
                "parent_commit": parent,
                "data_split": "DEV",
                "changed_paths": ["parser.py"],
                "excluded_paths": [],
                "diff_stat": " parser.py | 2 +-",
                "historical_subject": "fix parser",
                "extraction_command": "git diff --stat ...",
                "review_status": "not_reviewed",
            }
            review = {
                "id": "review-example-repo-111111111111",
                "source_diff_id": source["id"],
                "source_repo_url": source["source_repo_url"],
                "source_commit": commit,
                "parent_commit": parent,
                "data_split": "DEV",
                "changed_paths": ["parser.py"],
                "decision": "accepted_for_teacher_labeling",
                "reasons": ["small_source_fix"],
                "notes": "test",
                "reviewer": "reviewer@example.com",
                "review_timestamp": "2026-06-18T20:00:00Z",
                "review_protocol": "source-diff-smoke-review-v0.1",
            }
            (root / "artifacts/smoke/source-diffs.smoke.jsonl").write_text(
                json.dumps(source) + "\n",
                encoding="utf-8",
            )
            (root / "reviews/source-diffs.smoke.review.jsonl").write_text(
                json.dumps(review) + "\n",
                encoding="utf-8",
            )

            output_path = create_smoke_teacher_inputs(root)
            summary = validate_smoke_teacher_inputs(root)
            record = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(summary["teacher_input_records"], 1)
            self.assertIn("Generate one Conventional Commit message", record["user_message"])
            self.assertIn("diff --git", record["diff"])
            self.assertEqual(record["input_status"], "ready_for_generation")
            self.assertEqual(record["teacher_model_id"], "ollama/qwen2.5-coder:7b")
            self.assertEqual(record["teacher_runtime"], "ollama")

    def test_creates_and_validates_named_teacher_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "sources/github.com/example/repo"
            repo.mkdir(parents=True)
            self._run(["git", "init"], cwd=repo)
            self._run(["git", "config", "user.email", "test@example.com"], cwd=repo)
            self._run(["git", "config", "user.name", "Test User"], cwd=repo)
            (repo / "parser.py").write_text("old = True\n", encoding="utf-8")
            self._run(["git", "add", "parser.py"], cwd=repo)
            self._run(["git", "commit", "-m", "initial"], cwd=repo)
            parent = self._run(["git", "rev-parse", "HEAD"], cwd=repo).strip()
            (repo / "parser.py").write_text("old = False\n", encoding="utf-8")
            self._run(["git", "commit", "-am", "fix parser"], cwd=repo)
            commit = self._run(["git", "rev-parse", "HEAD"], cwd=repo).strip()

            (root / "artifacts/pilot").mkdir(parents=True)
            (root / "reviews").mkdir()
            source = {
                "id": "example-repo-111111111111",
                "source_repo_url": "https://github.com/example/repo",
                "source_license": "MIT",
                "manifest_revision": commit,
                "source_commit": commit,
                "parent_commit": parent,
                "data_split": "DEV",
                "changed_paths": ["parser.py"],
                "excluded_paths": [],
                "diff_stat": " parser.py | 2 +-",
                "historical_subject": "fix parser",
                "extraction_command": "git diff --stat ...",
                "review_status": "not_reviewed",
            }
            review = {
                "id": "review-example-repo-111111111111",
                "source_diff_id": source["id"],
                "source_repo_url": source["source_repo_url"],
                "source_commit": commit,
                "parent_commit": parent,
                "data_split": "DEV",
                "changed_paths": ["parser.py"],
                "decision": "accepted_for_teacher_labeling",
                "reasons": ["small_source_fix"],
                "notes": "test",
                "reviewer": "reviewer@example.com",
                "review_timestamp": "2026-06-18T20:00:00Z",
                "review_protocol": "source-diff-pilot-review-v0.1",
            }
            (root / "artifacts/pilot/source-diffs.pilot.jsonl").write_text(
                json.dumps(source) + "\n",
                encoding="utf-8",
            )
            (root / "reviews/source-diffs.pilot.review.jsonl").write_text(
                json.dumps(review) + "\n",
                encoding="utf-8",
            )

            output_path = create_teacher_inputs(root, artifact_name="pilot")
            summary = validate_teacher_inputs(root, artifact_name="pilot")

            self.assertEqual(output_path.name, "teacher-inputs.pilot.jsonl")
            self.assertEqual(summary["artifact_name"], "pilot")
            self.assertEqual(summary["teacher_input_records"], 1)

    def test_validates_source_cache_for_accepted_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "sources/github.com/example/repo"
            repo.mkdir(parents=True)
            self._run(["git", "init"], cwd=repo)
            self._run(["git", "config", "user.email", "test@example.com"], cwd=repo)
            self._run(["git", "config", "user.name", "Test User"], cwd=repo)
            (repo / "parser.py").write_text("old = True\n", encoding="utf-8")
            self._run(["git", "add", "parser.py"], cwd=repo)
            self._run(["git", "commit", "-m", "initial"], cwd=repo)
            parent = self._run(["git", "rev-parse", "HEAD"], cwd=repo).strip()
            (repo / "parser.py").write_text("old = False\n", encoding="utf-8")
            self._run(["git", "commit", "-am", "fix parser"], cwd=repo)
            commit = self._run(["git", "rev-parse", "HEAD"], cwd=repo).strip()

            (root / "artifacts/pilot").mkdir(parents=True)
            (root / "reviews").mkdir()
            source = {
                "id": "example-repo-111111111111",
                "source_repo_url": "https://github.com/example/repo",
                "source_license": "MIT",
                "manifest_revision": commit,
                "source_commit": commit,
                "parent_commit": parent,
                "data_split": "DEV",
                "changed_paths": ["parser.py"],
                "excluded_paths": [],
                "diff_stat": " parser.py | 2 +-",
                "historical_subject": "fix parser",
                "extraction_command": "git diff --stat ...",
                "review_status": "not_reviewed",
            }
            review = {
                "id": "review-example-repo-111111111111",
                "source_diff_id": source["id"],
                "source_repo_url": source["source_repo_url"],
                "source_commit": commit,
                "parent_commit": parent,
                "data_split": "DEV",
                "changed_paths": ["parser.py"],
                "decision": "accepted_for_teacher_labeling",
                "reasons": ["small_source_fix"],
                "notes": "test",
                "reviewer": "reviewer@example.com",
                "review_timestamp": "2026-06-18T20:00:00Z",
                "review_protocol": "source-diff-pilot-review-v0.1",
            }
            (root / "artifacts/pilot/source-diffs.pilot.jsonl").write_text(
                json.dumps(source) + "\n",
                encoding="utf-8",
            )
            (root / "reviews/source-diffs.pilot.review.jsonl").write_text(
                json.dumps(review) + "\n",
                encoding="utf-8",
            )

            summary = validate_source_cache(root, artifact_name="pilot")

            self.assertEqual(summary["accepted_review_records"], 1)
            self.assertEqual(summary["required_repos"], 1)

    def test_source_cache_validation_rejects_missing_clone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/pilot").mkdir(parents=True)
            (root / "reviews").mkdir()
            source = {
                "id": "example-repo-111111111111",
                "source_repo_url": "https://github.com/example/repo",
                "source_license": "MIT",
                "manifest_revision": "1111111111111111111111111111111111111111",
                "source_commit": "1111111111111111111111111111111111111111",
                "parent_commit": "0000000000000000000000000000000000000000",
                "data_split": "DEV",
                "changed_paths": ["parser.py"],
                "excluded_paths": [],
                "diff_stat": " parser.py | 2 +-",
                "historical_subject": "fix parser",
                "extraction_command": "git diff --stat ...",
                "review_status": "not_reviewed",
            }
            review = {
                "id": "review-example-repo-111111111111",
                "source_diff_id": source["id"],
                "source_repo_url": source["source_repo_url"],
                "source_commit": source["source_commit"],
                "parent_commit": source["parent_commit"],
                "data_split": "DEV",
                "changed_paths": ["parser.py"],
                "decision": "accepted_for_teacher_labeling",
                "reasons": ["small_source_fix"],
                "notes": "test",
                "reviewer": "reviewer@example.com",
                "review_timestamp": "2026-06-18T20:00:00Z",
                "review_protocol": "source-diff-pilot-review-v0.1",
            }
            (root / "artifacts/pilot/source-diffs.pilot.jsonl").write_text(
                json.dumps(source) + "\n",
                encoding="utf-8",
            )
            (root / "reviews/source-diffs.pilot.review.jsonl").write_text(
                json.dumps(review) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                validate_source_cache(root, artifact_name="pilot")

    def _run(self, args: list[str], *, cwd: Path) -> str:
        return subprocess.check_output(args, cwd=cwd, text=True)


if __name__ == "__main__":
    unittest.main()

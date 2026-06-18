import json
import tempfile
import unittest
from pathlib import Path

from gitctx.data_artifacts import (
    create_generated_label_review_template,
    create_named_generated_label_review_template,
    create_smoke_review_template,
    create_source_review_template,
    normalize_smoke_report,
    normalize_source_report,
    validate_generated_label_review,
    validate_named_generated_label_review,
    validate_source_artifact,
    validate_source_review,
    validate_smoke_review,
    validate_smoke_artifact,
    write_checksums,
)


class DataArtifactTests(unittest.TestCase):
    def test_normalizes_validates_and_checksums_smoke_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/smoke").mkdir(parents=True)
            (root / "manifests").mkdir()
            (root / "lineage").mkdir()

            (root / "artifacts/smoke/source-diffs.smoke.report.json").write_text(
                json.dumps(
                    {
                        "data_dir": "/private/path",
                        "manifest_path": "manifests/source-manifest.audit.jsonl",
                        "output_path": "/private/path/artifacts/smoke/source-diffs.smoke.jsonl",
                        "written_records": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "artifacts/smoke/source-diffs.smoke.jsonl").write_text(
                json.dumps(
                    {
                        "id": "example-repo-111111111111",
                        "source_repo_url": "https://github.com/example/repo",
                        "source_license": "MIT",
                        "manifest_revision": "1111111111111111111111111111111111111111",
                        "source_commit": "1111111111111111111111111111111111111111",
                        "parent_commit": "0000000000000000000000000000000000000000",
                        "data_split": "DEV",
                        "changed_paths": ["parser.py"],
                        "excluded_paths": [],
                        "diff_stat": " parser.py | 1 +",
                        "historical_subject": "fix(parser): reject empty values",
                        "extraction_command": "git diff --stat ...",
                        "review_status": "not_reviewed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "manifests/source-manifest.audit.jsonl").write_text(
                json.dumps(
                    {
                        "repo_url": "https://github.com/example/repo",
                        "default_branch": "main",
                        "source_license": "MIT",
                        "license_url": "https://example.com/license",
                        "license_review_date": "2026-06-18",
                        "reviewer": "Test User",
                        "review_status": "approved_for_audit",
                        "source_revision": "1111111111111111111111111111111111111111",
                        "allowed_splits": ["DEV"],
                        "exclude_globs": ["vendor/**"],
                        "notes": "test",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "lineage/gitctx-public-commit.txt").write_text(
                "21c0ea9\n",
                encoding="utf-8",
            )

            report = normalize_smoke_report(root)
            summary = validate_smoke_artifact(root)
            review_path = create_smoke_review_template(root, reviewer="reviewer@example.com")
            review_summary = validate_smoke_review(root)
            checksum_path = write_checksums(root)

            self.assertEqual(report["data_dir"], "$GITCTX_DATA_DIR")
            self.assertEqual(summary["source_records"], 1)
            self.assertTrue(review_path.exists())
            self.assertEqual(review_summary["needs_review"], 1)
            self.assertTrue(checksum_path.exists())
            self.assertIn("source-diffs.smoke.jsonl", checksum_path.read_text())
            self.assertIn("source-diffs.smoke.review.jsonl", checksum_path.read_text())

    def test_normalizes_and_validates_named_source_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/pilot").mkdir(parents=True)
            (root / "manifests").mkdir()
            (root / "lineage").mkdir()
            source_record = {
                "id": "example-repo-111111111111",
                "source_repo_url": "https://github.com/example/repo",
                "source_license": "MIT",
                "manifest_revision": "1111111111111111111111111111111111111111",
                "source_commit": "1111111111111111111111111111111111111111",
                "parent_commit": "0000000000000000000000000000000000000000",
                "data_split": "DEV",
                "changed_paths": ["parser.py"],
                "excluded_paths": [],
                "diff_stat": " parser.py | 1 +",
                "historical_subject": "fix(parser): reject empty values",
                "extraction_command": "git diff --stat ...",
                "review_status": "not_reviewed",
            }
            manifest_record = {
                "repo_url": "https://github.com/example/repo",
                "default_branch": "main",
                "source_license": "MIT",
                "license_url": "https://example.com/license",
                "license_review_date": "2026-06-18",
                "reviewer": "Test User",
                "review_status": "approved_for_audit",
                "source_revision": "1111111111111111111111111111111111111111",
                "allowed_splits": ["DEV"],
                "exclude_globs": ["vendor/**"],
                "notes": "test",
            }
            (root / "artifacts/pilot/source-diffs.pilot.jsonl").write_text(
                json.dumps(source_record) + "\n",
                encoding="utf-8",
            )
            (root / "artifacts/pilot/source-diffs.pilot.report.json").write_text(
                json.dumps(
                    {
                        "data_dir": "/private/path",
                        "manifest_path": "manifests/source-manifest.audit.jsonl",
                        "output_path": "/private/path/artifacts/pilot/source-diffs.pilot.jsonl",
                        "written_records": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "manifests/source-manifest.audit.jsonl").write_text(
                json.dumps(manifest_record) + "\n",
                encoding="utf-8",
            )
            (root / "lineage/gitctx-public-commit.txt").write_text("abc1234\n", encoding="utf-8")

            report = normalize_source_report(root, artifact_name="pilot")
            summary = validate_source_artifact(root, artifact_name="pilot")
            review_path = create_source_review_template(
                root,
                artifact_name="pilot",
                reviewer="reviewer@example.com",
            )
            review_summary = validate_source_review(root, artifact_name="pilot")
            checksum_path = write_checksums(root)

            self.assertEqual(report["output_path"], "artifacts/pilot/source-diffs.pilot.jsonl")
            self.assertEqual(summary["artifact_name"], "pilot")
            self.assertTrue(review_path.exists())
            self.assertEqual(review_summary["artifact_name"], "pilot")
            self.assertEqual(review_summary["needs_review"], 1)
            self.assertIn("source-diffs.pilot.jsonl", checksum_path.read_text())
            self.assertIn("source-diffs.pilot.review.jsonl", checksum_path.read_text())

    def test_creates_and_validates_generated_label_review_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/teacher").mkdir(parents=True)
            (root / "reviews").mkdir()
            (root / "artifacts/teacher/generated-labels.smoke.jsonl").write_text(
                json.dumps(
                    {
                        "id": "generated-example-repo-111111111111",
                        "source_repo_url": "https://github.com/example/repo",
                        "source_license": "MIT",
                        "source_commit": "1111111111111111111111111111111111111111",
                        "parent_commit": "0000000000000000000000000000000000000000",
                        "data_split": "DEV",
                        "changed_paths": ["src/parser.py"],
                        "teacher_model_id": "ollama/qwen2.5-coder:7b",
                        "teacher_runtime": "ollama",
                        "teacher_runtime_model_id": "qwen2.5-coder:7b",
                        "teacher_revision": "dae161e27b0e",
                        "teacher_license": "Apache-2.0",
                        "teacher_size": "4.7 GB",
                        "teacher_context_length": "32K",
                        "prompt_version": "commit-message-teacher-v0.1",
                        "decoding_config": {"temperature": 0.0},
                        "generation_timestamp": "2026-06-18T04:11:54Z",
                        "header": "fix(parser): handle empty values",
                        "body": [],
                        "footers": [],
                        "type": "fix",
                        "scope": "parser",
                        "confidence": 1.0,
                        "warnings": [],
                        "evidence_paths": ["src/parser.py"],
                        "parser_result": {},
                        "verifier_score": 1.0,
                        "human_review_status": "not_reviewed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            review_path = create_generated_label_review_template(
                root,
                reviewer="reviewer@example.com",
            )
            summary = validate_generated_label_review(root)

            self.assertTrue(review_path.exists())
            self.assertEqual(summary["generated_label_records"], 1)
            self.assertEqual(summary["needs_review"], 1)

    def test_creates_and_validates_named_generated_label_review_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/teacher").mkdir(parents=True)
            (root / "reviews").mkdir()
            (root / "artifacts/teacher/generated-labels.pilot.jsonl").write_text(
                json.dumps(
                    {
                        "id": "generated-example-repo-111111111111",
                        "source_repo_url": "https://github.com/example/repo",
                        "source_license": "MIT",
                        "source_commit": "1111111111111111111111111111111111111111",
                        "parent_commit": "0000000000000000000000000000000000000000",
                        "data_split": "DEV",
                        "changed_paths": ["src/parser.py"],
                        "teacher_model_id": "ollama/qwen2.5-coder:7b",
                        "teacher_runtime": "ollama",
                        "teacher_runtime_model_id": "qwen2.5-coder:7b",
                        "teacher_revision": "dae161e27b0e",
                        "teacher_license": "Apache-2.0",
                        "teacher_size": "4.7 GB",
                        "teacher_context_length": "32K",
                        "prompt_version": "commit-message-teacher-v0.1",
                        "decoding_config": {"temperature": 0.0},
                        "generation_timestamp": "2026-06-18T04:11:54Z",
                        "header": "fix(parser): handle empty values",
                        "body": [],
                        "footers": [],
                        "type": "fix",
                        "scope": "parser",
                        "confidence": 1.0,
                        "warnings": [],
                        "evidence_paths": ["src/parser.py"],
                        "parser_result": {},
                        "verifier_score": 1.0,
                        "human_review_status": "not_reviewed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            review_path = create_named_generated_label_review_template(
                root,
                artifact_name="pilot",
                reviewer="reviewer@example.com",
            )
            summary = validate_named_generated_label_review(root, artifact_name="pilot")

            self.assertEqual(review_path.name, "generated-labels.pilot.review.jsonl")
            self.assertEqual(summary["artifact_name"], "pilot")
            self.assertEqual(summary["generated_label_records"], 1)
            self.assertEqual(summary["needs_review"], 1)

    def test_review_validation_rejects_missing_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/smoke").mkdir(parents=True)
            (root / "reviews").mkdir()
            (root / "artifacts/smoke/source-diffs.smoke.jsonl").write_text(
                json.dumps(
                    {
                        "id": "example-repo-111111111111",
                        "source_repo_url": "https://github.com/example/repo",
                        "source_license": "MIT",
                        "manifest_revision": "1111111111111111111111111111111111111111",
                        "source_commit": "1111111111111111111111111111111111111111",
                        "parent_commit": "0000000000000000000000000000000000000000",
                        "data_split": "DEV",
                        "changed_paths": ["parser.py"],
                        "excluded_paths": [],
                        "diff_stat": " parser.py | 1 +",
                        "historical_subject": "fix(parser): reject empty values",
                        "extraction_command": "git diff --stat ...",
                        "review_status": "not_reviewed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "reviews/source-diffs.smoke.review.jsonl").write_text("", encoding="utf-8")

            with self.assertRaises(SystemExit):
                validate_smoke_review(root)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from gitctx.data_artifacts import (
    apply_generated_label_review_policy,
    apply_source_review_policy,
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


def _source_record(
    record_id: str,
    changed_paths: list[str],
    historical_subject: str,
    diff_stat: str,
    *,
    data_split: str,
) -> dict[str, object]:
    return {
        "id": f"example-repo-{record_id}",
        "source_repo_url": "https://github.com/example/repo",
        "source_license": "MIT",
        "manifest_revision": "1111111111111111111111111111111111111111",
        "source_commit": "1111111111111111111111111111111111111111",
        "parent_commit": "0000000000000000000000000000000000000000",
        "data_split": data_split,
        "changed_paths": changed_paths,
        "excluded_paths": [],
        "diff_stat": diff_stat,
        "historical_subject": historical_subject,
        "extraction_command": "git diff --stat ...",
        "review_status": "not_reviewed",
    }


def _generated_label(
    suffix: str,
    *,
    header: str = "fix(parser): handle empty values",
    verifier_score: float = 1.0,
    parser_errors: object = None,
) -> dict[str, object]:
    return {
        "id": f"generated-example-repo-{suffix}",
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
        "header": header,
        "body": [],
        "footers": [],
        "type": header.split(":", 1)[0].split("(", 1)[0],
        "scope": "parser",
        "confidence": 1.0,
        "warnings": [],
        "evidence_paths": ["src/parser.py"],
        "parser_result": {"errors": parser_errors or []},
        "verifier_score": verifier_score,
        "human_review_status": "not_reviewed",
    }


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

    def test_named_source_artifact_can_use_custom_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/next").mkdir(parents=True)
            (root / "manifests").mkdir()
            (root / "lineage").mkdir()
            source_record = {
                "id": "example-repo-222222222222",
                "source_repo_url": "https://github.com/example/repo",
                "source_license": "MIT",
                "manifest_revision": "2222222222222222222222222222222222222222",
                "source_commit": "2222222222222222222222222222222222222222",
                "parent_commit": "1111111111111111111111111111111111111111",
                "data_split": "REPORT",
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
                "license_review_date": "2026-06-20",
                "reviewer": "Test User",
                "review_status": "approved_for_audit",
                "source_revision": "2222222222222222222222222222222222222222",
                "allowed_splits": ["DEV", "REPORT"],
                "exclude_globs": ["vendor/**"],
                "notes": "test",
            }
            (root / "artifacts/next/source-diffs.next.jsonl").write_text(
                json.dumps(source_record) + "\n",
                encoding="utf-8",
            )
            (root / "artifacts/next/source-diffs.next.report.json").write_text(
                json.dumps(
                    {
                        "data_dir": "/private/path",
                        "manifest_path": "/private/path/source-manifest.next.jsonl",
                        "output_path": "/private/path/artifacts/next/source-diffs.next.jsonl",
                        "split_plan_path": str(root / "manifests/split-plan.next.json"),
                        "written_records": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "manifests/source-manifest.audit.jsonl").write_text("", encoding="utf-8")
            (root / "manifests/source-manifest.next.jsonl").write_text(
                json.dumps(manifest_record) + "\n",
                encoding="utf-8",
            )
            (root / "manifests/split-plan.next.json").write_text("{}\n", encoding="utf-8")
            (root / "lineage/gitctx-public-commit.txt").write_text("abc1234\n", encoding="utf-8")

            report = normalize_source_report(
                root,
                artifact_name="next",
                manifest_path="manifests/source-manifest.next.jsonl",
            )
            summary = validate_source_artifact(
                root,
                artifact_name="next",
                manifest_path="manifests/source-manifest.next.jsonl",
            )
            checksum_path = write_checksums(root)

            self.assertEqual(report["manifest_path"], "manifests/source-manifest.next.jsonl")
            self.assertEqual(report["split_plan_path"], "manifests/split-plan.next.json")
            self.assertEqual(summary["manifest_path"], "manifests/source-manifest.next.jsonl")
            checksum_text = checksum_path.read_text()
            self.assertIn("manifests/source-manifest.audit.jsonl", checksum_text)
            self.assertIn("manifests/source-manifest.next.jsonl", checksum_text)

    def test_named_source_artifact_rejects_repo_missing_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/next").mkdir(parents=True)
            (root / "manifests").mkdir()
            (root / "artifacts/next/source-diffs.next.jsonl").write_text(
                json.dumps(
                    {
                        "id": "missing-repo-222222222222",
                        "source_repo_url": "https://github.com/missing/repo",
                        "source_license": "MIT",
                        "manifest_revision": "2222222222222222222222222222222222222222",
                        "source_commit": "2222222222222222222222222222222222222222",
                        "parent_commit": "1111111111111111111111111111111111111111",
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
            (root / "manifests/source-manifest.next.jsonl").write_text(
                json.dumps(
                    {
                        "repo_url": "https://github.com/example/repo",
                        "default_branch": "main",
                        "source_license": "MIT",
                        "license_url": "https://example.com/license",
                        "license_review_date": "2026-06-20",
                        "reviewer": "Test User",
                        "review_status": "approved_for_audit",
                        "source_revision": "2222222222222222222222222222222222222222",
                        "allowed_splits": ["DEV"],
                        "exclude_globs": ["vendor/**"],
                        "notes": "test",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                validate_source_artifact(
                    root,
                    artifact_name="next",
                    manifest_path="manifests/source-manifest.next.jsonl",
                )

    def test_applies_source_review_policy_to_pending_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/pilot").mkdir(parents=True)
            (root / "reviews").mkdir()
            records = [
                _source_record(
                    "code-change",
                    ["src/parser.py", "tests/test_parser.py"],
                    "fix parser edge case",
                    "2 files changed, 10 insertions(+), 2 deletions(-)",
                    data_split="DEV",
                ),
                _source_record(
                    "held-out",
                    ["src/parser.py"],
                    "fix held out edge case",
                    "1 file changed, 3 insertions(+)",
                    data_split="HELD_OUT",
                ),
                _source_record(
                    "docs-only",
                    ["docs/usage.md"],
                    "clarify docs",
                    "1 file changed, 5 insertions(+)",
                    data_split="REPORT",
                ),
                _source_record(
                    "dep-bump",
                    ["src/parser.py"],
                    "bump parser dependency",
                    "1 file changed, 2 insertions(+)",
                    data_split="DEV",
                ),
            ]
            (root / "artifacts/pilot/source-diffs.pilot.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            create_source_review_template(
                root,
                artifact_name="pilot",
                reviewer="reviewer@example.com",
            )

            dry_summary = apply_source_review_policy(
                root,
                artifact_name="pilot",
                reviewer="reviewer@example.com",
            )
            dry_review = validate_source_review(root, artifact_name="pilot")
            self.assertEqual(dry_summary["accepted_for_teacher_labeling"], 1)
            self.assertEqual(dry_review["needs_review"], 4)

            summary = apply_source_review_policy(
                root,
                artifact_name="pilot",
                reviewer="reviewer@example.com",
                review_timestamp="2026-06-21T00:00:00Z",
                write=True,
            )
            review_summary = validate_source_review(root, artifact_name="pilot")
            decisions = [
                json.loads(line)["decision"]
                for line in (root / "reviews/source-diffs.pilot.review.jsonl").read_text().splitlines()
            ]

            self.assertEqual(summary["accepted_for_teacher_labeling"], 1)
            self.assertEqual(summary["rejected"], 3)
            self.assertEqual(review_summary["needs_review"], 0)
            self.assertEqual(decisions.count("accepted_for_teacher_labeling"), 1)
            self.assertEqual(decisions.count("rejected"), 3)

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

    def test_applies_generated_label_review_policy_to_pending_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/teacher").mkdir(parents=True)
            (root / "reviews").mkdir()
            labels = [
                _generated_label("perfect", verifier_score=1.0, parser_errors=[]),
                _generated_label(
                    "long-subject",
                    header=(
                        "feat(parser): add a very long subject that should stay out "
                        "of the proof artifact"
                    ),
                    verifier_score=0.6666666666666666,
                    parser_errors=["subject length is outside the configured limit"],
                ),
            ]
            (root / "artifacts/teacher/generated-labels.pilot.jsonl").write_text(
                "\n".join(json.dumps(label) for label in labels) + "\n",
                encoding="utf-8",
            )
            create_named_generated_label_review_template(
                root,
                artifact_name="pilot",
                reviewer="reviewer@example.com",
            )

            dry_summary = apply_generated_label_review_policy(
                root,
                artifact_name="pilot",
                reviewer="reviewer@example.com",
            )
            dry_review = validate_named_generated_label_review(root, artifact_name="pilot")
            self.assertEqual(dry_summary["accept"], 1)
            self.assertEqual(dry_summary["reject"], 1)
            self.assertEqual(dry_review["needs_review"], 2)

            summary = apply_generated_label_review_policy(
                root,
                artifact_name="pilot",
                reviewer="reviewer@example.com",
                review_timestamp="2026-06-23T00:00:00Z",
                write=True,
            )
            review_summary = validate_named_generated_label_review(root, artifact_name="pilot")
            reviews = [
                json.loads(line)
                for line in (
                    root / "reviews/generated-labels.pilot.review.jsonl"
                ).read_text().splitlines()
            ]

            self.assertEqual(summary["accept"], 1)
            self.assertEqual(summary["reject"], 1)
            self.assertEqual(review_summary["needs_review"], 0)
            self.assertEqual(reviews[0]["decision"], "accept")
            self.assertEqual(reviews[1]["decision"], "reject")
            self.assertIn("subject_issue", reviews[1]["issues"])

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

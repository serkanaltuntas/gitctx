from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gitctx.artifact_eval import evaluate_training_artifact


class ArtifactEvalTests(unittest.TestCase):
    def test_evaluates_targets_teacher_labels_and_historical_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_artifacts(root)

            report = evaluate_training_artifact(root, artifact_name="pilot")

            self.assertEqual(report["training_records"], 2)
            self.assertEqual(report["changed_from_teacher_records"], 1)
            self.assertEqual(report["target"]["format_validity"]["true"], 2)
            self.assertEqual(report["teacher"]["format_validity"]["true"], 2)
            self.assertEqual(report["historical"]["format_validity"]["false"], 1)
            self.assertEqual(report["label_source_counts"]["teacher_generated_human_edited"], 1)
            self.assertEqual(report["by_data_split"]["DEV"]["training_records"], 1)
            self.assertEqual(report["by_data_split"]["DEV"]["target"]["format_validity"]["true"], 1)
            self.assertEqual(report["by_data_split"]["REPORT"]["training_records"], 1)
            self.assertEqual(
                report["by_data_split"]["REPORT"]["historical"]["format_validity"]["false"],
                1,
            )
            self.assertTrue((root / "artifacts/eval/sft.pilot.v0.baseline.report.json").exists())

    def _write_artifacts(self, root: Path) -> None:
        (root / "artifacts/train").mkdir(parents=True)
        (root / "artifacts/teacher").mkdir(parents=True)
        training_records = [
            {
                "id": "sft-example-repo-111111111111",
                "artifact_name": "pilot",
                "artifact_version": "v0",
                "source_diff_id": "example-repo-111111111111",
                "teacher_input_id": "teacher-input-example-repo-111111111111",
                "generated_label_id": "generated-example-repo-111111111111",
                "generated_label_review_id": "review-generated-example-repo-111111111111",
                "source_repo_url": "https://github.com/example/repo",
                "source_license": "MIT",
                "source_commit": "1" * 40,
                "parent_commit": "0" * 40,
                "data_split": "REPORT",
                "changed_paths": ["src/parser.py"],
                "diff_stat": " src/parser.py | 1 +",
                "historical_subject": "fix parser",
                "instruction": "Write one Conventional Commit message for the provided Git diff.",
                "diff": "diff --git a/src/parser.py b/src/parser.py\n",
                "diff_sha256": "0" * 64,
                "messages": [],
                "target_header": "fix(parser): handle empty values",
                "target_body": [],
                "target_footers": [],
                "target_message": "fix(parser): handle empty values",
                "label_source": "teacher_generated_human_accepted",
                "review_decision": "accept",
                "review_issues": [],
                "review_notes": "",
                "reviewer": "reviewer@example.com",
                "review_timestamp": "2026-06-18T20:00:00Z",
                "teacher_model_id": "ollama/qwen2.5-coder:7b",
                "teacher_runtime": "ollama",
                "teacher_runtime_model_id": "qwen2.5-coder:7b",
                "teacher_revision": "dae161e27b0e",
                "teacher_license": "Apache-2.0",
                "teacher_size": "4.7 GB",
                "teacher_context_length": "32K",
                "prompt_version": "commit-message-teacher-v0.1",
                "decoding_config": {},
                "generation_timestamp": "2026-06-18T04:11:54Z",
                "confidence": 0.9,
                "verifier_score": 1.0,
                "evidence_paths": ["src/parser.py"],
                "warnings": [],
                "parser_result": {},
            },
            {
                "id": "sft-example-repo-222222222222",
                "artifact_name": "pilot",
                "artifact_version": "v0",
                "source_diff_id": "example-repo-222222222222",
                "teacher_input_id": "teacher-input-example-repo-222222222222",
                "generated_label_id": "generated-example-repo-222222222222",
                "generated_label_review_id": "review-generated-example-repo-222222222222",
                "source_repo_url": "https://github.com/example/repo",
                "source_license": "MIT",
                "source_commit": "2" * 40,
                "parent_commit": "1" * 40,
                "data_split": "DEV",
                "changed_paths": ["src/parser.py"],
                "diff_stat": " src/parser.py | 1 +",
                "historical_subject": "fix(parser): reject blank input",
                "instruction": "Write one Conventional Commit message for the provided Git diff.",
                "diff": "diff --git a/src/parser.py b/src/parser.py\n",
                "diff_sha256": "0" * 64,
                "messages": [],
                "target_header": "fix(parser): reject blank input",
                "target_body": ["Reject blank parser input before tokenizing."],
                "target_footers": [],
                "target_message": "fix(parser): reject blank input\n\n"
                "Reject blank parser input before tokenizing.",
                "label_source": "teacher_generated_human_edited",
                "review_decision": "edit",
                "review_issues": ["subject_issue"],
                "review_notes": "",
                "reviewer": "reviewer@example.com",
                "review_timestamp": "2026-06-18T20:00:00Z",
                "teacher_model_id": "ollama/qwen2.5-coder:7b",
                "teacher_runtime": "ollama",
                "teacher_runtime_model_id": "qwen2.5-coder:7b",
                "teacher_revision": "dae161e27b0e",
                "teacher_license": "Apache-2.0",
                "teacher_size": "4.7 GB",
                "teacher_context_length": "32K",
                "prompt_version": "commit-message-teacher-v0.1",
                "decoding_config": {},
                "generation_timestamp": "2026-06-18T04:11:54Z",
                "confidence": 0.9,
                "verifier_score": 1.0,
                "evidence_paths": ["src/parser.py"],
                "warnings": [],
                "parser_result": {},
            },
        ]
        generated_labels = [
            {
                "id": "generated-example-repo-111111111111",
                "header": "fix(parser): handle empty values",
                "body": [],
                "footers": [],
            },
            {
                "id": "generated-example-repo-222222222222",
                "header": "fix(parser): handle blank values",
                "body": [],
                "footers": [],
            },
        ]
        self._write_jsonl(root / "artifacts/train/sft.pilot.v0.jsonl", training_records)
        self._write_jsonl(root / "artifacts/teacher/generated-labels.pilot.jsonl", generated_labels)

    def _write_jsonl(self, path: Path, records: list[dict[str, object]]) -> None:
        path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()

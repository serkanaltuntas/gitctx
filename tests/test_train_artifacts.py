from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from gitctx.train_artifacts import (
    create_training_artifact,
    format_commit_message,
    merge_training_artifact_inputs,
    validate_training_artifact,
)


class TrainArtifactTests(unittest.TestCase):
    def test_creates_and_validates_training_artifact_from_reviewed_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_artifact_inputs(root)

            report = create_training_artifact(root, artifact_name="pilot")
            summary = validate_training_artifact(root, artifact_name="pilot")
            records = self._load_jsonl(root / "artifacts/train/sft.pilot.v0.jsonl")

            self.assertEqual(report["training_records"], 2)
            self.assertEqual(summary["training_records"], 2)
            self.assertEqual(report["accept"], 1)
            self.assertEqual(report["edit"], 1)
            self.assertEqual(report["reject"], 1)
            self.assertEqual(report["needs_review"], 0)
            self.assertEqual(records[0]["label_source"], "teacher_generated_human_accepted")
            self.assertEqual(records[1]["label_source"], "teacher_generated_human_edited")
            self.assertEqual(records[1]["target_header"], "fix(parser): reject blank input")
            self.assertEqual(records[1]["target_body"], ["Reject blank parser input before tokenizing."])
            self.assertNotIn("generated-example-repo-333333333333", {r["generated_label_id"] for r in records})

    def test_refuses_to_build_when_review_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_artifact_inputs(root, include_needs_review=True)

            with self.assertRaises(SystemExit):
                create_training_artifact(root, artifact_name="pilot")

    def test_format_commit_message_adds_blank_lines_between_sections(self) -> None:
        message = format_commit_message(
            "feat(cli): add inspect command",
            ["Print the resolved repository context."],
            ["Refs: #12"],
        )

        self.assertEqual(
            message,
            "feat(cli): add inspect command\n\n"
            "Print the resolved repository context.\n\n"
            "Refs: #12",
        )

    def test_merges_reviewed_artifact_inputs_and_skips_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_artifact_inputs(root, artifact_name="base")
            self._write_artifact_inputs(
                root,
                artifact_name="extra",
                suffixes=("111111111111", "444444444444", "555555555555"),
            )

            report = merge_training_artifact_inputs(
                root,
                input_artifact_names=["base", "extra"],
                output_artifact_name="combined",
            )
            summary = validate_training_artifact(root, artifact_name="combined")
            records = self._load_jsonl(root / "artifacts/train/sft.combined.v0.jsonl")

            self.assertEqual(report["training_records"], 3)
            self.assertEqual(summary["training_records"], 3)
            self.assertEqual(report["source_diffs"]["skipped_duplicate_records"], 1)
            self.assertEqual(report["generated_label_reviews"]["skipped_duplicate_records"], 1)
            self.assertEqual({record["artifact_name"] for record in records}, {"combined"})
            self.assertEqual(
                {record["source_diff_id"] for record in records},
                {
                    "example-repo-111111111111",
                    "example-repo-222222222222",
                    "example-repo-444444444444",
                },
            )

    def _write_artifact_inputs(
        self,
        root: Path,
        *,
        artifact_name: str = "pilot",
        include_needs_review: bool = False,
        suffixes: tuple[str, str, str] = (
            "111111111111",
            "222222222222",
            "333333333333",
        ),
    ) -> None:
        (root / f"artifacts/{artifact_name}").mkdir(parents=True, exist_ok=True)
        (root / "artifacts/teacher").mkdir(parents=True, exist_ok=True)
        (root / "reviews").mkdir(parents=True, exist_ok=True)
        source_records = [
            self._source_record(suffixes[0], historical_subject="fix parser"),
            self._source_record(suffixes[1], historical_subject="fix blank input"),
            self._source_record(suffixes[2], historical_subject="refactor parser"),
        ]
        if include_needs_review:
            source_records.append(self._source_record("444444444444", historical_subject="docs"))
        teacher_inputs = [self._teacher_input(record) for record in source_records]
        generated_labels = [
            self._generated_label(
                teacher_inputs[0],
                header="fix(parser): handle empty values",
                body=[],
            ),
            self._generated_label(
                teacher_inputs[1],
                header="fix(parser): handle blank values",
                body=[],
            ),
            self._generated_label(
                teacher_inputs[2],
                header="refactor(parser): simplify parser",
                body=[],
            ),
        ]
        review_records = [
            self._review(generated_labels[0], decision="accept"),
            self._review(
                generated_labels[1],
                decision="edit",
                edited_header="fix(parser): reject blank input",
                edited_body=["Reject blank parser input before tokenizing."],
            ),
            self._review(generated_labels[2], decision="reject"),
        ]
        if include_needs_review:
            generated_labels.append(
                self._generated_label(
                    teacher_inputs[3],
                    header="docs(parser): update parser notes",
                    body=[],
                )
            )
            review_records.append(self._review(generated_labels[3], decision="needs_review"))

        self._write_jsonl(
            root / f"artifacts/{artifact_name}/source-diffs.{artifact_name}.jsonl",
            source_records,
        )
        self._write_jsonl(
            root / f"artifacts/teacher/teacher-inputs.{artifact_name}.jsonl",
            teacher_inputs,
        )
        self._write_jsonl(
            root / f"artifacts/teacher/generated-labels.{artifact_name}.jsonl",
            generated_labels,
        )
        self._write_jsonl(
            root / f"reviews/generated-labels.{artifact_name}.review.jsonl",
            review_records,
        )

    def _source_record(self, suffix: str, *, historical_subject: str) -> dict[str, object]:
        commit = suffix.ljust(40, "1")
        parent = "0" * 40
        return {
            "id": f"example-repo-{suffix}",
            "source_repo_url": "https://github.com/example/repo",
            "source_license": "MIT",
            "manifest_revision": commit,
            "source_commit": commit,
            "parent_commit": parent,
            "data_split": "DEV",
            "changed_paths": ["src/parser.py", "tests/test_parser.py"],
            "excluded_paths": [],
            "diff_stat": " src/parser.py | 2 +-",
            "historical_subject": historical_subject,
            "extraction_command": "git diff --stat ...",
            "review_status": "not_reviewed",
        }

    def _teacher_input(self, source: dict[str, object]) -> dict[str, object]:
        diff = f"diff --git a/src/parser.py b/src/parser.py\n+{source['id']}\n"
        return {
            "id": f"teacher-input-{source['id']}",
            "source_diff_id": source["id"],
            "review_decision_id": f"review-{source['id']}",
            "source_repo_url": source["source_repo_url"],
            "source_license": source["source_license"],
            "source_commit": source["source_commit"],
            "parent_commit": source["parent_commit"],
            "data_split": source["data_split"],
            "changed_paths": source["changed_paths"],
            "diff_stat": source["diff_stat"],
            "historical_subject": source["historical_subject"],
            "teacher_model_id": "ollama/qwen2.5-coder:7b",
            "teacher_runtime": "ollama",
            "teacher_runtime_model_id": "qwen2.5-coder:7b",
            "teacher_revision": "dae161e27b0e",
            "teacher_license": "Apache-2.0",
            "teacher_size": "4.7 GB",
            "teacher_context_length": "32K",
            "prompt_version": "commit-message-teacher-v0.1",
            "prompt_path": "prompts/commit-message-teacher-v0.1.md",
            "decoding_config": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 256},
            "system_message": "Return JSON only.",
            "user_message": "Generate one Conventional Commit message.",
            "diff": diff,
            "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
            "input_status": "ready_for_generation",
        }

    def _generated_label(
        self,
        teacher_input: dict[str, object],
        *,
        header: str,
        body: list[str],
    ) -> dict[str, object]:
        return {
            "id": f"generated-{teacher_input['source_diff_id']}",
            "source_repo_url": teacher_input["source_repo_url"],
            "source_license": teacher_input["source_license"],
            "source_commit": teacher_input["source_commit"],
            "parent_commit": teacher_input["parent_commit"],
            "data_split": teacher_input["data_split"],
            "changed_paths": teacher_input["changed_paths"],
            "teacher_model_id": teacher_input["teacher_model_id"],
            "teacher_runtime": teacher_input["teacher_runtime"],
            "teacher_runtime_model_id": teacher_input["teacher_runtime_model_id"],
            "teacher_revision": teacher_input["teacher_revision"],
            "teacher_license": teacher_input["teacher_license"],
            "teacher_size": teacher_input["teacher_size"],
            "teacher_context_length": teacher_input["teacher_context_length"],
            "prompt_version": teacher_input["prompt_version"],
            "decoding_config": teacher_input["decoding_config"],
            "generation_timestamp": "2026-06-18T04:11:54Z",
            "header": header,
            "body": body,
            "footers": [],
            "type": header.split("(", 1)[0],
            "scope": "parser",
            "confidence": 0.9,
            "warnings": [],
            "evidence_paths": ["src/parser.py"],
            "parser_result": {},
            "verifier_score": 1.0,
            "human_review_status": "not_reviewed",
        }

    def _review(
        self,
        label: dict[str, object],
        *,
        decision: str,
        edited_header: str | None = None,
        edited_body: list[str] | None = None,
    ) -> dict[str, object]:
        return {
            "id": f"review-{label['id']}",
            "generated_label_id": label["id"],
            "source_repo_url": label["source_repo_url"],
            "source_commit": label["source_commit"],
            "teacher_model_id": label["teacher_model_id"],
            "prompt_version": label["prompt_version"],
            "header": label["header"],
            "verifier_score": label["verifier_score"],
            "decision": decision,
            "issues": [],
            "edited_header": edited_header,
            "edited_body": edited_body,
            "notes": "",
            "reviewer": "reviewer@example.com",
            "review_timestamp": "2026-06-18T20:00:00Z",
            "review_protocol": "generated-label-pilot-review-v0.1",
        }

    def _write_jsonl(self, path: Path, records: list[dict[str, object]]) -> None:
        path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )

    def _load_jsonl(self, path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()

import unittest

from gitctx.provenance import (
    load_jsonl,
    validate_generated_label_matches_input,
    validate_generated_label_review_decision,
    validate_generated_label_record,
    validate_source_diff_review_decision,
    validate_source_manifest_entry,
)


class SourceManifestValidationTests(unittest.TestCase):
    def test_example_source_manifest_entry_is_valid(self) -> None:
        records = load_jsonl("examples/source-manifest.example.jsonl")

        self.assertEqual(len(records), 1)
        self.assertEqual(validate_source_manifest_entry(records[0]), ())

    def test_rejects_held_out_for_audit_source(self) -> None:
        record = load_jsonl("examples/source-manifest.example.jsonl")[0]
        record["allowed_splits"] = ["DEV", "HELD_OUT"]

        errors = validate_source_manifest_entry(record)

        self.assertIn("approved_for_audit sources must not include HELD_OUT by default", errors)

    def test_audit_source_manifest_entries_are_valid(self) -> None:
        records = load_jsonl("manifests/source-manifest.audit.jsonl")

        self.assertEqual(len(records), 5)
        for record in records:
            self.assertEqual(validate_source_manifest_entry(record), ())
            self.assertEqual(record["review_status"], "approved_for_audit")


class GeneratedLabelValidationTests(unittest.TestCase):
    def test_example_generated_label_is_valid(self) -> None:
        records = load_jsonl("examples/generated-label.example.jsonl")

        self.assertEqual(len(records), 1)
        self.assertEqual(validate_generated_label_record(records[0]), ())

    def test_rejects_teacher_label_in_held_out_split(self) -> None:
        record = load_jsonl("examples/generated-label.example.jsonl")[0]
        record["data_split"] = "HELD_OUT"

        errors = validate_generated_label_record(record)

        self.assertIn("teacher-generated labels must not be stored as HELD_OUT labels", errors)

    def test_rejects_label_id_that_does_not_match_teacher_input(self) -> None:
        teacher_input = {
            "source_diff_id": "example-repo-111111111111",
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
            "decoding_config": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 256},
        }
        label = {
            "id": "generated-other-repo-222222222222",
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
            "header": "fix(parser): handle empty values",
            "body": [],
            "footers": [],
            "type": "fix",
            "scope": "parser",
            "confidence": 0.9,
            "warnings": [],
            "evidence_paths": ["src/parser.py"],
            "parser_result": {},
            "verifier_score": 1.0,
            "human_review_status": "not_reviewed",
        }

        errors = validate_generated_label_matches_input(label, teacher_input)

        self.assertIn("id does not match teacher input source_diff_id", errors)


class SourceDiffReviewValidationTests(unittest.TestCase):
    def test_example_source_diff_review_is_valid(self) -> None:
        records = load_jsonl("examples/source-diff-review.example.jsonl")

        self.assertEqual(len(records), 1)
        self.assertEqual(validate_source_diff_review_decision(records[0]), ())

    def test_rejects_unknown_decision(self) -> None:
        record = load_jsonl("examples/source-diff-review.example.jsonl")[0]
        record["decision"] = "maybe"

        errors = validate_source_diff_review_decision(record)

        self.assertIn("invalid decision: maybe", errors)


class GeneratedLabelReviewValidationTests(unittest.TestCase):
    def test_generated_label_review_decision_is_valid(self) -> None:
        record = {
            "id": "review-generated-example-repo-111111111111",
            "generated_label_id": "generated-example-repo-111111111111",
            "source_repo_url": "https://github.com/example/repo",
            "source_commit": "1111111111111111111111111111111111111111",
            "teacher_model_id": "ollama/qwen2.5-coder:7b",
            "prompt_version": "commit-message-teacher-v0.1",
            "header": "fix(parser): handle empty values",
            "verifier_score": 1.0,
            "decision": "accept",
            "issues": [],
            "edited_header": None,
            "edited_body": None,
            "notes": "",
            "reviewer": "reviewer@example.com",
            "review_timestamp": "2026-06-18T20:00:00Z",
            "review_protocol": "generated-label-smoke-review-v0.1",
        }

        self.assertEqual(validate_generated_label_review_decision(record), ())

    def test_generated_label_review_rejects_unknown_issue(self) -> None:
        record = {
            "id": "review-generated-example-repo-111111111111",
            "generated_label_id": "generated-example-repo-111111111111",
            "source_repo_url": "https://github.com/example/repo",
            "source_commit": "1111111111111111111111111111111111111111",
            "teacher_model_id": "ollama/qwen2.5-coder:7b",
            "prompt_version": "commit-message-teacher-v0.1",
            "header": "fix(parser): handle empty values",
            "verifier_score": 1.0,
            "decision": "accept",
            "issues": ["surprise_issue"],
            "edited_header": None,
            "edited_body": None,
            "notes": "",
            "reviewer": "reviewer@example.com",
            "review_timestamp": "2026-06-18T20:00:00Z",
            "review_protocol": "generated-label-smoke-review-v0.1",
        }

        errors = validate_generated_label_review_decision(record)

        self.assertIn("invalid issues: ['surprise_issue']", errors)


if __name__ == "__main__":
    unittest.main()

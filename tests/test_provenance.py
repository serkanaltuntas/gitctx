import unittest

from gitctx.provenance import (
    load_jsonl,
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


if __name__ == "__main__":
    unittest.main()

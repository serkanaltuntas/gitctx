import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from gitctx.delta_targets import generate, render
from gitctx.reference_overlay import artifact_hash, build_override
from gitctx.reference_review import sha
from gitctx.teacher_response import decode_response
from test_student_input import record


class TeacherResponseTests(unittest.TestCase):
    def test_text_preserves_content_and_breaking_marker(self):
        raw = "fix(parser)!: Preserve  two spaces  \r\n"
        fields, target = decode_response(raw, "text")
        self.assertEqual(target, raw[:-2])
        self.assertEqual(fields["subject"], "Preserve  two spaces  ")
        self.assertTrue(fields["breaking"])

    def test_does_not_extract_a_header_from_explanations_or_fences(self):
        for raw in ("Here is a header:\nfix: change value", "```\nfix: change value\n```",
                    " fix: change value", "fix: change value\n\nExplanation",
                    "fix: change\u2028value", "unknown: change value", "fix(a/b): change value",
                    "fix:   "):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_response(raw, "text")

    def test_json_analysis_is_preserved_but_not_target_text(self):
        fields = {"type": "fix", "scope": "", "subject": "change value", "analysis": "unverified"}
        self.assertEqual(decode_response(json.dumps(fields)), (fields, "fix: change value"))
        for raw in ("null", "[]", '{"type":"fix"}'):
            with self.assertRaises(ValueError):
                decode_response(raw)

    def test_generation_omits_grammar_only_in_text_mode(self):
        class Teacher:
            def encode(self, text):
                return type("Encoded", (), {"ids": [1] * 50})()
        class Student:
            def encode(self, text):
                return [1] * 10
        r = record(); r["data_split"] = "DEV"
        response = {"response": "fix(a): use new value\n", "done_reason": "stop", "prompt_eval_count": 50}
        with patch("gitctx.delta_targets.request_json", side_effect=[
            {"models": [{"name": "teacher", "digest": "digest"}]}, response
        ]) as request:
            c = generate(r, tokenizer=Teacher(), student_tokenizer=Student(), model="teacher",
                         model_digest="digest", model_license="Apache-2.0", output_format="text")
        self.assertNotIn("format", request.call_args_list[-1].args[1])
        self.assertEqual(c["target"], "fix(a): use new value")
        self.assertEqual(c["response"], response)
        self.assertEqual(c["validation_errors"], [])
        self.assertIn("header only", render(r, output_format="text"))
        a = {"decision": "accept", "method": "full_diff_review", "reviewer_kind": "assistant",
             "candidate_sha256": artifact_hash(c), "source_diff_sha256": sha(r["diff"]),
             "timestamp": "2026-01-01", "note": "Checked the complete source.",
             "claims": [{"text": c["target"], "decision": "supported",
                         "evidence": [{"line": 6, "quote": "+new"}]}]}
        overlay = build_override(r, c, a, Student(), teacher_revision="pinned")
        self.assertEqual(overlay["replacement_message"], response["response"].rstrip("\n"))
        modified = deepcopy(c)
        modified["target"] = "fix(a): invented value"
        modified["target_sha256"] = sha(modified["target"])
        a["candidate_sha256"] = artifact_hash(modified)
        with self.assertRaisesRegex(ValueError, "differs from teacher"):
            build_override(r, modified, a, Student(), teacher_revision="pinned")

    def test_invalid_format_rejected_before_network(self):
        with patch("gitctx.delta_targets.request_json") as request:
            with self.assertRaises(ValueError):
                generate({"data_split": "DEV"}, tokenizer=None, student_tokenizer=None,
                         model="test", model_digest="x", model_license="Apache-2.0", output_format="unknown")
        request.assert_not_called()

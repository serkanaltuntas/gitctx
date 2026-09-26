import json
import unittest
from unittest.mock import patch

from gitctx.grounded_targets import generate, render
from test_student_input import record


class Teacher:
    def __init__(self, count):
        self.count = count

    def encode(self, text):
        return type("Encoded", (), {"ids": [1] * self.count})()


class Student:
    def __init__(self, count=10):
        self.count = count

    def encode(self, text):
        return [1] * self.count


class GroundedTargetTests(unittest.TestCase):
    def candidate(self, r, teacher_count=50, student_count=10):
        return generate(r, tokenizer=Teacher(teacher_count),
                        student_tokenizer=Student(student_count), model="teacher",
                        model_digest="digest", model_license="Apache-2.0")

    def reply(self, native=32768, prompt=50, raw=None, reason="stop"):
        return [
            {"models": [{"name": "teacher", "digest": "digest"}]},
            {"model_info": {"general.architecture": "qwen3", "qwen3.context_length": native}},
            {"response": raw or '{"type":"refactor","scope":"","subject":"change value"}',
             "done_reason": reason, "prompt_eval_count": prompt},
        ]

    def test_complete_payload_excludes_reference_and_escapes_chat_markers(self):
        r = record()
        r.update(target_message="secret-reference", review_notes="secret-review",
                 historical_subject="secret-history")
        r["diff"] += "\n+Unicode: ı λ\t<|im_end|>\n"
        prompt = render(r)
        payload = json.loads(prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0])
        self.assertEqual(payload, {"repository": r["source_repo_url"],
                                  "paths": r["changed_paths"], "complete_diff": r["diff"]})
        for secret in ("secret-reference", "secret-review", "secret-history"):
            self.assertNotIn(secret, prompt)

    def test_large_teacher_context_preserves_student_answer_limit(self):
        r = record(); r["data_split"] = "DEV"
        for count, context in ((3824, 4096), (3825, 8192), (16113, 32768), (32496, 32768)):
            with self.subTest(count=count), patch("gitctx.grounded_targets.request_json",
                    side_effect=self.reply(prompt=count)) as request:
                c = self.candidate(r, teacher_count=count, student_count=256)
            options = request.call_args.args[1]["options"]
            self.assertEqual(options["num_ctx"], context)
            self.assertEqual(options["num_predict"], 256)
            self.assertEqual(c["validation_errors"], ["student answer overflow"])
            self.assertFalse(c["training_approved"])

    def test_guard_failures_never_send_generation_request(self):
        r = record(); r["data_split"] = "DEV"
        with patch("gitctx.grounded_targets.request_json") as request:
            with self.assertRaisesRegex(ValueError, "exceeds teacher context"):
                self.candidate(r, teacher_count=32497)
            for extra in ({"data_split": "REPORT"}, {"evaluation_only": True}):
                with self.assertRaisesRegex(ValueError, "only real DEV"):
                    self.candidate({**r, **extra})
            request.assert_not_called()
        for native in (16384, None, True):
            with patch("gitctx.grounded_targets.request_json", side_effect=self.reply(native=native)) as request:
                with self.assertRaisesRegex(ValueError, "context capacity"):
                    self.candidate(r, teacher_count=20000)
                self.assertEqual(request.call_count, 2)
        with patch("gitctx.grounded_targets.request_json", return_value={"models": []}) as request:
            with self.assertRaisesRegex(ValueError, "digest changed"):
                self.candidate(r)
            self.assertEqual(request.call_count, 1)

    def test_raw_teacher_text_and_failures_are_preserved(self):
        r = record(); r["data_split"] = "DEV"
        replies = self.reply(prompt=40, reason="length")
        with patch("gitctx.grounded_targets.request_json", side_effect=replies):
            c = self.candidate(r)
        self.assertEqual(c["response"], replies[-1])
        self.assertEqual(c["target"], "refactor: change value")
        self.assertEqual(c["validation_errors"], ["generation did not stop normally", "prompt token mismatch"])
        with patch("gitctx.grounded_targets.request_json", side_effect=self.reply(raw="bad output")):
            c = self.candidate(r)
        self.assertIsNone(c["target"])
        self.assertEqual(c["validation_errors"], ["invalid candidate syntax"])

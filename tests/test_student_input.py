from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

from gitctx.student_input import review_provenance, student_messages
from gitctx.student_readiness import SYNTHETIC, corpus, inspect
from gitctx.student_sequences import diff_units, inspect_budget, materialize, prompt_ids
from gitctx.student_tokenizer import OFFSET, SPECIAL, StudentTokenizer
from gitctx.train_artifacts import _label_source


def record(rid="example", diff=None):
    value = {
        "id": rid, "source_repo_url": "https://github.com/example/project",
        "changed_paths": ["src/İstanbul.py"],
        "diff": diff or "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-old\n+new\n",
        "target_message": "fix(a): preserve whitespace\n\nKeep tabs.\n",
        "source_license": "MIT", "teacher_license": "Apache-2.0",
        "teacher_model_id": "ollama/qwen2.5-coder:7b", "teacher_revision": "abcdef123",
        "prompt_version": "teacher-v1", "label_source": "teacher_generated_human_accepted",
        "review_decision": "accept", "review_notes": "generated-label-review-policy-v0.1; parser pass",
        "reviewer": "owner@example.com", "generated_label_id": "label-" + rid,
        "generated_label_review_id": "review-" + rid,
    }
    import hashlib
    value["diff_sha256"] = hashlib.sha256(value["diff"].encode()).hexdigest()
    return value


class StudentInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = StudentTokenizer.fit(corpus([record()], ["example"]), 512)

    def test_utf8_whitespace_control_bytes_and_literal_roles_roundtrip(self):
        rng = random.Random(41)
        fuzz = ["".join(chr(rng.choice([rng.randrange(0, 0xD800),
                                      rng.randrange(0xE000, 0x110000)])) for _ in range(200))
                for _ in range(20)]
        for value in [*SYNTHETIC, *fuzz]:
            with self.subTest(value=value[:30]):
                ids = self.tokenizer.encode(value)
                self.assertTrue(all(i >= OFFSET for i in ids))
                self.assertEqual(self.tokenizer.decode(ids), value)
                self.assertEqual(self.tokenizer.backend.decode([i - OFFSET for i in ids]), value)
        with self.assertRaises(ValueError):
            self.tokenizer.decode([SPECIAL["<assistant>"]])

    def test_saved_tokenizer_is_reproducible_and_rejects_lossy_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tokenizer.json"
            self.tokenizer.save(p)
            restored = StudentTokenizer.load(p)
            self.assertEqual(restored.encode(SYNTHETIC[2]), self.tokenizer.encode(SYNTHETIC[2]))
            config = json.loads(p.read_text())
            config["backend"]["normalizer"] = {"type": "Lowercase"}
            p.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                StudentTokenizer.load(p)

    def test_pathological_long_lines_are_bounded_and_lossless(self):
        from gitctx.student_tokenizer import content_pre_tokenizer
        text = "a" * 170000 + "\t\r\n" + "🚀" * 2000
        # ByteLevel may represent a character by up to four byte characters.
        pieces = content_pre_tokenizer().pre_tokenize_str(text)
        self.assertLessEqual(max(len(piece) for piece, _ in pieces), 4096)
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_decoding_does_not_rescan_vocabulary_for_every_token(self):
        from unittest.mock import Mock
        tokenizer = StudentTokenizer(self.tokenizer.backend)
        text = "unseen identifier 🚀 " * 500
        ids = tokenizer.encode(text)
        tokenizer.backend = Mock(wraps=tokenizer.backend)
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertLessEqual(tokenizer.backend.get_vocab_size.call_count, 1)

    def test_unicode_separators_do_not_change_git_line_numbers(self):
        diff = "diff --git a/a b/a\n@@ -1 +1 @@\n-old\u2028text\n+new\x85text\n"
        units = diff_units(diff)
        self.assertEqual(units[-1]["changed_line_indices"], [2, 3])
        self.assertEqual(units[-1]["end_line"], 4)

    def test_fitting_excludes_validation_and_reserved_text(self):
        train, validation = record("train"), record("val")
        a = StudentTokenizer.fit(corpus([train, validation], ["train"]), 512)
        validation["diff"] = "UNIQUE_VALIDATION_SECRET " * 1000
        validation["target_message"] = "another unseen reference"
        b = StudentTokenizer.fit(corpus([train, validation], ["train"]), 512)
        self.assertEqual(a.backend.to_str(), b.backend.to_str())
        self.assertEqual(b.decode(b.encode(validation["diff"])), validation["diff"])

    def test_prompt_never_depends_on_target_or_teacher_messages(self):
        a = record()
        b = {k: a[k] for k in ("source_repo_url", "changed_paths", "diff")}
        b["messages"] = [{"role": "system", "content": "Return JSON"}]
        b["historical_subject"] = "do not leak this"
        b["target_message"] = "X" * 3000
        self.assertEqual(student_messages(a), student_messages(b))
        self.assertEqual(prompt_ids(a, self.tokenizer), prompt_ids(b, self.tokenizer))
        self.assertNotIn("Return JSON", student_messages(a)[0]["content"])

    def test_fixed_budget_boundary_and_target_overflow_fail_closed(self):
        r = record()
        n = len(prompt_ids(r, self.tokenizer))
        reserve = len(self.tokenizer.encode(r["target_message"])) + 1
        seq = materialize(r, self.tokenizer, context=n + reserve, reserve=reserve)
        self.assertEqual(seq["input_ids"][:n], prompt_ids(r, self.tokenizer))
        self.assertEqual(seq["loss_mask"], [0] * n + [1] * reserve)
        with self.assertRaisesRegex(ValueError, "prompt overflow"):
            materialize(r, self.tokenizer, context=n + reserve - 1, reserve=reserve)
        with self.assertRaisesRegex(ValueError, "target overflow"):
            materialize(r, self.tokenizer, context=n + reserve, reserve=reserve - 1)

    def test_long_sequence_causal_shift_and_batch_padding(self):
        import torch
        from gitctx.proof_lm_train import _batch_for_torch
        r = record(diff="diff --git a/a b/a\n@@ -0,0 +1 @@\n+" + "x " * 3500 + "\n")
        seq = materialize(r, self.tokenizer)
        self.assertGreater(len(seq["input_ids"]), 6000)
        short = materialize(record(), self.tokenizer)
        batch = _batch_for_torch(torch, [seq, short], device="cpu")
        for i, s in enumerate([seq, short]):
            start = s["prompt_length"] - 1
            labels = batch["labels"][i].tolist()
            self.assertTrue(all(v == -100 for v in labels[:start]))
            self.assertEqual(labels[start:start + s["loss_tokens"]], s["input_ids"][start + 1:])
            self.assertTrue(all(v == -100 for v in labels[start + s["loss_tokens"]:]))
        logits = torch.randn(*batch["labels"].shape, self.tokenizer.vocab_size, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(logits.grad[batch["labels"] == -100].abs().sum().item(), 0)

    def test_all_hunk_lines_and_file_headers_are_accounted_without_cropping(self):
        diff = ("diff --git a/a b/a\r\n--- a/a\r\n+++ b/a\r\n"
                "@@ -1,2 +1,2 @@\r\n---literal\r\n+++literal\r\n same\r\n"
                "@@ -9 +9 @@\n-x\n+y\n"
                "diff --git a/b b/b\nBinary files a/b and b/b differ\n")
        units = diff_units(diff)
        self.assertEqual([u["kind"] for u in units], ["file_header", "hunk", "hunk", "file_header"])
        self.assertEqual(sum(len(u["changed_line_indices"]) for u in units), 4)
        self.assertEqual([i for u in units for i in range(u["start_line"], u["end_line"])],
                         list(range(len(diff.splitlines()))))
        result = inspect_budget(record(diff=diff), self.tokenizer, context=100, reserve=20)
        self.assertTrue(result["prompt_overflow"])
        self.assertEqual(result["removed_changed_lines"], 0)
        self.assertEqual(result["decision"], "requires_policy")

    def test_automatic_acceptance_and_email_cannot_claim_human_review(self):
        r = record()
        self.assertEqual(review_provenance(r)["reviewer_kind"], "automation")
        self.assertFalse(review_provenance(r)["independent_factuality_review"])
        r.update(decision="accept", notes=r["review_notes"])
        self.assertEqual(_label_source(r), "teacher_generated_automated_accepted")
        r.update(reviewer_kind="human", review_method="independent_diff_review",
                 review_evidence="review/123", review_timestamp="2026-01-01")
        self.assertFalse(review_provenance(r)["independent_factuality_review"])
        r["notes"] = r["review_notes"] = "Reviewed against full diff"
        self.assertTrue(review_provenance(r)["independent_factuality_review"])
        self.assertEqual(_label_source(r), "teacher_generated_human_accepted")
        r.pop("review_evidence")
        self.assertEqual(_label_source(r), "teacher_generated_unverified_accepted")

    def test_readiness_keeps_excluded_ids_and_unresolved_reference_queue(self):
        rows = [record("train"), record("val"), record("old-excluded")]
        p = {"arms": {"ordered": ["train"]}, "validation_ids": ["val"]}
        plans = {r["id"]: {"decision": "exclude_oversize" if r["id"] == "old-excluded"
                              else "use_full"} for r in rows}
        findings = [{"record_id": "val", "status": "confirmed_content_error", "partition": "validation"}]
        original = deepcopy(rows)
        report, coverage, provenance, queue = inspect(rows, p, plans, findings, self.tokenizer)
        self.assertEqual(rows, original)
        self.assertTrue(report["preparation_complete"])
        self.assertFalse(report["training_ready"])
        self.assertEqual(len(coverage), 3)
        self.assertEqual(report["counts"]["legacy_excluded"]["records"], 1)
        self.assertIsNone(queue[0]["correction"])
        self.assertEqual(queue[0]["training_use"], "prohibited_as_target")
        self.assertFalse(any(p["independent_factuality_review"] for p in provenance))


if __name__ == "__main__":
    unittest.main()

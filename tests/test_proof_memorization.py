import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gitctx.proof_memorization import (
    evaluate_examples, implementation_hashes, load_examples, run, select_records,
)
from gitctx.proof_lm_train import _load_torch, _sha256
from gitctx.proof_train_job import proof_trainer_job_path
from gitctx.proof_tokenizer import tokenize_text
from test_proof_lm_train import _prepare_trainer_job


def fixture(root, run_id="memorization"):
    _prepare_trainer_job(root, run_id="fixture")
    path = proof_trainer_job_path("fixture")
    job = json.loads((root / path).read_text())
    protocol = {
        "run_id": run_id, "source_job": {"path": str(path), "sha256": _sha256(root / path)},
        "model_contract": job["model_contract"], "implementation_hashes": implementation_hashes(),
        "records": [{"record_id": "one"}],
        "training": {"seed": 17, "attention_chunk_size": 16, "activation_checkpointing": True,
                     "learning_rate": 0.0003, "evaluate_every": 1, "max_epochs": 2},
        "evaluation": {"max_new_tokens": 16}, "limitations": ["unit-test fixture"],
    }
    folder = root / "artifacts/train-runs" / run_id
    folder.mkdir()
    (folder / "protocol.json").write_text(json.dumps(protocol))
    return protocol, folder


class ProofMemorizationTests(unittest.TestCase):
    def test_selection_is_deterministic_and_excludes_report(self):
        records, meta, vocab = [], {}, set()
        for index, split in enumerate(("DEV", "DEV", "DEV", "REPORT")):
            r = {"id": str(index), "data_split": split, "source_repo_url": f"repo-{index}",
                 "target_message": "fix(api): reject bad input in parser"}
            records.append(r)
            meta[r["id"]] = {"decision": "use_full", "input_length": 300}
            vocab.update(tokenize_text(r["target_message"]))
        selected = select_records(records, meta, vocab, {"fix": 2})
        self.assertEqual(selected, select_records(list(reversed(records)), meta, vocab, {"fix": 2}))
        self.assertTrue(all(r["data_split"] == "DEV" for r in selected))
        with self.assertRaisesRegex(ValueError, "not enough"):
            select_records(records, meta, vocab, {"fix": 4})
        with self.assertRaisesRegex(ValueError, "not enough"):
            select_records(records, meta, vocab - {"parser"}, {"fix": 1})

    def test_selected_report_is_rejected_and_prompt_matches_supervision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, _ = fixture(root)
            examples, _ = load_examples(root, protocol)
            ex = examples[0]
            first = ex["sequence"]["loss_mask"].index(1)
            self.assertEqual(ex["prompt_ids"], ex["sequence"]["input_ids"][:first])
            self.assertEqual(ex["target_ids"], ex["sequence"]["input_ids"][first:first + len(ex["target_ids"])])
            modified = copy.deepcopy(protocol)
            modified["records"] = [{"record_id": "three"}]
            with self.assertRaisesRegex(ValueError, "entirely.*DEV"):
                load_examples(root, modified)

    @unittest.skipIf(_load_torch() is None, "torch missing")
    def test_perfect_tokens_do_not_hide_lossy_scope_formatting(self):
        torch = _load_torch()
        target = "fix(src/api.py): handle invalid input"
        tokens = tokenize_text(target)
        vocab = {t:i for i,t in enumerate(dict.fromkeys(["<bos>", "<sep>", "<eos>", *tokens]))}
        ids = [vocab[t] for t in tokens]
        sequence = {"input_ids": [vocab["<bos>"], *ids, vocab["<sep>"], vocab["<eos>"]],
                    "loss_mask": [0] + [1] * (len(ids) + 2)}
        class PerfectModel:
            def eval(self):
                return self
            def __call__(self, inputs, mask):
                logits = torch.full((1, inputs.shape[1], len(vocab)), -100.0)
                for i, token in enumerate(sequence["input_ids"][1:]):
                    logits[0, i, token] = 100.0
                return logits
        example = {"record": {"id": "fixture"}, "sequence": sequence, "prompt_ids": [vocab["<bos>"]],
                   "target_ids": ids, "target": target}
        with patch("gitctx.proof_memorization.generate", return_value=(ids, "stop_token")):
            summary, predictions = evaluate_examples(torch, PerfectModel(), [example], vocab,
                                                     device="cpu", max_new_tokens=32)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["exact_tokens"], 1)
        self.assertEqual(summary["scope_tokens_match"], 1)
        self.assertEqual(summary["scope_match"], 0)
        self.assertEqual(summary["exact_text"], 0)
        self.assertIn("src / api. py", predictions[0]["message"])

    @unittest.skipIf(_load_torch() is None, "torch missing")
    def test_training_resume_matches_uninterrupted_and_protects_protocol(self):
        import gitctx.proof_memorization as module
        torch = _load_torch()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            root_a, root_b = Path(a), Path(b)
            _, folder_a = fixture(root_a)
            _, folder_b = fixture(root_b)
            first = run(root_a, "memorization", device="cpu")
            original = module.evaluate_examples
            calls = 0
            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("simulated interruption")
                return original(*args, **kwargs)
            with patch.object(module, "evaluate_examples", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    run(root_b, "memorization", device="cpu")
            resumed = run(root_b, "memorization", device="cpu", resume=True)
            self.assertEqual(first["history"], resumed["history"])
            self.assertEqual(resumed["optimizer_steps"], 2)
            self.assertEqual(resumed["report_records_used"], 0)
            self.assertGreater(resumed["first_gradient_norm"], 0)
            left = torch.load(folder_a / "latest.pt", weights_only=True)
            right = torch.load(folder_b / "latest.pt", weights_only=True)
            for key in left["model_state"]:
                torch.testing.assert_close(left["model_state"][key], right["model_state"][key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "has a checkpoint"):
                run(root_a, "memorization", device="cpu")
            path = folder_b / "protocol.json"
            protocol = json.loads(path.read_text())
            protocol["training"]["learning_rate"] = 0.1
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                run(root_b, "memorization", device="cpu", resume=True)


if __name__ == "__main__":
    unittest.main()

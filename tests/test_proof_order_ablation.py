import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gitctx.proof_lm_train import _load_torch, _sha256
from gitctx.proof_order_ablation import (
    compare, evaluate_arm, fit_tokenizer, leakage_keys, load_protocol, prepare, run_arm, split_dev,
)


def records():
    rows = []
    for i in range(20):
        target = "fix(api): handle missing value safely with parser input output"
        diff = f"diff --git a/api.py b/api.py\n- old_{i}\n+ new_{i}"
        rows.append({"id": str(i), "source_repo_url": f"repo-{i//4}", "source_commit": str(i),
            "diff":diff, "diff_sha256":hashlib.sha256(diff.encode()).hexdigest(), "data_split":"DEV",
            "target_message":target, "messages":[{"role":"system","content":"Write one Conventional Commit message."},
                {"role":"user","content":diff}, {"role":"assistant","content":target}]})
    return rows


def fixture(root):
    rows = records()
    rows.append({**rows[0], "id":"report", "data_split":"REPORT"})
    (root/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (root/"plan.jsonl").write_text("".join(json.dumps({"record_id":r["id"], "decision":"use_full", "context_tokens":512})+"\n" for r in rows))
    (root/"metadata.jsonl").write_text("")
    (root/"tokenizer.json").write_text("{}")
    (root/"handoff.json").write_text(json.dumps({"training_contract":{"train_split":"DEV"}}))
    paths = {"training_artifact":"rows.jsonl","sequence_plan":"plan.jsonl","sequence_metadata":"metadata.jsonl",
             "tokenizer":"tokenizer.json","handoff":"handoff.json"}
    model = {"architecture":"decoder-only transformer","context_tokens":512,"tokenizer_vocab_size":24,
             "layers":1,"hidden_size":16,"attention_heads":2,"kv_heads":1,"intermediate_size":32,
             "position_encoding":"rope","tie_input_output_embeddings":True}
    job = {"status":"ready_for_trainer","blockers":[],"model_contract":model,
           "inputs":{k:{"path":v,"sha256":_sha256(root/v)} for k,v in paths.items()}}
    (root/"source.json").write_text(json.dumps(job))
    return prepare(root, Path("source.json"), "test-order", validation_repos=2, validation_records=4)


class OrderAblationTests(unittest.TestCase):
    def test_repository_separation_and_duplicate_purge(self):
        rows = records()
        train, val, reserved, _ = split_dev(rows, validation_repos=2, validation_records=4)
        self.assertTrue(set(r["source_repo_url"] for r in train).isdisjoint(reserved))
        duplicate = copy.deepcopy(train[0])
        duplicate.update(id="duplicate", diff=val[0]["diff"], diff_sha256=val[0]["diff_sha256"])
        train, val, again, purged = split_dev([*rows,duplicate], validation_repos=2, validation_records=4)
        self.assertEqual(again,reserved)
        self.assertEqual([r["id"] for r in purged],["duplicate"])
        forbidden = set().union(*(leakage_keys(r) for r in val))
        self.assertFalse(any(leakage_keys(r)&forbidden for r in train))
        self.assertEqual([r["id"] for r in train], [r["id"] for r in rows if r["source_repo_url"] not in reserved])
        with self.assertRaisesRegex(ValueError,"DEV only"):
            split_dev([{**rows[0],"data_split":"REPORT"}])

    def test_vocabulary_does_not_include_validation_only_tokens(self):
        train, val, _, _ = split_dev(records(),validation_repos=2,validation_records=4)
        val[0]["messages"][-1]["content"] = "VALIDATION_SECRET " * 100
        tokenizer = fit_tokenizer(train,24)
        self.assertNotIn("VALIDATION_SECRET",{v["token"] for v in tokenizer["vocab"]})
        self.assertEqual(tokenizer["vocab_size"],24)

    @unittest.skipIf(_load_torch() is None,"torch missing")
    def test_prepare_paired_training_and_validation_end_to_end_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixture(root)
            folder,p=load_protocol(root,"test-order")
            with self.assertRaisesRegex(ValueError,"already exists"):
                prepare(root,Path("source.json"),"test-order")
            protocol_path=folder/"protocol.json"
            frozen_text=protocol_path.read_text()
            protocol_path.write_text(frozen_text+"\n")
            with self.assertRaisesRegex(ValueError,"frozen protocol changed"):
                load_protocol(root,"test-order")
            protocol_path.write_text(frozen_text)
            self.assertEqual(set(p["arms"]["ordered"]),set(p["arms"]["shuffled"]))
            self.assertNotEqual(p["arms"]["ordered"],p["arms"]["shuffled"])
            self.assertTrue(set(p["arms"]["ordered"]).isdisjoint(p["validation_ids"]))
            for arm in ("ordered","shuffled"):
                report=run_arm(root,"test-order",arm,device="cpu")
                self.assertEqual(report["status"],"trained")
                evaluated=evaluate_arm(root,"test-order",arm,device="cpu")
                self.assertEqual(evaluated["records"],4)
                self.assertGreater(evaluated["mean_loss"],0)
                with self.assertRaisesRegex(ValueError,"already exists"):
                    evaluate_arm(root,"test-order",arm,device="cpu")
            self.assertEqual(set(compare(root,"test-order")["arms"]),{"ordered","shuffled"})
            result_path=folder/"ordered.validation.json"
            result=json.loads(result_path.read_text())
            result["predictions"].pop()
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,"coverage mismatch"):
                compare(root,"test-order")
            path=root/"rows.jsonl"
            path.write_text(path.read_text()+"\n")
            with self.assertRaises(ValueError):
                load_protocol(root,"test-order")


if __name__=="__main__":
    unittest.main()

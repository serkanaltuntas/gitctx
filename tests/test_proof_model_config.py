import copy
import json
import tempfile
import unittest
from pathlib import Path

from gitctx.proof_model_config import (
    load_proof_model_config,
    main,
    validate_proof_model_config,
)


VALID_CONFIG = {
    "id": "gctx1-proof-model",
    "version": "v0",
    "status": "planned",
    "purpose": "First GCTX-1 proof-model training run after the strict reviewed SFT data gate.",
    "data": {
        "artifact_name": "gctx1-strict",
        "artifact_version": "v0",
        "train_split": "DEV",
        "eval_split": "REPORT",
        "reserved_split": "HELD_OUT",
        "minimum_dev_records": 10_000,
        "minimum_report_records": 1_000,
        "minimum_reserved_held_out_records": 1_000,
    },
    "model": {
        "target_rung": "GCTX-1",
        "target_parameter_range": "60M-100M",
        "architecture": "decoder-only transformer",
        "context_tokens": 8192,
        "tokenizer": "code-and-diff-aware tokenizer",
        "objective": "supervised fine-tuning on reviewed Conventional Commit examples",
    },
    "training": {
        "precision": "bf16 or fp16 depending on target runtime",
        "optimizer": "adamw",
        "checkpoint_policy": "save resumable checkpoints and final model artifacts",
        "reproducibility": (
            "record seed, git revision, training code revision, dataset checksum, and runtime"
        ),
    },
    "evaluation": {
        "locked_report_required": True,
        "metrics": [
            "format_validity",
            "type_match",
            "scope_quality",
            "specificity",
            "brevity",
            "exact_message_match",
        ],
        "quality_claim_policy": (
            "No public model-quality claim until a model card and eval card are written "
            "from locked REPORT results."
        ),
    },
    "release": {
        "public_model_release": False,
        "requires_model_card": True,
        "requires_eval_card": True,
        "requires_dataset_redistribution_review": True,
    },
}


class ProofModelConfigTests(unittest.TestCase):
    def test_accepts_valid_config(self) -> None:
        report = validate_proof_model_config(copy.deepcopy(VALID_CONFIG))

        self.assertTrue(report.valid)
        self.assertEqual(report.errors, [])
        self.assertEqual(report.summary["artifact_name"], "gctx1-strict")
        self.assertEqual(report.summary["target_parameter_range"], "60M-100M")

    def test_rejects_weakened_data_thresholds_and_wrong_splits(self) -> None:
        config = copy.deepcopy(VALID_CONFIG)
        config["data"]["train_split"] = "REPORT"
        config["data"]["minimum_dev_records"] = 999

        report = validate_proof_model_config(config)

        self.assertFalse(report.valid)
        self.assertIn("data.train_split: expected DEV, got REPORT", report.errors)
        self.assertIn("data.minimum_dev_records: expected at least 10000", report.errors)

    def test_rejects_model_contract_drift(self) -> None:
        config = copy.deepcopy(VALID_CONFIG)
        config["model"]["target_parameter_range"] = "3B-7B"
        config["model"]["context_tokens"] = 4096
        config["evaluation"]["metrics"].remove("scope_quality")

        report = validate_proof_model_config(config)

        self.assertFalse(report.valid)
        self.assertIn(
            "model.target_parameter_range: expected to stay within the GCTX-1 proof band",
            report.errors,
        )
        self.assertIn("model.context_tokens: expected at least 8192", report.errors)
        self.assertIn("evaluation.metrics: missing scope_quality", report.errors)

    def test_cli_returns_success_for_valid_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proof.json"
            path.write_text(json.dumps(VALID_CONFIG) + "\n", encoding="utf-8")

            self.assertEqual(main(["validate", str(path)]), 0)

    def test_cli_returns_failure_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proof.json"
            path.write_text("{", encoding="utf-8")

            self.assertEqual(main(["validate", str(path)]), 1)

    def test_load_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proof.json"
            path.write_text("[]\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_proof_model_config(path)


if __name__ == "__main__":
    unittest.main()

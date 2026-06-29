from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gitctx.proof_lm_train import (
    _load_torch,
    proof_lm_final_checkpoint_path,
    proof_lm_latest_checkpoint_path,
    proof_lm_train_report_path,
    run_proof_lm_training,
    validate_proof_lm_training,
)
from gitctx.proof_sequences import (
    build_proof_sequence_metadata,
    proof_sequence_metadata_path,
    proof_sequence_report_path,
)
from gitctx.proof_tokenizer import SPECIAL_TOKENS, record_tokens, tokenizer_artifact_path
from gitctx.proof_train import dry_run_proof_training, proof_train_sequence_plan_path
from gitctx.proof_train_job import proof_trainer_job_path
from gitctx.train_artifacts import training_examples_path


class ProofLmTrainTests(unittest.TestCase):
    def test_blocks_cleanly_when_torch_backend_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _prepare_trainer_job(root, run_id="test-proof-run")

            with patch("gitctx.proof_lm_train._load_torch", return_value=None):
                report = run_proof_lm_training(root, run_id="test-proof-run")

            self.assertEqual(report["status"], "blocked")
            self.assertIn("torch is not importable", report["blockers"])

    @unittest.skipIf(_load_torch() is None, "torch is not installed")
    def test_tiny_cpu_training_writes_report_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _prepare_trainer_job(root, run_id="test-proof-run")

            report = run_proof_lm_training(
                root,
                run_id="test-proof-run",
                device="cpu",
                max_records=2,
                max_steps=2,
                write=True,
            )
            validation = validate_proof_lm_training(root, run_id="test-proof-run")

            self.assertEqual(report["status"], "trained")
            self.assertTrue(validation["valid"])
            self.assertEqual(report["training"]["completed_records"], 2)
            self.assertGreater(report["training"]["loss_tokens"], 0)
            self.assertTrue((root / proof_lm_train_report_path("test-proof-run")).exists())
            self.assertTrue((root / proof_lm_latest_checkpoint_path("test-proof-run")).exists())
            self.assertTrue((root / proof_lm_final_checkpoint_path("test-proof-run")).exists())

    def test_paths_are_stable(self) -> None:
        self.assertEqual(
            proof_lm_train_report_path("gctx1-proof-model.v0.dry-run"),
            Path("artifacts/train-runs/gctx1-proof-model.v0.dry-run.trainer.report.json"),
        )
        self.assertEqual(
            proof_lm_latest_checkpoint_path("gctx1-proof-model.v0.dry-run"),
            Path("artifacts/train-runs/gctx1-proof-model.v0.dry-run/checkpoints/latest.json"),
        )
        self.assertEqual(
            proof_lm_final_checkpoint_path("gctx1-proof-model.v0.dry-run"),
            Path("artifacts/train-runs/gctx1-proof-model.v0.dry-run/checkpoints/final.json"),
        )


def _prepare_trainer_job(root: Path, *, run_id: str) -> None:
    handoff_path, tokenizer = _write_ready_handoff(root)
    dry_run_proof_training(root, handoff_path=handoff_path, run_id=run_id, write=True)
    build_proof_sequence_metadata(root, handoff_path=handoff_path, run_id=run_id, write=True)
    _write_trainer_job(root, run_id=run_id, tokenizer_vocab_size=tokenizer["vocab_size"])


def _write_ready_handoff(root: Path) -> tuple[Path, dict[str, object]]:
    training_path = root / training_examples_path("pilot")
    tokenizer_path = root / tokenizer_artifact_path("pilot")
    handoff_path = root / "artifacts/train-runs/gctx1-proof-model.v0.handoff.json"
    training_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        _record("one", "DEV", "feat(api): add request parser"),
        _record("two", "DEV", "fix(cli): handle missing path"),
        _record("three", "REPORT", "docs(api): explain parser behavior"),
    ]
    training_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    tokenizer = _tokenizer_for(records)
    tokenizer_path.write_text(json.dumps(tokenizer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    handoff = {
        "schema_version": "v0",
        "id": "gctx1-proof-model.v0.handoff",
        "status": "ready_for_training",
        "blockers": [],
        "code": {"training_code_revision": "abc123"},
        "inputs": {
            "training_artifact": {
                "path": training_examples_path("pilot").as_posix(),
                "sha256": _sha256(training_path),
            },
            "tokenizer": {
                "path": tokenizer_artifact_path("pilot").as_posix(),
                "sha256": _sha256(tokenizer_path),
            },
        },
        "training_contract": {
            "target_rung": "GCTX-1",
            "target_parameter_range": "0.01M-1M",
            "architecture": "decoder-only transformer",
            "context_tokens": 64,
            "train_split": "DEV",
            "eval_split": "REPORT",
            "reserved_split": "HELD_OUT",
            "optimizer": "adamw",
        },
    }
    handoff_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return handoff_path, tokenizer


def _write_trainer_job(root: Path, *, run_id: str, tokenizer_vocab_size: int) -> None:
    handoff_path = root / "artifacts/train-runs/gctx1-proof-model.v0.handoff.json"
    training_path = root / training_examples_path("pilot")
    tokenizer_path = root / tokenizer_artifact_path("pilot")
    sequence_plan_path = root / proof_train_sequence_plan_path(run_id)
    sequence_metadata_path = root / proof_sequence_metadata_path(run_id)
    sequence_report_path = root / proof_sequence_report_path(run_id)
    job_path = root / proof_trainer_job_path(run_id)
    job = {
        "schema_version": "v0",
        "trainer_job_id": "gctx1-proof-trainer-job-v0",
        "run_id": run_id,
        "status": "ready_for_trainer",
        "blockers": [],
        "seed": 17,
        "inputs": {
            "handoff": {
                "path": "artifacts/train-runs/gctx1-proof-model.v0.handoff.json",
                "sha256": _sha256(handoff_path),
            },
            "sequence_plan": {
                "path": str(proof_train_sequence_plan_path(run_id)),
                "sha256": _sha256(sequence_plan_path),
            },
            "sequence_metadata": {
                "path": str(proof_sequence_metadata_path(run_id)),
                "sha256": _sha256(sequence_metadata_path),
            },
            "sequence_materialization_report": {
                "path": str(proof_sequence_report_path(run_id)),
                "sha256": _sha256(sequence_report_path),
            },
            "tokenizer": {
                "path": tokenizer_artifact_path("pilot").as_posix(),
                "actual_sha256": _sha256(tokenizer_path),
                "expected_sha256": _sha256(tokenizer_path),
                "ok": True,
            },
            "training_artifact": {
                "path": training_examples_path("pilot").as_posix(),
                "actual_sha256": _sha256(training_path),
                "expected_sha256": _sha256(training_path),
                "ok": True,
            },
        },
        "model_contract": {
            "architecture": "decoder-only transformer",
            "target_parameter_range": "0.01M-1M",
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "context_tokens": 64,
            "layers": 1,
            "hidden_size": 16,
            "attention_heads": 2,
            "kv_heads": 1,
            "intermediate_size": 32,
            "activation": "silu",
            "normalization": "rmsnorm",
            "position_encoding": "learned_absolute_for_fixture",
            "tie_input_output_embeddings": True,
        },
        "data_contract": {
            "train_split": "DEV",
            "eval_split": "REPORT",
            "reserved_split": "HELD_OUT",
            "train_report_records": 0,
            "report_excluded_records": 0,
        },
    }
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tokenizer_for(records: list[dict[str, object]]) -> dict[str, object]:
    tokens = list(SPECIAL_TOKENS)
    seen = set(tokens)
    for record in records:
        for token in record_tokens(record):
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return {
        "tokenizer_kind": "dependency_free_regex_diff_frequency_tokenizer",
        "tokenizer_version": "regex-diff-v0",
        "special_tokens": {token: index for index, token in enumerate(SPECIAL_TOKENS)},
        "vocab_size": len(tokens),
        "vocab": [
            {"id": index, "token": token, "count": 1}
            for index, token in enumerate(tokens)
        ],
    }


def _record(record_id: str, split: str, target: str) -> dict[str, object]:
    return {
        "id": record_id,
        "source_diff_id": f"source-{record_id}",
        "source_repo_url": "https://github.com/example/repo",
        "source_commit": f"{record_id}cccccccccccc",
        "parent_commit": f"{record_id}pppppppppppp",
        "diff_sha256": hashlib.sha256(record_id.encode()).hexdigest(),
        "changed_paths": ["src/api.py"],
        "data_split": split,
        "messages": [
            {"role": "system", "content": "Write one Conventional Commit message."},
            {
                "role": "user",
                "content": "diff --git a/src/api.py b/src/api.py\n+def parse(value):\n+    return value",
            },
            {"role": "assistant", "content": target},
        ],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from gitctx.proof_sequences import proof_sequence_metadata_path, proof_sequence_report_path
from gitctx.proof_train import proof_train_sequence_plan_path
from gitctx.proof_train_job import (
    create_proof_trainer_job,
    proof_trainer_job_path,
    validate_proof_trainer_job,
)


class ProofTrainJobTests(unittest.TestCase):
    def test_creates_and_validates_ready_trainer_job_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = _write_ready_job_inputs(root)

            job = create_proof_trainer_job(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
                code_revision="abc123",
                write=True,
            )
            validation = validate_proof_trainer_job(root, run_id="test-proof-run")

            self.assertEqual(job["status"], "ready_for_trainer")
            self.assertEqual(job["blockers"], [])
            self.assertTrue(validation["valid"])
            self.assertEqual(job["data_contract"]["train_split"], "DEV")
            self.assertEqual(job["data_contract"]["eval_split"], "REPORT")
            self.assertEqual(job["data_contract"]["train_report_records"], 0)
            self.assertEqual(job["data_contract"]["report_excluded_records"], 0)
            self.assertGreaterEqual(job["model_contract"]["estimated_parameters"], 60_000_000)
            self.assertLessEqual(job["model_contract"]["estimated_parameters"], 100_000_000)
            self.assertEqual(job["model_contract"]["tokenizer_vocab_size"], 32_000)
            self.assertTrue((root / proof_trainer_job_path("test-proof-run")).exists())

    def test_blocks_when_report_records_would_be_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = _write_ready_job_inputs(root, report_excluded_records=1)

            job = create_proof_trainer_job(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
                code_revision="abc123",
            )

            self.assertEqual(job["status"], "blocked")
            self.assertIn("REPORT contains excluded records", job["blockers"])

    def test_blocks_if_handoff_train_split_drifts_from_dev(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = _write_ready_job_inputs(root, train_split="REPORT")

            job = create_proof_trainer_job(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
                code_revision="abc123",
            )

            self.assertEqual(job["status"], "blocked")
            self.assertIn("training contract train_split must be DEV", job["blockers"])
            self.assertIn("data contract train_split must be DEV", job["blockers"])

    def test_blocks_if_sequence_report_was_built_from_different_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = _write_ready_job_inputs(root, sequence_handoff_sha="0" * 64)

            job = create_proof_trainer_job(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
                code_revision="abc123",
            )

            self.assertEqual(job["status"], "blocked")
            self.assertIn("sequence report handoff sha256 mismatch", job["blockers"])

    def test_blocks_if_sequence_report_input_checks_are_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = _write_ready_job_inputs(root)
            report_path = root / proof_sequence_report_path("test-proof-run")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["input_checks"] = "malformed"
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            job = create_proof_trainer_job(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
                code_revision="abc123",
            )

            self.assertEqual(job["status"], "blocked")
            self.assertIn("sequence report sequence-plan sha256 is missing", job["blockers"])
            self.assertIn("sequence report handoff sha256 is missing", job["blockers"])

    def test_blocks_if_training_artifact_hash_does_not_match_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = _write_ready_job_inputs(root)
            (root / "artifacts/train/sft.gctx1-strict.v0.jsonl").write_text(
                "{\"changed\": true}\n",
                encoding="utf-8",
            )

            job = create_proof_trainer_job(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
                code_revision="abc123",
            )

            self.assertEqual(job["status"], "blocked")
            self.assertIn("training_artifact sha256 mismatch", job["blockers"])

    def test_validation_rejects_stale_sequence_metadata_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = _write_ready_job_inputs(root)
            create_proof_trainer_job(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
                code_revision="abc123",
                write=True,
            )
            (root / proof_sequence_metadata_path("test-proof-run")).write_text(
                "{\"changed\": true}\n",
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                validate_proof_trainer_job(root, run_id="test-proof-run")

    def test_validation_rejects_input_without_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = _write_ready_job_inputs(root)
            create_proof_trainer_job(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
                code_revision="abc123",
                write=True,
            )
            job_path = root / proof_trainer_job_path("test-proof-run")
            job = json.loads(job_path.read_text(encoding="utf-8"))
            del job["inputs"]["handoff"]["sha256"]
            job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                validate_proof_trainer_job(root, run_id="test-proof-run")

    def test_trainer_job_path_is_stable(self) -> None:
        self.assertEqual(
            proof_trainer_job_path("gctx1-proof-model.v0.dry-run"),
            Path("artifacts/train-runs/gctx1-proof-model.v0.dry-run.trainer-job.json"),
        )


def _write_ready_job_inputs(
    root: Path,
    *,
    train_split: str = "DEV",
    report_excluded_records: int = 0,
    sequence_handoff_sha: str | None = None,
) -> Path:
    tokenizer_path = root / "artifacts/tokenizers/regex-diff-v0.gctx1-strict.v0.json"
    training_path = root / "artifacts/train/sft.gctx1-strict.v0.jsonl"
    handoff_path = root / "artifacts/train-runs/gctx1-proof-model.v0.handoff.json"
    sequence_report_path = root / proof_sequence_report_path("test-proof-run")
    metadata_path = root / proof_sequence_metadata_path("test-proof-run")
    sequence_plan_path = root / proof_train_sequence_plan_path("test-proof-run")
    for path in (tokenizer_path, training_path, handoff_path, sequence_report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    vocab = [
        {"id": index, "token": f"tok{index}", "count": 1}
        for index in range(32_000)
    ]
    tokenizer = {
        "tokenizer_kind": "dependency_free_regex_diff_frequency_tokenizer",
        "tokenizer_version": "regex-diff-v0",
        "special_tokens": {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<sep>": 4,
            "<system>": 5,
            "<user>": 6,
            "<assistant>": 7,
            "<nl>": 8,
        },
        "vocab_size": 32_000,
        "vocab": vocab,
    }
    tokenizer_path.write_text(json.dumps(tokenizer, sort_keys=True) + "\n", encoding="utf-8")
    training_path.write_text("{}\n", encoding="utf-8")
    sequence_plan_path.write_text("{}\n", encoding="utf-8")
    metadata_path.write_text(
        json.dumps({"record_id": "one", "data_split": "DEV"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    handoff = {
        "schema_version": "v0",
        "id": "gctx1-proof-model.v0.handoff",
        "status": "ready_for_training",
        "blockers": [],
        "code": {"training_code_revision": "abc123"},
        "inputs": {
            "tokenizer": {
                "path": "artifacts/tokenizers/regex-diff-v0.gctx1-strict.v0.json",
                "sha256": _sha256(tokenizer_path),
            },
            "training_artifact": {
                "path": "artifacts/train/sft.gctx1-strict.v0.jsonl",
                "sha256": _sha256(training_path),
            },
        },
        "readiness": {
            "gates": {
                "dev_training_records": {
                    "actual": 2,
                    "minimum": 2,
                    "status": "pass",
                }
            }
        },
        "training_contract": {
            "target_rung": "GCTX-1",
            "target_parameter_range": "60M-100M",
            "architecture": "decoder-only transformer",
            "context_tokens": 8192,
            "objective": "supervised fine-tuning on reviewed Conventional Commit examples",
            "train_split": train_split,
            "eval_split": "REPORT",
            "reserved_split": "HELD_OUT",
            "optimizer": "adamw",
            "precision": "bf16 or fp16 depending on target runtime",
            "checkpoint_policy": "save resumable checkpoints and final model artifacts",
        },
    }
    handoff_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sequence_report = {
        "schema_version": "v0",
        "run_id": "test-proof-run",
        "status": "materialized",
        "context_tokens": 8192,
        "input_checks": {
            "handoff": {
                "path": "artifacts/train-runs/gctx1-proof-model.v0.handoff.json",
                "sha256": sequence_handoff_sha or _sha256(handoff_path),
            },
            "sequence_plan": {
                "path": str(proof_train_sequence_plan_path("test-proof-run")),
                "sha256": _sha256(sequence_plan_path),
            },
        },
        "materialization": {
            "kept_records": 3,
            "excluded_records": report_excluded_records,
            "full_records": 3,
            "truncated_records": 0,
            "input_tokens": 4096,
            "loss_tokens": 64,
            "unknown_tokens": 0,
            "max_input_length": 128,
            "by_split": {
                "DEV": {
                    "kept_records": 2,
                    "excluded_records": 0,
                    "full_records": 2,
                    "truncated_records": 0,
                    "input_tokens": 2048,
                    "loss_tokens": 32,
                    "unknown_tokens": 0,
                    "max_input_length": 128,
                },
                "REPORT": {
                    "kept_records": 1,
                    "excluded_records": report_excluded_records,
                    "full_records": 1,
                    "truncated_records": 0,
                    "input_tokens": 1024,
                    "loss_tokens": 16,
                    "unknown_tokens": 0,
                    "max_input_length": 128,
                },
            },
        },
        "outputs": {
            "metadata_path": str(proof_sequence_metadata_path("test-proof-run")),
            "metadata_sha256": _sha256(metadata_path),
            "report_path": str(proof_sequence_report_path("test-proof-run")),
        },
    }
    sequence_report_path.write_text(
        json.dumps(sequence_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return handoff_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()

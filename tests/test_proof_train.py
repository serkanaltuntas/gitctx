from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from gitctx.proof_train import (
    dry_run_proof_training,
    proof_train_checkpoint_path,
    proof_train_report_path,
    proof_train_sequence_plan_path,
    validate_proof_training_run,
)
from gitctx.proof_tokenizer import SPECIAL_TOKENS, record_tokens, tokenizer_artifact_path
from gitctx.train_artifacts import training_examples_path


class ProofTrainTests(unittest.TestCase):
    def test_dry_run_writes_and_validates_no_weight_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = _write_ready_handoff(root)

            report = dry_run_proof_training(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
                max_records_per_split=1,
                write=True,
            )
            validation = validate_proof_training_run(root, run_id="test-proof-run")
            checkpoint = json.loads(
                (root / proof_train_checkpoint_path("test-proof-run")).read_text(
                    encoding="utf-8"
                )
            )
            sequence_plan = _load_jsonl(root / proof_train_sequence_plan_path("test-proof-run"))

            self.assertEqual(report["status"], "dry_run_passed")
            self.assertTrue(validation["valid"])
            self.assertFalse(checkpoint["contains_model_weights"])
            self.assertEqual(len(sequence_plan), 2)
            self.assertEqual(report["tokenization"]["processed_records"], 2)
            self.assertEqual(report["tokenization"]["by_split"]["DEV"]["records"], 1)
            self.assertEqual(report["tokenization"]["by_split"]["REPORT"]["records"], 1)
            self.assertEqual(
                report["tokenization"]["sequence_policy"]["status"],
                "ready",
            )
            self.assertEqual(
                report["outputs"]["report_path"],
                "artifacts/train-runs/test-proof-run.report.json",
            )

    def test_dry_run_records_long_record_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            huge_record = _record(
                "huge",
                "DEV",
                "chore(assets): refresh generated fixture",
                user_content="diff --git a/generated.txt b/generated.txt\n" + "+value\n" * 200,
            )
            handoff_path = _write_ready_handoff(root, extra_records=[huge_record])

            report = dry_run_proof_training(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
                max_raw_record_tokens=80,
                write=True,
            )
            sequence_plan = _load_jsonl(root / proof_train_sequence_plan_path("test-proof-run"))
            huge_plan = next(record for record in sequence_plan if record["record_id"] == "huge")

            self.assertEqual(report["status"], "dry_run_passed")
            self.assertEqual(report["tokenization"]["by_split"]["DEV"]["excluded_records"], 1)
            self.assertEqual(
                report["tokenization"]["sequence_policy"]["excluded_records"],
                1,
            )
            self.assertEqual(huge_plan["decision"], "exclude_oversize")
            self.assertEqual(huge_plan["trainer_sequence_tokens"], 0)

    def test_dry_run_blocks_when_kept_dev_records_drop_below_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            huge_record = _record(
                "huge",
                "DEV",
                "chore(assets): refresh generated fixture",
                user_content="diff --git a/generated.txt b/generated.txt\n" + "+value\n" * 200,
            )
            handoff_path = _write_ready_handoff(
                root,
                extra_records=[huge_record],
                minimum_dev_records=3,
            )

            report = dry_run_proof_training(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
                max_raw_record_tokens=80,
            )

            self.assertEqual(report["status"], "blocked")
            self.assertIn(
                "sequence policy: kept DEV records fall below minimum (2 < 3)",
                report["blockers"],
            )

    def test_dry_run_blocks_on_input_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = _write_ready_handoff(root, tokenizer_sha="0" * 64)

            report = dry_run_proof_training(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
            )

            self.assertEqual(report["status"], "blocked")
            self.assertIn("input tokenizer: sha256 mismatch", report["blockers"])
            self.assertFalse((root / proof_train_report_path("test-proof-run")).exists())

    def test_paths_are_stable(self) -> None:
        self.assertEqual(
            proof_train_report_path("gctx1-proof-model.v0.dry-run"),
            Path("artifacts/train-runs/gctx1-proof-model.v0.dry-run.report.json"),
        )
        self.assertEqual(
            proof_train_checkpoint_path("gctx1-proof-model.v0.dry-run"),
            Path("artifacts/train-runs/gctx1-proof-model.v0.dry-run.checkpoint.json"),
        )
        self.assertEqual(
            proof_train_sequence_plan_path("gctx1-proof-model.v0.dry-run"),
            Path("artifacts/train-runs/gctx1-proof-model.v0.dry-run.sequence-plan.jsonl"),
        )


def _write_ready_handoff(
    root: Path,
    *,
    tokenizer_sha: str | None = None,
    extra_records: list[dict[str, object]] | None = None,
    minimum_dev_records: int | None = None,
) -> Path:
    training_path = root / training_examples_path("pilot")
    tokenizer_path = root / tokenizer_artifact_path("pilot")
    handoff_path = root / "artifacts/train-runs/gctx1-proof-model.v0.handoff.json"
    training_path.parent.mkdir(parents=True)
    tokenizer_path.parent.mkdir(parents=True)
    handoff_path.parent.mkdir(parents=True)

    records = [
        _record("one", "DEV", "feat(api): add request parser"),
        _record("two", "DEV", "fix(cli): handle missing path"),
        _record("three", "REPORT", "feat(api): add response parser"),
    ]
    if extra_records is not None:
        records.extend(extra_records)
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
                "sha256": tokenizer_sha if tokenizer_sha is not None else _sha256(tokenizer_path),
            },
        },
        "training_contract": {
            "target_rung": "GCTX-1",
            "architecture": "decoder-only transformer",
            "context_tokens": 16,
        },
    }
    if minimum_dev_records is not None:
        handoff["readiness"] = {
            "gates": {
                "dev_training_records": {
                    "actual": 2 + len(extra_records or []),
                    "minimum": minimum_dev_records,
                    "status": "pass",
                }
            }
        }
    handoff_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return handoff_path


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
        "vocab_size": len(tokens),
        "vocab": [
            {"id": index, "token": token, "count": 1}
            for index, token in enumerate(tokens)
        ],
    }


def _record(
    record_id: str,
    split: str,
    target: str,
    *,
    user_content: str | None = None,
) -> dict[str, object]:
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
                "content": user_content
                or "diff --git a/src/api.py b/src/api.py\n+def parse(value):\n+    return value",
            },
            {"role": "assistant", "content": target},
        ],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()

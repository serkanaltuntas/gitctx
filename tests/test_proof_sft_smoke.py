from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from gitctx.proof_sequences import build_proof_sequence_metadata
from gitctx.proof_sft_smoke import (
    proof_sft_smoke_checkpoint_path,
    proof_sft_smoke_report_path,
    run_proof_sft_smoke,
    validate_proof_sft_smoke,
)
from gitctx.proof_tokenizer import SPECIAL_TOKENS, record_tokens, tokenizer_artifact_path
from gitctx.proof_train import dry_run_proof_training
from gitctx.train_artifacts import training_examples_path


class ProofSftSmokeTests(unittest.TestCase):
    def test_trains_dev_sequences_and_writes_resumable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handoff_path = _prepare_sequence_artifacts(root, run_id="test-proof-run")

            report = run_proof_sft_smoke(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
                max_records=2,
                write=True,
            )
            validation = validate_proof_sft_smoke(root, run_id="test-proof-run")
            checkpoint = _load_json(root / proof_sft_smoke_checkpoint_path("test-proof-run"))

            self.assertEqual(report["status"], "trained")
            self.assertTrue(validation["valid"])
            self.assertEqual(report["training"]["train_split"], "DEV")
            self.assertEqual(report["training"]["selected_train_records"], 2)
            self.assertEqual(report["training"]["completed_train_records"], 2)
            self.assertGreater(report["training"]["optimizer_steps"], 0)
            self.assertTrue(checkpoint["contains_model_weights"])
            self.assertTrue((root / proof_sft_smoke_report_path("test-proof-run")).exists())

    def test_resume_matches_uninterrupted_smoke_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_handoff = _prepare_sequence_artifacts(root, run_id="full-run")
            resume_handoff = _prepare_sequence_artifacts(root, run_id="resume-run")

            run_proof_sft_smoke(
                root,
                handoff_path=full_handoff,
                run_id="full-run",
                max_records=3,
                write=True,
            )
            partial = run_proof_sft_smoke(
                root,
                handoff_path=resume_handoff,
                run_id="resume-run",
                max_records=3,
                stop_after_records=1,
                write=True,
            )
            resumed = run_proof_sft_smoke(
                root,
                handoff_path=resume_handoff,
                run_id="resume-run",
                max_records=3,
                resume=True,
                write=True,
            )

            full_checkpoint = _load_json(root / proof_sft_smoke_checkpoint_path("full-run"))
            resume_checkpoint = _load_json(root / proof_sft_smoke_checkpoint_path("resume-run"))

            self.assertEqual(partial["status"], "partial")
            self.assertEqual(resumed["status"], "trained")
            self.assertEqual(
                resume_checkpoint["model"]["weight_sha256"],
                full_checkpoint["model"]["weight_sha256"],
            )
            self.assertEqual(
                resume_checkpoint["optimizer"]["optimizer_steps"],
                full_checkpoint["optimizer"]["optimizer_steps"],
            )
            self.assertEqual(resume_checkpoint["record_cursor"], full_checkpoint["record_cursor"])


def _prepare_sequence_artifacts(root: Path, *, run_id: str) -> Path:
    handoff_path = _write_ready_handoff(root)
    dry_run_proof_training(
        root,
        handoff_path=handoff_path,
        run_id=run_id,
        write=True,
    )
    build_proof_sequence_metadata(
        root,
        handoff_path=handoff_path,
        run_id=run_id,
        write=True,
    )
    return handoff_path


def _write_ready_handoff(root: Path) -> Path:
    training_path = root / training_examples_path("pilot")
    tokenizer_path = root / tokenizer_artifact_path("pilot")
    handoff_path = root / "artifacts/train-runs/gctx1-proof-model.v0.handoff.json"
    training_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        _record("one", "DEV", "feat(api): add request parser"),
        _record("two", "DEV", "fix(cli): handle missing path"),
        _record("three", "DEV", "test(api): cover parser failures"),
        _record("four", "REPORT", "docs(api): explain parser behavior"),
    ]
    training_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    tokenizer_path.write_text(
        json.dumps(_tokenizer_for(records), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
            "architecture": "decoder-only transformer",
            "context_tokens": 128,
        },
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


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

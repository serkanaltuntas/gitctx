from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from gitctx.proof_sequences import (
    build_proof_sequence_metadata,
    materialize_training_sequence,
    proof_sequence_metadata_path,
    proof_sequence_report_path,
    validate_proof_sequence_metadata,
)
from gitctx.proof_tokenizer import SPECIAL_TOKENS, record_tokens
from gitctx.proof_train import proof_train_report_path, proof_train_sequence_plan_path
from gitctx.train_artifacts import training_examples_path


class ProofSequencesTests(unittest.TestCase):
    def test_materializes_full_record_without_cropping(self) -> None:
        record = _record("one", "DEV", "feat(api): add parser")
        tokenizer = _tokenizer_for([record])
        plan = _plan_for(record, decision="use_full")

        sequence = materialize_training_sequence(record, plan, tokenizer)

        self.assertEqual(sequence["tokens"], record_tokens(record))
        self.assertEqual(sequence["input_length"], len(record_tokens(record)))
        self.assertGreater(sequence["loss_tokens"], 0)
        self.assertEqual(sequence["crop"]["policy"], "full")

    def test_materializes_truncated_record_with_prefix_suffix_crop(self) -> None:
        record = _record(
            "two",
            "REPORT",
            "fix(api): handle blank parser input",
            user_content="diff --git a/src/api.py b/src/api.py\n" + "+value = parse(input)\n" * 80,
        )
        tokenizer = _tokenizer_for([record])
        plan = _plan_for(record, decision="use_truncated", context_tokens=48)

        sequence = materialize_training_sequence(record, plan, tokenizer, context_tokens=48)

        self.assertEqual(sequence["input_length"], 48)
        self.assertEqual(sequence["crop"]["policy"], "deterministic_prefix_suffix")
        self.assertGreater(sequence["crop"]["user_prefix_tokens"], 0)
        self.assertGreater(sequence["crop"]["user_suffix_tokens"], 0)
        self.assertIn("fix", sequence["tokens"])
        self.assertIn("blank", sequence["tokens"])

    def test_builds_and_validates_sequence_metadata_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = _record("one", "DEV", "feat(api): add parser")
            truncated = _record(
                "two",
                "REPORT",
                "fix(api): handle blank parser input",
                user_content="diff --git a/src/api.py b/src/api.py\n" + "+value = parse(input)\n" * 80,
            )
            excluded = _record(
                "three",
                "DEV",
                "chore(fixtures): refresh generated output",
                user_content="diff --git a/generated.txt b/generated.txt\n" + "+value\n" * 200,
            )
            handoff_path = _write_artifacts(root, [full, truncated, excluded])

            report = build_proof_sequence_metadata(
                root,
                handoff_path=handoff_path,
                run_id="test-proof-run",
                write=True,
            )
            validation = validate_proof_sequence_metadata(root, run_id="test-proof-run")
            metadata = _load_jsonl(root / proof_sequence_metadata_path("test-proof-run"))

            self.assertEqual(report["status"], "materialized")
            self.assertTrue(validation["valid"])
            self.assertEqual(report["materialization"]["kept_records"], 2)
            self.assertEqual(report["materialization"]["excluded_records"], 1)
            self.assertEqual(len(metadata), 2)
            self.assertTrue((root / proof_sequence_report_path("test-proof-run")).exists())


def _write_artifacts(root: Path, records: list[dict[str, object]]) -> Path:
    training_path = root / training_examples_path("pilot")
    tokenizer_path = root / "artifacts/tokenizers/regex-diff-v0.pilot.v0.json"
    handoff_path = root / "artifacts/train-runs/gctx1-proof-model.v0.handoff.json"
    report_path = root / proof_train_report_path("test-proof-run")
    sequence_plan_path = root / proof_train_sequence_plan_path("test-proof-run")
    training_path.parent.mkdir(parents=True)
    tokenizer_path.parent.mkdir(parents=True)
    handoff_path.parent.mkdir(parents=True)
    training_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    tokenizer_path.write_text(
        json.dumps(_tokenizer_for(records), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sequence_plan = [
        _plan_for(records[0], decision="use_full"),
        _plan_for(records[1], decision="use_truncated"),
        _plan_for(records[2], decision="exclude_oversize", trainer_sequence_tokens=0),
    ]
    sequence_plan_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in sequence_plan),
        encoding="utf-8",
    )
    handoff = {
        "schema_version": "v0",
        "id": "gctx1-proof-model.v0.handoff",
        "status": "ready_for_training",
        "inputs": {
            "training_artifact": {
                "path": training_examples_path("pilot").as_posix(),
                "sha256": _sha256(training_path),
            },
            "tokenizer": {
                "path": "artifacts/tokenizers/regex-diff-v0.pilot.v0.json",
                "sha256": _sha256(tokenizer_path),
            },
        },
    }
    handoff_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dry_run_report = {
        "schema_version": "v0",
        "run_id": "test-proof-run",
        "status": "dry_run_passed",
        "handoff": {"sha256": _sha256(handoff_path)},
        "outputs": {"sequence_plan_sha256": _sha256(sequence_plan_path)},
        "tokenization": {"context_tokens": 128, "processed_records": len(records)},
    }
    report_path.write_text(json.dumps(dry_run_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return handoff_path


def _plan_for(
    record: dict[str, object],
    *,
    decision: str,
    context_tokens: int = 128,
    trainer_sequence_tokens: int | None = None,
) -> dict[str, object]:
    raw_tokens = len(record_tokens(record))
    if trainer_sequence_tokens is None:
        trainer_sequence_tokens = raw_tokens if decision == "use_full" else context_tokens
    return {
        "schema_version": "v0",
        "policy_id": "gctx1-proof-sequence-policy-v0",
        "record_id": record["id"],
        "source_diff_id": record["source_diff_id"],
        "data_split": record["data_split"],
        "source_repo_url": record["source_repo_url"],
        "source_commit": record["source_commit"],
        "parent_commit": record["parent_commit"],
        "diff_sha256": record["diff_sha256"],
        "raw_token_count": raw_tokens,
        "raw_sequences_at_context": math.ceil(raw_tokens / context_tokens),
        "decision": decision,
        "reason": "test fixture",
        "trainer_context_tokens": context_tokens,
        "trainer_sequence_tokens": trainer_sequence_tokens,
        "unknown_token_count": 0,
    }


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

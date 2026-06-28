import json
import tempfile
import unittest
from pathlib import Path

from gitctx.proof_tokenizer import (
    build_proof_tokenizer,
    tokenize_text,
    tokenizer_artifact_path,
    validate_proof_tokenizer,
)
from gitctx.train_artifacts import training_examples_path


class ProofTokenizerTests(unittest.TestCase):
    def test_tokenizes_code_and_newlines(self) -> None:
        self.assertEqual(
            tokenize_text("feat(api): add foo_bar()\n+ return value >= 1"),
            [
                "feat",
                "(",
                "api",
                ")",
                ":",
                "add",
                "foo_bar",
                "(",
                ")",
                "<nl>",
                "+",
                "return",
                "value",
                ">=",
                "1",
            ],
        )

    def test_builds_and_validates_tokenizer_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_training_artifact(root)

            report = build_proof_tokenizer(
                root,
                artifact_name="pilot",
                vocab_size=64,
                min_frequency=1,
            )
            validation = validate_proof_tokenizer(
                root,
                artifact_name="pilot",
                min_eval_coverage=0.1,
            )

            self.assertEqual(report["coverage"]["DEV"]["records"], 2)
            self.assertEqual(report["coverage"]["REPORT"]["records"], 1)
            self.assertTrue(validation["valid"])
            artifact = json.loads(
                (root / tokenizer_artifact_path("pilot")).read_text(encoding="utf-8")
            )
            self.assertEqual(artifact["special_tokens"]["<pad>"], 0)
            self.assertEqual(artifact["vocab"][0]["token"], "<pad>")

    def test_rejects_too_small_vocab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_training_artifact(root)

            with self.assertRaises(ValueError):
                build_proof_tokenizer(root, artifact_name="pilot", vocab_size=4)


def _write_training_artifact(root: Path) -> None:
    output_path = root / training_examples_path("pilot")
    output_path.parent.mkdir(parents=True)
    records = [
        _record("one", "DEV", "feat(api): add request parser"),
        _record("two", "DEV", "fix(cli): handle missing path"),
        _record("three", "REPORT", "feat(api): add response parser"),
    ]
    output_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(record_id: str, split: str, target: str) -> dict[str, object]:
    return {
        "id": record_id,
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


if __name__ == "__main__":
    unittest.main()

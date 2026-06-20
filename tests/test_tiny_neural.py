from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gitctx.tiny_neural import evaluate_tiny_neural_model, train_tiny_neural_model


class TinyNeuralTests(unittest.TestCase):
    def test_trains_and_evaluates_tiny_neural_smoke_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_training_artifact(root)

            model = train_tiny_neural_model(root, artifact_name="next", epochs=8)
            report = evaluate_tiny_neural_model(root, artifact_name="next")

            self.assertEqual(model["model_kind"], "dependency_free_single_layer_softmax_classifier")
            self.assertEqual(model["training_records"], 4)
            self.assertLess(model["loss_history"][-1], model["loss_history"][0])
            self.assertEqual(report["eval_records"], 2)
            self.assertEqual(report["prediction"]["format_validity"]["true"], 2)
            self.assertTrue((root / "artifacts/models/tiny-softmax-v0.next.v0.json").exists())
            self.assertTrue(
                (root / "artifacts/eval/tiny-softmax-v0.next.v0.report.predictions.jsonl").exists()
            )
            self.assertTrue(
                (root / "artifacts/eval/tiny-softmax-v0.next.v0.report.report.json").exists()
            )

    def _write_training_artifact(self, root: Path) -> None:
        path = root / "artifacts/train/sft.next.v0.jsonl"
        path.parent.mkdir(parents=True)
        records = [
            {
                "id": "sft-example-1",
                "source_diff_id": "example-1",
                "data_split": "DEV",
                "changed_paths": ["src/parser.py"],
                "diff_stat": {"files_changed": 1, "insertions": 8, "deletions": 2},
                "target_message": "fix(parser): handle empty input",
            },
            {
                "id": "sft-example-2",
                "source_diff_id": "example-2",
                "data_split": "DEV",
                "changed_paths": ["src/parser.py"],
                "diff_stat": {"files_changed": 1, "insertions": 4, "deletions": 4},
                "target_message": "fix(parser): reject blank input",
            },
            {
                "id": "sft-example-3",
                "source_diff_id": "example-3",
                "data_split": "DEV",
                "changed_paths": ["tests/test_parser.py"],
                "diff_stat": {"files_changed": 1, "insertions": 12, "deletions": 0},
                "target_message": "test(parser): cover empty input",
            },
            {
                "id": "sft-example-4",
                "source_diff_id": "example-4",
                "data_split": "DEV",
                "changed_paths": ["tests/test_parser.py"],
                "diff_stat": {"files_changed": 1, "insertions": 6, "deletions": 1},
                "target_message": "test(parser): cover blank input",
            },
            {
                "id": "sft-example-5",
                "source_diff_id": "example-5",
                "data_split": "REPORT",
                "changed_paths": ["src/parser.py"],
                "diff_stat": {"files_changed": 1, "insertions": 5, "deletions": 1},
                "target_message": "fix(parser): handle whitespace input",
            },
            {
                "id": "sft-example-6",
                "source_diff_id": "example-6",
                "data_split": "REPORT",
                "changed_paths": ["tests/test_parser.py"],
                "diff_stat": {"files_changed": 1, "insertions": 7, "deletions": 0},
                "target_message": "test(parser): cover whitespace input",
            },
        ]
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()

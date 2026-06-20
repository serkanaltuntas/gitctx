from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gitctx.prototype_model import evaluate_prototype_model, train_prototype_model


class PrototypeModelTests(unittest.TestCase):
    def test_trains_and_evaluates_dependency_free_prototype(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_training_artifact(root)

            model = train_prototype_model(root, artifact_name="next")
            report = evaluate_prototype_model(root, artifact_name="next")

            self.assertEqual(model["training_records"], 2)
            self.assertEqual(report["eval_records"], 1)
            self.assertEqual(report["prediction"]["format_validity"]["true"], 1)
            self.assertTrue((root / "artifacts/models/path-type-v0.next.v0.json").exists())
            self.assertTrue(
                (root / "artifacts/eval/path-type-v0.next.v0.report.predictions.jsonl").exists()
            )
            self.assertTrue((root / "artifacts/eval/path-type-v0.next.v0.report.report.json").exists())

    def _write_training_artifact(self, root: Path) -> None:
        path = root / "artifacts/train/sft.next.v0.jsonl"
        path.parent.mkdir(parents=True)
        records = [
            {
                "id": "sft-example-1",
                "source_diff_id": "example-1",
                "data_split": "DEV",
                "changed_paths": ["src/parser.py"],
                "target_message": "fix(parser): handle empty input",
                "review_decision": "accept",
                "label_source": "teacher_generated_human_accepted",
            },
            {
                "id": "sft-example-2",
                "source_diff_id": "example-2",
                "data_split": "DEV",
                "changed_paths": ["tests/test_parser.py"],
                "target_message": "test(parser): cover empty input",
                "review_decision": "accept",
                "label_source": "teacher_generated_human_accepted",
            },
            {
                "id": "sft-example-3",
                "source_diff_id": "example-3",
                "data_split": "REPORT",
                "changed_paths": ["src/parser.py"],
                "target_message": "fix(parser): reject blank input",
                "review_decision": "accept",
                "label_source": "teacher_generated_human_accepted",
            },
        ]
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()

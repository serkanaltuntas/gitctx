import json
import tempfile
import unittest
from pathlib import Path

from gitctx.proof_readiness import (
    ProofThresholds,
    evaluate_proof_readiness,
    write_proof_readiness_report,
)


class ProofReadinessTests(unittest.TestCase):
    def test_reports_single_failed_dev_gate_for_smaller_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root, dev_records=4, report_records=2, repo_count=6)
            thresholds = ProofThresholds(
                min_dev_records=5,
                min_report_records=2,
                min_reserved_held_out_records=6,
                min_train_repos=4,
                min_report_repos=2,
                min_reserved_held_out_repos=2,
                min_ecosystems=2,
            )

            report = evaluate_proof_readiness(
                root,
                artifact_name="gctx1",
                thresholds=thresholds,
            )

            self.assertFalse(report["ready_for_gctx1_proof_run"])
            self.assertEqual(report["gates"]["dev_training_records"]["status"], "fail")
            self.assertEqual(report["gates"]["report_records"]["status"], "pass")
            self.assertEqual(report["gates"]["reserved_held_out_records"]["status"], "pass")
            self.assertEqual(report["gates"]["baseline_report_complete"]["status"], "pass")
            self.assertIn("smaller private proof model", report["recommended_next_action"])

    def test_passes_and_writes_report_when_all_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root, dev_records=6, report_records=2, repo_count=6)
            thresholds = ProofThresholds(
                min_dev_records=6,
                min_report_records=2,
                min_reserved_held_out_records=6,
                min_train_repos=5,
                min_report_repos=2,
                min_reserved_held_out_repos=2,
                min_ecosystems=2,
            )

            report = evaluate_proof_readiness(
                root,
                artifact_name="gctx1",
                thresholds=thresholds,
            )
            output_path = write_proof_readiness_report(root, report)

            self.assertTrue(report["ready_for_gctx1_proof_run"])
            self.assertEqual(report["gates"]["max_repo_train_fraction"]["status"], "pass")
            self.assertEqual(output_path, root / "artifacts/eval/sft.gctx1.v0.proof-readiness.report.json")
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written["training_artifact_path"], "artifacts/train/sft.gctx1.v0.jsonl")


def _write_artifact(
    root: Path,
    *,
    dev_records: int,
    report_records: int,
    repo_count: int,
) -> None:
    (root / "artifacts/train").mkdir(parents=True)
    (root / "artifacts/eval").mkdir(parents=True)
    (root / "manifests").mkdir()
    training_records = [
        _training_record(index, "DEV", repo_count=repo_count)
        for index in range(dev_records)
    ] + [
        _training_record(index + dev_records, "REPORT", repo_count=repo_count)
        for index in range(report_records)
    ]
    (root / "artifacts/train/sft.gctx1.v0.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in training_records),
        encoding="utf-8",
    )
    (root / "artifacts/eval/sft.gctx1.v0.baseline.report.json").write_text(
        json.dumps(
            {
                "artifact_name": "gctx1",
                "artifact_version": "v0",
                "training_records": len(training_records),
                "data_split_counts": {"DEV": dev_records, "REPORT": report_records},
                "target": {},
                "teacher": {},
                "historical": {},
                "by_data_split": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "manifests/source-manifest.gctx1.jsonl").write_text(
        "".join(
            json.dumps(_manifest_record(index, ecosystem="python" if index % 2 else "rust")) + "\n"
            for index in range(repo_count)
        ),
        encoding="utf-8",
    )
    (root / "manifests/split-plan.gctx1.json").write_text(
        json.dumps(
            {
                "id": "gctx1",
                "version": "v0",
                "created_at": "2026-06-23",
                "windows": [
                    *[_window(index, "DEV", target_records=10) for index in range(repo_count)],
                    *[_window(index, "REPORT", target_records=3) for index in range(2)],
                    *[_window(index, "HELD_OUT", target_records=3) for index in range(2)],
                ],
            }
        ),
        encoding="utf-8",
    )


def _training_record(index: int, split: str, *, repo_count: int) -> dict[str, object]:
    repo_index = index % repo_count
    return {
        "id": f"sft-example-{index}",
        "source_repo_url": f"https://github.com/example/repo-{repo_index}",
        "source_license": "MIT",
        "data_split": split,
        "label_source": "teacher_generated_human_accepted",
        "review_decision": "accept",
    }


def _manifest_record(index: int, *, ecosystem: str) -> dict[str, object]:
    return {
        "repo_url": f"https://github.com/example/repo-{index}",
        "default_branch": "main",
        "source_license": "MIT",
        "license_url": f"https://github.com/example/repo-{index}/blob/main/LICENSE",
        "license_review_date": "2026-06-21",
        "reviewer": "Serkan Altuntas",
        "review_status": "approved_for_training",
        "source_revision": f"{index:040x}"[-40:],
        "allowed_splits": ["DEV", "REPORT", "HELD_OUT"],
        "ecosystem": ecosystem,
        "exclude_globs": ["vendor/**"],
        "notes": "test source",
    }


def _window(index: int, split: str, *, target_records: int) -> dict[str, object]:
    ranges = {
        "HELD_OUT": ("2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
        "REPORT": ("2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
        "DEV": ("2024-01-01T00:00:00Z", "2100-01-01T00:00:00Z"),
    }
    start, end = ranges[split]
    return {
        "id": f"{split.lower()}-{index}",
        "repo_url": f"https://github.com/example/repo-{index}",
        "split": split,
        "start": start,
        "end": end,
        "reason": "test window",
        "target_records": target_records,
    }


if __name__ == "__main__":
    unittest.main()

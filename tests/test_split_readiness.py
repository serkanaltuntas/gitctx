import json
import tempfile
import unittest
from pathlib import Path

from gitctx.split_readiness import evaluate_split_readiness


class SplitReadinessTests(unittest.TestCase):
    def test_reports_gctx1_ready_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "source.jsonl"
            split_plan_path = root / "split.json"
            manifest_path.write_text(
                "".join(
                    json.dumps(_manifest_record(index, ecosystem="python" if index % 2 else "rust"))
                    + "\n"
                    for index in range(30)
                ),
                encoding="utf-8",
            )
            split_plan_path.write_text(
                json.dumps(
                    {
                        "id": "ready",
                        "version": "v0",
                        "created_at": "2026-06-21",
                        "windows": [
                            *[
                                _window(index, "DEV", target_records=400)
                                for index in range(25)
                            ],
                            *[
                                _window(index, "REPORT", start_month=3, end_month=4, target_records=200)
                                for index in range(25, 30)
                            ],
                            *[
                                _window(index, "HELD_OUT", start_month=5, end_month=6, target_records=200)
                                for index in range(25, 30)
                            ],
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_split_readiness(
                source_manifest_path=manifest_path,
                split_plan_path=split_plan_path,
            )

            self.assertTrue(report["ready_for_gctx1_planning"])
            self.assertEqual(report["gates"]["dev_records"]["status"], "pass")
            self.assertEqual(report["gates"]["held_out_repos"]["status"], "pass")
            self.assertEqual(report["gates"]["ecosystems"]["status"], "pass")

    def test_flags_missing_held_out_and_cross_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "source.jsonl"
            split_plan_path = root / "split.json"
            record = _manifest_record(1)
            record["allowed_splits"] = ["DEV"]
            manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            split_plan_path.write_text(
                json.dumps(
                    {
                        "id": "not-ready",
                        "version": "v0",
                        "created_at": "2026-06-21",
                        "windows": [_window(1, "REPORT")],
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_split_readiness(
                source_manifest_path=manifest_path,
                split_plan_path=split_plan_path,
            )

            self.assertFalse(report["ready_for_gctx1_planning"])
            self.assertEqual(report["gates"]["held_out_repos"]["status"], "fail")
            self.assertEqual(len(report["cross_errors"]), 1)

    def test_missing_target_records_make_record_gates_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "source.jsonl"
            split_plan_path = root / "split.json"
            manifest_path.write_text(json.dumps(_manifest_record(1)) + "\n", encoding="utf-8")
            window = _window(1, "DEV")
            window.pop("target_records")
            split_plan_path.write_text(
                json.dumps(
                    {
                        "id": "unknown-counts",
                        "version": "v0",
                        "created_at": "2026-06-21",
                        "windows": [window],
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_split_readiness(
                source_manifest_path=manifest_path,
                split_plan_path=split_plan_path,
            )

            self.assertEqual(report["gates"]["dev_records"]["status"], "unknown")
            self.assertEqual(report["gates"]["max_repo_train_fraction"]["status"], "unknown")


def _manifest_record(index: int, *, ecosystem: str = "python") -> dict[str, object]:
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


def _window(
    index: int,
    split: str,
    *,
    start_month: int = 1,
    end_month: int = 2,
    target_records: int = 100,
) -> dict[str, object]:
    return {
        "id": f"{split.lower()}-{index}",
        "repo_url": f"https://github.com/example/repo-{index}",
        "split": split,
        "start": f"2025-{start_month:02d}-{index % 20 + 1:02d}T00:00:00Z",
        "end": f"2025-{end_month:02d}-{index % 20 + 1:02d}T00:00:00Z",
        "reason": "test window",
        "target_records": target_records,
    }


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from gitctx.proof_handoff import create_proof_handoff, proof_handoff_path, write_proof_handoff


class ProofHandoffTests(unittest.TestCase):
    def test_creates_ready_handoff_for_valid_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "configs/gctx1-proof-model.v0.json"
            config_path.parent.mkdir()
            config_path.write_text(json.dumps(_valid_config()) + "\n", encoding="utf-8")
            _write_ready_artifacts(root)

            handoff = create_proof_handoff(
                root,
                config_path=config_path,
                source_manifest_path=root / "manifests/source-manifest.gctx1-strict.jsonl",
                split_plan_path=root / "manifests/split-plan.gctx1-strict.json",
                code_revision="abc123",
            )
            output_path = write_proof_handoff(root, handoff)

            self.assertEqual(handoff["status"], "ready_for_training")
            self.assertEqual(handoff["blockers"], [])
            self.assertEqual(handoff["code"]["training_code_revision"], "abc123")
            self.assertTrue(handoff["config"]["valid"])
            self.assertTrue(handoff["readiness"]["ready_for_gctx1_proof_run"])
            self.assertEqual(
                handoff["inputs"]["training_artifact"]["path"],
                "artifacts/train/sft.gctx1-strict.v0.jsonl",
            )
            self.assertEqual(len(handoff["inputs"]["training_artifact"]["sha256"]), 64)
            self.assertEqual(
                output_path,
                root / "artifacts/train-runs/gctx1-proof-model.v0.handoff.json",
            )

    def test_blocks_invalid_config_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config = _valid_config()
            config["evaluation"]["locked_report_required"] = False
            config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

            handoff = create_proof_handoff(
                root,
                config_path=config_path,
                code_revision="abc123",
            )

            self.assertEqual(handoff["status"], "blocked")
            self.assertIn(
                "config: evaluation.locked_report_required: expected True",
                handoff["blockers"],
            )

    def test_handoff_path_is_stable(self) -> None:
        self.assertEqual(
            proof_handoff_path("gctx1-proof-model", version="v0"),
            Path("artifacts/train-runs/gctx1-proof-model.v0.handoff.json"),
        )


def _valid_config() -> dict[str, object]:
    return {
        "id": "gctx1-proof-model",
        "version": "v0",
        "status": "planned",
        "purpose": "First GCTX-1 proof-model training run.",
        "data": {
            "artifact_name": "gctx1-strict",
            "artifact_version": "v0",
            "train_split": "DEV",
            "eval_split": "REPORT",
            "reserved_split": "HELD_OUT",
            "minimum_dev_records": 10_000,
            "minimum_report_records": 1_000,
            "minimum_reserved_held_out_records": 1_000,
        },
        "model": {
            "target_rung": "GCTX-1",
            "target_parameter_range": "60M-100M",
            "architecture": "decoder-only transformer",
            "context_tokens": 8192,
            "tokenizer": "code-and-diff-aware tokenizer",
            "objective": "supervised fine-tuning on reviewed Conventional Commit examples",
        },
        "training": {
            "precision": "bf16 or fp16 depending on target runtime",
            "optimizer": "adamw",
            "checkpoint_policy": "save resumable checkpoints and final model artifacts",
            "reproducibility": (
                "record seed, git revision, training code revision, dataset checksum, and runtime"
            ),
        },
        "evaluation": {
            "locked_report_required": True,
            "metrics": [
                "format_validity",
                "type_match",
                "scope_quality",
                "specificity",
                "brevity",
                "exact_message_match",
            ],
            "quality_claim_policy": (
                "No public model-quality claim until a model card and eval card are written "
                "from locked REPORT results."
            ),
        },
        "release": {
            "public_model_release": False,
            "requires_model_card": True,
            "requires_eval_card": True,
            "requires_dataset_redistribution_review": True,
        },
    }


def _write_ready_artifacts(root: Path) -> None:
    (root / "artifacts/train").mkdir(parents=True)
    (root / "artifacts/eval").mkdir(parents=True)
    (root / "manifests").mkdir()
    training_records = [
        _training_record(index, "DEV", repo_count=25)
        for index in range(10_000)
    ] + [
        _training_record(index + 10_000, "REPORT", repo_count=25)
        for index in range(1_000)
    ]
    (root / "artifacts/train/sft.gctx1-strict.v0.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in training_records),
        encoding="utf-8",
    )
    (root / "artifacts/eval/sft.gctx1-strict.v0.baseline.report.json").write_text(
        json.dumps(
            {
                "artifact_name": "gctx1-strict",
                "artifact_version": "v0",
                "training_records": len(training_records),
                "data_split_counts": {"DEV": 10_000, "REPORT": 1_000},
                "target": {},
                "teacher": {},
                "historical": {},
                "by_data_split": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "artifacts/eval/sft.gctx1-strict.v0.proof-readiness.report.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (root / "manifests/source-manifest.gctx1-strict.jsonl").write_text(
        "".join(
            json.dumps(_manifest_record(index, ecosystem="python" if index % 2 else "rust")) + "\n"
            for index in range(25)
        ),
        encoding="utf-8",
    )
    (root / "manifests/split-plan.gctx1-strict.json").write_text(
        json.dumps(
            {
                "id": "gctx1-strict",
                "version": "v0",
                "created_at": "2026-06-28",
                "windows": [
                    *[_window(index, "DEV", target_records=400) for index in range(25)],
                    *[_window(index, "REPORT", target_records=200) for index in range(5)],
                    *[_window(index, "HELD_OUT", target_records=200) for index in range(5)],
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
        "license_review_date": "2026-06-28",
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

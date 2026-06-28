"""Create proof-model training handoff manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from gitctx.proof_model_config import load_proof_model_config, validate_proof_model_config
from gitctx.proof_readiness import (
    ProofThresholds,
    evaluate_proof_readiness,
    proof_readiness_report_path,
)
from gitctx.train_artifacts import training_examples_path

RUN_DIR = Path("artifacts/train-runs")
SCHEMA_VERSION = "v0"


def proof_handoff_path(config_id: str, *, version: str) -> Path:
    """Return the handoff manifest path for a proof model config."""

    _validate_identifier(config_id, "config_id")
    _validate_identifier(version, "version")
    return RUN_DIR / f"{config_id}.{version}.handoff.json"


def create_proof_handoff(
    data_dir: str | Path,
    *,
    config_path: str | Path,
    source_manifest_path: str | Path | None = None,
    split_plan_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Create a training handoff manifest from config and proof-readiness gates."""

    data_dir = Path(data_dir)
    selected_config_path = Path(config_path)
    config = load_proof_model_config(selected_config_path)
    config_report = validate_proof_model_config(config)
    blockers = [f"config: {error}" for error in config_report.errors]
    readiness: dict[str, Any] | None = None

    artifact_name = config_report.summary.get("artifact_name")
    artifact_version = config_report.summary.get("artifact_version")
    if config_report.valid and isinstance(artifact_name, str) and isinstance(artifact_version, str):
        try:
            readiness = evaluate_proof_readiness(
                data_dir,
                artifact_name=artifact_name,
                version=artifact_version,
                source_manifest_path=source_manifest_path,
                split_plan_path=split_plan_path,
                thresholds=_thresholds_from_config(config),
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            blockers.append(f"readiness: {exc}")
    elif config_report.valid:
        blockers.append("config: data artifact_name and artifact_version are required")

    if readiness is not None and not readiness["ready_for_gctx1_proof_run"]:
        blockers.extend(
            f"readiness gate {name}: {gate['status']}"
            for name, gate in readiness["gates"].items()
            if gate["status"] != "pass"
        )
    if tokenizer_path is not None:
        selected_tokenizer_path = _data_path(data_dir, Path(tokenizer_path))
        if not selected_tokenizer_path.exists():
            blockers.append(f"tokenizer: missing {_display_path(selected_tokenizer_path, base=data_dir)}")

    status = "ready_for_training" if not blockers and readiness is not None else "blocked"
    revision = code_revision if code_revision is not None else _current_git_revision(Path.cwd())
    handoff = {
        "schema_version": SCHEMA_VERSION,
        "id": f"{config_report.summary.get('id')}.{config_report.summary.get('version')}.handoff",
        "status": status,
        "blockers": blockers,
        "config": {
            "path": _display_path(selected_config_path),
            "sha256": _sha256(selected_config_path),
            "valid": config_report.valid,
            "summary": config_report.summary,
            "errors": config_report.errors,
            "warnings": config_report.warnings,
        },
        "code": {
            "training_code_revision": revision,
        },
        "inputs": _input_files(
            data_dir,
            config_report.summary,
            readiness,
            source_manifest_path=source_manifest_path,
            split_plan_path=split_plan_path,
            tokenizer_path=tokenizer_path,
        ),
        "readiness": _readiness_summary(readiness),
        "training_contract": _training_contract(config),
        "required_outputs": _required_outputs(config),
    }
    return handoff


def write_proof_handoff(data_dir: str | Path, handoff: dict[str, Any]) -> Path:
    """Write a proof-model handoff manifest under the data artifact directory."""

    config_summary = handoff["config"]["summary"]
    config_id = config_summary.get("id")
    version = config_summary.get("version")
    if not isinstance(config_id, str):
        config_id = "proof-model-config"
    if not isinstance(version, str):
        version = "invalid"
    output_path = Path(data_dir) / proof_handoff_path(
        config_id,
        version=version,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _thresholds_from_config(config: dict[str, Any]) -> ProofThresholds:
    data = config["data"]
    return ProofThresholds(
        min_dev_records=data["minimum_dev_records"],
        min_report_records=data["minimum_report_records"],
        min_reserved_held_out_records=data["minimum_reserved_held_out_records"],
    )


def _input_files(
    data_dir: Path,
    config_summary: dict[str, Any],
    readiness: dict[str, Any] | None,
    *,
    source_manifest_path: str | Path | None,
    split_plan_path: str | Path | None,
    tokenizer_path: str | Path | None,
) -> dict[str, Any]:
    artifact_name = config_summary.get("artifact_name")
    artifact_version = config_summary.get("artifact_version")
    if not isinstance(artifact_name, str) or not isinstance(artifact_version, str):
        return {}

    files = {
        "training_artifact": _data_file_entry(
            data_dir,
            training_examples_path(artifact_name, version=artifact_version),
        ),
        "readiness_report": _data_file_entry(
            data_dir,
            proof_readiness_report_path(artifact_name, version=artifact_version),
        ),
    }
    if readiness is not None:
        for key in ("baseline_report_path", "source_manifest_path", "split_plan_path"):
            files[key.removesuffix("_path")] = _data_file_entry(data_dir, Path(readiness[key]))
    else:
        if source_manifest_path is not None:
            files["source_manifest"] = _possibly_external_file_entry(data_dir, Path(source_manifest_path))
        if split_plan_path is not None:
            files["split_plan"] = _possibly_external_file_entry(data_dir, Path(split_plan_path))
    if tokenizer_path is not None:
        files["tokenizer"] = _possibly_external_file_entry(data_dir, Path(tokenizer_path))
    return files


def _readiness_summary(readiness: dict[str, Any] | None) -> dict[str, Any]:
    if readiness is None:
        return {
            "ready_for_gctx1_proof_run": False,
            "gates": {},
            "recommended_next_action": "Resolve config or input artifact errors before handoff.",
        }
    return {
        "artifact_name": readiness["artifact_name"],
        "artifact_version": readiness["artifact_version"],
        "ready_for_gctx1_proof_run": readiness["ready_for_gctx1_proof_run"],
        "gates": readiness["gates"],
        "actual": readiness["actual"],
        "planned": readiness["planned"],
        "recommended_next_action": readiness["recommended_next_action"],
    }


def _training_contract(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data")
    model = config.get("model")
    training = config.get("training")
    if not isinstance(data, dict) or not isinstance(model, dict) or not isinstance(training, dict):
        return {}
    return {
        "target_rung": model.get("target_rung"),
        "target_parameter_range": model.get("target_parameter_range"),
        "architecture": model.get("architecture"),
        "context_tokens": model.get("context_tokens"),
        "objective": model.get("objective"),
        "train_split": data.get("train_split"),
        "eval_split": data.get("eval_split"),
        "reserved_split": data.get("reserved_split"),
        "optimizer": training.get("optimizer"),
        "precision": training.get("precision"),
        "checkpoint_policy": training.get("checkpoint_policy"),
    }


def _required_outputs(config: dict[str, Any]) -> list[str]:
    if not isinstance(config.get("release"), dict):
        return []
    return [
        "resumable training checkpoints",
        "final model artifact",
        "training run report with seed, code revision, dataset checksum, and runtime",
        "locked REPORT evaluation report",
        "model card before any public model-quality claim",
        "eval card before any public model-quality claim",
        "dataset redistribution review before publishing example-level data",
    ]


def _data_file_entry(data_dir: Path, relative_path: Path) -> dict[str, Any]:
    path = relative_path if relative_path.is_absolute() else data_dir / relative_path
    return _file_entry(path, display_base=data_dir)


def _possibly_external_file_entry(data_dir: Path, path: Path) -> dict[str, Any]:
    selected = _data_path(data_dir, path)
    return _file_entry(selected, display_base=data_dir)


def _data_path(data_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else data_dir / path


def _file_entry(path: Path, *, display_base: Path | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": _display_path(path, base=display_base),
        "exists": path.exists(),
    }
    if path.exists():
        entry["sha256"] = _sha256(path)
        entry["bytes"] = path.stat().st_size
    return entry


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _display_path(path: Path, *, base: Path | None = None) -> str:
    selected = path.resolve(strict=False)
    bases = [base, Path.cwd()]
    for candidate in bases:
        if candidate is None:
            continue
        try:
            return selected.relative_to(candidate.resolve()).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def _current_git_revision(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_identifier(value: str, name: str) -> None:
    if not value.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"{name} must be a stable alphanumeric identifier")


def _print_handoff(handoff: dict[str, Any]) -> None:
    print("id", handoff["id"])
    print("status", handoff["status"])
    print("training_code_revision", handoff["code"]["training_code_revision"])
    for blocker in handoff["blockers"]:
        print("blocker", blocker)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a GCTX proof-model handoff manifest.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--source-manifest")
    create.add_argument("--split-plan")
    create.add_argument("--tokenizer")
    create.add_argument("--code-revision")
    create.add_argument("--write", action="store_true")
    create.add_argument("--json", action="store_true")
    create.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "create":
        handoff = create_proof_handoff(
            args.data_dir,
            config_path=args.config,
            source_manifest_path=args.source_manifest,
            split_plan_path=args.split_plan,
            tokenizer_path=args.tokenizer,
            code_revision=args.code_revision,
        )
        if args.write:
            print("output_path", write_proof_handoff(args.data_dir, handoff))
        if args.json:
            print(json.dumps(handoff, indent=2, sort_keys=True))
        else:
            _print_handoff(handoff)
        if args.fail_on_blocked and handoff["status"] != "ready_for_training":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

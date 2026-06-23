"""Readiness checks for GCTX proof-model training artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from gitctx.artifact_eval import eval_report_path
from gitctx.provenance import load_jsonl
from gitctx.split_readiness import evaluate_split_readiness
from gitctx.train_artifacts import training_examples_path

EVAL_DIR = Path("artifacts/eval")


@dataclass(frozen=True)
class ProofThresholds:
    min_dev_records: int = 10_000
    min_report_records: int = 1_000
    min_reserved_held_out_records: int = 1_000
    min_train_repos: int = 25
    min_report_repos: int = 5
    min_reserved_held_out_repos: int = 5
    min_ecosystems: int = 2
    max_repo_train_fraction: float = 0.25


def proof_readiness_report_path(artifact_name: str, *, version: str = "v0") -> Path:
    """Return the proof-readiness report path for a named training artifact."""

    _validate_identifier(artifact_name, "artifact_name")
    _validate_identifier(version, "version")
    return EVAL_DIR / f"sft.{artifact_name}.{version}.proof-readiness.report.json"


def evaluate_proof_readiness(
    data_dir: str | Path,
    *,
    artifact_name: str,
    version: str = "v0",
    source_manifest_path: str | Path | None = None,
    split_plan_path: str | Path | None = None,
    thresholds: ProofThresholds | None = None,
) -> dict[str, Any]:
    """Return a readiness report for a GCTX proof-model training artifact."""

    _validate_identifier(artifact_name, "artifact_name")
    _validate_identifier(version, "version")
    data_dir = Path(data_dir)
    thresholds = thresholds or ProofThresholds()
    training_path = data_dir / training_examples_path(artifact_name, version=version)
    baseline_path = data_dir / eval_report_path(artifact_name, version=version)
    source_manifest = _defaulted_data_path(
        data_dir,
        source_manifest_path,
        Path("manifests") / f"source-manifest.{artifact_name}.jsonl",
    )
    split_plan = _defaulted_data_path(
        data_dir,
        split_plan_path,
        Path("manifests") / f"split-plan.{artifact_name}.json",
    )

    training_records = load_jsonl(training_path)
    source_manifest_records = load_jsonl(source_manifest) if source_manifest.exists() else []
    source_manifest_by_repo = {
        _normalize_repo_url(record["repo_url"]): record for record in source_manifest_records
    }
    split_readiness = (
        evaluate_split_readiness(source_manifest_path=source_manifest, split_plan_path=split_plan)
        if source_manifest.exists() and split_plan.exists()
        else None
    )

    actual = _actual_artifact_summary(training_records, source_manifest_by_repo)
    baseline = _baseline_summary(
        baseline_path,
        expected_training_records=len(training_records),
        expected_split_counts=actual["split_counts"],
    )
    planned = _planned_summary(split_readiness)
    gates = _gates(
        actual=actual,
        baseline=baseline,
        planned=planned,
        thresholds=thresholds,
        source_manifest_exists=source_manifest.exists(),
        split_plan_exists=split_plan.exists(),
    )
    ready = all(gate["status"] == "pass" for gate in gates.values())
    report = {
        "artifact_name": artifact_name,
        "artifact_version": version,
        "training_artifact_path": _display_path(data_dir, training_path),
        "baseline_report_path": _display_path(data_dir, baseline_path),
        "source_manifest_path": _display_path(data_dir, source_manifest),
        "split_plan_path": _display_path(data_dir, split_plan),
        "thresholds": asdict(thresholds),
        "actual": actual,
        "planned": planned,
        "baseline": baseline,
        "gates": gates,
        "ready_for_gctx1_proof_run": ready,
        "recommended_next_action": _recommended_next_action(gates),
        "output_path": str(proof_readiness_report_path(artifact_name, version=version)),
    }
    return report


def write_proof_readiness_report(
    data_dir: str | Path,
    report: dict[str, Any],
) -> Path:
    """Write a proof-readiness report under the data artifact directory."""

    output_path = Path(data_dir) / proof_readiness_report_path(
        report["artifact_name"],
        version=report["artifact_version"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _actual_artifact_summary(
    training_records: list[dict[str, Any]],
    source_manifest_by_repo: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    split_counts: Counter[str] = Counter(record["data_split"] for record in training_records)
    repos_by_split: dict[str, set[str]] = {"DEV": set(), "REPORT": set(), "HELD_OUT": set()}
    repo_counts_by_split: dict[str, Counter[str]] = {
        "DEV": Counter(),
        "REPORT": Counter(),
        "HELD_OUT": Counter(),
    }
    license_counts: Counter[str] = Counter()
    label_source_counts: Counter[str] = Counter()
    review_decision_counts: Counter[str] = Counter()
    ecosystems: set[str] = set()
    missing_manifest_repos: set[str] = set()

    for record in training_records:
        split = record["data_split"]
        repo_url = _normalize_repo_url(record["source_repo_url"])
        repos_by_split.setdefault(split, set()).add(repo_url)
        repo_counts_by_split.setdefault(split, Counter())[repo_url] += 1
        license_counts[record["source_license"]] += 1
        label_source_counts[record["label_source"]] += 1
        review_decision_counts[record["review_decision"]] += 1
        manifest_record = source_manifest_by_repo.get(repo_url)
        if manifest_record is None:
            missing_manifest_repos.add(repo_url)
        else:
            ecosystem = manifest_record.get("ecosystem")
            if isinstance(ecosystem, str) and ecosystem:
                ecosystems.add(ecosystem)

    dev_repo_counts = repo_counts_by_split["DEV"]
    max_repo_fraction = _max_fraction(dev_repo_counts)
    return {
        "training_records": len(training_records),
        "split_counts": {split: split_counts[split] for split in ("DEV", "REPORT", "HELD_OUT")},
        "repos_by_split": {
            split: len(repos_by_split.get(split, set())) for split in ("DEV", "REPORT", "HELD_OUT")
        },
        "dev_repo_record_counts_top10": dict(dev_repo_counts.most_common(10)),
        "max_repo_train_fraction": max_repo_fraction,
        "source_license_counts": dict(sorted(license_counts.items())),
        "label_source_counts": dict(sorted(label_source_counts.items())),
        "review_decision_counts": dict(sorted(review_decision_counts.items())),
        "ecosystems": sorted(ecosystems),
        "missing_manifest_repos": sorted(missing_manifest_repos),
    }


def _baseline_summary(
    baseline_path: Path,
    *,
    expected_training_records: int,
    expected_split_counts: dict[str, int],
) -> dict[str, Any]:
    if not baseline_path.exists():
        return {
            "exists": False,
            "complete": False,
            "errors": ["baseline report is missing"],
        }
    report = json.loads(baseline_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if report.get("training_records") != expected_training_records:
        errors.append("baseline training_records does not match training artifact length")
    baseline_split_counts = report.get("data_split_counts")
    normalized_baseline_split_counts = {
        split: int(baseline_split_counts.get(split, 0))
        for split in ("DEV", "REPORT", "HELD_OUT")
    } if isinstance(baseline_split_counts, dict) else None
    if normalized_baseline_split_counts != expected_split_counts:
        errors.append("baseline data_split_counts does not match training artifact splits")
    for key in ("target", "teacher", "historical", "by_data_split"):
        if key not in report:
            errors.append(f"baseline report is missing {key}")
    return {
        "exists": True,
        "complete": not errors,
        "training_records": report.get("training_records"),
        "data_split_counts": normalized_baseline_split_counts,
        "errors": errors,
    }


def _planned_summary(split_readiness: dict[str, Any] | None) -> dict[str, Any]:
    if split_readiness is None:
        return {
            "exists": False,
            "reserved_held_out_records": None,
            "reserved_held_out_repos": None,
            "report_windows": None,
            "errors": ["source manifest or split plan is missing"],
        }
    held_out = split_readiness["splits"]["HELD_OUT"]
    report = split_readiness["splits"]["REPORT"]
    errors = [
        *[
            f"manifest: {entry['repo_url']}: {', '.join(entry['errors'])}"
            for entry in split_readiness["manifest_errors"]
        ],
        *[
            f"window: {entry['window_id']}: {', '.join(entry['errors'])}"
            for entry in split_readiness["cross_errors"]
        ],
    ]
    return {
        "exists": True,
        "reserved_held_out_records": held_out["target_records"],
        "reserved_held_out_repos": held_out["repos"],
        "report_windows": report["windows"],
        "split_plan_ready_for_gctx1_planning": split_readiness["ready_for_gctx1_planning"],
        "errors": errors,
    }


def _gates(
    *,
    actual: dict[str, Any],
    baseline: dict[str, Any],
    planned: dict[str, Any],
    thresholds: ProofThresholds,
    source_manifest_exists: bool,
    split_plan_exists: bool,
) -> dict[str, dict[str, Any]]:
    return {
        "source_manifest_available": _bool_gate(source_manifest_exists),
        "split_plan_available": _bool_gate(split_plan_exists),
        "dev_training_records": _min_gate(
            actual["split_counts"]["DEV"],
            thresholds.min_dev_records,
        ),
        "report_records": _min_gate(
            actual["split_counts"]["REPORT"],
            thresholds.min_report_records,
        ),
        "held_out_excluded_from_training": _exact_gate(actual["split_counts"]["HELD_OUT"], 0),
        "reserved_held_out_records": _min_gate_or_unknown(
            planned["reserved_held_out_records"],
            thresholds.min_reserved_held_out_records,
        ),
        "train_repos": _min_gate(actual["repos_by_split"]["DEV"], thresholds.min_train_repos),
        "report_repos": _min_gate(
            actual["repos_by_split"]["REPORT"],
            thresholds.min_report_repos,
        ),
        "reserved_held_out_repos": _min_gate_or_unknown(
            planned["reserved_held_out_repos"],
            thresholds.min_reserved_held_out_repos,
        ),
        "ecosystems": _min_gate(len(actual["ecosystems"]), thresholds.min_ecosystems),
        "max_repo_train_fraction": _max_gate_or_unknown(
            actual["max_repo_train_fraction"],
            thresholds.max_repo_train_fraction,
        ),
        "baseline_report_complete": _bool_gate(baseline["complete"]),
        "no_missing_manifest_repos": _bool_gate(not actual["missing_manifest_repos"]),
        "no_split_plan_errors": _bool_gate(not planned["errors"]),
    }


def _recommended_next_action(gates: dict[str, dict[str, Any]]) -> str:
    failing = {name for name, gate in gates.items() if gate["status"] != "pass"}
    if not failing:
        return "Proceed to a GCTX-1 proof-model training run and evaluate on locked REPORT."
    if failing == {"dev_training_records"}:
        return (
            "Expand reviewed DEV training records to the GCTX-1 minimum, or record "
            "a written decision to run a smaller private proof model with weaker claims."
        )
    if "baseline_report_complete" in failing:
        return "Regenerate the baseline report before training."
    if "reserved_held_out_records" in failing or "reserved_held_out_repos" in failing:
        return "Reserve enough HELD_OUT candidates before any release or generalization claim."
    return "Resolve failing proof-readiness gates before treating the artifact as GCTX-1-ready."


def _min_gate(actual: int | float, minimum: int) -> dict[str, Any]:
    return {
        "actual": actual,
        "minimum": minimum,
        "status": "pass" if actual >= minimum else "fail",
    }


def _min_gate_or_unknown(actual: int | None, minimum: int) -> dict[str, Any]:
    if actual is None:
        return {
            "actual": None,
            "minimum": minimum,
            "status": "unknown",
        }
    return _min_gate(actual, minimum)


def _max_gate_or_unknown(actual: float | None, maximum: float) -> dict[str, Any]:
    if actual is None:
        return {
            "actual": None,
            "maximum": maximum,
            "status": "unknown",
        }
    return {
        "actual": round(actual, 6),
        "maximum": maximum,
        "status": "pass" if actual <= maximum else "fail",
    }


def _exact_gate(actual: int | float, expected: int | float) -> dict[str, Any]:
    return {
        "actual": actual,
        "expected": expected,
        "status": "pass" if actual == expected else "fail",
    }


def _bool_gate(value: bool) -> dict[str, Any]:
    return {
        "actual": value,
        "expected": True,
        "status": "pass" if value else "fail",
    }


def _max_fraction(counter: Counter[str]) -> float | None:
    total = sum(counter.values())
    if total <= 0:
        return None
    return max(counter.values()) / total


def _defaulted_data_path(data_dir: Path, path: str | Path | None, default: Path) -> Path:
    selected = Path(path) if path is not None else default
    return selected if selected.is_absolute() else data_dir / selected


def _display_path(data_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(data_dir))
    except ValueError:
        return str(path)


def _normalize_repo_url(repo_url: str) -> str:
    return repo_url.rstrip("/").removesuffix(".git")


def _validate_identifier(value: str, name: str) -> None:
    if not value.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"{name} must be a stable alphanumeric identifier")


def _print_report(report: dict[str, Any]) -> None:
    print("artifact_name", report["artifact_name"])
    print("artifact_version", report["artifact_version"])
    print("training_records", report["actual"]["training_records"])
    print("dev_training_records", report["actual"]["split_counts"]["DEV"])
    print("report_records", report["actual"]["split_counts"]["REPORT"])
    print("reserved_held_out_records", report["planned"]["reserved_held_out_records"])
    for name, gate in report["gates"].items():
        print(f"gate_{name}", gate["status"])
    print("ready_for_gctx1_proof_run", report["ready_for_gctx1_proof_run"])
    print("recommended_next_action", report["recommended_next_action"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate gitctx proof-model readiness.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--artifact-name", required=True)
    evaluate.add_argument("--version", default="v0")
    evaluate.add_argument("--source-manifest")
    evaluate.add_argument("--split-plan")
    evaluate.add_argument("--write", action="store_true")
    evaluate.add_argument("--json", action="store_true")
    evaluate.add_argument("--fail-on-not-ready", action="store_true")
    evaluate.add_argument("--min-dev-records", type=int, default=ProofThresholds.min_dev_records)
    evaluate.add_argument("--min-report-records", type=int, default=ProofThresholds.min_report_records)
    evaluate.add_argument(
        "--min-reserved-held-out-records",
        type=int,
        default=ProofThresholds.min_reserved_held_out_records,
    )
    evaluate.add_argument("--min-train-repos", type=int, default=ProofThresholds.min_train_repos)
    evaluate.add_argument("--min-report-repos", type=int, default=ProofThresholds.min_report_repos)
    evaluate.add_argument(
        "--min-reserved-held-out-repos",
        type=int,
        default=ProofThresholds.min_reserved_held_out_repos,
    )
    evaluate.add_argument("--min-ecosystems", type=int, default=ProofThresholds.min_ecosystems)
    evaluate.add_argument(
        "--max-repo-train-fraction",
        type=float,
        default=ProofThresholds.max_repo_train_fraction,
    )
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        thresholds = ProofThresholds(
            min_dev_records=args.min_dev_records,
            min_report_records=args.min_report_records,
            min_reserved_held_out_records=args.min_reserved_held_out_records,
            min_train_repos=args.min_train_repos,
            min_report_repos=args.min_report_repos,
            min_reserved_held_out_repos=args.min_reserved_held_out_repos,
            min_ecosystems=args.min_ecosystems,
            max_repo_train_fraction=args.max_repo_train_fraction,
        )
        report = evaluate_proof_readiness(
            args.data_dir,
            artifact_name=args.artifact_name,
            version=args.version,
            source_manifest_path=args.source_manifest,
            split_plan_path=args.split_plan,
            thresholds=thresholds,
        )
        if args.write:
            output_path = write_proof_readiness_report(args.data_dir, report)
            print("output_path", output_path)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _print_report(report)
        if args.fail_on_not_ready and not report["ready_for_gctx1_proof_run"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

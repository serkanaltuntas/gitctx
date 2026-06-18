"""Build supervised fine-tuning artifacts from reviewed teacher labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from gitctx.data_artifacts import generated_label_review_path
from gitctx.ollama_generate import generated_labels_path
from gitctx.provenance import (
    load_jsonl,
    validate_generated_label_matches_input,
    validate_generated_label_review_decision,
    validate_source_diff_record,
    validate_teacher_input_record,
)
from gitctx.teacher_inputs import source_diffs_path, teacher_inputs_path

TRAIN_ARTIFACT_DIR = Path("artifacts/train")
TRAINING_INSTRUCTION = "Write one Conventional Commit message for the provided Git diff."
TRAINING_LABEL_SOURCES = frozenset(
    {
        "teacher_generated_human_accepted",
        "teacher_generated_human_edited",
    }
)


def training_examples_path(artifact_name: str, *, version: str = "v0") -> Path:
    """Return the training-example JSONL path for a named artifact."""

    _validate_artifact_name(artifact_name)
    _validate_version(version)
    return TRAIN_ARTIFACT_DIR / f"sft.{artifact_name}.{version}.jsonl"


def training_report_path(artifact_name: str, *, version: str = "v0") -> Path:
    """Return the training-example report path for a named artifact."""

    _validate_artifact_name(artifact_name)
    _validate_version(version)
    return TRAIN_ARTIFACT_DIR / f"sft.{artifact_name}.{version}.report.json"


def create_training_artifact(
    data_dir: str | Path,
    *,
    artifact_name: str,
    version: str = "v0",
) -> dict[str, Any]:
    """Create supervised fine-tuning records from reviewed generated labels."""

    _validate_artifact_name(artifact_name)
    _validate_version(version)
    data_dir = Path(data_dir)
    source_records = load_jsonl(data_dir / source_diffs_path(artifact_name))
    teacher_inputs = load_jsonl(data_dir / teacher_inputs_path(artifact_name))
    generated_labels = load_jsonl(data_dir / generated_labels_path(artifact_name))
    review_records = load_jsonl(data_dir / generated_label_review_path(artifact_name))
    source_by_id = {record["id"]: record for record in source_records}
    input_by_source_id = {record["source_diff_id"]: record for record in teacher_inputs}
    label_by_id = {record["id"]: record for record in generated_labels}
    review_ids: set[str] = set()
    failures: list[tuple[str, tuple[str, ...]]] = []
    decision_counts = _decision_counts()
    output_records: list[dict[str, Any]] = []

    for review in review_records:
        review_id = str(review.get("id", "<missing-id>"))
        errors = list(validate_generated_label_review_decision(review))
        label_id = review.get("generated_label_id")
        if isinstance(label_id, str):
            if label_id in review_ids:
                errors.append(f"duplicate review for generated_label_id: {label_id}")
            review_ids.add(label_id)
        label = label_by_id.get(label_id)
        if label is None:
            errors.append(f"unknown generated_label_id: {label_id}")
            failures.append((review_id, tuple(errors)))
            continue

        source_diff_id = _source_diff_id_from_generated_label_id(label["id"])
        teacher_input = input_by_source_id.get(source_diff_id)
        source = source_by_id.get(source_diff_id)
        if teacher_input is None:
            errors.append(f"missing teacher input for source_diff_id: {source_diff_id}")
        if source is None:
            errors.append(f"missing source diff for source_diff_id: {source_diff_id}")
        if teacher_input is not None:
            errors.extend(validate_teacher_input_record(teacher_input))
            errors.extend(validate_generated_label_matches_input(label, teacher_input))
        if source is not None:
            errors.extend(validate_source_diff_record(source))

        decision = review.get("decision")
        if isinstance(decision, str) and decision in decision_counts:
            decision_counts[decision] += 1
        if decision == "needs_review":
            errors.append("generated label review is still needs_review")
        if decision == "edit" and review.get("edited_header") is None and review.get("edited_body") is None:
            errors.append("edit review must provide edited_header or edited_body")
        if decision == "accept" and (
            review.get("edited_header") is not None or review.get("edited_body") is not None
        ):
            errors.append("accept review must not include edited text")

        if errors:
            failures.append((review_id, tuple(errors)))
            continue
        if decision not in {"accept", "edit"}:
            continue

        assert teacher_input is not None
        assert source is not None
        output_records.append(
            _build_training_record(
                artifact_name=artifact_name,
                version=version,
                source=source,
                teacher_input=teacher_input,
                label=label,
                review=review,
            )
        )

    missing_reviews = sorted(set(label_by_id) - review_ids)
    if missing_reviews:
        failures.append(("<missing-generated-label-reviews>", tuple(missing_reviews)))

    if failures:
        for name, errors in failures:
            print(f"FAIL {name}: {errors}")
        raise SystemExit(1)

    output_path = data_dir / training_examples_path(artifact_name, version=version)
    report_path = data_dir / training_report_path(artifact_name, version=version)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in output_records) + "\n",
        encoding="utf-8",
    )
    report = {
        "artifact_name": artifact_name,
        "artifact_version": version,
        "source_records": len(source_records),
        "teacher_input_records": len(teacher_inputs),
        "generated_label_records": len(generated_labels),
        "review_records": len(review_records),
        "training_records": len(output_records),
        "output_path": str(training_examples_path(artifact_name, version=version)),
        "report_path": str(training_report_path(artifact_name, version=version)),
        **decision_counts,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for key in (
        "artifact_name",
        "artifact_version",
        "generated_label_records",
        "review_records",
        "training_records",
        "accept",
        "edit",
        "reject",
        "needs_review",
    ):
        print(key, report[key])
    return report


def validate_training_artifact(
    data_dir: str | Path,
    *,
    artifact_name: str,
    version: str = "v0",
) -> dict[str, Any]:
    """Validate supervised fine-tuning records against the reviewed labels."""

    _validate_artifact_name(artifact_name)
    _validate_version(version)
    data_dir = Path(data_dir)
    teacher_inputs = load_jsonl(data_dir / teacher_inputs_path(artifact_name))
    generated_labels = load_jsonl(data_dir / generated_labels_path(artifact_name))
    review_records = load_jsonl(data_dir / generated_label_review_path(artifact_name))
    training_records = load_jsonl(data_dir / training_examples_path(artifact_name, version=version))
    report = json.loads(
        (data_dir / training_report_path(artifact_name, version=version)).read_text(
            encoding="utf-8"
        )
    )
    input_by_source_id = {record["source_diff_id"]: record for record in teacher_inputs}
    label_by_id = {record["id"]: record for record in generated_labels}
    review_by_label_id = {record["generated_label_id"]: record for record in review_records}
    accepted_label_ids = {
        record["generated_label_id"]
        for record in review_records
        if record.get("decision") in {"accept", "edit"}
    }
    seen_ids: set[str] = set()
    seen_label_ids: set[str] = set()
    failures: list[tuple[str, tuple[str, ...]]] = []

    for record in training_records:
        record_id = str(record.get("id", "<missing-id>"))
        errors = list(validate_training_record(record))
        label_id = record.get("generated_label_id")
        review_id = record.get("generated_label_review_id")
        source_diff_id = record.get("source_diff_id")
        label = label_by_id.get(label_id)
        review = review_by_label_id.get(label_id)
        teacher_input = input_by_source_id.get(source_diff_id)
        if record_id in seen_ids:
            errors.append(f"duplicate training id: {record_id}")
        seen_ids.add(record_id)
        if isinstance(label_id, str):
            if label_id in seen_label_ids:
                errors.append(f"duplicate generated_label_id: {label_id}")
            seen_label_ids.add(label_id)
        if label is None:
            errors.append(f"unknown generated_label_id: {label_id}")
        if review is None:
            errors.append(f"missing review for generated_label_id: {label_id}")
        elif review.get("id") != review_id:
            errors.append("generated_label_review_id does not match review record")
        if teacher_input is None:
            errors.append(f"unknown source_diff_id: {source_diff_id}")
        elif teacher_input.get("id") != record.get("teacher_input_id"):
            errors.append("teacher_input_id does not match teacher input")
        if label is not None and review is not None and teacher_input is not None:
            errors.extend(_validate_training_record_lineage(record, label, review, teacher_input))
        if errors:
            failures.append((record_id, tuple(errors)))

    missing = sorted(accepted_label_ids - seen_label_ids)
    if missing:
        failures.append(("<missing-training-records>", tuple(missing)))

    expected_training_records = sum(
        1 for record in review_records if record.get("decision") in {"accept", "edit"}
    )
    if report.get("training_records") != len(training_records):
        failures.append(("<report>", ("training_records does not match JSONL length",)))
    if len(training_records) != expected_training_records:
        failures.append(("<training-record-count>", (str(expected_training_records),)))

    if failures:
        for name, errors in failures:
            print(f"FAIL {name}: {errors}")
        raise SystemExit(1)

    summary = {
        "artifact_name": artifact_name,
        "artifact_version": version,
        "training_records": len(training_records),
        "accepted_or_edited_reviews": expected_training_records,
    }
    for key, value in summary.items():
        print(key, value)
    return summary


def validate_training_record(record: dict[str, Any]) -> tuple[str, ...]:
    """Return validation errors for one training example record."""

    errors: list[str] = []
    for key in (
        "id",
        "artifact_name",
        "artifact_version",
        "source_diff_id",
        "teacher_input_id",
        "generated_label_id",
        "generated_label_review_id",
        "source_repo_url",
        "source_license",
        "source_commit",
        "parent_commit",
        "data_split",
        "instruction",
        "diff",
        "diff_sha256",
        "target_header",
        "target_message",
        "label_source",
        "review_decision",
        "reviewer",
        "review_timestamp",
        "teacher_model_id",
        "teacher_runtime",
        "teacher_runtime_model_id",
        "teacher_revision",
        "teacher_license",
        "prompt_version",
        "generation_timestamp",
    ):
        _require_str(record, key, errors)

    if record.get("data_split") == "HELD_OUT":
        errors.append("training examples must not come from HELD_OUT teacher labels")
    if record.get("instruction") != TRAINING_INSTRUCTION:
        errors.append("instruction does not match the training artifact contract")
    if record.get("label_source") not in TRAINING_LABEL_SOURCES:
        errors.append(f"invalid label_source: {record.get('label_source')}")
    if record.get("review_decision") not in {"accept", "edit"}:
        errors.append(f"invalid review_decision: {record.get('review_decision')}")
    for key in (
        "changed_paths",
        "target_body",
        "target_footers",
        "review_issues",
        "evidence_paths",
        "warnings",
    ):
        _require_list(record, key, errors)
    for key in ("decoding_config", "parser_result"):
        if not isinstance(record.get(key), dict):
            errors.append(f"{key} must be an object")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append("messages must contain system, user, and assistant messages")
    elif [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        errors.append("messages roles must be system, user, assistant")
    elif messages[2].get("content") != record.get("target_message"):
        errors.append("assistant message does not match target_message")
    diff = record.get("diff")
    diff_sha256 = record.get("diff_sha256")
    if isinstance(diff, str) and isinstance(diff_sha256, str):
        actual_digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        if actual_digest != diff_sha256:
            errors.append("diff_sha256 does not match diff")
    target_header = record.get("target_header")
    target_body = record.get("target_body")
    target_footers = record.get("target_footers")
    if isinstance(target_header, str) and isinstance(target_body, list) and isinstance(
        target_footers, list
    ):
        expected = format_commit_message(target_header, target_body, target_footers)
        if record.get("target_message") != expected:
            errors.append("target_message does not match target fields")
    return tuple(errors)


def format_commit_message(header: str, body: list[str], footers: list[str]) -> str:
    """Format Conventional Commit fields as the final assistant target."""

    lines = [header]
    if body:
        lines.extend(["", *body])
    if footers:
        lines.extend(["", *footers])
    return "\n".join(lines)


def _build_training_record(
    *,
    artifact_name: str,
    version: str,
    source: dict[str, Any],
    teacher_input: dict[str, Any],
    label: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    decision = review["decision"]
    target_header = review["edited_header"] if review["edited_header"] is not None else label["header"]
    target_body = review["edited_body"] if review["edited_body"] is not None else label["body"]
    target_footers = label["footers"]
    target_message = format_commit_message(target_header, target_body, target_footers)
    return {
        "id": f"sft-{source['id']}",
        "artifact_name": artifact_name,
        "artifact_version": version,
        "source_diff_id": source["id"],
        "teacher_input_id": teacher_input["id"],
        "generated_label_id": label["id"],
        "generated_label_review_id": review["id"],
        "source_repo_url": source["source_repo_url"],
        "source_license": source["source_license"],
        "source_commit": source["source_commit"],
        "parent_commit": source["parent_commit"],
        "data_split": source["data_split"],
        "changed_paths": source["changed_paths"],
        "diff_stat": source["diff_stat"],
        "historical_subject": source["historical_subject"],
        "instruction": TRAINING_INSTRUCTION,
        "diff": teacher_input["diff"],
        "diff_sha256": teacher_input["diff_sha256"],
        "messages": [
            {"role": "system", "content": teacher_input["system_message"]},
            {"role": "user", "content": teacher_input["user_message"]},
            {"role": "assistant", "content": target_message},
        ],
        "target_header": target_header,
        "target_body": target_body,
        "target_footers": target_footers,
        "target_message": target_message,
        "label_source": _label_source(decision),
        "review_decision": decision,
        "review_issues": review["issues"],
        "review_notes": review["notes"],
        "reviewer": review["reviewer"],
        "review_timestamp": review["review_timestamp"],
        "teacher_model_id": label["teacher_model_id"],
        "teacher_runtime": label["teacher_runtime"],
        "teacher_runtime_model_id": label["teacher_runtime_model_id"],
        "teacher_revision": label["teacher_revision"],
        "teacher_license": label["teacher_license"],
        "teacher_size": label["teacher_size"],
        "teacher_context_length": label["teacher_context_length"],
        "prompt_version": label["prompt_version"],
        "decoding_config": label["decoding_config"],
        "generation_timestamp": label["generation_timestamp"],
        "confidence": label["confidence"],
        "verifier_score": label["verifier_score"],
        "evidence_paths": label["evidence_paths"],
        "warnings": label["warnings"],
        "parser_result": label["parser_result"],
    }


def _validate_training_record_lineage(
    record: dict[str, Any],
    label: dict[str, Any],
    review: dict[str, Any],
    teacher_input: dict[str, Any],
) -> tuple[str, ...]:
    errors = []
    for key in (
        "source_repo_url",
        "source_license",
        "source_commit",
        "parent_commit",
        "data_split",
        "changed_paths",
        "teacher_model_id",
        "teacher_runtime",
        "teacher_runtime_model_id",
        "teacher_revision",
        "teacher_license",
        "prompt_version",
        "decoding_config",
    ):
        if record.get(key) != label.get(key):
            errors.append(f"{key} does not match generated label")
    for key in ("diff", "diff_sha256"):
        if record.get(key) != teacher_input.get(key):
            errors.append(f"{key} does not match teacher input")
    if record.get("review_decision") != review.get("decision"):
        errors.append("review_decision does not match review")
    if record.get("reviewer") != review.get("reviewer"):
        errors.append("reviewer does not match review")
    if record.get("review_timestamp") != review.get("review_timestamp"):
        errors.append("review_timestamp does not match review")
    if review.get("decision") == "accept" and record.get("target_header") != label.get("header"):
        errors.append("accepted target_header does not match generated label")
    if review.get("decision") == "edit":
        edited_header = review.get("edited_header")
        edited_body = review.get("edited_body")
        if edited_header is not None and record.get("target_header") != edited_header:
            errors.append("edited target_header does not match review")
        if edited_body is not None and record.get("target_body") != edited_body:
            errors.append("edited target_body does not match review")
    if record.get("target_footers") != label.get("footers"):
        errors.append("target_footers does not match generated label")
    return tuple(errors)


def _source_diff_id_from_generated_label_id(label_id: str) -> str:
    if not label_id.startswith("generated-"):
        raise ValueError(f"generated label id does not start with generated-: {label_id}")
    return label_id.removeprefix("generated-")


def _label_source(decision: str) -> str:
    if decision == "accept":
        return "teacher_generated_human_accepted"
    if decision == "edit":
        return "teacher_generated_human_edited"
    raise ValueError(f"unsupported training label decision: {decision}")


def _decision_counts() -> dict[str, int]:
    return {
        "accept": 0,
        "edit": 0,
        "reject": 0,
        "needs_review": 0,
    }


def _require_str(record: dict[str, Any], key: str, errors: list[str]) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"{key} must be a non-empty string")


def _require_list(record: dict[str, Any], key: str, errors: list[str]) -> None:
    value = record.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{key} must be a list of strings")


def _validate_artifact_name(artifact_name: str) -> None:
    if not artifact_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("artifact_name must be a stable alphanumeric identifier")


def _validate_version(version: str) -> None:
    if not version.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("version must be a stable alphanumeric identifier")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build gitctx training artifacts.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--artifact-name", required=True)
    create.add_argument("--version", default="v0")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--artifact-name", required=True)
    validate.add_argument("--version", default="v0")
    args = parser.parse_args(argv)

    if args.command == "create":
        create_training_artifact(
            args.data_dir,
            artifact_name=args.artifact_name,
            version=args.version,
        )
    elif args.command == "validate":
        validate_training_artifact(
            args.data_dir,
            artifact_name=args.artifact_name,
            version=args.version,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

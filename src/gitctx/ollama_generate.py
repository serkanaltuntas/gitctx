"""Generate teacher labels from teacher-input artifacts through Ollama."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import urllib.error
import urllib.request
from typing import Any

from gitctx.conventional import CommitContext, score_commit_message
from gitctx.provenance import (
    load_jsonl,
    validate_generated_label_matches_input,
)
from gitctx.teacher_inputs import SMOKE_TEACHER_INPUTS

SMOKE_GENERATED_LABELS = Path("artifacts/teacher/generated-labels.smoke.jsonl")
SMOKE_GENERATED_REPORT = Path("artifacts/teacher/generated-labels.smoke.report.json")


def generate_smoke_labels(
    data_dir: str | Path,
    *,
    ollama_url: str = "http://127.0.0.1:11434",
    limit: int | None = None,
    ollama_options: dict[str, Any] | None = None,
    request_timeout: int = 600,
    resume: bool = True,
    think: bool = False,
) -> dict[str, Any]:
    """Generate labels for smoke teacher inputs and write JSONL plus report."""

    data_dir = Path(data_dir)
    input_path = data_dir / SMOKE_TEACHER_INPUTS
    output_path = data_dir / SMOKE_GENERATED_LABELS
    report_path = data_dir / SMOKE_GENERATED_REPORT
    teacher_inputs = load_jsonl(input_path)
    if limit is not None:
        teacher_inputs = teacher_inputs[:limit]

    existing_records = _load_existing(output_path) if resume else []
    existing_ids = {record["id"] for record in existing_records}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    with output_path.open("a" if resume else "w", encoding="utf-8") as f:
        for teacher_input in teacher_inputs:
            output_id = f"generated-{teacher_input['source_diff_id']}"
            if output_id in existing_ids:
                skipped += 1
                continue
            try:
                raw_content = _call_ollama(
                    ollama_url,
                    teacher_input,
                    ollama_options=ollama_options,
                    request_timeout=request_timeout,
                    think=think,
                )
                candidate = _parse_teacher_json(raw_content)
                record = _build_generated_label(output_id, teacher_input, candidate)
                errors = validate_generated_label_matches_input(record, teacher_input)
                if errors:
                    raise ValueError("; ".join(errors))
                f.write(json.dumps(record, sort_keys=True) + "\n")
                generated += 1
            except Exception as exc:  # noqa: BLE001 - preserve per-record failures.
                failures.append(
                    {
                        "id": output_id,
                        "source_diff_id": teacher_input.get("source_diff_id"),
                        "error": str(exc),
                    }
                )

    report = {
        "generated_records": generated,
        "input_records": len(teacher_inputs),
        "output_path": str(SMOKE_GENERATED_LABELS),
        "failed_records": len(failures),
        "failures": failures,
        "skipped_existing_records": skipped,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for key in (
        "input_records",
        "generated_records",
        "skipped_existing_records",
        "failed_records",
    ):
        print(key, report[key])
    if failures:
        raise SystemExit(1)
    return report


def validate_smoke_generated_labels(data_dir: str | Path) -> dict[str, int]:
    """Validate generated-label smoke artifacts against teacher inputs."""

    data_dir = Path(data_dir)
    teacher_inputs = load_jsonl(data_dir / SMOKE_TEACHER_INPUTS)
    generated_labels = load_jsonl(data_dir / SMOKE_GENERATED_LABELS)
    inputs_by_source_id = {record["source_diff_id"]: record for record in teacher_inputs}
    seen_source_ids: set[str] = set()
    failures: list[tuple[str, tuple[str, ...]]] = []

    for label in generated_labels:
        record_id = label.get("id", "<missing-id>")
        source_diff_id = str(record_id).removeprefix("generated-")
        teacher_input = inputs_by_source_id.get(source_diff_id)
        errors: list[str] = []
        if teacher_input is None:
            errors.append(f"unknown generated source_diff_id: {source_diff_id}")
        else:
            errors.extend(validate_generated_label_matches_input(label, teacher_input))
            if source_diff_id in seen_source_ids:
                errors.append(f"duplicate generated label for source_diff_id: {source_diff_id}")
            seen_source_ids.add(source_diff_id)
        if errors:
            failures.append((str(record_id), tuple(errors)))

    missing = sorted(set(inputs_by_source_id) - seen_source_ids)
    if missing:
        failures.append(("<missing-generated-labels>", tuple(missing)))

    if failures:
        for name, errors in failures:
            print(f"FAIL {name}: {errors}")
        raise SystemExit(1)

    summary = {
        "teacher_input_records": len(teacher_inputs),
        "generated_label_records": len(generated_labels),
    }
    for key, value in summary.items():
        print(key, value)
    return summary


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def _call_ollama(
    ollama_url: str,
    teacher_input: dict[str, Any],
    *,
    ollama_options: dict[str, Any] | None = None,
    request_timeout: int = 600,
    think: bool = False,
) -> str:
    payload = {
        "model": teacher_input["teacher_runtime_model_id"],
        "stream": False,
        "format": "json",
        "options": _ollama_options(teacher_input["decoding_config"], ollama_options),
        "think": think,
        "messages": [
            {"role": "system", "content": teacher_input["system_message"]},
            {"role": "user", "content": teacher_input["user_message"]},
        ],
    }
    request = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    try:
        return str(body["message"]["content"])
    except KeyError as exc:
        raise ValueError(f"unexpected Ollama response shape: {body}") from exc


def _ollama_options(
    decoding_config: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for key in ("temperature", "top_p", "seed"):
        if key in decoding_config:
            options[key] = decoding_config[key]
    if "num_predict" in decoding_config:
        options["num_predict"] = decoding_config["num_predict"]
    elif "max_new_tokens" in decoding_config:
        options["num_predict"] = decoding_config["max_new_tokens"]
    if overrides:
        options.update({key: value for key, value in overrides.items() if value is not None})
    return options


def _parse_teacher_json(raw_content: str) -> dict[str, Any]:
    content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    first = content.find("{")
    last = content.rfind("}")
    if first == -1 or last == -1 or last < first:
        raise ValueError("teacher output did not contain a JSON object")
    value = json.loads(content[first : last + 1])
    if not isinstance(value, dict):
        raise ValueError("teacher output JSON must be an object")
    return value


def _build_generated_label(
    output_id: str,
    teacher_input: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    header = _require_candidate_str(candidate, "header")
    body = _require_candidate_list(candidate, "body")
    footers = _require_candidate_list(candidate, "footers")
    warnings = _require_candidate_list(candidate, "warnings")
    evidence_paths = _require_candidate_list(candidate, "evidence_paths")
    score = score_commit_message(
        "\n".join([header, *body, *footers]),
        CommitContext(changed_paths=tuple(teacher_input["changed_paths"])),
    )
    parsed = score.parsed
    parser_result = {
        "body_presence": score.body_presence,
        "breaking_change_detection": score.breaking_change_detection,
        "brevity": score.brevity,
        "errors": list(score.errors),
        "factuality": score.factuality,
        "format_validity": score.format_validity,
        "footer_presence": score.footer_presence,
        "mixed_change_warning": score.mixed_change_warning,
        "scope_quality": score.scope_quality,
        "specificity": score.specificity,
        "type_accuracy": score.type_accuracy,
    }
    verifier_score = _verifier_score(parser_result, evidence_paths, teacher_input["changed_paths"])
    return {
        "id": output_id,
        "source_repo_url": teacher_input["source_repo_url"],
        "source_license": teacher_input["source_license"],
        "source_commit": teacher_input["source_commit"],
        "parent_commit": teacher_input["parent_commit"],
        "data_split": teacher_input["data_split"],
        "changed_paths": teacher_input["changed_paths"],
        "teacher_model_id": teacher_input["teacher_model_id"],
        "teacher_runtime": teacher_input["teacher_runtime"],
        "teacher_runtime_model_id": teacher_input["teacher_runtime_model_id"],
        "teacher_revision": teacher_input["teacher_revision"],
        "teacher_license": teacher_input["teacher_license"],
        "teacher_size": teacher_input["teacher_size"],
        "teacher_context_length": teacher_input["teacher_context_length"],
        "prompt_version": teacher_input["prompt_version"],
        "decoding_config": teacher_input["decoding_config"],
        "generation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "header": header,
        "body": body,
        "footers": footers,
        "type": str(candidate.get("type") or (parsed.type if parsed else "")),
        "scope": candidate.get("scope"),
        "confidence": _candidate_confidence(candidate),
        "warnings": warnings,
        "evidence_paths": evidence_paths,
        "parser_result": parser_result,
        "verifier_score": verifier_score,
        "human_review_status": "not_reviewed",
    }


def _require_candidate_str(candidate: dict[str, Any], key: str) -> str:
    value = candidate.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"teacher output {key} must be a non-empty string")
    return value


def _require_candidate_list(candidate: dict[str, Any], key: str) -> list[str]:
    value = candidate.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"teacher output {key} must be a list of strings")
    return value


def _candidate_confidence(candidate: dict[str, Any]) -> float:
    value = candidate.get("confidence")
    if not isinstance(value, (int, float)):
        raise ValueError("teacher output confidence must be numeric")
    return max(0.0, min(1.0, float(value)))


def _verifier_score(
    parser_result: dict[str, Any],
    evidence_paths: list[str],
    changed_paths: list[str],
) -> float:
    checks = [
        parser_result["format_validity"],
        parser_result["specificity"],
        parser_result["brevity"],
        not parser_result["errors"],
        bool(evidence_paths),
        set(evidence_paths).issubset(set(changed_paths)),
    ]
    return sum(1 for check in checks if check) / len(checks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate gitctx labels through Ollama.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate-smoke")
    generate.add_argument("--limit", type=int)
    generate.add_argument("--no-resume", action="store_true")
    generate.add_argument("--num-ctx", type=int)
    generate.add_argument("--num-predict", type=int)
    generate.add_argument("--request-timeout", type=int, default=600)
    generate.add_argument("--think", action="store_true")
    subparsers.add_parser("validate-smoke")
    args = parser.parse_args(argv)

    if args.command == "generate-smoke":
        generate_smoke_labels(
            args.data_dir,
            ollama_url=args.ollama_url,
            limit=args.limit,
            ollama_options={"num_ctx": args.num_ctx, "num_predict": args.num_predict},
            request_timeout=args.request_timeout,
            resume=not args.no_resume,
            think=args.think,
        )
    elif args.command == "validate-smoke":
        validate_smoke_generated_labels(args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

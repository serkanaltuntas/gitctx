"""Generate teacher labels from teacher-input artifacts through Ollama."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import urllib.error
import urllib.request
from typing import Any, Callable

from gitctx.conventional import CommitContext, parse_commit_message, score_commit_message
from gitctx.provenance import (
    load_jsonl,
    validate_generated_label_matches_input,
)
from gitctx.teacher_inputs import teacher_inputs_path

SMOKE_GENERATED_LABELS = Path("artifacts/teacher/generated-labels.smoke.jsonl")
SMOKE_GENERATED_REPORT = Path("artifacts/teacher/generated-labels.smoke.report.json")
ProgressCallback = Callable[[dict[str, Any]], None]


def generate_smoke_labels(
    data_dir: str | Path,
    *,
    ollama_url: str = "http://127.0.0.1:11434",
    limit: int | None = None,
    ollama_options: dict[str, Any] | None = None,
    progress_every: int = 25,
    request_timeout: int = 600,
    resume: bool = True,
    think: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Generate labels for smoke teacher inputs and write JSONL plus report."""

    return generate_labels(
        data_dir,
        artifact_name="smoke",
        ollama_url=ollama_url,
        limit=limit,
        ollama_options=ollama_options,
        progress_every=progress_every,
        request_timeout=request_timeout,
        resume=resume,
        think=think,
        progress_callback=progress_callback,
    )


def generate_labels(
    data_dir: str | Path,
    *,
    artifact_name: str,
    ollama_url: str = "http://127.0.0.1:11434",
    limit: int | None = None,
    ollama_options: dict[str, Any] | None = None,
    progress_every: int = 25,
    request_timeout: int = 600,
    resume: bool = True,
    think: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Generate labels for a named teacher-input artifact and write JSONL plus report."""

    _validate_artifact_name(artifact_name)
    data_dir = Path(data_dir)
    output_relative_path = generated_labels_path(artifact_name)
    report_relative_path = generated_report_path(artifact_name)
    input_path = data_dir / teacher_inputs_path(artifact_name)
    output_path = data_dir / output_relative_path
    report_path = data_dir / report_relative_path
    teacher_inputs = load_jsonl(input_path)
    if limit is not None:
        teacher_inputs = teacher_inputs[:limit]

    existing_records = _load_existing(output_path) if resume else []
    existing_ids = {record["id"] for record in existing_records}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _report_generation_progress(
        processed=0,
        total=len(teacher_inputs),
        generated=0,
        skipped=0,
        failed=0,
        force=True,
        progress_callback=progress_callback,
    )

    generated = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    with output_path.open("a" if resume else "w", encoding="utf-8") as f:
        for index, teacher_input in enumerate(teacher_inputs, 1):
            output_id = f"generated-{teacher_input['source_diff_id']}"
            if output_id in existing_ids:
                skipped += 1
                _report_generation_progress(
                    processed=index,
                    total=len(teacher_inputs),
                    generated=generated,
                    skipped=skipped,
                    failed=len(failures),
                    progress_every=progress_every,
                    current_record_id=output_id,
                    progress_callback=progress_callback,
                )
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
                f.flush()
                generated += 1
            except Exception as exc:  # noqa: BLE001 - preserve per-record failures.
                failures.append(
                    {
                        "id": output_id,
                        "source_diff_id": teacher_input.get("source_diff_id"),
                        "error": str(exc),
                    }
                )
            _report_generation_progress(
                processed=index,
                total=len(teacher_inputs),
                generated=generated,
                skipped=skipped,
                failed=len(failures),
                progress_every=progress_every,
                current_record_id=output_id,
                progress_callback=progress_callback,
            )

    report = {
        "artifact_name": artifact_name,
        "generated_records": generated,
        "input_records": len(teacher_inputs),
        "output_path": str(output_relative_path),
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

    return validate_generated_labels(data_dir, artifact_name="smoke")


def validate_generated_labels(data_dir: str | Path, *, artifact_name: str) -> dict[str, int]:
    """Validate generated-label artifacts against named teacher inputs."""

    _validate_artifact_name(artifact_name)
    data_dir = Path(data_dir)
    teacher_inputs = load_jsonl(data_dir / teacher_inputs_path(artifact_name))
    generated_labels = load_jsonl(data_dir / generated_labels_path(artifact_name))
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
        "artifact_name": artifact_name,
        "teacher_input_records": len(teacher_inputs),
        "generated_label_records": len(generated_labels),
    }
    for key, value in summary.items():
        print(key, value)
    return summary


def generated_labels_path(artifact_name: str) -> Path:
    """Return the generated-label JSONL path for a named artifact."""

    _validate_artifact_name(artifact_name)
    return Path("artifacts/teacher") / f"generated-labels.{artifact_name}.jsonl"


def generated_report_path(artifact_name: str) -> Path:
    """Return the generated-label report path for a named artifact."""

    _validate_artifact_name(artifact_name)
    return Path("artifacts/teacher") / f"generated-labels.{artifact_name}.report.json"


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def _report_generation_progress(
    *,
    processed: int,
    total: int,
    generated: int,
    skipped: int,
    failed: int,
    progress_every: int = 25,
    current_record_id: str | None = None,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> None:
    percent = 100.0 if total == 0 else processed / total * 100.0
    if not force and progress_every > 0 and processed != total and processed % progress_every != 0:
        return
    print(
        "progress "
        f"{processed}/{total} "
        f"({percent:.2f}%) "
        f"generated={generated} "
        f"skipped={skipped} "
        f"failed={failed}",
        flush=True,
    )
    if progress_callback is not None:
        event: dict[str, Any] = {
            "processed": processed,
            "total": total,
            "generated": generated,
            "skipped": skipped,
            "failed": failed,
            "percent": percent,
        }
        if current_record_id is not None:
            event["current_record_id"] = current_record_id
        progress_callback(event)


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
        return _parse_plain_commit_candidate(content)
    value = json.loads(content[first : last + 1])
    if not isinstance(value, dict):
        raise ValueError("teacher output JSON must be an object")
    return value


def _parse_plain_commit_candidate(content: str) -> dict[str, Any]:
    """Recover the common teacher failure mode: plain Conventional Commit text."""

    content = _extract_plain_commit_message(content)
    try:
        parsed = parse_commit_message(content)
    except ValueError as exc:
        raise ValueError("teacher output did not contain a JSON object") from exc
    return {
        "header": content.splitlines()[0],
        "body": list(parsed.body),
        "footers": list(parsed.footers),
        "type": parsed.type,
        "scope": parsed.scope,
        "confidence": 0.5,
        "warnings": [
            "teacher output was plain Conventional Commit text; normalized to JSON candidate"
        ],
        "evidence_paths": [],
    }


def _extract_plain_commit_message(content: str) -> str:
    """Return the first parseable Conventional Commit message from text output."""

    stripped = content.strip()
    if not stripped:
        return stripped
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, str):
        stripped = decoded.strip()

    candidates = [stripped]
    lines = stripped.splitlines()
    for index, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        candidates.append("\n".join([line, *lines[index + 1 :]]).strip())

    for candidate in candidates:
        candidate = _strip_surrounding_code_fence(candidate)
        try:
            parse_commit_message(candidate)
        except ValueError:
            continue
        return candidate
    return stripped


def _strip_surrounding_code_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?", "", stripped).strip()
    return re.sub(r"```$", "", stripped).strip()


def _build_generated_label(
    output_id: str,
    teacher_input: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    header = _require_candidate_str(candidate, "header")
    body = _require_candidate_list(candidate, "body", allow_string=True)
    footers = _require_candidate_list(candidate, "footers", allow_string=True)
    warnings = _require_candidate_list(candidate, "warnings", allow_string=True)
    evidence_paths, evidence_warnings = _sanitize_evidence_paths(
        _require_candidate_list(candidate, "evidence_paths"),
        teacher_input["changed_paths"],
    )
    warnings = [*warnings, *evidence_warnings]
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


def _require_candidate_list(
    candidate: dict[str, Any],
    key: str,
    *,
    allow_string: bool = False,
) -> list[str]:
    value = candidate.get(key)
    if allow_string and isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"teacher output {key} must be a list of strings")
    return value


def _candidate_confidence(candidate: dict[str, Any]) -> float:
    value = candidate.get("confidence")
    if not isinstance(value, (int, float)):
        raise ValueError("teacher output confidence must be numeric")
    return max(0.0, min(1.0, float(value)))


def _sanitize_evidence_paths(
    evidence_paths: list[str],
    changed_paths: list[str],
) -> tuple[list[str], list[str]]:
    changed_path_set = set(changed_paths)
    normalized_paths: list[str] = []
    dropped_paths: list[str] = []

    for path in evidence_paths:
        normalized = _normalize_evidence_path(path)
        if normalized in changed_path_set:
            if normalized not in normalized_paths:
                normalized_paths.append(normalized)
        else:
            dropped_paths.append(path)

    warnings = []
    if dropped_paths:
        warnings.append(
            "dropped evidence_paths not present in changed_paths: "
            + ", ".join(sorted(dropped_paths))
        )
    return normalized_paths, warnings


def _normalize_evidence_path(path: str) -> str:
    return re.sub(r"(#L\d+(?:-L?\d+)?|:\d+(?::\d+)?)$", "", path)


def _verifier_score(
    parser_result: dict[str, Any],
    evidence_paths: list[str],
    changed_paths: list[str],
) -> float:
    fatal_errors = [
        error
        for error in parser_result["errors"]
        if not error.startswith("scope ") or " is not visible in changed paths" not in error
    ]
    checks = [
        parser_result["format_validity"],
        parser_result["specificity"],
        parser_result["brevity"],
        not fatal_errors,
        bool(evidence_paths),
        set(evidence_paths).issubset(set(changed_paths)),
    ]
    return sum(1 for check in checks if check) / len(checks)


def _validate_artifact_name(artifact_name: str) -> None:
    if not artifact_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("artifact_name must be a stable alphanumeric identifier")


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
    generate.add_argument("--progress-every", type=int, default=25)
    generate.add_argument("--request-timeout", type=int, default=600)
    generate.add_argument("--think", action="store_true")
    subparsers.add_parser("validate-smoke")
    generate_named = subparsers.add_parser("generate")
    generate_named.add_argument("--artifact-name", required=True)
    generate_named.add_argument("--limit", type=int)
    generate_named.add_argument("--no-resume", action="store_true")
    generate_named.add_argument("--num-ctx", type=int)
    generate_named.add_argument("--num-predict", type=int)
    generate_named.add_argument("--progress-every", type=int, default=25)
    generate_named.add_argument("--request-timeout", type=int, default=600)
    generate_named.add_argument("--think", action="store_true")
    validate_named = subparsers.add_parser("validate")
    validate_named.add_argument("--artifact-name", required=True)
    args = parser.parse_args(argv)

    if args.command == "generate-smoke":
        generate_smoke_labels(
            args.data_dir,
            ollama_url=args.ollama_url,
            limit=args.limit,
            ollama_options={"num_ctx": args.num_ctx, "num_predict": args.num_predict},
            progress_every=args.progress_every,
            request_timeout=args.request_timeout,
            resume=not args.no_resume,
            think=args.think,
        )
    elif args.command == "validate-smoke":
        validate_smoke_generated_labels(args.data_dir)
    elif args.command == "generate":
        generate_labels(
            args.data_dir,
            artifact_name=args.artifact_name,
            ollama_url=args.ollama_url,
            limit=args.limit,
            ollama_options={"num_ctx": args.num_ctx, "num_predict": args.num_predict},
            progress_every=args.progress_every,
            request_timeout=args.request_timeout,
            resume=not args.no_resume,
            think=args.think,
        )
    elif args.command == "validate":
        validate_generated_labels(args.data_dir, artifact_name=args.artifact_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

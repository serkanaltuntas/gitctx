"""Build teacher prompt input artifacts from reviewed source diffs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, NamedTuple

from gitctx.provenance import load_jsonl, validate_teacher_input_record

DEFAULT_TEACHER_MODEL_ID = "ollama/qwen2.5-coder:7b"
DEFAULT_TEACHER_RUNTIME = "ollama"
DEFAULT_TEACHER_RUNTIME_MODEL_ID = "qwen2.5-coder:7b"
DEFAULT_TEACHER_REVISION = "dae161e27b0e"
DEFAULT_TEACHER_LICENSE = "Apache-2.0"
DEFAULT_TEACHER_SIZE = "4.7 GB"
DEFAULT_TEACHER_CONTEXT_LENGTH = "32K"
PROMPT_VERSION = "commit-message-teacher-v0.1"
PROMPT_PATH = Path("prompts/commit-message-teacher-v0.1.md")
DECODING_CONFIG = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 256,
}

SMOKE_SOURCE_DIFFS = Path("artifacts/smoke/source-diffs.smoke.jsonl")
SMOKE_REVIEW = Path("reviews/source-diffs.smoke.review.jsonl")
SMOKE_TEACHER_INPUTS = Path("artifacts/teacher/teacher-inputs.smoke.jsonl")


class TeacherConfig(NamedTuple):
    model_id: str
    runtime: str
    runtime_model_id: str
    revision: str
    license: str
    size: str
    context_length: str


def create_smoke_teacher_inputs(
    data_dir: str | Path,
    *,
    prompt_path: str | Path = PROMPT_PATH,
    teacher_config: TeacherConfig | None = None,
) -> Path:
    """Write teacher input prompts for accepted smoke source diffs."""

    data_dir = Path(data_dir)
    teacher_config = teacher_config or default_teacher_config()
    prompt = _load_prompt_template(prompt_path)
    source_records = load_jsonl(data_dir / SMOKE_SOURCE_DIFFS)
    review_records = load_jsonl(data_dir / SMOKE_REVIEW)
    source_by_id = {record["id"]: record for record in source_records}
    accepted_reviews = [
        record
        for record in review_records
        if record.get("decision") == "accepted_for_teacher_labeling"
    ]

    output_records = []
    for review in accepted_reviews:
        source = source_by_id[review["source_diff_id"]]
        diff = _read_git_diff(data_dir, source)
        user_message = _render_user_message(
            prompt["user_template"],
            {
                "repository": source["source_repo_url"],
                "default_branch": _default_branch(source["source_repo_url"]),
                "source_license": source["source_license"],
                "changed_paths": json.dumps(source["changed_paths"], ensure_ascii=False),
                "diff_stat": source["diff_stat"],
                "diff": diff,
            },
        )
        output_records.append(
            {
                "id": f"teacher-input-{source['id']}",
                "source_diff_id": source["id"],
                "review_decision_id": review["id"],
                "source_repo_url": source["source_repo_url"],
                "source_license": source["source_license"],
                "source_commit": source["source_commit"],
                "parent_commit": source["parent_commit"],
                "data_split": source["data_split"],
                "changed_paths": source["changed_paths"],
                "diff_stat": source["diff_stat"],
                "historical_subject": source["historical_subject"],
                "teacher_model_id": teacher_config.model_id,
                "teacher_runtime": teacher_config.runtime,
                "teacher_runtime_model_id": teacher_config.runtime_model_id,
                "teacher_revision": teacher_config.revision,
                "teacher_license": teacher_config.license,
                "teacher_size": teacher_config.size,
                "teacher_context_length": teacher_config.context_length,
                "prompt_version": PROMPT_VERSION,
                "prompt_path": str(prompt_path),
                "decoding_config": DECODING_CONFIG,
                "system_message": prompt["system_message"],
                "user_message": user_message,
                "diff": diff,
                "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
                "input_status": "ready_for_generation",
            }
        )

    output_path = data_dir / SMOKE_TEACHER_INPUTS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in output_records) + "\n",
        encoding="utf-8",
    )
    return output_path


def default_teacher_config() -> TeacherConfig:
    """Return the first local smoke teacher config."""

    return TeacherConfig(
        model_id=DEFAULT_TEACHER_MODEL_ID,
        runtime=DEFAULT_TEACHER_RUNTIME,
        runtime_model_id=DEFAULT_TEACHER_RUNTIME_MODEL_ID,
        revision=DEFAULT_TEACHER_REVISION,
        license=DEFAULT_TEACHER_LICENSE,
        size=DEFAULT_TEACHER_SIZE,
        context_length=DEFAULT_TEACHER_CONTEXT_LENGTH,
    )


def validate_smoke_teacher_inputs(data_dir: str | Path) -> dict[str, int]:
    """Validate smoke teacher input records."""

    data_dir = Path(data_dir)
    source_records = load_jsonl(data_dir / SMOKE_SOURCE_DIFFS)
    review_records = load_jsonl(data_dir / SMOKE_REVIEW)
    input_records = load_jsonl(data_dir / SMOKE_TEACHER_INPUTS)
    source_by_id = {record["id"]: record for record in source_records}
    accepted_source_ids = {
        record["source_diff_id"]
        for record in review_records
        if record.get("decision") == "accepted_for_teacher_labeling"
    }
    seen_source_ids: set[str] = set()
    failures: list[tuple[str, tuple[str, ...]]] = []

    for record in input_records:
        record_id = record.get("id", "<missing-id>")
        errors = list(validate_teacher_input_record(record))
        source_id = record.get("source_diff_id")
        source = source_by_id.get(source_id)
        if source is None:
            errors.append(f"unknown source_diff_id: {source_id}")
        elif source_id not in accepted_source_ids:
            errors.append(f"source_diff_id is not accepted for teacher labeling: {source_id}")
        else:
            for key in (
                "source_repo_url",
                "source_license",
                "source_commit",
                "parent_commit",
                "data_split",
                "changed_paths",
                "diff_stat",
                "historical_subject",
            ):
                if record.get(key) != source.get(key):
                    errors.append(f"{key} does not match source diff record")
        if isinstance(source_id, str):
            if source_id in seen_source_ids:
                errors.append(f"duplicate teacher input for source_diff_id: {source_id}")
            seen_source_ids.add(source_id)
        diff = record.get("diff")
        diff_sha256 = record.get("diff_sha256")
        if isinstance(diff, str) and isinstance(diff_sha256, str):
            actual_digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()
            if actual_digest != diff_sha256:
                errors.append("diff_sha256 does not match diff")
        if errors:
            failures.append((str(record_id), tuple(errors)))

    missing = sorted(accepted_source_ids - seen_source_ids)
    if missing:
        failures.append(("<missing-teacher-inputs>", tuple(missing)))

    if failures:
        for name, errors in failures:
            print(f"FAIL {name}: {errors}")
        raise SystemExit(1)

    summary = {
        "source_records": len(source_records),
        "accepted_review_records": len(accepted_source_ids),
        "teacher_input_records": len(input_records),
    }
    for key, value in summary.items():
        print(key, value)
    return summary


def _load_prompt_template(path: str | Path) -> dict[str, str]:
    text = Path(path).read_text(encoding="utf-8")
    system_marker = "## System Message"
    user_marker = "## User Template"
    notes_marker = "## Notes"
    if system_marker not in text or user_marker not in text or notes_marker not in text:
        raise ValueError(f"{path} does not have the expected prompt sections")
    system_section = text.split(system_marker, 1)[1].split(user_marker, 1)[0].strip()
    user_section = text.split(user_marker, 1)[1].split(notes_marker, 1)[0]
    return {
        "system_message": system_section,
        "user_template": _extract_fenced_text(user_section),
    }


def _extract_fenced_text(text: str) -> str:
    fence = "```text"
    if fence not in text:
        raise ValueError("prompt user template must contain a ```text fenced block")
    return text.split(fence, 1)[1].split("```", 1)[0].strip()


def _render_user_message(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _read_git_diff(data_dir: Path, source_record: dict[str, Any]) -> str:
    repo_dir = data_dir / _repo_cache_path(source_record["source_repo_url"])
    command = [
        "git",
        "diff",
        "--no-ext-diff",
        "--find-renames",
        source_record["parent_commit"],
        source_record["source_commit"],
        "--",
        *source_record["changed_paths"],
    ]
    return subprocess.check_output(command, cwd=repo_dir, text=True, errors="replace")


def _repo_cache_path(repo_url: str) -> Path:
    if not repo_url.startswith("https://github.com/"):
        raise ValueError(f"unsupported repo URL: {repo_url}")
    owner_repo = repo_url.removeprefix("https://github.com/")
    return Path("sources/github.com") / owner_repo


def _default_branch(repo_url: str) -> str:
    del repo_url
    return "main"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create teacher input artifacts.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, default=PROMPT_PATH)
    parser.add_argument("--teacher-model-id", default=DEFAULT_TEACHER_MODEL_ID)
    parser.add_argument("--teacher-runtime", default=DEFAULT_TEACHER_RUNTIME)
    parser.add_argument("--teacher-runtime-model-id", default=DEFAULT_TEACHER_RUNTIME_MODEL_ID)
    parser.add_argument("--teacher-revision", default=DEFAULT_TEACHER_REVISION)
    parser.add_argument("--teacher-license", default=DEFAULT_TEACHER_LICENSE)
    parser.add_argument("--teacher-size", default=DEFAULT_TEACHER_SIZE)
    parser.add_argument("--teacher-context-length", default=DEFAULT_TEACHER_CONTEXT_LENGTH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("create-smoke")
    subparsers.add_parser("validate-smoke")
    args = parser.parse_args(argv)

    if args.command == "create-smoke":
        print(
            create_smoke_teacher_inputs(
                args.data_dir,
                prompt_path=args.prompt_path,
                teacher_config=TeacherConfig(
                    model_id=args.teacher_model_id,
                    runtime=args.teacher_runtime,
                    runtime_model_id=args.teacher_runtime_model_id,
                    revision=args.teacher_revision,
                    license=args.teacher_license,
                    size=args.teacher_size,
                    context_length=args.teacher_context_length,
                ),
            )
        )
    elif args.command == "validate-smoke":
        validate_smoke_teacher_inputs(args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

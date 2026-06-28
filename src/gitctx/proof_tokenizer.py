"""Build dependency-free tokenizer artifacts for proof-model handoff."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable

from gitctx.provenance import load_jsonl
from gitctx.train_artifacts import training_examples_path

TOKENIZER_DIR = Path("artifacts/tokenizers")
DEFAULT_TOKENIZER_VERSION = "regex-diff-v0"
DEFAULT_VOCAB_SIZE = 32_000
DEFAULT_MIN_FREQUENCY = 2
DEFAULT_TRAIN_SPLIT = "DEV"
DEFAULT_EVAL_SPLIT = "REPORT"
SPECIAL_TOKENS = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<system>",
    "<user>",
    "<assistant>",
    "<nl>",
)
TOKEN_RE = re.compile(
    r"\n|[A-Za-z_][A-Za-z0-9_]*|[0-9a-f]{7,64}|\d+(?:\.\d+)?|"
    r"==|!=|<=|>=|->|=>|::|\+\+|--|&&|\|\||[^\s]"
)


def tokenizer_artifact_path(
    artifact_name: str,
    *,
    artifact_version: str = "v0",
    tokenizer_version: str = DEFAULT_TOKENIZER_VERSION,
) -> Path:
    """Return the tokenizer artifact path for a named SFT artifact."""

    _validate_identifier(artifact_name, "artifact_name")
    _validate_identifier(artifact_version, "artifact_version")
    _validate_identifier(tokenizer_version, "tokenizer_version")
    return TOKENIZER_DIR / f"{tokenizer_version}.{artifact_name}.{artifact_version}.json"


def tokenizer_report_path(
    artifact_name: str,
    *,
    artifact_version: str = "v0",
    tokenizer_version: str = DEFAULT_TOKENIZER_VERSION,
) -> Path:
    """Return the tokenizer report path for a named SFT artifact."""

    _validate_identifier(artifact_name, "artifact_name")
    _validate_identifier(artifact_version, "artifact_version")
    _validate_identifier(tokenizer_version, "tokenizer_version")
    return TOKENIZER_DIR / f"{tokenizer_version}.{artifact_name}.{artifact_version}.report.json"


def tokenize_text(text: str) -> list[str]:
    """Tokenize text with the proof tokenizer regex."""

    tokens = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        tokens.append("<nl>" if token == "\n" else token)
    return tokens


def build_proof_tokenizer(
    data_dir: str | Path,
    *,
    artifact_name: str,
    artifact_version: str = "v0",
    tokenizer_version: str = DEFAULT_TOKENIZER_VERSION,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    train_split: str = DEFAULT_TRAIN_SPLIT,
    eval_split: str = DEFAULT_EVAL_SPLIT,
) -> dict[str, Any]:
    """Build a deterministic tokenizer vocabulary from reviewed DEV records."""

    _validate_identifier(artifact_name, "artifact_name")
    _validate_identifier(artifact_version, "artifact_version")
    _validate_identifier(tokenizer_version, "tokenizer_version")
    _validate_split(train_split)
    _validate_split(eval_split)
    if train_split == eval_split:
        raise ValueError("train_split and eval_split must be distinct")
    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError("vocab_size must be larger than the special-token count")
    if min_frequency < 1:
        raise ValueError("min_frequency must be positive")

    data_dir = Path(data_dir)
    records_path = data_dir / training_examples_path(artifact_name, version=artifact_version)
    records = load_jsonl(records_path)
    token_counts: Counter[str] = Counter()
    split_record_counts: Counter[str] = Counter()
    split_token_counts: Counter[str] = Counter()

    for record in records:
        split = record["data_split"]
        split_record_counts[split] += 1
        tokens = record_tokens(record)
        split_token_counts[split] += len(tokens)
        if split == train_split:
            token_counts.update(tokens)

    if split_record_counts[train_split] <= 0:
        raise ValueError(f"no training records found for split {train_split}")
    if split_record_counts[eval_split] <= 0:
        raise ValueError(f"no evaluation records found for split {eval_split}")

    vocab_tokens = list(SPECIAL_TOKENS)
    for token, count in sorted(token_counts.items(), key=lambda item: (-item[1], item[0])):
        if token in SPECIAL_TOKENS or count < min_frequency:
            continue
        vocab_tokens.append(token)
        if len(vocab_tokens) >= vocab_size:
            break

    token_to_id = {token: index for index, token in enumerate(vocab_tokens)}
    coverage = {
        split: _coverage_for_split(records, split=split, token_to_id=token_to_id)
        for split in (train_split, eval_split)
    }
    artifact = {
        "tokenizer_kind": "dependency_free_regex_diff_frequency_tokenizer",
        "tokenizer_version": tokenizer_version,
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "intended_use": "GCTX-1 proof-model training handoff; not a final public tokenizer claim",
        "train_split": train_split,
        "eval_split": eval_split,
        "vocab_size": len(vocab_tokens),
        "requested_vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": {token: token_to_id[token] for token in SPECIAL_TOKENS},
        "normalization": "case-sensitive regex tokenization; newline becomes <nl>",
        "input_format": "chat messages in role order from reviewed SFT records",
        "vocab": [
            {
                "id": index,
                "token": token,
                "count": token_counts.get(token, 0),
            }
            for index, token in enumerate(vocab_tokens)
        ],
    }
    report = {
        "tokenizer_version": tokenizer_version,
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "training_records": len(records),
        "split_record_counts": {
            split: split_record_counts[split] for split in sorted(split_record_counts)
        },
        "split_token_counts": {
            split: split_token_counts[split] for split in sorted(split_token_counts)
        },
        "coverage": coverage,
        "top_train_tokens": [
            {"token": token, "count": count}
            for token, count in sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))[:50]
        ],
        "output_path": str(tokenizer_artifact_path(
            artifact_name,
            artifact_version=artifact_version,
            tokenizer_version=tokenizer_version,
        )),
        "report_path": str(tokenizer_report_path(
            artifact_name,
            artifact_version=artifact_version,
            tokenizer_version=tokenizer_version,
        )),
    }

    output_path = data_dir / tokenizer_artifact_path(
        artifact_name,
        artifact_version=artifact_version,
        tokenizer_version=tokenizer_version,
    )
    report_path = data_dir / tokenizer_report_path(
        artifact_name,
        artifact_version=artifact_version,
        tokenizer_version=tokenizer_version,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_report(report, artifact)
    return report


def validate_proof_tokenizer(
    data_dir: str | Path,
    *,
    artifact_name: str,
    artifact_version: str = "v0",
    tokenizer_version: str = DEFAULT_TOKENIZER_VERSION,
    min_eval_coverage: float = 0.95,
) -> dict[str, Any]:
    """Validate a proof tokenizer artifact and coverage report."""

    if not 0 <= min_eval_coverage <= 1:
        raise ValueError("min_eval_coverage must be between 0 and 1")
    data_dir = Path(data_dir)
    artifact_path = data_dir / tokenizer_artifact_path(
        artifact_name,
        artifact_version=artifact_version,
        tokenizer_version=tokenizer_version,
    )
    report_path = data_dir / tokenizer_report_path(
        artifact_name,
        artifact_version=artifact_version,
        tokenizer_version=tokenizer_version,
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if artifact.get("tokenizer_kind") != "dependency_free_regex_diff_frequency_tokenizer":
        errors.append("unexpected tokenizer_kind")
    if artifact.get("tokenizer_version") != tokenizer_version:
        errors.append("tokenizer_version mismatch")
    if artifact.get("artifact_name") != artifact_name:
        errors.append("artifact_name mismatch")
    if artifact.get("artifact_version") != artifact_version:
        errors.append("artifact_version mismatch")
    vocab = artifact.get("vocab")
    if not isinstance(vocab, list) or len(vocab) != artifact.get("vocab_size"):
        errors.append("vocab_size does not match vocab length")
    else:
        _validate_vocab(vocab, errors)
    special_tokens = artifact.get("special_tokens")
    if not isinstance(special_tokens, dict):
        errors.append("special_tokens must be an object")
    else:
        for expected_id, token in enumerate(SPECIAL_TOKENS):
            if special_tokens.get(token) != expected_id:
                errors.append(f"special token {token} must have id {expected_id}")
    eval_split = artifact.get("eval_split")
    coverage = report.get("coverage", {})
    if not isinstance(eval_split, str) or not isinstance(coverage, dict):
        errors.append("coverage report is malformed")
    else:
        eval_coverage = coverage.get(eval_split, {}).get("known_token_fraction")
        if not isinstance(eval_coverage, (int, float)):
            errors.append(f"coverage for {eval_split} is missing")
        elif eval_coverage < min_eval_coverage:
            errors.append(f"coverage for {eval_split} is below {min_eval_coverage}")
    validation = {
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "tokenizer_version": tokenizer_version,
        "valid": not errors,
        "errors": errors,
        "artifact_path": str(tokenizer_artifact_path(
            artifact_name,
            artifact_version=artifact_version,
            tokenizer_version=tokenizer_version,
        )),
        "report_path": str(tokenizer_report_path(
            artifact_name,
            artifact_version=artifact_version,
            tokenizer_version=tokenizer_version,
        )),
    }
    for key, value in validation.items():
        if key != "errors":
            print(key, value)
    for error in errors:
        print("error", error)
    if errors:
        raise SystemExit(1)
    return validation


def _validate_vocab(vocab: list[Any], errors: list[str]) -> None:
    seen_tokens: set[str] = set()
    for expected_id, entry in enumerate(vocab):
        if not isinstance(entry, dict):
            errors.append(f"vocab entry {expected_id} must be an object")
            continue
        if entry.get("id") != expected_id:
            errors.append(f"vocab entry {expected_id} has a non-contiguous id")
        token = entry.get("token")
        if not isinstance(token, str) or not token:
            errors.append(f"vocab entry {expected_id} has an invalid token")
        elif token in seen_tokens:
            errors.append(f"duplicate vocab token: {token}")
        else:
            seen_tokens.add(token)


def record_tokens(record: dict[str, Any]) -> list[str]:
    """Return tokenizer tokens for one reviewed SFT record."""

    tokens: list[str] = ["<bos>"]
    for message in record.get("messages", []):
        role = message.get("role")
        if role in {"system", "user", "assistant"}:
            tokens.append(f"<{role}>")
        content = message.get("content")
        if isinstance(content, str):
            tokens.extend(tokenize_text(content))
        tokens.append("<sep>")
    tokens.append("<eos>")
    return tokens


def _coverage_for_split(
    records: Iterable[dict[str, Any]],
    *,
    split: str,
    token_to_id: dict[str, int],
) -> dict[str, Any]:
    total = 0
    unknown = 0
    records_seen = 0
    for record in records:
        if record["data_split"] != split:
            continue
        records_seen += 1
        for token in record_tokens(record):
            total += 1
            if token not in token_to_id:
                unknown += 1
    known = total - unknown
    return {
        "records": records_seen,
        "tokens": total,
        "known_tokens": known,
        "unknown_tokens": unknown,
        "known_token_fraction": round(known / total, 6) if total else 0.0,
        "unknown_token_fraction": round(unknown / total, 6) if total else 0.0,
    }


def _print_report(report: dict[str, Any], artifact: dict[str, Any]) -> None:
    print("tokenizer_version", artifact["tokenizer_version"])
    print("artifact_name", artifact["artifact_name"])
    print("artifact_version", artifact["artifact_version"])
    print("vocab_size", artifact["vocab_size"])
    for split, coverage in report["coverage"].items():
        print(f"{split}_records", coverage["records"])
        print(f"{split}_known_token_fraction", coverage["known_token_fraction"])
    print("output_path", report["output_path"])
    print("report_path", report["report_path"])


def _validate_split(split: str) -> None:
    if split not in {"DEV", "REPORT", "HELD_OUT"}:
        raise ValueError("split must be DEV, REPORT, or HELD_OUT")


def _validate_identifier(value: str, name: str) -> None:
    if not value.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"{name} must be a stable alphanumeric identifier")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate a GCTX proof tokenizer.")
    parser.add_argument("--data-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--artifact-name", required=True)
    build.add_argument("--artifact-version", default="v0")
    build.add_argument("--tokenizer-version", default=DEFAULT_TOKENIZER_VERSION)
    build.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    build.add_argument("--min-frequency", type=int, default=DEFAULT_MIN_FREQUENCY)
    build.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    build.add_argument("--eval-split", default=DEFAULT_EVAL_SPLIT)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--artifact-name", required=True)
    validate.add_argument("--artifact-version", default="v0")
    validate.add_argument("--tokenizer-version", default=DEFAULT_TOKENIZER_VERSION)
    validate.add_argument("--min-eval-coverage", type=float, default=0.95)
    args = parser.parse_args(argv)

    if args.command == "build":
        build_proof_tokenizer(
            args.data_dir,
            artifact_name=args.artifact_name,
            artifact_version=args.artifact_version,
            tokenizer_version=args.tokenizer_version,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            train_split=args.train_split,
            eval_split=args.eval_split,
        )
    elif args.command == "validate":
        validate_proof_tokenizer(
            args.data_dir,
            artifact_name=args.artifact_name,
            artifact_version=args.artifact_version,
            tokenizer_version=args.tokenizer_version,
            min_eval_coverage=args.min_eval_coverage,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

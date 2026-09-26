"""Target-independent prompt budget and explicit diff boundary accounting."""
from __future__ import annotations

import hashlib

from gitctx.student_input import student_messages
from gitctx.student_tokenizer import SPECIAL, StudentTokenizer

CONTEXT = 8192
ANSWER_RESERVE = 256  # Includes EOS; never derived from reference length.


def prompt_ids(record: dict, tokenizer: StudentTokenizer) -> list[int]:
    ids = [SPECIAL["<bos>"]]
    for message in student_messages(record):
        ids.append(SPECIAL[f"<{message['role']}>"])
        ids.extend(tokenizer.encode(message["content"]))
        ids.append(SPECIAL["<sep>"])
    return ids + [SPECIAL["<assistant>"]]


def materialize(record: dict, tokenizer: StudentTokenizer, *,
                context: int = CONTEXT, reserve: int = ANSWER_RESERVE) -> dict:
    """Reject overflow, never truncate a prompt or silently discard a target."""
    if not 1 < reserve < context:
        raise ValueError("invalid fixed context/answer budget")
    prompt = prompt_ids(record, tokenizer)
    if len(prompt) > context - reserve:
        raise ValueError("prompt overflow requires an explicit evidence/window policy")
    target = tokenizer.encode(record["target_message"]) + [SPECIAL["<eos>"]]
    if len(target) > reserve:
        raise ValueError("target overflow requires a common answer policy revision")
    return {"record_id": record["id"], "input_ids": prompt + target,
            "loss_mask": [0] * len(prompt) + [1] * len(target),
            "prompt_length": len(prompt), "loss_tokens": len(target)}


def diff_units(diff: str) -> list[dict]:
    """Partition every original line into file headers and complete hunks.

    Indices are zero-based, half-open; no reconstruction or newline normalization.
    Metadata/binary chunks stay explicit rather than masquerading as text hunks.
    """
    lines = physical_lines(diff)
    units = []
    start, file, kind = 0, None, "preamble"
    for i, line in enumerate(lines):
        if line.startswith("diff --git ") or line.startswith("@@ "):
            if i > start:
                units.append(_unit(lines, start, i, file, kind))
            start = i
            if line.startswith("diff --git "):
                file, kind = line.rstrip("\r\n"), "file_header"
            else:
                kind = "hunk"
        # A malformed/non-unified patch is still preserved and accounted for.
    if len(lines) > start:
        units.append(_unit(lines, start, len(lines), file, kind))
    return units


def _unit(lines, start, end, file, kind):
    changed = [i for i in range(start, end) if kind == "hunk"
               and lines[i].startswith(("+", "-"))]
    text = "".join(lines[start:end])
    return {"start_line": start, "end_line": end, "file": file, "kind": kind,
            "changed_line_indices": changed,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def physical_lines(text: str) -> list[str]:
    """Git patch line numbers follow LF, not Unicode text separators."""
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def inspect_budget(record: dict, tokenizer: StudentTokenizer, *,
                   context: int = CONTEXT, reserve: int = ANSWER_RESERVE) -> dict:
    prompt = prompt_ids(record, tokenizer)
    target = tokenizer.encode(record["target_message"])
    units = diff_units(record["diff"])
    prompt_overflow = len(prompt) > context - reserve
    return {
        "record_id": record["id"], "prompt_tokens": len(prompt),
        "target_tokens_including_eos": len(target) + 1,
        "prompt_overflow": prompt_overflow, "target_overflow": len(target) + 1 > reserve,
        "context_tokens": context, "answer_reserve": reserve,
        "decision": "requires_policy" if prompt_overflow or len(target) + 1 > reserve
                    else "fits_full",
        "prompt_sha256": hashlib.sha256(bytes(str(prompt), "ascii")).hexdigest(),
        "diff_sha256": hashlib.sha256(record["diff"].encode("utf-8")).hexdigest(),
        "diff_lines": len(physical_lines(record["diff"])),
        "changed_lines": sum(len(u["changed_line_indices"]) for u in units),
        "removed_changed_lines": 0, "removed_lines": 0,
        "crop_policy": "none; full evidence retained; overflow blocks training",
        "boundaries": units,
    }

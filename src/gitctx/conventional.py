"""Conventional Commit parsing and first-pass scoring.

This module intentionally stays small and dependency-free. It is the first
measurable core for gitctx: before training any model, we need deterministic
rules that say whether a proposed commit message is parseable and minimally
grounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

HEADER_RE = re.compile(
    r"^(?P<type>[a-z][a-z0-9-]*)(?:\((?P<scope>[^()\r\n]+)\))?"
    r"(?P<breaking>!)?: (?P<subject>[^\r\n]+)$"
)

DEFAULT_TYPES = frozenset(
    {
        "build",
        "chore",
        "ci",
        "docs",
        "feat",
        "fix",
        "perf",
        "refactor",
        "revert",
        "style",
        "test",
    }
)

VAGUE_SUBJECTS = frozenset(
    {
        "change files",
        "changes",
        "fix",
        "fix bug",
        "fix issue",
        "misc",
        "update",
        "update files",
        "updates",
        "wip",
        "work in progress",
    }
)


@dataclass(frozen=True)
class ParsedCommit:
    """Parsed Conventional Commit message."""

    raw: str
    type: str
    scope: str | None
    subject: str
    body: tuple[str, ...] = ()
    footers: tuple[str, ...] = ()
    breaking: bool = False


@dataclass(frozen=True)
class CommitContext:
    """Minimal context available to the scorer."""

    changed_paths: tuple[str, ...] = ()
    expected_type: str | None = None
    expected_scope: str | None = None
    expect_mixed_change_warning: bool = False
    expect_breaking_change: bool = False
    forbidden_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommitScore:
    """Deterministic score signals for one candidate message."""

    format_validity: bool
    type_accuracy: bool | None
    scope_quality: bool | None
    factuality: bool | None
    specificity: bool
    brevity: bool
    mixed_change_warning: bool | None
    breaking_change_detection: bool | None
    parsed: ParsedCommit | None = None
    errors: tuple[str, ...] = field(default_factory=tuple)


def parse_commit_message(message: str) -> ParsedCommit:
    """Parse a Conventional Commit message.

    The parser follows the common header shape:

    ``type(scope)!: subject``

    Body and footers are preserved as lines. This is deliberately strict for the
    first scorer: invalid format should be visible early.
    """

    normalized = message.strip("\n")
    lines = tuple(normalized.splitlines())
    if not lines:
        raise ValueError("empty commit message")

    match = HEADER_RE.match(lines[0])
    if match is None:
        raise ValueError("invalid Conventional Commit header")

    body_lines: list[str] = []
    footer_lines: list[str] = []
    in_footer = False

    for line in lines[1:]:
        if _is_footer_line(line):
            in_footer = True
        if in_footer:
            footer_lines.append(line)
        else:
            body_lines.append(line)

    breaking = bool(match.group("breaking")) or any(
        line.startswith("BREAKING CHANGE:") for line in footer_lines
    )
    return ParsedCommit(
        raw=normalized,
        type=match.group("type"),
        scope=match.group("scope"),
        subject=match.group("subject").strip(),
        body=tuple(body_lines),
        footers=tuple(footer_lines),
        breaking=breaking,
    )


def score_commit_message(
    message: str,
    context: CommitContext | None = None,
    *,
    allowed_types: Iterable[str] = DEFAULT_TYPES,
    max_subject_length: int = 72,
) -> CommitScore:
    """Score a candidate commit message against deterministic signals."""

    context = context or CommitContext()
    errors: list[str] = []
    allowed_type_set = set(allowed_types)

    try:
        parsed = parse_commit_message(message)
    except ValueError as exc:
        return CommitScore(
            format_validity=False,
            type_accuracy=False if context.expected_type else None,
            scope_quality=False if context.expected_scope else None,
            factuality=False if context.forbidden_claims else None,
            specificity=False,
            brevity=False,
            mixed_change_warning=False if context.expect_mixed_change_warning else None,
            breaking_change_detection=False if context.expect_breaking_change else None,
            errors=(str(exc),),
        )

    if parsed.type not in allowed_type_set:
        errors.append(f"unknown type: {parsed.type}")

    type_accuracy = None
    if context.expected_type is not None:
        type_accuracy = parsed.type == context.expected_type
        if not type_accuracy:
            errors.append(f"expected type {context.expected_type!r}, got {parsed.type!r}")

    scope_quality = _score_scope(parsed, context, errors)
    factuality = _score_factuality(parsed, context, errors)
    specificity = _is_specific(parsed.subject)
    if not specificity:
        errors.append("subject is too vague")

    brevity = 0 < len(parsed.subject) <= max_subject_length
    if not brevity:
        errors.append("subject length is outside the configured limit")

    mixed_change_warning = None
    if context.expect_mixed_change_warning:
        mixed_change_warning = _mentions_mixed_change(parsed)
        if not mixed_change_warning:
            errors.append("expected mixed-change warning")

    breaking_change_detection = None
    if context.expect_breaking_change:
        breaking_change_detection = parsed.breaking
        if not breaking_change_detection:
            errors.append("expected breaking-change marker")

    if parsed.type not in allowed_type_set:
        format_validity = False
    else:
        format_validity = True

    return CommitScore(
        format_validity=format_validity,
        type_accuracy=type_accuracy,
        scope_quality=scope_quality,
        factuality=factuality,
        specificity=specificity,
        brevity=brevity,
        mixed_change_warning=mixed_change_warning,
        breaking_change_detection=breaking_change_detection,
        parsed=parsed,
        errors=tuple(errors),
    )


def _is_footer_line(line: str) -> bool:
    return bool(re.match(r"^(BREAKING CHANGE|[A-Za-z-]+)(: | #).+", line))


def _score_scope(parsed: ParsedCommit, context: CommitContext, errors: list[str]) -> bool | None:
    if context.expected_scope is not None:
        ok = parsed.scope == context.expected_scope
        if not ok:
            errors.append(f"expected scope {context.expected_scope!r}, got {parsed.scope!r}")
        return ok

    if not context.changed_paths or parsed.scope is None:
        return None

    path_parts = {
        part
        for path in context.changed_paths
        for part in path.replace("\\", "/").split("/")
        if part and "." not in part
    }
    ok = parsed.scope in path_parts
    if not ok:
        errors.append(f"scope {parsed.scope!r} is not visible in changed paths")
    return ok


def _score_factuality(parsed: ParsedCommit, context: CommitContext, errors: list[str]) -> bool | None:
    if not context.forbidden_claims:
        return None

    haystack = parsed.raw.lower()
    forbidden_hits = [claim for claim in context.forbidden_claims if claim.lower() in haystack]
    if forbidden_hits:
        errors.append("message contains claims not grounded in the fixture")
        return False
    return True


def _is_specific(subject: str) -> bool:
    normalized = re.sub(r"\s+", " ", subject.strip().lower())
    if normalized in VAGUE_SUBJECTS:
        return False
    return len(normalized.split()) >= 2


def _mentions_mixed_change(parsed: ParsedCommit) -> bool:
    text = parsed.raw.lower()
    return "mixed" in text or "split" in text or "separate commit" in text

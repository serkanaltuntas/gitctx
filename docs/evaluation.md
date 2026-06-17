# Evaluation Plan

gitctx needs repo-shaped evaluations, not general language benchmarks.

## Splits

- `DEV`: fast iteration, inspectable.
- `REPORT`: public progress reporting.
- `HELD_OUT`: private until release claims are made.

## Metrics

| Metric | Meaning |
|---|---|
| Format validity | Parses as Conventional Commit |
| Type accuracy | Correct `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, or `chore` |
| Scope quality | Scope matches touched module when one is clear |
| Factuality | Message does not claim changes absent from the diff |
| Specificity | Avoids vague subjects like `update files` |
| Brevity | Subject is concise and editor-friendly |
| Mixed-change warning | Detects staged diffs that should be split |
| Breaking-change detection | Flags API, schema, or contract breaks |
| Human accept rate | Human accepts directly or with light edit |
| Latency | Time to suggestion on small, medium, and large diffs |

Exact match against historical commit messages is not enough. Many different
messages can correctly describe the same diff.

## First Scorer API

The first deterministic scorer lives in `src/gitctx/conventional.py`.

Core functions:

- `parse_commit_message(message)` parses a strict Conventional Commit header.
- `score_commit_message(message, context)` returns deterministic signals for
  format validity, type accuracy, scope quality, factuality, specificity,
  brevity, mixed-change warnings, and breaking-change markers.

The first fixture format is JSONL:

```json
{
  "id": "valid_scoped_fix",
  "split": "DEV",
  "message": "fix(parser): reject malformed headers",
  "context": {
    "changed_paths": ["src/gitctx/parser.py"],
    "expected_type": "fix",
    "expected_scope": "parser"
  },
  "expected": {
    "format_validity": true,
    "type_accuracy": true,
    "scope_quality": true,
    "specificity": true,
    "brevity": true
  }
}
```

The initial fixture file is `fixtures/dev/commit_message_cases.jsonl`.

## Release Gates

| Gate | Requirement |
|---|---|
| GCTX-A | Parser, scorer, and template baseline work on fixtures |
| GCTX-B | Public eval fixtures and split policy exist |
| GCTX-C | Data pipeline emits licensed examples and data cards |
| GCTX-D | Teacher-label pipeline records complete provenance |
| GCTX-E | 60M-100M model beats the template baseline on `REPORT` |
| GCTX-F | 150M-300M model reaches public-beta threshold |
| GCTX-G | Model card, data card, eval card, and held-out pass are signed |

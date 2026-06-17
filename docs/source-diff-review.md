# Source Diff Review Decisions

gitctx stores source extraction output separately from review decisions.
Extraction artifacts should stay reproducible and append-only; human or agent
judgment lives in review JSONL files.

The first smoke review artifact is:

```text
reviews/source-diffs.smoke.review.jsonl
```

Each record maps to exactly one `source-diffs.smoke.jsonl` record by
`source_diff_id`.

Allowed decisions:

- `needs_review`: default template state; no teacher-label generation yet.
- `accepted_for_teacher_labeling`: eligible for the first teacher-label audit.
- `rejected`: not eligible for teacher-label generation.

Acceptance guidance:

- prefer small, coherent commits with clear intent;
- prefer changes with useful tests or obvious behavioral evidence;
- prefer source code changes over vendored/generated/bulk formatting changes;
- reject ambiguous mixed-purpose commits unless the intent is still clear;
- reject commits where the historical subject suggests revert, merge-only,
  release bookkeeping, generated files, dependency bumps, or formatting-only
  churn.

The review decision does not approve public redistribution. It only decides
whether a source diff is eligible for the private teacher-label audit.

Create the template:

```bash
make smoke-review-template REVIEWER="reviewer@example.com"
```

Validate it:

```bash
make smoke-review-check
```

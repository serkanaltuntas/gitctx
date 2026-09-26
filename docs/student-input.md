# Student input preparation

`commit-student-v1` renders plain Conventional Commit instructions from repository,
changed paths and the unmodified diff. It never copies teacher JSON instructions,
historical commit subjects, labels or review notes into the prompt. Both training
materialization and inference use `student_sequences.prompt_ids`.

The content tokenizer is a locally fitted byte-level BPE using pinned
`tokenizers==0.22.1` (Apache-2.0). Its complete 256-byte alphabet, absent normalizer,
and byte decoder preserve UTF-8 and whitespace. The 32,000 model IDs comprise
31,993 content IDs and seven structural IDs. Structural tokens are inserted as
integers outside content encoding: literal `<assistant>` or `<eos>` strings in
code cannot terminate a message. Vocabulary/merges use frozen DEV training IDs
only; validation and reserved examples are audit inputs, never fitting inputs.
Before byte pre-tokenization, input is split into at most 1,024 Unicode characters
per piece, including whitespace. This bounds pathological minified/generated
lines without dropping characters; fitting and encoding share this rule.
Decoding uses a cached vocabulary bound, joins cached byte pieces once, then
decodes strict UTF-8. It never decodes partial code points separately.
See the [upstream ByteLevel API](https://huggingface.co/docs/tokenizers/v0.20.3/api/pre-tokenizers#tokenizers.pre_tokenizers.ByteLevel).

The prompt budget is always 8,192 minus 256 tokens; the answer reserve includes
EOS. The reference length cannot expand the prompt budget. Prompt tokens have
zero loss; target tokens and EOS have loss, shifted once by the existing trainer
batch adapter. Overflow raises an error. This preparation does not wire the new
tokenizer into legacy job files: new weights, an explicit compatible training
job and a fixed protocol are required before training.

## Reproducible CPU preparation

Use a frozen repository-isolated DEV ordering protocol and diagnostic findings:

```bash
uv run --locked python -m gitctx.student_readiness prepare \
  --data-dir "$DATA_DIR" --parent "$PARENT_RUN" --run "$NEW_RUN" \
  --findings "artifacts/train-runs/$DIAGNOSTIC_RUN/review-findings.jsonl"
uv run --locked python -m gitctx.student_readiness verify \
  --data-dir "$DATA_DIR" --run "$NEW_RUN"
```

Existing output directories are refused. `verify` checks frozen source/output,
code and dependency hashes, reloads the tokenizer and recomputes every report.
Add `--require-ready` to fail with exit 2 when training blockers remain. Normal
verification success means the preparation is reproducible, not training-ready.

Outputs include tokenizer, hash/lineage manifest, per-record coverage and
provenance, a pending review queue, and `readiness.json`. They belong alongside
the source data under its existing access and redistribution restrictions. No
network inference, label generation, GPU training or dataset release is performed.

## Coverage and reference gates

Every original DEV ID is accounted for, including reserved repositories and IDs
excluded by the old length policy. Full diffs are retained; no training subset is
silently promoted. Coverage gives prompt/target lengths, overflow IDs, full-file
headers and hunk boundaries, source line intervals, changed-line indices and
content hashes. Removed-line counts are zero because overflow is blocked rather
than cropped. These are source line counts, not claims of semantic coverage.

For overflowing evidence, a subsequent window policy must keep file headers and
complete hunks where possible, record every source line assigned to every window,
and explicitly handle individual hunks or metadata larger than the budget.
Repeating the whole-commit target on each partial window is not a valid default:
each target claim must have supporting evidence in that window, or an explicitly
designed aggregation stage must see all relevant windows. Freeze aligned targets,
the complete ID/window manifest and common answer budget before training. Until
then the overflow IDs remain blocked, including previously excluded IDs.

New SFT artifacts separate automatic parser acceptance, unspecified legacy review,
and independently documented human review. An email/accept flag alone is never
human factuality evidence. Explicit human review requires method, kind, reviewer,
timestamp and an evidence reference. Legacy artifacts remain readable without
retroactively changing their labels or claiming factual verification.

Diagnostic assistant findings are review requests, never replacement training
labels. Resolve them through a license-approved open teacher plus verification,
or explicit human corrections. Review validation references independently, keep
original references/scores and any corrected version side by side, and do not
remove difficult examples. The current source/teacher license recipe remains in
force; preparation does not authorize redistribution of a fitted tokenizer or
its training data.

Tests cover unseen Unicode, whitespace/control characters, literal role markers,
train-only fitting, persistence, target-independent budgets, overflow rejection,
long-sequence masks/causal shift/padding and review provenance.

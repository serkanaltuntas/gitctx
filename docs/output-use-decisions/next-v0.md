# Output-Use Decision: Next v0

Status: approved for private pipeline, eval, and model-artifact validation; not
approved for public dataset or model release.
Date: 2026-06-21.

## Decision

The `next-v0` reviewed SFT artifact may be used for private development of the
gitctx data, evaluation, teacher-labeling, review, and model-artifact pipeline.

It may also be used for dependency-free smoke training that validates the data
flow from reviewed examples into a model artifact and report.

It must not be published as a full example dataset and must not be used to make
public model-quality claims.

## Rationale

The artifact improves on `pilot-v0` in three concrete ways:

- it expands from 114 to 335 promoted training records;
- it includes a locked `REPORT` split with 33 records;
- it has passed baseline evaluation, record-level `REPORT` inspection, and a
  dependency-free training smoke.

The artifact still includes full source diffs from upstream repositories and has
no `HELD_OUT` split. It is therefore useful for private pipeline validation and
early proof-model planning, but insufficient for public dataset release or
quality claims.

## Allowed Uses

- Run deterministic baseline/eval reports.
- Compare reviewed targets, raw teacher labels, and historical commit subjects.
- Debug training record format, packing, tokenization, loader behavior, and
  model-artifact creation.
- Train throwaway local smoke models whose outputs are not released as a model
  claim.
- Publish aggregate counts, this decision, and the matching data card.

## Disallowed Uses

- Publish full JSONL records.
- Publish a model trained only on this artifact as a useful gitctx release.
- Mix this artifact into a public training corpus without a follow-up release
  review.
- Add records to `REPORT` or `HELD_OUT` after teacher generation.
- Treat historical commit subjects as ground truth.
- Treat the dependency-free training smoke as evidence of neural model quality.

## Next Review Gate

Before moving from `next-v0` to a release-track GCTX-1 proof model:

1. Decide whether the next implementation milestone is tiny neural training or
   another data-expansion pass.
2. Create a larger reviewed artifact with a locked `REPORT` split.
3. Reserve a real `HELD_OUT` split before teacher generation.
4. Complete redistribution review for any examples intended for public release.
5. Write an updated data card and output-use decision for the new artifact.

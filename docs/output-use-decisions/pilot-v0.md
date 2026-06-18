# Output-Use Decision: Pilot v0

Status: approved for private pipeline/eval use; not approved for public dataset
release.
Date: 2026-06-18.

## Decision

The `pilot-v0` reviewed SFT artifact may be used for private development of the
gitctx data, evaluation, and proof-model pipeline.

It must not be published as a full example dataset and must not be used to make
public model-quality claims.

## Rationale

The artifact has complete provenance for the current pilot:

- permissive source manifest;
- source-diff review;
- local/open teacher metadata;
- prompt version;
- generated-label validation;
- human `accept` / `edit` review;
- reviewed SFT target records.

However, it includes full source diffs from upstream repositories and only
contains `DEV` records. That makes it useful for internal pipeline validation
but insufficient for public release or model-quality claims.

## Allowed Uses

- Run deterministic baseline/eval reports.
- Compare reviewed targets, raw teacher labels, and historical commit subjects.
- Debug training record format, packing, tokenization, and loader behavior.
- Train throwaway local smoke models whose outputs are not released as a model
  claim.
- Publish aggregate counts and this card.

## Disallowed Uses

- Publish full JSONL records.
- Publish a model trained only on this artifact as a useful gitctx release.
- Mix this artifact into a public training corpus without a follow-up release
  review.
- Add records to `REPORT` or `HELD_OUT` after teacher generation.
- Treat historical commit subjects as ground truth.

## Next Review Gate

Before moving from `pilot-v0` to a real GCTX-1 proof model:

1. Produce the baseline/eval report.
2. Define split rules for `DEV`, `REPORT`, and `HELD_OUT`.
3. Expand the source manifest and artifact volume.
4. Write an updated data card for the larger artifact.
5. Decide which full examples, if any, can be redistributed.

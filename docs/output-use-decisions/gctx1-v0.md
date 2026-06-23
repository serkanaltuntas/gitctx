# Output-Use Decision: GCTX-1 v0

Status: approved for private proof-model training and pipeline validation; not
approved for public dataset or model release.
Date: 2026-06-23.

## Decision

The `gctx1-v0` reviewed SFT artifact may be used for private development of the
gitctx data, evaluation, training, and proof-model pipeline.

It may also be used to train private experimental small language models, provided
the outputs are described as experiments and not as public release-quality
models.

It must not be published as a full example dataset and must not be used to make
public model-quality claims.

## Rationale

The artifact improves on `next-v0` in concrete ways:

- it expands from 335 to 6,727 promoted training records;
- it includes 1,129 locked `REPORT` records;
- it has passed generated-label validation with six recorded missing labels;
- it has passed deterministic generated-label review policy with zero
  `needs_review` records;
- it has passed baseline evaluation, record-level `REPORT` inspection,
  dependency-free prototype smoke, and tiny-softmax neural smoke.

The artifact still includes full source diffs from upstream repositories and has
no promoted `HELD_OUT` split. The generated-label review policy is conservative
and deterministic, not a full manual quality review. It is therefore suitable
for private proof-model development, but insufficient for public dataset release
or model-quality claims.

## Allowed Uses

- Run deterministic baseline/eval reports.
- Compare reviewed targets, raw teacher labels, and historical commit subjects.
- Train private proof models and smoke models.
- Debug training record format, packing, tokenization, loader behavior, and
  model-artifact creation.
- Iterate prompt, review policy, tokenizer, and model-training code.
- Publish aggregate counts, this decision, and the matching data card.

## Disallowed Uses

- Publish full JSONL records.
- Publish a model trained only on this artifact as a useful gitctx release.
- Mix this artifact into a public training corpus without a follow-up release
  review.
- Add records to `REPORT` or `HELD_OUT` after teacher generation.
- Treat historical commit subjects as ground truth.
- Treat the dependency-free prototype or tiny-softmax smoke model as evidence of
  useful model quality.
- Claim `HELD_OUT` generalization.

## Next Review Gate

Before moving from `gctx1-v0` to a release-track proof model:

1. Decide whether GCTX-1 remains a smaller proof run or requires another data
   expansion pass.
2. Train a real small language-model configuration.
3. Create or reserve a separate `HELD_OUT` artifact before release claims.
4. Run model evals against locked `REPORT` and `HELD_OUT` splits.
5. Complete redistribution review for any examples intended for public release.
6. Write a model card, eval card, release license decision, and updated data
   card.

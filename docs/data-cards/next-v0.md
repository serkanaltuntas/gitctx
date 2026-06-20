# Data Card: gitctx Next v0

Status: private expanded artifact, public aggregate card.
Date: 2026-06-21.

## Summary

`next-v0` is the first expanded reviewed supervised fine-tuning artifact for
gitctx. It follows the `pilot-v0` recipe, adds a locked `REPORT` split, and
exercises the full private source extraction, teacher-labeling, human-review,
SFT-artifact, baseline-eval, split-inspection, and dependency-free training
smoke flow.

The full JSONL artifacts are private operational data. This card is public so
the source recipe, license posture, review counts, and quality limits are
inspectable without redistributing full diff examples.

## Artifact

Private artifact paths:

```text
artifacts/train/sft.next.v0.jsonl
artifacts/train/sft.next.v0.report.json
artifacts/eval/sft.next.v0.baseline.report.json
artifacts/eval/sft.next.v0.report.inspection.jsonl
artifacts/models/path-type-v0.next.v0.json
artifacts/eval/path-type-v0.next.v0.report.report.json
```

Aggregate counts:

| Field | Count |
|---|---:|
| Source-diff records extracted | 1,000 |
| Source diffs accepted for teacher labeling | 356 |
| Teacher input records | 356 |
| Generated labels | 356 |
| Human-reviewed generated labels | 356 |
| Training records | 335 |
| Human accepted as-is | 301 |
| Human edited | 34 |
| Human rejected | 21 |
| Pending review | 0 |

Split distribution in the promoted training artifact:

| Split | Training records |
|---|---:|
| DEV | 302 |
| REPORT | 33 |
| HELD_OUT | 0 |

`HELD_OUT` is intentionally absent from this artifact. `next-v0` is not a
release-quality generalization benchmark.

## Sources

The artifact uses the same permissive-license source-selection standard as the
pilot: source repositories must be recorded in a manifest with pinned
revisions, source licenses, review status, and allowed splits.

The full source manifest is private operational data for now because the
artifact includes full source diffs and has not completed redistribution
review. Public release of full examples requires a separate source-license and
dataset-redistribution decision.

## Labeling

Teacher:

```text
teacher_model_id: ollama/qwen2.5-coder:7b
teacher_runtime: ollama
teacher_runtime_model_id: qwen2.5-coder:7b
teacher_revision: dae161e27b0e
teacher_license: Apache-2.0
prompt_version: commit-message-teacher-v0.1
```

Generation is one diff per teacher call. Batching unrelated diffs into one
prompt is not part of this artifact recipe.

Human review promoted generated labels only when the decision was `accept` or
`edit`. `reject` records are excluded from the SFT artifact. Any
`needs_review` record blocks artifact creation.

## Baseline Quality Checks

Reviewed target records:

| Check | Result |
|---|---:|
| All target messages format-valid | 335 / 335 |
| REPORT target messages format-valid | 33 / 33 |
| REPORT target scope errors | 0 |

Teacher and historical-comparison checks:

| Check | Result |
|---|---:|
| Raw teacher labels format-valid | 333 / 335 |
| REPORT raw teacher labels format-valid | 33 / 33 |
| Historical subjects format-valid | 9 / 335 |
| REPORT historical subjects format-valid | 0 / 33 |

Reviewed target type distribution:

| Type | Training records |
|---|---:|
| chore | 34 |
| docs | 9 |
| feat | 43 |
| fix | 54 |
| perf | 2 |
| refactor | 184 |
| test | 9 |

## Training Smoke

The dependency-free training smoke trained a small aggregate prototype on the
`DEV` split and evaluated on the `REPORT` split.

This prototype is not a neural model and is not a model-quality claim. It exists
to validate that reviewed SFT artifacts can be consumed by a model-artifact
pipeline and evaluated against held-back `REPORT` records.

Smoke results:

| Field | Count |
|---|---:|
| Training records | 302 |
| Eval records | 33 |
| Prediction messages format-valid | 33 / 33 |
| Prediction scope errors | 0 |
| Type matches | 17 / 33 |
| Exact message matches | 0 / 33 |

## Intended Use

Allowed:

- private data/eval/model-artifact pipeline validation;
- baseline comparison against reviewed targets, raw teacher labels, and
  historical commit subjects;
- data-expansion planning for a future GCTX-1 proof model;
- aggregate public reporting.

Not allowed:

- public redistribution of full examples from this artifact;
- training a public release model without a follow-up data card and
  redistribution review;
- quality claims on `HELD_OUT` splits;
- claims that gitctx has a useful trained neural model;
- treating the dependency-free training smoke as a trained neural model.

## Known Gaps

- The artifact is still small: 335 training records and 33 `REPORT` records.
- There is no `HELD_OUT` split.
- It is not yet ecosystem-diverse enough for a public model release.
- Teacher labels come from one local/open teacher model and one prompt version.
- The dependency-free smoke prototype is intentionally weak and only validates
  artifact flow.
- The artifact includes full source diffs, so redistribution remains restricted
  until source-license and teacher-output reviews approve a release path.

## Next Requirements

Before proof-model training is treated as meaningful:

- decide whether the next milestone is a tiny neural training smoke or a larger
  data expansion;
- create a real `HELD_OUT` split before any release or generalization claim;
- expand toward an order-of-magnitude larger reviewed artifact;
- keep `REPORT` and `HELD_OUT` records locked before teacher generation;
- write a new data card and output-use decision for each promoted artifact.

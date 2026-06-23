# Data Card: gitctx GCTX-1 v0

Status: private proof artifact, public aggregate card.
Date: 2026-06-23.

## Summary

`gctx1-v0` is the first GCTX-scale reviewed supervised fine-tuning artifact for
gitctx. It expands the `next-v0` recipe to a larger multi-ecosystem source set,
keeps a locked `REPORT` split, and exercises the full private source
extraction, source review, teacher-labeling, generated-label review, SFT
artifact, baseline-eval, split-inspection, and smoke-model flow.

The full JSONL artifacts are private operational data. This card is public so
the source recipe, license posture, review policy, aggregate counts, and quality
limits are inspectable without redistributing full diff examples.

## Artifact

Private artifact paths:

```text
artifacts/train/sft.gctx1.v0.jsonl
artifacts/train/sft.gctx1.v0.report.json
artifacts/eval/sft.gctx1.v0.baseline.report.json
artifacts/eval/sft.gctx1.v0.report.inspection.jsonl
artifacts/models/path-type-v0.gctx1.v0.json
artifacts/eval/path-type-v0.gctx1.v0.report.report.json
artifacts/models/tiny-softmax-v0.gctx1.v0.json
artifacts/eval/tiny-softmax-v0.gctx1.v0.report.report.json
```

Aggregate counts:

| Field | Count |
|---|---:|
| Source-diff records extracted | 17,511 |
| Source diffs accepted for teacher labeling | 7,989 |
| Source diffs rejected before teacher labeling | 9,522 |
| Teacher input records | 7,989 |
| Generated labels | 7,983 |
| Recorded teacher-generation failures | 6 |
| Generated-label reviews | 7,983 |
| Review-policy accepted | 6,727 |
| Review-policy rejected | 1,256 |
| Pending review | 0 |
| Training records | 6,727 |

Split distribution in the promoted training artifact:

| Split | Training records |
|---|---:|
| DEV | 5,598 |
| REPORT | 1,129 |
| HELD_OUT | 0 |

`HELD_OUT` is intentionally absent from this artifact. `gctx1-v0` is not a
release-quality generalization benchmark.

## Sources

The artifact uses a permissive-license source-selection standard with pinned
source revisions, source licenses, review status, allowed splits, and split-plan
lineage. The private source manifest spans 37 repositories across Python,
TypeScript/JavaScript, Rust, and Go ecosystems.

The full source manifest and source-diff JSONL remain private operational data
for now because the artifact includes full source diffs and has not completed
redistribution review. Public release of full examples requires a separate
source-license and dataset-redistribution decision.

Source-license distribution in the promoted training artifact:

| License | Training records |
|---|---:|
| Apache-2.0 | 1,320 |
| BSD-3-Clause | 1,105 |
| MIT | 4,302 |

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

Six teacher calls failed deterministically and remain recorded in
`artifacts/teacher/generated-labels.gctx1.report.json`. They were not silently
dropped.

Generated-label review used a conservative deterministic policy:

```text
accept: verifier_score == 1.0 and parser_result.errors == []
reject: every other generated label
```

This policy is intentionally stricter than a human review pass. Borderline
records can be recovered only by a later explicit review-policy version or
manual review.

## Baseline Quality Checks

Reviewed target records:

| Check | Result |
|---|---:|
| All target messages format-valid | 6,727 / 6,727 |
| REPORT target messages format-valid | 1,129 / 1,129 |
| All target messages specific | 6,727 / 6,727 |
| REPORT target messages specific | 1,129 / 1,129 |
| All target scope-quality true | 4,972 / 6,727 |
| REPORT target scope-quality true | 801 / 1,129 |

Teacher and historical-comparison checks:

| Check | Result |
|---|---:|
| Raw teacher labels format-valid | 6,727 / 6,727 |
| REPORT raw teacher labels format-valid | 1,129 / 1,129 |
| Historical subjects format-valid | 1,785 / 6,727 |
| REPORT historical subjects format-valid | 112 / 1,129 |

Reviewed target type distribution:

| Type | Training records |
|---|---:|
| chore | 679 |
| docs | 277 |
| feat | 1,251 |
| fix | 633 |
| refactor | 3,885 |
| style | 1 |
| test | 1 |

## Smoke Models

The dependency-free prototype and tiny-softmax neural smoke runs validate that
the reviewed artifact can be consumed by model-artifact and eval pipelines. They
are not useful language models and are not model-quality claims.

Dependency-free prototype smoke:

| Field | Count |
|---|---:|
| Training records | 5,598 |
| Eval records | 1,129 |
| Prediction messages format-valid | 1,129 / 1,129 |
| Prediction type matches | 679 / 1,129 |
| Exact message matches | 0 / 1,129 |

Tiny-softmax neural smoke:

| Field | Count |
|---|---:|
| Training records | 5,598 |
| Eval records | 1,129 |
| Loss first | 1.100265 |
| Loss last | 0.829156 |
| Prediction messages format-valid | 1,129 / 1,129 |
| Prediction type matches | 601 / 1,129 |
| Exact message matches | 0 / 1,129 |

## Intended Use

Allowed:

- private GCTX proof-model training experiments;
- private data/eval/model-artifact pipeline validation;
- baseline comparison against reviewed targets, raw teacher labels, and
  historical commit subjects;
- prompt, review-policy, tokenizer, and training-code iteration;
- aggregate public reporting.

Not allowed:

- public redistribution of full examples from this artifact;
- public model release claims from this artifact alone;
- claims on `HELD_OUT` generalization;
- claims that the smoke models are useful language models;
- treating historical commit subjects as ground truth;
- mixing this artifact into a public training corpus without a follow-up release
  review.

## Known Gaps

- The promoted training set has 6,727 records, below the original 10k DEV target.
- There is no `HELD_OUT` split in the promoted training artifact.
- The generated-label review pass is deterministic and conservative, not a full
  manual review.
- Teacher labels come from one local/open teacher model and one prompt version.
- The dataset is refactor-heavy because the review policy selected only
  perfect verifier/parser outputs.
- The artifact includes full source diffs, so redistribution remains restricted
  until source-license and teacher-output reviews approve a release path.
- Smoke-model results validate plumbing only; they do not establish useful model
  behavior.

## Next Requirements

Before a public GCTX model release or quality claim:

- train a real small language-model configuration, not only smoke classifiers;
- evaluate against locked `REPORT` and a separate `HELD_OUT` artifact;
- decide whether to expand beyond 6,727 reviewed records or accept GCTX-1 as a
  smaller proof run;
- document the training configuration, tokenizer, model card, eval card, and
  release license;
- complete redistribution review for any examples intended for public release.

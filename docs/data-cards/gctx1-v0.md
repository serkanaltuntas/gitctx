# Data Card: gitctx GCTX-1 v0

Status: private proof artifact, public aggregate card.
Date: 2026-06-28.

## Summary

`gctx1-v0` is the first GCTX-scale reviewed supervised fine-tuning artifact for
gitctx. The current promoted proof artifact is `gctx1-strict`: a merged strict
artifact built from the reviewed GCTX expansion artifacts. It keeps a locked
`REPORT` split, reserves `HELD_OUT` candidates in the split plan, and exercises
the full private source extraction, source review, teacher-labeling,
generated-label review, SFT artifact, baseline-eval, proof-readiness, and
smoke-model flow.

The full JSONL artifacts are private operational data. This card is public so
the source recipe, license posture, review policy, aggregate counts, and quality
limits are inspectable without redistributing full diff examples.

## Artifact

Private artifact paths:

```text
artifacts/train/sft.gctx1.v0.jsonl
artifacts/train/sft.gctx1.v0.report.json
artifacts/train/sft.gctx1-strict.v0.jsonl
artifacts/train/sft.gctx1-strict.v0.report.json
artifacts/eval/sft.gctx1.v0.baseline.report.json
artifacts/eval/sft.gctx1.v0.report.inspection.jsonl
artifacts/eval/sft.gctx1-strict.v0.baseline.report.json
artifacts/eval/sft.gctx1-strict.v0.proof-readiness.report.json
artifacts/models/path-type-v0.gctx1.v0.json
artifacts/eval/path-type-v0.gctx1.v0.report.report.json
artifacts/models/tiny-softmax-v0.gctx1.v0.json
artifacts/eval/tiny-softmax-v0.gctx1.v0.report.report.json
artifacts/models/path-type-v0.gctx1-strict.v0.json
artifacts/eval/path-type-v0.gctx1-strict.v0.report.report.json
artifacts/models/tiny-softmax-v0.gctx1-strict.v0.json
artifacts/eval/tiny-softmax-v0.gctx1-strict.v0.report.report.json
```

Aggregate counts:

| Field | Count |
|---|---:|
| Source-diff records extracted | 33,152 |
| Teacher input records | 14,104 |
| Generated labels | 14,090 |
| Missing generated labels | 14 |
| Generated-label reviews | 14,090 |
| Review-policy accepted | 11,926 |
| Review-policy rejected | 2,164 |
| Pending review | 0 |
| Training records | 11,926 |

Split distribution in the promoted training artifact:

| Split | Training records |
|---|---:|
| DEV | 10,299 |
| REPORT | 1,627 |
| HELD_OUT | 0 |

`HELD_OUT` is intentionally absent from the training artifact. The split plan
reserves 1,295 `HELD_OUT` candidates across 37 repositories for later release
claims; they are not teacher-labeled in the promoted training artifact.

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
| Apache-2.0 | 2,497 |
| BSD-3-Clause | 1,118 |
| MIT | 8,311 |

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

Fourteen teacher inputs are missing generated labels across the merged strict
artifact lineage. They remain recorded in the generated-label reports and were
not silently dropped into training.

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
| All target messages format-valid | 11,926 / 11,926 |
| REPORT target messages format-valid | 1,627 / 1,627 |
| All target messages specific | 11,926 / 11,926 |
| REPORT target messages specific | 1,627 / 1,627 |
| All target scope-quality true | 9,044 / 11,926 |
| REPORT target scope-quality true | 1,179 / 1,627 |

Teacher and historical-comparison checks:

| Check | Result |
|---|---:|
| Raw teacher labels format-valid | 11,926 / 11,926 |
| REPORT raw teacher labels format-valid | 1,627 / 1,627 |
| Historical subjects format-valid | 3,517 / 11,926 |
| REPORT historical subjects format-valid | 285 / 1,627 |

Reviewed target type distribution:

| Type | Training records |
|---|---:|
| chore | 1,033 |
| docs | 539 |
| feat | 2,393 |
| fix | 1,031 |
| refactor | 6,926 |
| style | 3 |
| test | 1 |

## Smoke Models

The dependency-free prototype and tiny-softmax neural smoke runs validate that
the reviewed artifact can be consumed by model-artifact and eval pipelines. They
are not useful language models and are not model-quality claims.

Dependency-free prototype smoke:

| Field | Count |
|---|---:|
| Training records | 10,299 |
| Eval records | 1,627 |
| Prediction messages format-valid | 1,627 / 1,627 |
| Prediction type matches | 932 / 1,627 |
| Exact message matches | 0 / 1,627 |

Tiny-softmax neural smoke:

| Field | Count |
|---|---:|
| Training records | 10,299 |
| Eval records | 1,627 |
| Loss first | 1.061279 |
| Loss last | 0.820147 |
| Prediction messages format-valid | 1,627 / 1,627 |
| Prediction type matches | 988 / 1,627 |
| Exact message matches | 0 / 1,627 |

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

- The promoted strict training artifact passes the original 10k `DEV` target,
  but it is still small for a useful release model.
- There is no `HELD_OUT` split in the promoted training artifact; `HELD_OUT`
  candidates are reserved in the split plan for later release claims.
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
- document the training configuration, tokenizer, model card, eval card, and
  release license;
- complete redistribution review for any examples intended for public release.

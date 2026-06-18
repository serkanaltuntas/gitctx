# Data Card: gitctx Pilot v0

Status: private pilot artifact, public aggregate card.
Date: 2026-06-18.

## Summary

`pilot-v0` is the first reviewed supervised fine-tuning artifact for gitctx. It
exists to test whether a small model can learn to write grounded Conventional
Commit messages from Git diffs.

The full JSONL artifact is private operational data. This card is public so the
source recipe, license posture, and quality limits are inspectable without
redistributing full diff examples.

## Artifact

Private artifact path:

```text
artifacts/train/sft.pilot.v0.jsonl
artifacts/train/sft.pilot.v0.report.json
```

Aggregate counts:

| Field | Count |
|---|---:|
| Source-diff records extracted | 250 |
| Source diffs accepted for teacher labeling | 114 |
| Source diffs rejected before teacher labeling | 136 |
| Teacher input records | 114 |
| Generated labels | 114 |
| Human-reviewed generated labels | 114 |
| Training records | 114 |
| Human accepted as-is | 105 |
| Human edited | 9 |
| Human rejected | 0 |
| Pending review | 0 |

All records are `DEV` split. There is no `REPORT` or `HELD_OUT` claim in this
pilot artifact.

## Sources

The pilot uses a small permissive-license audit manifest:

- `pallets/click`;
- `psf/requests`;
- `pytest-dev/pluggy`;
- `encode/httpx`;
- `python-attrs/attrs`.

The first pilot extraction yielded records from the first three repositories.
The source manifest records pinned revisions, source licenses, review status,
and allowed splits.

Source-license distribution in the promoted training artifact:

| License | Training records |
|---|---:|
| Apache-2.0 | 41 |
| BSD-3-Clause | 63 |
| MIT | 10 |

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

## Intended Use

Allowed:

- deterministic data/eval pipeline validation;
- baseline comparison against historical commit subjects and raw teacher labels;
- proof-of-process experiments for a future GCTX-1 model;
- aggregate public reporting.

Not allowed:

- public redistribution of full examples from this artifact;
- training a public release model without a follow-up data card and
  redistribution review;
- quality claims on `REPORT` or `HELD_OUT` splits;
- claims that gitctx has a useful trained model.

## Known Gaps

- The artifact is tiny: 114 training records.
- It is Python-heavy and not ecosystem-diverse enough for a public model.
- All records are `DEV`; no held-out generalization is measured.
- Historical commit messages are kept only as weak comparison context.
- Teacher labels come from one local/open teacher model and one prompt version.
- The artifact includes full source diffs, so redistribution remains restricted
  until source-license and teacher-output reviews approve a release path.

## Next Requirements

Before proof-model training is treated as meaningful:

- add a baseline/eval report for target, teacher, and historical messages;
- create a `REPORT` split policy before reporting model progress;
- reserve HELD_OUT repositories or time ranges before any release claim;
- expand beyond the first tiny audit source set.

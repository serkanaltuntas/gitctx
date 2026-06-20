# Model Plan

gitctx trains a small model for one narrow job: produce grounded Conventional
Commit messages from Git context.

## Target

Input:

- repository metadata;
- repository instructions or detected commit conventions;
- `git status`;
- diffstat;
- unified diff;
- changed file paths;
- optional issue, pull request, or changelog context.

Output:

- Conventional Commit subject;
- optional body;
- optional `BREAKING CHANGE` footer;
- confidence and warnings.

## Size Ladder

| Rung | Size | Purpose |
|---|---:|---|
| GCTX-0 | rules/templates | deterministic baseline |
| GCTX-1 | 60M-100M | data and eval proof model |
| GCTX-2 | 150M-300M | first serious public model |
| GCTX-3 | 500M | expanded repo-operator tasks |
| GCTX-4 | 1B | only if the project grows beyond commit messages |

The project should not start with 1B. The first proof model should falsify data
and evaluation assumptions cheaply.

## Proof-Model Gate

GCTX-1 is not unlocked by the existence of a tiny `DEV` artifact. A local smoke
model may be trained earlier to test code paths, but it carries no quality
claim.

## Training Pipeline Smoke

Before training a neural model, gitctx uses a dependency-free prototype model to
validate the artifact contract:

```bash
make training-smoke PILOT_ARTIFACT=next
```

This trains `path-type-v0` on the `DEV` records in a reviewed SFT artifact and
evaluates deterministic predictions on `REPORT`. The prototype model learns only
aggregate path-token/type statistics and emits simple Conventional Commit
messages. It is intentionally weak. Its purpose is to prove that model
artifacts, prediction artifacts, and eval reports can be produced from the SFT
artifact without changing the data lineage.

Artifacts:

```text
artifacts/models/path-type-v0.<artifact>.v0.json
artifacts/eval/path-type-v0.<artifact>.v0.report.predictions.jsonl
artifacts/eval/path-type-v0.<artifact>.v0.report.report.json
```

This is a training/eval pipeline smoke, not a model-quality benchmark and not a
public model release candidate.

Minimum GCTX-1 proof-run conditions:

- 10,000 reviewed `DEV` training records;
- 1,000 `REPORT` records;
- 1,000 reserved `HELD_OUT` candidates;
- at least 25 training repositories;
- at least 5 `REPORT` repositories or non-overlapping report windows;
- at least 5 `HELD_OUT` repositories;
- at least two programming-language ecosystems;
- no single repository contributes more than 25% of training records;
- deterministic, raw-teacher, historical-subject, and reviewed-target
  baselines are recorded.

The split policy is defined in [`split-contract.md`](split-contract.md).

## Recommended First Public Model

- 150M-300M decoder-only model;
- 8K-16K context;
- code/diff-aware tokenizer;
- quantized local runtime;
- trained from high-quality human labels plus license-approved teacher labels.

## Non-Goals

- General chat.
- Fully autonomous coding.
- Automatic commit or push.
- Training on closed-model outputs.
- Private repository ingestion without explicit permission.

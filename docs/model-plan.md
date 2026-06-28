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

Current GCTX-1 proof status:

- `gctx1-strict` is the current private proof artifact.
- The strict data-readiness gate is passed: 10,299 `DEV` training records,
  1,627 locked `REPORT` records, and 1,295 reserved `HELD_OUT` candidates.
- The dependency-free prototype and tiny-softmax smoke paths have run on
  `gctx1-strict`; they validate training/eval plumbing only.
- The next model milestone is a real 60M-100M decoder-only proof model trained
  from the reviewed `DEV` split and evaluated on locked `REPORT`.

The public proof-model config placeholder is:

```text
configs/gctx1-proof-model.v0.json
```

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

## Tiny Neural Smoke

After the dependency-free prototype, gitctx can run a tiny dependency-free
neural smoke:

```bash
make neural-smoke PILOT_ARTIFACT=next
```

This trains `tiny-softmax-v0`, a single-layer softmax classifier, on reviewed
`DEV` records and evaluates Conventional Commit predictions on `REPORT`.
Training uses deterministic gradient descent over path and diff-stat features.
It writes a checkpoint-like JSON model artifact, prediction JSONL, and eval
report.

Artifacts:

```text
artifacts/models/tiny-softmax-v0.<artifact>.v0.json
artifacts/eval/tiny-softmax-v0.<artifact>.v0.report.predictions.jsonl
artifacts/eval/tiny-softmax-v0.<artifact>.v0.report.report.json
```

This is the first real neural-style training loop in the public repo, but it is
still a smoke test. It is not a language model, not a model-quality benchmark,
and not a public model release candidate.

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
Run `make split-readiness` against any GCTX-1 planning manifest and split plan
before extraction.

After reviewed SFT artifact creation, run proof readiness against the promoted
artifact:

```bash
make proof-readiness PILOT_ARTIFACT=gctx1
```

`split-readiness` checks whether the planned DEV/REPORT/HELD_OUT windows are
valid before extraction. `proof-readiness` checks the actual promoted SFT
artifact and records whether the GCTX-1 proof-run gates are met. A failed
readiness report is still a useful artifact: it identifies the exact missing
condition before expensive training starts.

For the current strict proof artifact, use the named targets:

```bash
make gctx1-proof-config-check
make gctx1-proof-readiness
make gctx1-proof-smoke
make gctx1-proof-smoke-check
```

These targets do not train the 60M-100M proof language model. They validate the
proof config, rerun readiness against the strict artifact, and run the current
pipeline smoke models on locked `REPORT`.

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

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


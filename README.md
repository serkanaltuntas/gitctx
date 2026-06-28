# gitctx

gitctx is a model-first open-source project for generating high-quality
Conventional Commit messages from Git context.

## Canonical resource

- **Preferred name:** GCTX
- **Repository / CLI name:** gitctx
- **Canonical resource page:** https://serkan.ai/projects/gitctx/
- **Short definition:** GCTX is a from-scratch small language model family for understanding Git diffs and writing Conventional Commit messages.
- **Status:** Building; there is no useful public model release yet.
- **License:** Code and documentation are Apache-2.0; model artifacts, generated labels, and datasets will carry release-specific licenses and cards after upstream license review.

The goal is not just to publish model weights. The project aims to publish the
source code, data-generation recipes, teacher-labeling prompts and provenance,
training configuration, evaluation harness, model artifacts, and release cards
needed for independent inspection and reproduction.

## Status

The project has a private `gctx1-strict` proof artifact that passes the GCTX-1
data-readiness gate: 10,299 `DEV` training records, 1,627 locked `REPORT`
records, and 1,295 reserved `HELD_OUT` candidates. The public repository
contains the source code, schemas, prompts, runbooks, aggregate data cards, and
output-use decisions needed to inspect the recipe.

There is no useful public model release yet. The CLI shell comes later, after
model behavior is measurable.

## Principles

- Local-first by default.
- No automatic commits or pushes.
- No network calls unless the user explicitly opts in.
- No training on closed-model outputs.
- Distillation only from models whose licenses allow generated outputs to train
  downstream models.
- Public data only when the source license and redistribution rights allow it.
- Honest release language: distinguish open weights, source available,
  reproducible open recipe, and full open-source releases.

## Planned Product Shape

Eventual commands:

```bash
gitctx suggest
gitctx inspect
gitctx hook install
gitctx doctor
```

The first useful public artifact should be a model and evaluation report, not a
thin CLI wrapper.

## Roadmap

1. Define the evaluation schema, parser, and scorer.
2. Build fixture repositories and public eval splits.
3. Extract diffs from permissively licensed repositories.
4. Generate teacher labels with license-approved local/open models.
5. Train a small proof model.
6. Publish model, data, and eval cards.
7. Package the CLI around a useful local model.

## Development

Run the current dependency-free test suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Run the first deterministic fixture evaluation:

```bash
PYTHONPATH=src python3 -m gitctx.eval fixtures/dev/commit_message_cases.jsonl
```

Run local source-diff smoke extraction:

```bash
make smoke
make smoke-check
make smoke-finalize
make pilot-source PILOT_RECORDS=250 PILOT_PER_REPO_LIMIT=100
make pilot-source-check
make pilot-source-finalize
make pilot-review-template REVIEWER="reviewer@example.com"
make pilot-review-check
make smoke-review-template REVIEWER="reviewer@example.com"
make smoke-review-check
make teacher-inputs
make teacher-input-check
make pilot-teacher-inputs
make pilot-teacher-input-check
make teacher-generate
make teacher-generate-check
make pilot-teacher-generate
make pilot-teacher-generate-check
make generated-review-template REVIEWER="reviewer@example.com"
make generated-review-check
make pilot-generated-review-template REVIEWER="reviewer@example.com"
make pilot-generated-review-check
make pilot-train-artifact
make pilot-train-artifact-check
make pilot-eval-baseline
make artifact-split-inspection PILOT_ARTIFACT=next DATA_SPLIT=REPORT
make training-smoke PILOT_ARTIFACT=next
make gctx1-proof-config-check
make gctx1-proof-readiness
make gctx1-proof-handoff
make gctx1-proof-handoff-check
make gctx1-proof-smoke
make gctx1-proof-smoke-check
```

For non-smoke artifacts that contain `REPORT` or `HELD_OUT` candidates, pass a
prelocked split plan:

```bash
make pilot-source SPLIT_PLAN="/path/to/split-plan.json"
```

See:

- [docs/model-plan.md](docs/model-plan.md)
- [configs/gctx1-proof-model.v0.json](configs/gctx1-proof-model.v0.json)
- [docs/data-and-distillation.md](docs/data-and-distillation.md)
- [docs/source-manifest.md](docs/source-manifest.md)
- [docs/source-diff-review.md](docs/source-diff-review.md)
- [docs/teacher-inputs.md](docs/teacher-inputs.md)
- [docs/training-artifacts.md](docs/training-artifacts.md)
- [docs/data-cards/pilot-v0.md](docs/data-cards/pilot-v0.md)
- [docs/data-cards/next-v0.md](docs/data-cards/next-v0.md)
- [docs/data-cards/gctx1-v0.md](docs/data-cards/gctx1-v0.md)
- [docs/output-use-decisions/pilot-v0.md](docs/output-use-decisions/pilot-v0.md)
- [docs/output-use-decisions/next-v0.md](docs/output-use-decisions/next-v0.md)
- [docs/output-use-decisions/gctx1-v0.md](docs/output-use-decisions/gctx1-v0.md)
- [schemas/generated-label-review.schema.json](schemas/generated-label-review.schema.json)
- [schemas/training-example.schema.json](schemas/training-example.schema.json)
- [docs/local-worker-runbook.md](docs/local-worker-runbook.md)
- [docs/teacher-model-shortlist.md](docs/teacher-model-shortlist.md)
- [docs/teacher-audit-plan.md](docs/teacher-audit-plan.md)
- [docs/teacher-decisions/ollama-qwen2.5-coder-7b-smoke.md](docs/teacher-decisions/ollama-qwen2.5-coder-7b-smoke.md)
- [docs/evaluation.md](docs/evaluation.md)
- [docs/split-contract.md](docs/split-contract.md)
- [docs/open-source-standard.md](docs/open-source-standard.md)
- [docs/release-checklist.md](docs/release-checklist.md)

## License

Code and documentation in this repository are licensed under the Apache License
2.0 unless a file states otherwise. Model artifacts, generated labels, and
datasets will carry release-specific licenses and cards after upstream license
review.

# gitctx

gitctx is a model-first open-source project for generating high-quality
Conventional Commit messages from Git context.

The goal is not just to publish model weights. The project aims to publish the
source code, data-generation recipes, teacher-labeling prompts and provenance,
training configuration, evaluation harness, model artifacts, and release cards
needed for independent inspection and reproduction.

## Status

Early planning and repository bootstrap. The first implementation work is the
data, evaluation, teacher-labeling, and model loop. The CLI shell comes later,
after model behavior is measurable.

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

See:

- [docs/model-plan.md](docs/model-plan.md)
- [docs/data-and-distillation.md](docs/data-and-distillation.md)
- [docs/evaluation.md](docs/evaluation.md)
- [docs/open-source-standard.md](docs/open-source-standard.md)
- [docs/release-checklist.md](docs/release-checklist.md)

## License

Code and documentation in this repository are licensed under the Apache License
2.0 unless a file states otherwise. Model artifacts, generated labels, and
datasets will carry release-specific licenses and cards after upstream license
review.

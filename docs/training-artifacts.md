# Training Artifacts

Training artifacts are the first model-ready output of the data pipeline. They
are built from reviewed generated labels, not directly from raw teacher output.

The first pilot artifact is:

```text
artifacts/train/sft.pilot.v0.jsonl
artifacts/train/sft.pilot.v0.report.json
```

The JSONL record format is defined in
[`schemas/training-example.schema.json`](../schemas/training-example.schema.json).

## Promotion Rule

A generated label becomes a training example only when its human review decision
is:

- `accept`; or
- `edit`.

Generated labels with `reject` are skipped. Generated labels with
`needs_review` stop artifact creation because the review is incomplete.

For `accept`, the target message is the generated Conventional Commit message.
For `edit`, the target message uses `edited_header` and/or `edited_body` from
the review record while preserving generated footers.

## Record Contents

Each training example records:

- source repository, license, commit, parent commit, split, changed paths, and
  historical subject;
- full source diff text and `diff_sha256`;
- system, user, and assistant messages for chat-style supervised fine-tuning;
- final target header, body, footers, and rendered message;
- generated-label id, review id, review decision, issues, reviewer, and review
  timestamp;
- teacher model id, runtime, revision, license, prompt version, decoding config,
  generation timestamp, verifier score, evidence paths, and warnings.

This is intentionally verbose. The training record must be reproducible,
auditable, and removable by source repository, teacher model, prompt version, or
review decision.

Splits are governed by [`split-contract.md`](split-contract.md). Training
artifacts must not move examples across `DEV`, `REPORT`, or `HELD_OUT` after
teacher generation.

## Commands

Create the pilot training artifact after generated-label review is complete:

```bash
make pilot-train-artifact
```

Validate it and refresh checksums:

```bash
make pilot-train-artifact-check
```

Evaluate reviewed targets against raw teacher labels and historical subjects:

```bash
make pilot-eval-baseline
```

For another named artifact:

```bash
make artifact-eval-baseline PILOT_ARTIFACT=next
```

The baseline report includes aggregate scores plus `by_data_split`, so `REPORT`
quality can be inspected separately from `DEV`.

For record-by-record split inspection:

```bash
make artifact-split-inspection PILOT_ARTIFACT=next DATA_SPLIT=REPORT
```

The inspection artifact is private review material. It is useful for deciding
whether the current SFT artifact is ready for a proof-model pipeline run, but it
is not itself a public model-quality claim.

Run the dependency-free training smoke after the baseline and split inspection:

```bash
make training-smoke PILOT_ARTIFACT=next
```

This consumes the reviewed SFT artifact, trains a small aggregate prototype on
`DEV`, and evaluates predictions on `REPORT`. The resulting artifacts validate
the training/eval flow; they do not establish model quality.

Run the tiny dependency-free neural smoke after the prototype smoke:

```bash
make neural-smoke PILOT_ARTIFACT=next
```

This trains `tiny-softmax-v0`, a single-layer softmax classifier, on `DEV` and
evaluates on `REPORT`. It validates the first gradient-descent model-artifact
path, but it is still not a language model and not a model-quality claim.

Equivalent direct commands:

```bash
PYTHONPATH=src python3 -m gitctx.train_artifacts \
  --data-dir "$GITCTX_DATA_DIR" \
  create \
  --artifact-name pilot \
  --version v0

PYTHONPATH=src python3 -m gitctx.train_artifacts \
  --data-dir "$GITCTX_DATA_DIR" \
  validate \
  --artifact-name pilot \
  --version v0
```

## Publication

Training artifacts include full diff text and teacher-generated targets. Keep
them private until source-license, teacher-license, and redistribution reviews
decide what can be published. Public releases should publish aggregate stats,
schemas, prompts, data cards, and reproducibility recipes even when full example
redistribution is not approved.

The first pilot data card and output-use decision are:

- [`data-cards/pilot-v0.md`](data-cards/pilot-v0.md)
- [`output-use-decisions/pilot-v0.md`](output-use-decisions/pilot-v0.md)

The first expanded artifact card and output-use decision are:

- [`data-cards/next-v0.md`](data-cards/next-v0.md)
- [`output-use-decisions/next-v0.md`](output-use-decisions/next-v0.md)

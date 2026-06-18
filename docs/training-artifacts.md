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

## Commands

Create the pilot training artifact after generated-label review is complete:

```bash
make pilot-train-artifact
```

Validate it and refresh checksums:

```bash
make pilot-train-artifact-check
```

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

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

Check whether a reviewed artifact is ready for a GCTX proof-model run:

```bash
make proof-readiness PILOT_ARTIFACT=gctx1
```

This writes:

```text
artifacts/eval/sft.gctx1.v0.proof-readiness.report.json
```

The report checks actual reviewed `DEV`/`REPORT` counts, repository diversity,
reserved `HELD_OUT` planning coverage, max per-repository contribution, source
manifest linkage, and baseline availability. It does not fail the command when
the artifact is not ready; use the report to decide whether to expand data or
record a smaller private proof-run decision.

The current GCTX-1 proof artifact is `gctx1-strict`. It is built by merging the
reviewed GCTX expansion artifacts and rebuilding the reviewed SFT artifact from
the merged source, teacher-input, generated-label, and generated-label-review
records. Its proof-readiness report passes the GCTX-1 gate:

| Field | Count |
|---|---:|
| Training records | 11,926 |
| DEV training records | 10,299 |
| REPORT records | 1,627 |
| Reserved HELD_OUT candidates | 1,295 |

Use the named strict proof targets for handoff:

```bash
make gctx1-proof-config-check
make gctx1-proof-readiness
make gctx1-tokenizer
make gctx1-tokenizer-check
make gctx1-proof-handoff
make gctx1-proof-handoff-check
make gctx1-proof-train-dry-run
make gctx1-proof-train-dry-run-check
make gctx1-proof-sequences
make gctx1-proof-sequences-check
make gctx1-proof-sft-smoke
make gctx1-proof-sft-smoke-check
make gctx1-proof-trainer-job
make gctx1-proof-trainer-job-check
make gctx1-proof-smoke
make gctx1-proof-smoke-check
```

`gctx1-proof-config-check` validates the proof-run contract in
`configs/gctx1-proof-model.v0.json`: the GCTX-1 parameter band, strict data
artifact, DEV/REPORT/HELD_OUT split roles, minimum record thresholds,
locked-REPORT policy, reproducibility fields, and model/eval-card release
preconditions.

`gctx1-tokenizer` writes a dependency-free proof tokenizer artifact and coverage
report under `artifacts/tokenizers/` in the private data repository. It learns
the vocabulary only from reviewed `DEV` records. `REPORT` is used only to record
known-token coverage before a real proof-model training job starts.

`gctx1-proof-handoff` writes
`artifacts/train-runs/gctx1-proof-model.v0.handoff.json` in the private data
repository. The handoff manifest records config and input hashes, proof-readiness
gates, tokenizer hash, training code revision, training contract, and required
trainer outputs. It is the artifact to copy into an actual proof-model training
job.

`gctx1-proof-train-dry-run` consumes the handoff manifest and writes a
no-weights proof-trainer run under `artifacts/train-runs/`. It verifies the
handoff input hashes, reports the local Python/Torch runtime state, counts
tokenized records and context windows by split, and writes a checkpoint skeleton
that deliberately contains no model weights. This validates the trainer artifact
contract before the 60M-100M decoder-only proof training job exists.

The dry-run also writes:

```text
artifacts/train-runs/gctx1-proof-model.v0.dry-run.sequence-plan.jsonl
```

The sequence plan is the long-record policy artifact. Each reviewed SFT record
receives one deterministic decision:

- `use_full`: the raw tokenized record fits the proof context;
- `use_truncated`: the record is over context but below the raw-token cap, so
  the real trainer must apply deterministic prefix/suffix cropping while
  preserving the system instruction and assistant target;
- `exclude_oversize`: the raw tokenized record is above the raw-token cap and is
  excluded from the first proof run.

`REPORT` exclusions block model-quality claims. `DEV` exclusions are acceptable
only when the kept `DEV` count remains above the proof-run minimum. The default
raw-token cap is 65,536 tokens for the GCTX-1 dry-run target.

`gctx1-proof-sequences` consumes the sequence plan and materializes deterministic
trainer-input metadata. The actual token-id arrays are generated in memory and
hashed; the data repository stores only metadata, crop accounting, loss-token
counts, and input/loss-mask hashes. This proves the exact crop and loss-mask
contract without committing a large token-id payload. The real trainer must use
the same materializer before replacing the no-weights dry-run checkpoint with
resumable model checkpoints.

`gctx1-proof-sft-smoke` is the first bounded checkpoint/resume proof over those
materialized sequences. It trains only on a deterministic `DEV` sample, updates a
small dependency-free hashed weight vector, and writes:

```text
artifacts/train-runs/gctx1-proof-model.v0.dry-run.sft-smoke.report.json
artifacts/train-runs/gctx1-proof-model.v0.dry-run.sft-smoke.checkpoint.json
```

Use `GCTX1_PROOF_SFT_SMOKE_MAX_RECORDS` to change the bounded sample size. Use
`GCTX1_PROOF_SFT_SMOKE_STOP_AFTER` plus
`GCTX1_PROOF_SFT_SMOKE_RESUME=1` to exercise the resume path. This target proves
trainer sequence consumption, optimizer accounting, checkpoint writing, and
resume determinism. It deliberately does not train on `REPORT`, does not train
the 60M-100M proof language model, and does not establish model quality.

`gctx1-proof-trainer-job` writes the actual proof-trainer job contract:

```text
artifacts/train-runs/gctx1-proof-model.v0.dry-run.trainer-job.json
```

The manifest records the bounded decoder-only model shape, estimated parameter
count, sequence metadata hash, checkpoint output paths, resume requirements,
and locked `REPORT` eval outputs. It is the handoff from artifact preparation to
the future real training job. A ready job manifest still does not mean training
has happened; it means the inputs and expected trainer outputs are now explicit
and validated before spending GPU time.

`gctx1-proof-smoke` runs the dependency-free prototype and tiny-softmax smoke
models against `gctx1-strict`. It is still a pipeline proof, not the 60M-100M
proof language-model run.

## Expansion Artifacts

When a proof-readiness report shows that a promoted artifact is below its
training-record target, do not overwrite the original artifact. Create a new
expansion artifact and exclude already extracted source ids:

```bash
make pilot-source \
  PILOT_ARTIFACT=gctx1-dev2 \
  SMOKE_MANIFEST="$GITCTX_DATA_DIR/manifests/source-manifest.gctx1.jsonl" \
  SPLIT_PLAN="$GITCTX_DATA_DIR/manifests/split-plan.gctx1.json" \
  EXCLUDE_SOURCE_ARTIFACT="$GITCTX_DATA_DIR/artifacts/gctx1/source-diffs.gctx1.jsonl"
```

After source review, teacher generation, generated-label review, and SFT
artifact creation for the expansion artifact, merge reviewed artifact inputs
into a new strict artifact:

```bash
make merge-train-artifact \
  PILOT_ARTIFACT=gctx1-strict \
  MERGE_INPUT_ARTIFACTS="gctx1 gctx1-dev2"
```

The merge command combines source diffs, teacher inputs, generated labels, and
generated-label reviews, skips duplicate source/generated ids, and then builds a
fresh reviewed SFT artifact. It keeps `gctx1` immutable and makes the strict
artifact reproducible.

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

The first GCTX-scale proof artifact card and output-use decision are:

- [`data-cards/gctx1-v0.md`](data-cards/gctx1-v0.md)
- [`output-use-decisions/gctx1-v0.md`](output-use-decisions/gctx1-v0.md)

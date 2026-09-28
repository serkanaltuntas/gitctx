# Supervised reviewed DEV training

`python -m gitctx.reviewed_job` runs an explicitly authorized, hash-pinned
protocol on a Linux CUDA host. It does not select an epoch/time budget, contact
services, produce labels or evaluate protected splits. Use `--data-root`,
`--protocol` and `--protocol-sha256`; `--resume` is explicit and never resets the
original launch deadline. A per-run file lock prevents concurrent workers.

The selected protocol binds every input, model/software settings, source module
hashes, frozen dataset identity, 1/2/4 epoch cap, wall seconds and free-disk floor.
Run it under a supervisor with the same maximum runtime. The worker checks time
and disk before each training/validation record; SIGTERM/SIGINT preserve the last
successful checkpoint. Failed saves retain orphan files for inspection. No
successful checkpoint means a fresh attempt must preserve the prior failed run.

Before the first update, score the entire frozen validation split. Checkpoint the
first step and then every configured interval, preserving the original trainer's
shuffle, optimizer/RNG state, per-commit objective and exact resume contract.
At every epoch, score and generate every validation record. Resumable prediction
files bind the checkpoint and selected protocol. Decode failures and empty
outputs remain in the evaluation denominator.

Stop automatically when a complete header occupies over 80% of validation
outputs or validation loss worsens over 10% relative to a preceding epoch.
Before a further epoch, require an explicit source-grounded review of the frozen
factuality panel. The review binds prediction/panel hashes, covers every panel
record exactly once and quotes source lines. Assistant reviews remain labeled;
automatic metrics cannot impersonate semantic review. Accuracy below 40% stops
the run. Final/stopped-quality checkpoints also generate the frozen mismatched
input controls. Full release/continuation success metrics require subsequent
review of those controls and the experiment's predeclared criteria.

Live status, launch deadline, baseline, epoch predictions and terminal result
are written under the private output directory selected by the caller. Weights,
optimizer states and intermediate predictions belong in an ignored cache.
Inspect the supervisor process as well as progress files: a file alone does not
prove a worker is alive. Do not extend a budget, clear terminal results or
silently restart a failed run. Training and data/model publication remain
separate authorizations.

# Reviewed lossless dataset adapter

`ReviewedDataset` joins an independently frozen active partition map and review
selection with original records, reviewed reference artifacts and lossless input
windows. All required references must resolve before the adapter exposes any
examples. Missing or unresolved review entries never fall back to old labels.
Every validation reference requires explicit review. Non-selected train targets
retain their original open-teacher provenance and are marked historical automatic
acceptance, never human review.

The adapter keeps source inputs separate from the one complete-commit target.
`example()` returns all window prompts and one answer ending in EOS; it does not
assign a whole-commit target independently to partial windows. Optional prepared
group hashes are reconstructed at access. Missing windows, altered partitions,
source mismatches and answer overflow fail explicitly. Epoch ordering must include
every partition member once. Dataset fingerprints bind source input, partition,
reference, tokenizer and window/group content for later run/resume checks.

`backward_example()` calls the joint-window gradient implementation and refuses
validation examples. The calling trainer owns zeroing gradients, clipping and
optimizer updates. `score_example()` evaluates the same joint probability with
no gradients and returns summed NLL and answer token count for weighted metrics.
`predict_record()` uses the same input preparation and joint decoding without
consulting labels or reference artifacts. Invalid structural tokens or incomplete
UTF-8 outputs are reported explicitly, preserving output IDs.

This is a data-loader and model-operation adapter. It does not by itself provide
a full epoch/checkpoint/resume runner, audit input file provenance, authorize a
training run or establish semantic quality. The caller must pin source files,
review artifacts, the independent split/selection and tokenizer; the full readiness
audit and a separately selected budget remain required before real training.

`reviewed_training.run_epochs()` supplies a bounded 1/2/4-epoch loop: one update
per complete commit, seeded shuffling, gradient clipping and exact joint validation
likelihood. Checkpoints bind dataset and run identity, optimizer settings, parameter
shapes, RNG state and an exact epoch cursor. State files are immutable and hashed;
`latest.json` is replaced atomically. Resume refuses changed input identities or
checkpoint bytes. A bounded invocation may stop mid-epoch and continue with the
same order and optimizer state. This API requires an explicit execution flag;
that flag is not a substitute for actual user authorization of a real run.

The data-file manifest loader, corpus readiness audit, generation-quality metrics
and measured resource proposal must still surround this runner before real use.
Tests exercise only small synthetic fixtures, including bit-exact interrupted vs
uninterrupted parameters, optimizer moments and validation metrics.

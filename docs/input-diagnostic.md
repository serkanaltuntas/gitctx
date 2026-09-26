# DEV input diagnostic

Use a completed, frozen ordering-ablation checkpoint to distinguish input
sensitivity from useful generation. This tool does not train or edit references.

```bash
PYTHONPATH=src uv run --locked --no-sync python -m gitctx.proof_input_diagnostic prepare DATA_DIR PARENT_EXPERIMENT DIAGNOSTIC_ID
PYTHONPATH=src uv run --locked --no-sync python -m gitctx.proof_input_diagnostic run DATA_DIR PARENT_EXPERIMENT DIAGNOSTIC_ID
```

Preparation validates the parent's code, inputs and split boundaries. It freezes
128 repository/type/length-stratified reserved DEV validation IDs, 32 training
review IDs, cross-repository donors, the checkpoint, code hash and metrics before
inference. All parent training and validation rows receive CPU coverage checks.
The implementation currently targets the existing 8,192-token proof contract.

Four conditions use the same target, system instruction and greedy decoding:

- Real input, required to reproduce the parent's saved prediction exactly.
- Empty diff, retaining repository, paths, statistics and output instructions.
- Another repository's diff, retaining original metadata. The nearest raw diff
  length is selected deterministically; donors may repeat.
- Another repository's entire user message, as a distinct metadata control.

Conditional reference CE uses exactly the same prompt as inference, with a fixed
256-token output reserve; reference length does not select prompt contents.
Loss includes target tokens and the two terminal tokens. Paired CE differences,
output identity and reference type/scope matches are diagnostics, not semantic
acceptance. Perturbations can create contradictions or out-of-distribution
inputs; they are not a clean causal estimate of every source of model collapse.

Coverage records unknown tokens, lossy target round trips, output-instruction
conflicts, training/inference prompt differences, mask alignment and retention of
changed diff lines under prefix/suffix cropping. A retained line's tokens do not
prove its surrounding context survived. Aggregate token rates are descriptive;
the stratified inference sample is not an unbiased estimate of production use.

The frozen first-version line counter excludes every line starting with `+++`
or `---`, including changed source lines with those prefixes. Treat its counts
as lexical estimates; an independent hunk-aware recount is needed for exact
changed-line totals. Preserve the original coverage output when attaching a
corrected audit rather than rewriting the frozen run.

Review findings must retain example IDs and evidence, distinguish an assistant
assessment from human approval, and leave original labels/provenance unchanged.
They are not replacement training targets. No REPORT or HELD_OUT evaluation,
network calls, tokenizer fitting, checkpoint writes, commits or pushes occur.

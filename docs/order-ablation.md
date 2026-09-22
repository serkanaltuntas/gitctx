# Internal DEV order ablation

This paired experiment tests original artifact order against a fixed seed-17
shuffle. Each arm trains from scratch for one pass on exactly the same records,
with identical model, tokenizer, seed, optimizer, precision and token budget.
The original model size and context capacity are preserved. No long examples
are additionally filtered to fit hardware.

Preparation reserves eight entire DEV repositories by SHA256(URL), samples 256
validation rows by ID hash in repository round-robin order, and uses all other
eligible DEV rows for training. Cross-repository duplicate commits, exact diffs
and whitespace-normalized tokenized diffs are purged from training. Unselected
rows in validation repositories remain reserved. Original DEV/REPORT/HELD_OUT
assignments are unchanged. The tokenizer is refitted on training rows alone at
the original vocabulary size and shared across arms. Insufficient vocabulary
blocks preparation instead of shrinking the model.

```bash
uv run --locked python -m gitctx.proof_order_ablation prepare \
  --data-dir /path/to/data --source-job artifacts/train-runs/SOURCE.trainer-job.json \
  --experiment-id dev-order-v1
uv run --locked python -m gitctx.proof_order_ablation train \
  --data-dir /path/to/data --experiment-id dev-order-v1 --arm ordered
uv run --locked python -m gitctx.proof_order_ablation evaluate \
  --data-dir /path/to/data --experiment-id dev-order-v1 --arm ordered
# Repeat train/evaluate with --arm shuffled, then:
uv run --locked python -m gitctx.proof_order_ablation compare \
  --data-dir /path/to/data --experiment-id dev-order-v1
```

Add `--resume` to training after interruption. The unchanged trainer checkpoints
every 100 steps, including optimizer/RNG state. A blocked training command exits
nonzero. Model checkpoints remain outside Git. Protocol/input/code hashes are
frozen before execution; a changed implementation requires a new experiment.
After preparation, `bash scripts/run-order-ablation.sh /path/to/data dev-order-v1`
runs both arms, validates them and writes the comparison sequentially. Rerunning
the script resumes checkpoints and retains completed validation outputs.

Final validation measures token-weighted supervised CE, format, type, raw scope,
exact text, and dominant type/scope concentration with gold-free prompts and a
fixed 256-token generation limit. The predeclared decision favors shuffling only
when CE is at least 5% lower and dominant type/scope concentration at least ten
percentage points lower. Otherwise report tradeoffs or an inconclusive result.
Invalid outputs count as one concentration category, never as diversity.

One seed, repository-balanced validation and one pass limit interpretation.
These are exploratory DEV results, not public quality or release gates. The
fold tokenizer also differs from the earlier full run, so only the paired arms
isolate ordering. Duplicate checks do not rule out every semantic near-duplicate.
REPORT/HELD_OUT are not used. Existing data licensing/provenance still applies.

## Decoder policy

`scope-punctuation-v1` removes reconstruction spaces around `. / \\ : @ -`
inside a valid Conventional Commit header scope. It preserves word boundaries
in multiword scopes and leaves the legacy subject/body decoding unchanged.
`decode_tokens_legacy` remains available for inspecting historical token outputs;
replaying frozen experiments requires their recorded source revision. Existing
evaluation identity checks prevent silently replacing old REPORT results.

Arbitrary original whitespace cannot be recovered from this regex tokenizer.
This scope repair does not claim lossless text reconstruction.

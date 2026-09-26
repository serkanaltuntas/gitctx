# DEV memorization diagnostic

Use this diagnostic to check whether the proof model can learn a small set of
seen examples before spending more compute on generalization experiments.
It reuses the proof model, assistant-only loss, tokenizer, and production
generation helpers. It does not alter the original training run.

```bash
uv run --locked python -m gitctx.proof_memorization prepare \
  --data-dir /path/to/data \
  --source-job artifacts/train-runs/SOURCE_RUN.trainer-job.json \
  --run-id dev-memorization-v1
uv run --locked python -m gitctx.proof_memorization run \
  --data-dir /path/to/data --run-id dev-memorization-v1 --device cuda
```

Preparation freezes input and implementation hashes, model contract, selection,
training settings, and success criteria in `protocol.json`. The selection uses
16 distinct DEV repositories: four `feat`, four `fix`, three `docs`, three
`chore`, and two `refactor` examples. Inputs are 256–1024 tokens, untruncated,
with 8–96 known target tokens. Neither REPORT nor HELD_OUT influences selection
or receives training or generation.

Training starts fresh with seed 17, FP32, batch one, and AdamW at 0.0003. Each
epoch shuffles the selected examples deterministically. Evaluation occurs
before training and every ten epochs, up to 100 epochs. It uses greedy decoding
with a fixed 128-token budget and checks that inference prompts equal training
prefixes. Success requires all 16 target token sequences, parsed types, and
token-normalized scopes to match, with assistant/terminal-token cross-entropy
at most 0.1. Raw scope equality is also reported. The legacy decoder inserted
spaces in dotted or slash-separated scopes, even for perfect target tokens;
the current [scope policy](order-ablation.md#decoder-policy) repairs these cases.
Token-normalized matching remains separate from raw equality. Replay historical
results with their frozen source revision; do not rewrite the original metrics.

Outputs are separate per-checkpoint evaluations, a summary report, and a local
checkpoint containing weights, optimizer state, and RNG state. Add `--resume`
to recover from the last saved evaluation boundary. Changing the protocol,
implementation, runtime identity, or source inputs is rejected. Weights and
optimizer checkpoints must remain outside Git.

Exact token matches and exact text matches are reported separately because the
proof tokenizer normalizes whitespace. A pass demonstrates memorization of
deliberately easy, short, seen examples only. It establishes neither unseen
quality nor long-input quality, and does not satisfy a release gate. Existing
input license and provenance restrictions still apply; the diagnostic grants
no new redistribution rights and generates no teacher labels.

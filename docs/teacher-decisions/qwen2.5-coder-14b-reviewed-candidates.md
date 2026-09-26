# Qwen2.5-Coder 14B reviewed candidates

Review date: 2026-09-26. Status: approved for local DEV candidate generation
and individually reviewed reference overlays; not an automatic label verifier.

## Pinned sources

- Upstream: [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/tree/aedcc2d42b622764e023cf882b6652e646b95671).
- [Model card](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/blob/aedcc2d42b622764e023cf882b6652e646b95671/README.md): SHA-256 `2bb69760ec931f9adbe560e95b5b9848dc35198508825547ffdc19934fc6f261`.
- [License](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/blob/aedcc2d42b622764e023cf882b6652e646b95671/LICENSE): Apache-2.0, SHA-256 `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`.
- Runtime distribution: [Ollama qwen2.5-coder:14b-instruct-q4_K_M](https://ollama.com/library/qwen2.5-coder:14b-instruct-q4_K_M).

The reviewed license and card contain no restriction against using generated
outputs for downstream model training. The project therefore permits that use
subject to source eligibility, traceability and explicit content review. This
is an output-use decision, not a claim that all generated text is correct or
that redistribution rights follow automatically from the teacher license.
Apache notice and redistribution obligations still apply where relevant.

## Allowed workflow

Generate from complete DEV diffs using the reference-blind grounded prompt.
Pin the installed Ollama digest separately from the upstream card/tokenizer
revision: the upstream revision does not identify the quantized weight bytes.
Before generation, compare cached license/tokenizer hashes and runtime digest,
check advertised context capacity, and preserve the exact raw response.

Each candidate requires explicit full-diff review and evidence before inclusion
in an overlay. Assistant screening must remain identified as assistant review;
it is not independent human approval. Retain rejected attempts and unchanged
source references. No assistant-authored replacement text is a training label.
Training requires the separate data-readiness gate and selected run budget.

## Bounds

Use local inference only, never hosted teacher APIs. Do not use REPORT or
HELD_OUT to tune candidates. Start with a bounded local capacity pilot before
larger batches. The card's default context is 32,768 tokens; do not infer 128K
runtime support from its optional YaRN discussion. Teacher context changes do
not change student capacity or answer-token limits.

This decision does not grant automatic semantic approval, change the separate
7B smoke decision, launch student training, or authorize model/data publication.
Any release needs the applicable source-license, data-card and redistribution
review.

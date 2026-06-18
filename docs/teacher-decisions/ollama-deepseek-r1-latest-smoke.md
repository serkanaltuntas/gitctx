# Teacher Decision: Ollama DeepSeek-R1 Latest Smoke

Status: replaced after smoke generation.
Review date: 2026-06-18.
Review type: engineering license review, not legal advice.

## Decision

Use the local Ollama model `deepseek-r1:latest` for the first 17-record smoke
teacher run. This decision has been superseded by
[`ollama-qwen2.5-coder-7b-smoke.md`](ollama-qwen2.5-coder-7b-smoke.md) after
the R1 smoke run showed unreliable JSON/schema behavior.

This was a low-cost local smoke teacher, not a final training-data approval. The
goal is to test prompt shape, JSON validity, deterministic verifiers,
provenance capture, and human-review workflow before spending time on larger
teacher models.

## Smoke Result

The first generation smoke produced 5 generated records and 12 failed records
from 17 inputs. Failure modes included missing JSON, malformed output shape,
hallucinated evidence paths, and invalid Conventional Commit headers.

Do not use this teacher as the primary smoke or pilot generator. It remains
available only as a possible reviewer/fallback candidate after a separate
decision.

## Model Record

| Field | Value |
|---|---|
| teacher_model_id | `ollama/deepseek-r1:latest` |
| runtime | `ollama` |
| runtime_model_id | `deepseek-r1:latest` |
| ollama_tag_id | `6995872bfe4c` |
| size | `5.2 GB` |
| context_length | `128K` |
| source_url | <https://ollama.com/library/deepseek-r1> |
| tags_url | <https://ollama.com/library/deepseek-r1/tags> |
| license | MIT |
| license_signal | Ollama model metadata and DeepSeek-R1 family license signal |
| local_or_hosted | local/self-hosted only for this approval |
| prompt_version | `commit-message-teacher-v0.1` |
| prompt_path | `prompts/commit-message-teacher-v0.1.md` |

## Output-Use Decision

| Question | Decision |
|---|---|
| output_training_allowed | yes, for the bounded smoke audit |
| redistribution_of_outputs_allowed | conditional on source repository license and later data-card review |
| hosted_api_outputs_allowed | no |
| chain_of_thought_storage_allowed | no |
| held_out_label_use_allowed | no |
| release_dataset_allowed | no, requires a separate decision |

Rationale: this smoke uses a local Ollama DeepSeek-R1 tag with an MIT license
signal. Because generated labels are derived from source diffs, any public
release remains conditional on source-license and data-card review.

## Approval Scope

Allowed:

- run local Ollama inference with `deepseek-r1:latest`;
- generate labels for the 17 accepted smoke source-diff records;
- store prompt, decoding config, source commit id, teacher tag id, verifier
  results, and human-review status for every generated label;
- use the smoke sample to compare local teacher quality against deterministic
  validators and later larger-teacher outputs.

Not allowed yet:

- train a public gitctx model on these labels;
- publish a full derived dataset;
- use hosted teacher APIs;
- include teacher outputs in HELD_OUT labels;
- store chain-of-thought or long reasoning traces;
- continue beyond smoke scale without a follow-up decision.

## Required Generation Config

```yaml
teacher_model_id: ollama/deepseek-r1:latest
teacher_runtime: ollama
teacher_runtime_model_id: deepseek-r1:latest
teacher_revision: 6995872bfe4c
teacher_license: MIT
teacher_size: 5.2 GB
teacher_context_length: 128K
prompt_version: commit-message-teacher-v0.1
temperature: 0.0
top_p: 1.0
max_new_tokens: 256
output_format: json
store_reasoning_trace: false
sample_limit: 17
```

## Exit Criteria

Before these labels can become training data, record:

- exact generated-label artifact id and checksums;
- verifier pass/fail rates;
- human review of all 17 smoke labels;
- failure categories and prompt changes;
- whether this local teacher is good enough for the 250-500 record pilot or
  should be replaced by a larger teacher.

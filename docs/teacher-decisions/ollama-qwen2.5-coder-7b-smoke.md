# Teacher Decision: Ollama Qwen2.5-Coder 7B Smoke

Status: approved for smoke and bounded pilot audit only.
Review date: 2026-06-18.

Use the local Ollama model `qwen2.5-coder:7b` as the primary small teacher for
the first commit-message smoke and pilot audit.

This is a local smoke/pilot teacher decision, not a final training-data
approval. Promotion to training data still requires human review, a data card,
artifact checksums, and a follow-up approval decision.

## Model Record

| Field | Value |
|---|---|
| teacher_model_id | `ollama/qwen2.5-coder:7b` |
| runtime | `ollama` |
| runtime_model_id | `qwen2.5-coder:7b` |
| ollama_tag_id | `dae161e27b0e` |
| size | `4.7 GB` |
| context_length | `32K` |
| source_url | <https://ollama.com/library/qwen2.5-coder> |
| source_model_card | <https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct> |
| license_url | <https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/LICENSE> |
| license | `Apache-2.0` |
| prompt_version | `commit-message-teacher-v0.1` |
| prompt_path | `prompts/commit-message-teacher-v0.1.md` |

## Decision

Allowed:

- run local Ollama inference with `qwen2.5-coder:7b`;
- generate the smoke artifact and next bounded pilot audit labels;
- keep generated outputs in private data artifacts while review is pending;
- store prompt, decoding config, source commit id, teacher tag id, verifier
  score, and parser result with every output.

Not allowed:

- use hosted teacher APIs;
- include teacher outputs in HELD_OUT labels;
- publish private diff-bearing artifacts before a data-card and redistribution
  review;
- promote generated labels to training data without a follow-up approval.

## Smoke Result

The first local smoke generation completed with:

- input records: 17;
- generated records: 17;
- failed records: 0;
- schema/provenance validator: passed.

Some valid outputs received partial verifier scores because scope names were
more semantic than path-derived. Treat this as prompt and verifier tuning input
before scaling to the 250-500 record pilot.

## Required Metadata

```text
teacher_model_id: ollama/qwen2.5-coder:7b
teacher_runtime: ollama
teacher_runtime_model_id: qwen2.5-coder:7b
teacher_revision: dae161e27b0e
teacher_license: Apache-2.0
teacher_size: 4.7 GB
teacher_context_length: 32K
prompt_version: commit-message-teacher-v0.1
temperature: 0.0
top_p: 1.0
max_new_tokens: 256
output_format: json
```

## Exit Criteria Before Training Approval

- human review of at least 100 generated examples;
- measured accept/edit/reject rates;
- top verifier and human-review failure categories;
- explicit scope-quality decision;
- artifact checksums recorded in the private data repo;
- source-license and teacher-license summary in the data card;
- follow-up decision marked `approved_for_training`.

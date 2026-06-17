# Teacher Decision: DeepSeek-R1-0528 Audit

Status: approved for audit only.
Review date: 2026-06-17.
Review type: engineering license review, not legal advice.

## Decision

Use `deepseek-ai/DeepSeek-R1-0528` for the first 1K-5K teacher-label audit
sample, subject to the constraints below.

This is not an approval for a release dataset or final model training run. It
only allows a bounded audit sample so gitctx can test prompt quality,
provenance capture, deterministic verifiers, and human review workflow.

## Model Record

| Field | Value |
|---|---|
| model_id | `deepseek-ai/DeepSeek-R1-0528` |
| source_url | <https://huggingface.co/deepseek-ai/DeepSeek-R1-0528> |
| raw_model_card | <https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/raw/main/README.md> |
| revision | `4236a6af538feda4548eca9ab308586007567f52` |
| branch | `main` |
| last_modified | `2025-05-29T11:37:44.000Z` |
| license | MIT |
| license_signal | Hugging Face tag `license:mit`; model card license section |
| local_or_hosted | local/self-hosted only for this approval |
| prompt_version | `commit-message-teacher-v0.1` |
| prompt_path | `prompts/commit-message-teacher-v0.1.md` |

## Output-Use Decision

| Question | Decision |
|---|---|
| output_training_allowed | yes, for the bounded audit sample |
| redistribution_of_outputs_allowed | conditional on source repository license and later data-card review |
| hosted_api_outputs_allowed | no, not under this decision |
| chain_of_thought_storage_allowed | no |
| held_out_label_use_allowed | no |
| release_dataset_allowed | no, requires a separate decision |

Rationale: the model card states that the DeepSeek-R1 series is under MIT and
supports commercial use and distillation. Because teacher labels also include
source-code-derived context, publication of full derived examples remains
conditional on the source repository license and release data-card review.

## Approval Scope

Allowed:

- run local or self-hosted inference with the pinned revision above;
- generate 1K-5K audit labels from eligible permissively licensed source diffs;
- store prompt, decoding config, source commit id, teacher revision, verifier
  results, and human review status for every generated label;
- use the audit sample to measure label quality, prompt failures, verifier
  coverage, and data-pipeline correctness.

Not allowed yet:

- train a public gitctx model on these labels;
- publish a full derived dataset;
- use hosted teacher APIs;
- include teacher outputs in HELD_OUT labels;
- store chain-of-thought or long reasoning traces;
- continue beyond 5K generated labels without a follow-up decision.

## Required Generation Config

Initial audit runs should use the following baseline config unless a later
decision records a change:

```yaml
teacher_model_id: deepseek-ai/DeepSeek-R1-0528
teacher_revision: 4236a6af538feda4548eca9ab308586007567f52
prompt_version: commit-message-teacher-v0.1
temperature: 0.0
top_p: 1.0
max_new_tokens: 256
output_format: json
store_reasoning_trace: false
sample_limit: 5000
```

## Exit Criteria

Before these labels can become training data, record a follow-up decision with:

- exact source repository manifest and licenses;
- audit report over 1K-5K generated labels;
- verifier pass/fail rates;
- human review of at least 100 examples;
- failure categories and prompt changes;
- redistribution decision for derived examples;
- model-training approval or rejection.

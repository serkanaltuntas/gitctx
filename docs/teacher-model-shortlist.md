# Teacher Model License Shortlist

Status: initial shortlist. Do not generate training labels from any teacher
until the exact model/version license decision is recorded in this document.

## Policy

Teacher labels are training data. A teacher model is usable only when its exact
model/version license and terms allow generated outputs to train downstream
models.

Rules:

- prefer self-hosted/local inference;
- do not use closed-model outputs;
- do not use hosted APIs unless the provider terms explicitly allow downstream
  training/distillation from outputs;
- record model id, version/date, source URL, license, output-use decision, and
  review date before generation;
- keep teacher outputs out of held-out eval labels;
- preserve prompt, decoding config, source commit id, and verifier result for
  every generated label.

## Current Decision

Start with a small audit using a teacher whose license explicitly permits
distillation. Do not run large-scale generation until prompts, verifier filters,
and provenance records have passed the 1K-5K audit sample.

The first local smoke decision is recorded in
[`teacher-decisions/ollama-deepseek-r1-latest-smoke.md`](teacher-decisions/ollama-deepseek-r1-latest-smoke.md).
The larger DeepSeek-R1-0528 audit decision remains available as a higher-cost
quality anchor in
[`teacher-decisions/deepseek-r1-0528-audit.md`](teacher-decisions/deepseek-r1-0528-audit.md).

## Candidate Table

| Candidate | License signal | Output/distillation decision | Status |
|---|---|---|---|
| `ollama/deepseek-r1:latest` | MIT license signal; local Ollama model tag `6995872bfe4c` | Approved for the 17-record local smoke audit only | First smoke teacher |
| `deepseek-ai/DeepSeek-R1-0528` | MIT license; model card says DeepSeek-R1 series supports commercial use and distillation | Approved for local audit after recording exact artifact hash and prompt version | Preferred first teacher candidate |
| `deepseek-ai/DeepSeek-R1` | MIT license; model card says derivative work including distillation is allowed | Approved for local audit if `0528` is unavailable or too costly | Backup first teacher candidate |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | Apache-2.0 model license | Candidate, but record an explicit output-use decision before labels are kept | Strong code teacher candidate |
| `Qwen/Qwen2.5-Coder-14B-Instruct` or `7B-Instruct` | Apache-2.0 family license signal | Candidate, useful when local compute is smaller; output-use decision still required | Smaller code teacher candidate |
| `ibm-granite/granite-20b-code-instruct-8k` | Apache-2.0 license signal and permissive instruction-data note | Candidate, but output-use decision and exact model-card license review required | Enterprise-friendly code teacher candidate |
| `ibm-granite/granite-8b-code-instruct-4k` | Apache-2.0 license signal | Candidate for low-cost audit; output-use decision still required | Smaller backup |
| OLMo instruct models | Apache-2.0 base family, but some instruct cards note third-party generated outputs with additional terms | Avoid for first teacher unless exact downstream output rights are reviewed | Not first choice |

## Source Links

- DeepSeek-R1-0528: <https://huggingface.co/deepseek-ai/DeepSeek-R1-0528>
- DeepSeek-R1: <https://huggingface.co/deepseek-ai/DeepSeek-R1>
- Qwen2.5-Coder-32B-Instruct: <https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct>
- Qwen2.5-Coder-32B-Instruct license: <https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct/blob/main/LICENSE>
- Granite 20B Code Instruct: <https://huggingface.co/ibm-granite/granite-20b-code-instruct-8k>
- Granite 8B Code Instruct: <https://huggingface.co/ibm-granite/granite-8b-code-instruct-4k>
- OLMo 2 7B Instruct caveat example: <https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct>

## First Audit Recommendation

Use `ollama/deepseek-r1:latest` for the first 17-record local smoke run. Use
`deepseek-ai/DeepSeek-R1-0528` as the larger quality anchor if the local smoke
teacher fails JSON validity, factuality, or human-review quality.

Audit target:

- 1K-5K diffs;
- at least two prompt templates;
- deterministic decoding config first;
- JSON-only output schema;
- fixture scorer pass/fail report;
- human review sample of at least 100 examples;
- explicit rejection categories for hallucinated APIs, vague subjects, invalid
  Conventional Commit format, missing body/footer, and mixed-change misses.

## License Decision Template

```text
model_id:
version_or_revision:
source_url:
license:
review_date:
reviewer:
local_or_hosted:
output_training_allowed: yes | no | unclear
redistribution_of_outputs_allowed: yes | no | unclear
notes:
decision: approved_for_audit | approved_for_training | rejected | pending
```

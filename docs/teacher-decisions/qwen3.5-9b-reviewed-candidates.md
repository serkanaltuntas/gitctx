# Qwen3.5 9B reviewed candidates

Review date: 2026-09-26. The project permits local DEV candidate generation and
individually reviewed reference overlays from the pinned open distribution.
This decision does not automatically approve any generated content.

The reviewed upstream is [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a).
Its [license](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/LICENSE)
is Apache-2.0, including the Alibaba Cloud copyright notice. The
[model card](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md)
identifies the same license. No additional output-use restriction against training
downstream models was found in the reviewed license/card. Individual source and
output rights still require review; model weights are not published by this work.

Pin the installed [Ollama 9B distribution](https://ollama.com/library/qwen3.5:9b)
digest independently from the upstream metadata/tokenizer revision. Retain hashes
of the license, card, tokenizer, tokenizer configuration and native template;
compare the runtime distribution license with the reviewed upstream text. The
reviewed runtime has the standard Apache appendix placeholder where upstream
substitutes "Copyright 2026 Alibaba Cloud"; normalized terms otherwise match.
Retain the actual upstream copyright notice. A tag
alone does not establish provenance or exact upstream quantized weight lineage.

Use complete DEV source diffs, never original references, historical subjects or
assistant review notes. Record the exact native non-thinking prompt, full raw
response, generation parameters and runtime token counts. Native prompt rendering
must match the pinned template, including the empty thinking block. Context and
output limits are bounded locally; upstream benchmark recommendations do not
establish performance under these constraints.

Only the exact decoded open-teacher message fields may become a candidate target.
All candidates need separate full-source review. Assistant screening is not
independent human approval. No hosted inference, REPORT/HELD_OUT tuning, bulk
acceptance, new student training or model/data publication is authorized here.
Readiness, resource-budget selection and release decisions remain separate.

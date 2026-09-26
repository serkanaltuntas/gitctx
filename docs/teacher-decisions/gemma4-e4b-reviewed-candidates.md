# Gemma 4 E4B reviewed candidates

Review date: 2026-09-26. Approved for local DEV candidate generation and
individually reviewed reference overlays, not automatic factual approval.

The reviewed upstream is [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it/tree/ee0ef6023621cff504d758262d4e04895a5af4a2).
Its [card](https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/README.md)
identifies Apache-2.0 and links Google's [license](https://ai.google.dev/gemma/apache_2).
The local Ollama distribution license matches the canonical Apache-2.0 text
apart from whitespace. Neither the reviewed license nor the card adds a
restriction against using generated text for downstream training. The project
permits that use subject to source eligibility and explicit content review.
Output correctness and redistribution rights still need their own review.

Pin the installed [Ollama E4B distribution](https://ollama.com/library/gemma4:e4b)
digest separately from the upstream card/tokenizer revision. Record license,
card, tokenizer, tokenizer configuration and native template hashes in each
experiment. A mutable tag alone is insufficient. The non-thinking text-only
rendering must match the pinned upstream template; verify exact prompt token
counts against the runtime. Do not use a Qwen chat template with this model.

Generate from complete DEV source diffs without original references, historical
subjects or assistant review notes. Retain raw outputs, invalid attempts and
explicit evidence for every accepted candidate. Assistant review is not
independent human approval. No hosted inference, REPORT/HELD_OUT tuning,
automatic bulk promotion, new student training or model/data publication is
authorized by this teacher decision. Separate readiness, budget and release
gates remain applicable.

# Evidence windows and reference review

`gitctx.window_readiness` builds lossless, target-independent windows using an
existing frozen student tokenizer and split. It verifies the parent input,
output and implementation hashes before preparing a new immutable run directory.
It does not fit a tokenizer, modify references, or launch training.

```bash
uv run python -m gitctx.window_readiness \
  --data-dir /path/to/data --parent STUDENT_INPUT_RUN --run WINDOW_RUN
```

The student contract remains 8,192 tokens, with 256 reserved for the answer and
EOS. A full prompt that fits retains the shared training/inference format.
Otherwise file headers and whole hunks are packed into windows. Oversized hunks
split at physical LF boundaries; a single oversized line splits at Unicode
character boundaries. Primary source intervals cover the original diff exactly,
with no gaps, overlaps, or lost bytes. Repeated file/hunk/neighbor context is
bounded separately and carries explicit source coordinates. Global changed paths
remain visible. Metadata that cannot fit fails explicitly.

Every prepared window has `target: null`. Exact quotation-to-window mapping is
structural evidence, not a factuality verdict. `materialize_window` requires a
separate verified alignment with source/target/review hashes, permitted target
origin, every nonempty target line, and supporting quotations visible in that
window. Callers must establish semantic correctness and trustworthy review
provenance before constructing this attestation. A serialized `verified` field
is not a cryptographic approval. A whole-commit message must never be copied to
partial windows merely because the original record was accepted. Claims that
require several windows need a verified window-specific target or a separately
designed aggregation stage; this tool does not synthesize one automatically.

`gitctx.review_batch` records local open-model audit candidates for an explicit
frozen list of DEV IDs. It is an opt-in network command to local Ollama, separate
from the eventual product CLI. Use a license-reviewed Apache-2.0 Instruct model
whose chat template matches the ChatML rendering in `reference_review`, its exact
matching tokenizer, and a recorded upstream revision. Mutable tags alone are
insufficient: the batch protocol pins and checks the installed digest, tokenizer,
license, source, selection, prompt version and implementation hashes. The
reviewer must be distinct from the original label generator when assessing
independent model review; it still does not constitute human review.

```bash
uv run python -m gitctx.review_batch \
  --source /path/to/sft.jsonl --selection /path/to/dev-ids.json \
  --tokenizer /path/to/reviewer-tokenizer.json --license /path/to/LICENSE \
  --model EXACT_INSTRUCT_TAG --model-revision UPSTREAM_REVISION \
  --output /path/to/review-output
```

References are split into nonempty-line claims. Long source diffs are partitioned
without dropping source text. Each part is explicitly marked partial, and cannot
produce a whole-commit correction. This partitioning is independent of student
windows and is not a target alignment. A valid review requires complete claim
indices, exact quotations inside the supplied source, changed-line evidence for
decisive verdicts, normal generation termination, and agreement between local
and runtime prompt token counts. Invalid output is retained as unresolved.
These checks do not establish that a model interpreted the evidence correctly.

Resume accepts only the same protocol and retains prior results, including
failures. A damaged JSONL tail fails rather than being silently discarded. Original
references and historical scores remain immutable. Open-teacher corrections are
candidates until verified; assistant prose and screening notes are never training
labels. No review result automatically promotes a label or opens the training
gate. Source and generated-output redistribution still require a release-specific
license review; local operational artifacts are not a public dataset release.

`window_targets.generate` can propose a concise window target using a separately
reviewed Apache-2.0 teacher. Its prompt does not include the original reference,
historical subject, or screening notes. It checks the installed model digest,
prompt/answer budgets, generation termination and Conventional Commit syntax.
It always returns a candidate requiring separate semantic verification. Grammar
constraints establish output shape only; they cannot correct an invented change.
Before scaling a reviewer, qualify it on known errors and positive controls and
retain failed or interrupted pilots. A failed qualification blocks reference
promotion; it does not establish a corpus-wide label error rate.

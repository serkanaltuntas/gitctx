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
retain failed or interrupted pilots. A failed qualification blocks automatic
reference promotion; it does not establish a corpus-wide label error rate.

`delta_review` separates factual review from correction generation. It preserves
all source rows and uses source line identifiers as evidence; quotations are
looked up from the source, never copied from model text. Complete inputs choose
an adequate reviewer context without truncation. An input exceeding the reviewed
limit fails explicitly. Synthetic positive/negative controls belong exclusively
to evaluation, and a structurally valid decision never automatically approves a
reference or training target. Model/tokenizer/license identity and qualification
results must be pinned by the calling experiment protocol.

`delta_targets.generate` produces reference-blind, full-diff candidates with either
clean before/after hunks (`source_format="hunks"`) or the original unified diff
(`"unified"`). Both preserve the complete supplied source. The hunk view removes
patch prefixes and keeps metadata separately, so source row identifiers cannot
be mistaken for changed code. Record the chosen format, prompt implementation,
temperature and seed in the experiment protocol; retain every rejected attempt.
Only real DEV records are eligible. Teachers must have an explicitly reviewed
Apache-2.0 license, a pinned installed digest and matching tokenizer. Generated
output licensing still needs a release-specific review before redistribution.

`grounded_targets.generate` provides a separate versioned JSON-header prompt
that explicitly distinguishes documentation, tests, annotations, preserved
context and actual runtime changes. It preserves the original complete unified
diff and excludes references, historical subjects and reviewer notes. It supports
teacher contexts up to 32,768 tokens, checks the installed model's advertised
context capacity, and verifies runtime prompt token counts. This does not enlarge
the student's 8,192-token context or 256-token answer reserve. Pin the module,
decoder, schema dependency, tokenizer, teacher metadata and generation settings
in the experiment protocol. Prior prompt versions remain available for replay.
The stricter instructions are a generation method, not a factuality guarantee.

`reference_overlay.build_override` binds a selected candidate to an explicit
full-diff semantic attestation. It reconstructs the message from the raw teacher
JSON fields and refuses substituted reviewer text. Each target line must have
supporting source evidence, including an actual changed line. Candidate, source,
original reference, replacement and verification hashes remain separate; the
original reference is never edited in place. The attestation must truthfully
identify assistant versus human review. This is a provenance and binding check,
not an automatic factuality judge: a caller must actually examine the full diff.
An assistant-verified open-teacher overlay remains distinct from an independently
human-reviewed reference and does not by itself open the training gate. Assistant
review prose is never a training target. Review selection must also retain the
frozen training/validation assignment; preparing an overlay does not move a
validation reference into training or rewrite historical evaluation scores.

`reviewed_references.resolve_reference` selects a retained original or reviewed
open-teacher overlay using an audited index row and its hash-bound artifact.
The caller supplies the frozen partition independently; mismatched source,
original target, artifact, teacher provenance or partition fails. The selected
target carries explicit assistant/human provenance without mutating the source
record or granting run approval. Callers must still audit and pin the complete
index and artifact collection; a matching hash is not semantic verification.

`materialize_reviewed_full` feeds that selection through the common student
prompt, fixed answer reserve and assistant-only loss mask. It explicitly rejects
long inputs; it never copies a whole-commit target onto partial windows. A
separate window alignment or verified aggregation path remains required before
those inputs can be consumed by a trainer.

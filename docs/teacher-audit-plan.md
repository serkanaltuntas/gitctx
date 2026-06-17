# Teacher Audit Plan

Status: initial plan for the first teacher-label audit.

The first audit tests whether a licensed teacher can produce useful
Conventional Commit labels from real Git diffs before gitctx spends compute on
large-scale data generation or proof-model training.

## Inputs

Use only source repositories that pass the permissive-license manifest review.
For the first audit, prefer small and medium diffs with clear intent.

The manifest format and source-selection rules are defined in
[`source-manifest.md`](source-manifest.md).

Exclude:

- generated or vendored files;
- lockfile-only changes unless the expected message is dependency-related;
- massive formatting-only changes;
- merge commits;
- commits selected for HELD_OUT evaluation;
- files whose license or redistribution status is unclear.

## Sample Size

| Stage | Size | Purpose |
|---|---:|---|
| smoke | 25-50 diffs | validate prompt, schema, parser, and provenance writer |
| pilot | 250-500 diffs | find common teacher and verifier failures |
| audit | 1K-5K diffs | decide whether labels can become training data |

Stop at each stage if schema validity, factuality, or human accept rate is
poor enough that more generation would only create cleanup work.

## Teacher

The first approved audit teacher is recorded in
[`teacher-decisions/deepseek-r1-0528-audit.md`](teacher-decisions/deepseek-r1-0528-audit.md).

No other teacher may generate kept labels until it has its own decision record.

## Prompt

Use `prompts/commit-message-teacher-v0.1.md` for the first smoke run.

The prompt must require:

- JSON-only output;
- a Conventional Commit header;
- optional body lines when the diff needs explanation;
- optional footers for breaking changes or issue references;
- no chain-of-thought or long reasoning trace;
- warnings when the diff looks mixed and should be split.

## Required Output Schema

```json
{
  "header": "feat(scope): concise subject",
  "body": ["optional explanatory line"],
  "footers": ["BREAKING CHANGE: optional contract note"],
  "type": "feat",
  "scope": "scope",
  "confidence": 0.0,
  "warnings": ["optional warning"],
  "evidence_paths": ["src/example.py"]
}
```

Rules:

- `header` is required and must parse as Conventional Commit.
- `body`, `footers`, `warnings`, and `evidence_paths` are arrays.
- `type` must match the header type.
- `scope` must match the header scope, or be `null` if no scope is used.
- `confidence` is a number from 0.0 to 1.0.
- The teacher must not include markdown fences around JSON.

## Required Provenance

Every generated label must preserve:

- source repository URL;
- source repository license;
- source commit id;
- parent commit id;
- diff extraction command or artifact id;
- source data split;
- teacher model id;
- teacher model revision;
- teacher model license;
- prompt version;
- decoding config;
- generation timestamp;
- parser result;
- verifier score;
- human-review status when available.

Generated-label records must satisfy
[`schemas/generated-label.schema.json`](../schemas/generated-label.schema.json)
and the dependency-free validator in `src/gitctx/provenance.py`.

## Verifier Checks

Run deterministic checks before human review:

- JSON schema validity;
- Conventional Commit parse validity;
- allowed type;
- scope matches touched paths when clear;
- subject is specific and concise;
- body present when the fixture or heuristic expects it;
- footer present when breaking-change cues exist;
- warnings present for likely mixed changes;
- forbidden claims absent from the diff;
- evidence paths exist in the diff context.

## Human Review

Review at least 100 examples before any training-data decision.

Record:

- accept as-is;
- accept with light edit;
- wrong type;
- wrong scope;
- vague subject;
- factual hallucination;
- missing body;
- missing footer;
- mixed change missed;
- too verbose;
- invalid JSON or invalid Conventional Commit.

## Promotion Gate

The audit can be promoted to training-data production only after a follow-up
decision records:

- source manifest and license summary;
- exact teacher decision;
- prompt version and config;
- audit sample size;
- verifier pass rates;
- human accept/edit/reject rates;
- top failure categories;
- redistribution decision;
- training approval.

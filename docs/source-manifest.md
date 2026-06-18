# Source Manifest

Status: initial format for source selection and generated-label provenance.

The source manifest decides which repositories and revisions may enter a
gitctx audit or training dataset. It is intentionally separate from teacher
selection: a generated label is usable only when both the source manifest and
teacher decision allow the intended use.

## First-Audit Source Policy

For the first teacher audit, prefer repositories that are:

- public;
- independently useful as software projects;
- permissively licensed with a clear SPDX license id;
- small enough to inspect manually when needed;
- active enough to contain meaningful diffs;
- not selected for HELD_OUT evaluation.

Default allowed source licenses for the first audit:

- `Apache-2.0`;
- `MIT`;
- `BSD-2-Clause`;
- `BSD-3-Clause`;
- `ISC`.

`MPL-2.0` requires explicit review before use. GPL, AGPL, LGPL, custom,
unknown, missing, or source-available-only licenses are rejected for the first
audit unless a later written decision changes this policy.

## Exclusion Rules

Exclude repositories or commits when:

- license files are missing, generated, contradictory, or unclear;
- the repository mainly mirrors vendored code;
- the relevant files have different per-file license headers;
- commit diffs are mostly generated artifacts, vendored files, lockfiles, or
  formatting churn;
- the commit is a merge commit;
- the commit is part of a HELD_OUT evaluation slice;
- terms of service or dataset redistribution rules are unclear.

## Manifest Entry

The JSONL source manifest uses one object per repository revision.

Required fields are defined in
[`schemas/source-manifest-entry.schema.json`](../schemas/source-manifest-entry.schema.json).
The dependency-free Python validator lives in `src/gitctx/provenance.py`.

Example:

```json
{
  "repo_url": "https://example.com/owner/repo",
  "default_branch": "main",
  "source_license": "Apache-2.0",
  "license_url": "https://example.com/owner/repo/LICENSE",
  "license_review_date": "2026-06-17",
  "reviewer": "Serkan Altuntas",
  "review_status": "approved_for_audit",
  "source_revision": "0123456789abcdef0123456789abcdef01234567",
  "allowed_splits": ["DEV", "REPORT"],
  "exclude_globs": ["vendor/**", "dist/**", "*.lock"],
  "notes": "Format example only; not an approved real source."
}
```

## Generated-Label Provenance

Every teacher-generated label must carry enough provenance to reproduce,
filter, reject, or audit the example later.

Required fields are defined in
[`schemas/generated-label.schema.json`](../schemas/generated-label.schema.json).

The record must include:

- source repository URL and license;
- source commit and parent commit;
- source data split;
- changed paths;
- teacher model id, revision, and license;
- prompt version;
- decoding config;
- generated message fields;
- parser/verifier outputs;
- human review status.

Teacher-generated labels must not be stored as HELD_OUT labels.

## Source-Diff Extraction

The first extraction scaffold lives in `src/gitctx/source_extract.py`.

It reads a local Git clone plus one source-manifest entry and emits JSONL
source-diff records. These records contain commit ids, parent ids, changed
paths, diffstat, and the historical subject. They intentionally do not include
full diff text; full diffs should be generated as local artifacts only after
redistribution review.

Required fields are defined in
[`schemas/source-diff.schema.json`](../schemas/source-diff.schema.json).

When a split-plan file is provided, extraction assigns `data_split` from the
matching repository/time window and skips commits outside the plan. This is the
required mode for any artifact used for `REPORT` or `HELD_OUT` claims. The
split-plan contract is documented in
[`docs/split-contract.md`](split-contract.md).

Example command shape:

```bash
python -m gitctx.source_extract /path/to/local/repo /path/to/source-entry.json --limit 25
```

Worker command shape with a split plan:

```bash
make pilot-source \
  GITCTX_DATA_DIR="$HOME/LAB/gitctx-data" \
  SPLIT_PLAN="$HOME/LAB/gitctx-data/manifests/split-plan.next.json"
```

## Example Files

- `examples/source-manifest.example.jsonl`
- `examples/generated-label.example.jsonl`

These are format examples only. They are not approved source data.

## First Audit Manifest

The first real audit source candidates are recorded in
`manifests/source-manifest.audit.jsonl`.

Current approved-for-audit repositories:

| Repository | License | Revision |
|---|---|---|
| `https://github.com/pallets/click` | `BSD-3-Clause` | `8a1b1a33d739be05b7e91251e3c0dde77c5e152f` |
| `https://github.com/psf/requests` | `Apache-2.0` | `d64b9ad4bf1c14e21e0df3f0f4320fec81180e91` |
| `https://github.com/pytest-dev/pluggy` | `MIT` | `7fce99cb955846901b22b051909aa4f30dc16128` |
| `https://github.com/encode/httpx` | `BSD-3-Clause` | `b5addb64f0161ff6bfe94c124ef76f6a1fba5254` |
| `https://github.com/python-attrs/attrs` | `MIT` | `89fae8300f484544c1b7678cea5efe58c551fbb9` |

These entries allow audit extraction only. They do not approve publishing full
diff-derived examples, generating more than the bounded audit sample, or using
labels for model training.

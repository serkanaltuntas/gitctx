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

## Example Files

- `examples/source-manifest.example.jsonl`
- `examples/generated-label.example.jsonl`

These are format examples only. They are not approved source data.

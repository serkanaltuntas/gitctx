# Data and Distillation Plan

The hard problem is not finding public repositories. The hard problem is
obtaining high-quality commit labels that are legally usable and semantically
grounded in diffs.

## Source Strategy

Use permissively licensed public repositories as sources of:

- diffs;
- file paths;
- repository metadata;
- weak historical commit messages;
- pull request and issue context when terms and licenses permit.

Historical commit messages are not automatically truth. Many are vague,
inconsistent, or unrelated to the actual diff.

Source eligibility is tracked in [`source-manifest.md`](source-manifest.md).
Do not generate kept labels from a repository revision that is missing from the
source manifest.

## Default Label Strategy

1. Extract eligible diffs and repository context.
2. Keep strong human Conventional Commit messages as gold or auxiliary labels.
3. Generate fresh Conventional Commit labels for most eligible diffs with a
   license-approved local/open teacher model.
4. Generate multiple candidates per diff when affordable.
5. Score candidates with deterministic verifiers.
6. Keep original human messages as comparison/context, not as automatic truth.
7. Reserve held-out repositories and time windows before teacher generation.

## Teacher Policy

- Self-hosted teacher inference is preferred.
- Hosted APIs are allowed only when output terms explicitly permit downstream
  training and privacy requirements are acceptable.
- Closed-model outputs are not allowed.
- Every teacher model and version requires a recorded license decision before
  use.

The initial teacher shortlist is tracked in
[`teacher-model-shortlist.md`](teacher-model-shortlist.md).

The first audit decision is tracked in
[`teacher-decisions/deepseek-r1-0528-audit.md`](teacher-decisions/deepseek-r1-0528-audit.md),
and the audit workflow is tracked in
[`teacher-audit-plan.md`](teacher-audit-plan.md).

## Required Provenance

Every generated label must record:

- source repository URL;
- source repository license;
- source commit id;
- source data split;
- teacher model id and version;
- teacher model license;
- prompt template version;
- decoding configuration;
- verifier score;
- human-review status when available.

The generated-label record format is defined in
[`source-manifest.md`](source-manifest.md) and
[`schemas/generated-label.schema.json`](../schemas/generated-label.schema.json).

## Publication Rule

Publish source manifests, scripts, filters, prompts, cards, and aggregate stats
by default. Publish full derived examples only after confirming that source and
teacher licenses allow redistribution.

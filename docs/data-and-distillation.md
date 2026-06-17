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

## Publication Rule

Publish source manifests, scripts, filters, prompts, cards, and aggregate stats
by default. Publish full derived examples only after confirming that source and
teacher licenses allow redistribution.

# AGENTS.md - Working on gitctx

This repository is intended to become a public open-source project. Treat every
file as public by default.

## Rules

- Keep all documents, code, comments, identifiers, and commit messages in
  English.
- Do not reference private projects, internal codenames, private repositories,
  customer data, private evals, or unpublished strategy notes.
- Do not train on closed-model outputs.
- Use only public repositories, datasets, and teacher models whose licenses
  permit the intended downstream use.
- Record source licenses, teacher licenses, prompt versions, data splits, and
  generated-label provenance.
- Prefer model/eval/data work before CLI polish.
- The CLI must not commit, push, or make network calls by default.
- Commits must use `serkan@altuntas.dev` as the author/committer email.

## Repository scope and synchronization

- Read this file and README.md before starting work. This repository holds
  shareable source code, schemas, tests, reusable training/evaluation tools and
  public documentation. Keep private inputs, detailed experiment outputs and
  personal planning records outside this repository. Never commit credentials,
  environment files, weights, checkpoints or caches.
- Standing maintainer instruction, 2026-09-24: after each meaningful completed
  unit of authorized work, review and validate its changes, create a GPG-signed
  commit and push it to the corresponding branch on `origin`. Routine pushes
  do not need another confirmation. Use the current task branch; for a new
  branch, push with upstream tracking. Do not automatically merge into `main`,
  tag a release or publish model/data artifacts.
- Fetch before committing/pushing and check the remote branch. Fast-forward a
  clean checkout when appropriate; preserve local and concurrent work when
  resolving divergence. Never force-push, rewrite shared history or overwrite
  someone else's changes. Stage only completed files belonging to the task;
  inspect the index and all outgoing commits for public suitability.
- Verify the push succeeded and report the branch and commit. If signing,
  authentication or networking fails, preserve the local work and explicitly
  report that synchronization is incomplete. Do not claim a local commit is
  backed up remotely.
- This is the development agent's completion workflow, not a scheduled job.
  The product CLI's no-automatic-commit/push/network behavior remains intact.

## Definition of Done

- The change is suitable for a public repository.
- License implications are documented when data, model weights, prompts, or
  generated labels are touched.
- No private names or private paths are introduced.
- Tests or documented verification are added for behavior changes.

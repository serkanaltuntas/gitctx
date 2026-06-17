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

## Definition of Done

- The change is suitable for a public repository.
- License implications are documented when data, model weights, prompts, or
  generated labels are touched.
- No private names or private paths are introduced.
- Tests or documented verification are added for behavior changes.


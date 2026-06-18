# Local Worker Runbook

Status: first smoke workflow.

This runbook describes how to run source-diff smoke extraction on any local
worker. It intentionally uses generic machine names only.

## Roles

- **Control workstation**: edits gitctx, reviews outputs, commits public
  changes.
- **Local worker**: clones approved source repositories and produces local
  artifacts.
- **Teacher worker**: reads source-diff artifacts and produces generated-label
  artifacts after a teacher decision permits it.

A single physical machine may play more than one role. Public gitctx docs do
not name private hardware, hostnames, or personal machine details.

## Data Directory

The Makefile uses a data directory outside the gitctx source repository:

```bash
make data-dir
```

Expected layout:

```text
$(GITCTX_DATA_DIR)/
  sources/
    github.com/
      owner/repo/
  artifacts/
    smoke/
      source-diffs.smoke.jsonl
      source-diffs.smoke.report.json
  cache/
```

Do not place source clones or generated artifacts inside the public gitctx
repository.

## Setup

```bash
git clone git@github.com:serkanaltuntas/gitctx.git
cd gitctx
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Source-Diff Smoke Extraction

Run:

```bash
make smoke
```

Outputs:

- `$(GITCTX_DATA_DIR)/artifacts/smoke/source-diffs.smoke.jsonl`
- `$(GITCTX_DATA_DIR)/artifacts/smoke/source-diffs.smoke.report.json`

The smoke artifact contains commit metadata, changed paths, diffstat, and
historical subjects. It does not contain full diff text.

Override defaults when needed:

```bash
make smoke GITCTX_DATA_DIR="$HOME/work/gitctx-data" SMOKE_RECORDS=25
```

Use `PYTHON=python` if your virtual environment exposes `python` but not
`python3`.

## After The Run

Review the report first:

```bash
make smoke-check
```

Then send the report and smoke JSONL back to the control workstation for review.
Do not publish the local artifact until redistribution review is complete.

## Finalize Private Artifact

If the smoke run writes into a private data repository, normalize machine-local
paths, validate records, and refresh checksums:

```bash
make smoke-finalize GITCTX_DATA_DIR="$HOME/LAB/gitctx-data"
```

Then inspect and commit the private data repository:

```bash
cd "$HOME/LAB/gitctx-data"
git status --short --ignored
git diff -- artifacts/smoke/source-diffs.smoke.report.json checksums/sha256.txt
```

## Prepare Smoke Review Decisions

After the smoke artifact is committed to the private data repository, create a
separate review-decision JSONL template. Keep this file private until the
source redistribution and data-card review says otherwise.

```bash
make smoke-review-template \
  GITCTX_DATA_DIR="$HOME/LAB/gitctx-data" \
  REVIEWER="reviewer@example.com"
```

Edit `reviews/source-diffs.smoke.review.jsonl` in the private data repository.
Each source-diff record starts as `needs_review`. Change it to
`accepted_for_teacher_labeling` only when the diff is small/coherent enough for
the first teacher-label audit. Use `rejected` for noisy, oversized, generated,
vendor, formatting-only, or ambiguous changes.

Then validate and refresh checksums:

```bash
make smoke-review-check GITCTX_DATA_DIR="$HOME/LAB/gitctx-data"
```

## Prepare Teacher Inputs

After source-diff review decisions are complete, create teacher input prompts
for only the records marked `accepted_for_teacher_labeling`:

```bash
make teacher-inputs GITCTX_DATA_DIR="$HOME/LAB/gitctx-data"
make teacher-input-check GITCTX_DATA_DIR="$HOME/LAB/gitctx-data"
```

This writes `artifacts/teacher/teacher-inputs.smoke.jsonl` in the private data
repository. It includes full diff text and must remain private until a later
data-card and redistribution review approves any public release shape.

The default smoke teacher identity is the local Ollama model
`ollama/qwen2.5-coder:7b` with tag id `dae161e27b0e`. If the worker uses a
different local tag, override the teacher metadata when creating inputs and add
or update the matching teacher decision before keeping outputs.

## Generate Smoke Labels With Ollama

After teacher inputs are ready and Ollama has the approved local model, generate
one label per input record:

```bash
make teacher-generate GITCTX_DATA_DIR="$HOME/LAB/gitctx-data"
make teacher-generate-check GITCTX_DATA_DIR="$HOME/LAB/gitctx-data"
```

This writes:

```text
artifacts/teacher/generated-labels.smoke.jsonl
artifacts/teacher/generated-labels.smoke.report.json
```

If the worker stops, rerun `make teacher-generate`; it resumes by skipping
existing generated-label ids.

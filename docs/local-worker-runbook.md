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

Set a data directory outside the gitctx source repository:

```bash
export GITCTX_DATA_DIR="$HOME/LAB/gitctx-data"
mkdir -p "$GITCTX_DATA_DIR"
```

Expected layout:

```text
$GITCTX_DATA_DIR/
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
python -m gitctx.worker_smoke \
  --manifest manifests/source-manifest.audit.jsonl \
  --data-dir "$GITCTX_DATA_DIR" \
  --records 50
```

Outputs:

- `$GITCTX_DATA_DIR/artifacts/smoke/source-diffs.smoke.jsonl`
- `$GITCTX_DATA_DIR/artifacts/smoke/source-diffs.smoke.report.json`

The smoke artifact contains commit metadata, changed paths, diffstat, and
historical subjects. It does not contain full diff text.

## After The Run

Review the report first:

```bash
python -m json.tool "$GITCTX_DATA_DIR/artifacts/smoke/source-diffs.smoke.report.json"
wc -l "$GITCTX_DATA_DIR/artifacts/smoke/source-diffs.smoke.jsonl"
```

Then send the report and smoke JSONL back to the control workstation for review.
Do not publish the local artifact until redistribution review is complete.

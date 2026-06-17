GITCTX_DATA_DIR ?= $(HOME)/LAB/gitctx-data
PYTHON ?= python3
SMOKE_MANIFEST ?= manifests/source-manifest.audit.jsonl
SMOKE_RECORDS ?= 50
SMOKE_REPORT = $(GITCTX_DATA_DIR)/artifacts/smoke/source-diffs.smoke.report.json
SMOKE_JSONL = $(GITCTX_DATA_DIR)/artifacts/smoke/source-diffs.smoke.jsonl

.PHONY: data-dir smoke smoke-check test fixture-eval

data-dir:
	mkdir -p "$(GITCTX_DATA_DIR)"

smoke: data-dir
	$(PYTHON) -m gitctx.worker_smoke \
		--manifest "$(SMOKE_MANIFEST)" \
		--data-dir "$(GITCTX_DATA_DIR)" \
		--records "$(SMOKE_RECORDS)"

smoke-check:
	$(PYTHON) -m json.tool "$(SMOKE_REPORT)"
	wc -l "$(SMOKE_JSONL)"

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests

fixture-eval:
	PYTHONPATH=src $(PYTHON) -m gitctx.eval fixtures/dev/commit_message_cases.jsonl

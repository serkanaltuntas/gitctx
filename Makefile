GITCTX_DATA_DIR ?= $(HOME)/LAB/gitctx-data
PYTHON ?= python3
SMOKE_MANIFEST ?= manifests/source-manifest.audit.jsonl
SMOKE_RECORDS ?= 50
SMOKE_REPORT = $(GITCTX_DATA_DIR)/artifacts/smoke/source-diffs.smoke.report.json
SMOKE_JSONL = $(GITCTX_DATA_DIR)/artifacts/smoke/source-diffs.smoke.jsonl
REVIEWER ?= reviewer@example.com
OLLAMA_NUM_CTX ?= 8192
OLLAMA_NUM_PREDICT ?= 384
OLLAMA_REQUEST_TIMEOUT ?= 300

.PHONY: data-dir smoke smoke-check smoke-finalize smoke-review-template smoke-review-check teacher-inputs teacher-input-check teacher-generate teacher-generate-check test fixture-eval

data-dir:
	mkdir -p "$(GITCTX_DATA_DIR)"

smoke: data-dir
	PYTHONPATH=src $(PYTHON) -m gitctx.worker_smoke \
		--manifest "$(SMOKE_MANIFEST)" \
		--data-dir "$(GITCTX_DATA_DIR)" \
		--records "$(SMOKE_RECORDS)"

smoke-check:
	$(PYTHON) -m json.tool "$(SMOKE_REPORT)"
	wc -l "$(SMOKE_JSONL)"

smoke-normalize:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" normalize-smoke

smoke-validate:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" validate-smoke

smoke-checksum:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

smoke-finalize: smoke-normalize smoke-validate smoke-checksum

smoke-review-template:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" create-smoke-review-template --reviewer "$(REVIEWER)"

smoke-review-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" validate-smoke-review
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

teacher-inputs:
	PYTHONPATH=src $(PYTHON) -m gitctx.teacher_inputs --data-dir "$(GITCTX_DATA_DIR)" create-smoke

teacher-input-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.teacher_inputs --data-dir "$(GITCTX_DATA_DIR)" validate-smoke
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

teacher-generate:
	PYTHONPATH=src $(PYTHON) -m gitctx.ollama_generate --data-dir "$(GITCTX_DATA_DIR)" generate-smoke \
		--num-ctx "$(OLLAMA_NUM_CTX)" \
		--num-predict "$(OLLAMA_NUM_PREDICT)" \
		--request-timeout "$(OLLAMA_REQUEST_TIMEOUT)"

teacher-generate-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.ollama_generate --data-dir "$(GITCTX_DATA_DIR)" validate-smoke
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests

fixture-eval:
	PYTHONPATH=src $(PYTHON) -m gitctx.eval fixtures/dev/commit_message_cases.jsonl

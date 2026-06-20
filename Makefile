GITCTX_DATA_DIR ?= $(HOME)/LAB/gitctx-data
PYTHON ?= python3
SMOKE_MANIFEST ?= manifests/source-manifest.audit.jsonl
SMOKE_RECORDS ?= 50
PILOT_RECORDS ?= 250
PILOT_PER_REPO_LIMIT ?= 100
SPLIT_PLAN ?=
SPLIT_PLAN_FLAG = $(if $(SPLIT_PLAN),--split-plan "$(SPLIT_PLAN)")
SMOKE_REPORT = $(GITCTX_DATA_DIR)/artifacts/smoke/source-diffs.smoke.report.json
SMOKE_JSONL = $(GITCTX_DATA_DIR)/artifacts/smoke/source-diffs.smoke.jsonl
PILOT_REPORT = $(GITCTX_DATA_DIR)/artifacts/pilot/source-diffs.pilot.report.json
PILOT_JSONL = $(GITCTX_DATA_DIR)/artifacts/pilot/source-diffs.pilot.jsonl
REVIEWER ?= reviewer@example.com
OLLAMA_NUM_CTX ?= 8192
OLLAMA_NUM_PREDICT ?= 1024
OLLAMA_REQUEST_TIMEOUT ?= 300
TRAIN_VERSION ?= v0

.PHONY: data-dir smoke smoke-check smoke-finalize pilot-source pilot-source-check pilot-source-finalize pilot-review-template pilot-review-check smoke-review-template smoke-review-check teacher-inputs teacher-input-check pilot-teacher-inputs pilot-teacher-input-check teacher-generate teacher-generate-check pilot-teacher-generate pilot-teacher-generate-check generated-review-template generated-review-check pilot-generated-review-template pilot-generated-review-check pilot-train-artifact pilot-train-artifact-check pilot-eval-baseline test fixture-eval

data-dir:
	mkdir -p "$(GITCTX_DATA_DIR)"

smoke: data-dir
	PYTHONPATH=src $(PYTHON) -m gitctx.worker_smoke \
		--manifest "$(SMOKE_MANIFEST)" \
		--data-dir "$(GITCTX_DATA_DIR)" \
		--records "$(SMOKE_RECORDS)" \
		--artifact-name smoke \
		$(SPLIT_PLAN_FLAG)

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

pilot-source: data-dir
	PYTHONPATH=src $(PYTHON) -m gitctx.worker_smoke \
		--manifest "$(SMOKE_MANIFEST)" \
		--data-dir "$(GITCTX_DATA_DIR)" \
		--records "$(PILOT_RECORDS)" \
		--per-repo-limit "$(PILOT_PER_REPO_LIMIT)" \
		--artifact-name pilot \
		$(SPLIT_PLAN_FLAG)

pilot-source-check:
	$(PYTHON) -m json.tool "$(PILOT_REPORT)"
	wc -l "$(PILOT_JSONL)"

pilot-source-normalize:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" normalize-source --artifact-name pilot --manifest "$(SMOKE_MANIFEST)"

pilot-source-validate:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" validate-source --artifact-name pilot --manifest "$(SMOKE_MANIFEST)"

pilot-source-finalize: pilot-source-normalize pilot-source-validate smoke-checksum

pilot-review-template:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" create-source-review-template --artifact-name pilot --reviewer "$(REVIEWER)"

pilot-review-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" validate-source-review --artifact-name pilot
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

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

pilot-teacher-inputs:
	PYTHONPATH=src $(PYTHON) -m gitctx.teacher_inputs --data-dir "$(GITCTX_DATA_DIR)" create --artifact-name pilot

pilot-teacher-input-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.teacher_inputs --data-dir "$(GITCTX_DATA_DIR)" validate --artifact-name pilot
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

teacher-generate:
	PYTHONPATH=src $(PYTHON) -m gitctx.ollama_generate --data-dir "$(GITCTX_DATA_DIR)" generate-smoke \
		--num-ctx "$(OLLAMA_NUM_CTX)" \
		--num-predict "$(OLLAMA_NUM_PREDICT)" \
		--request-timeout "$(OLLAMA_REQUEST_TIMEOUT)"

teacher-generate-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.ollama_generate --data-dir "$(GITCTX_DATA_DIR)" validate-smoke
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

pilot-teacher-generate:
	PYTHONPATH=src $(PYTHON) -m gitctx.ollama_generate --data-dir "$(GITCTX_DATA_DIR)" generate --artifact-name pilot \
		--num-ctx "$(OLLAMA_NUM_CTX)" \
		--num-predict "$(OLLAMA_NUM_PREDICT)" \
		--request-timeout "$(OLLAMA_REQUEST_TIMEOUT)"

pilot-teacher-generate-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.ollama_generate --data-dir "$(GITCTX_DATA_DIR)" validate --artifact-name pilot
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

generated-review-template:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" create-generated-label-review-template --reviewer "$(REVIEWER)"

generated-review-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" validate-generated-label-review
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

pilot-generated-review-template:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" create-named-generated-label-review-template --artifact-name pilot --reviewer "$(REVIEWER)"

pilot-generated-review-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" validate-named-generated-label-review --artifact-name pilot
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

pilot-train-artifact:
	PYTHONPATH=src $(PYTHON) -m gitctx.train_artifacts --data-dir "$(GITCTX_DATA_DIR)" create \
		--artifact-name pilot \
		--version "$(TRAIN_VERSION)"

pilot-train-artifact-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.train_artifacts --data-dir "$(GITCTX_DATA_DIR)" validate \
		--artifact-name pilot \
		--version "$(TRAIN_VERSION)"
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

pilot-eval-baseline:
	PYTHONPATH=src $(PYTHON) -m gitctx.artifact_eval --data-dir "$(GITCTX_DATA_DIR)" evaluate \
		--artifact-name pilot \
		--version "$(TRAIN_VERSION)"
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests

fixture-eval:
	PYTHONPATH=src $(PYTHON) -m gitctx.eval fixtures/dev/commit_message_cases.jsonl

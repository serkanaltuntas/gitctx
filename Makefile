GITCTX_DATA_DIR ?= $(HOME)/LAB/gitctx-data
PYTHON ?= python3
SMOKE_MANIFEST ?= manifests/source-manifest.audit.jsonl
SMOKE_RECORDS ?= 50
PILOT_RECORDS ?= 250
PILOT_PER_REPO_LIMIT ?= 100
PILOT_ARTIFACT ?= pilot
DATA_SPLIT ?= REPORT
SPLIT_PLAN ?=
SPLIT_PLAN_FLAG = $(if $(SPLIT_PLAN),--split-plan "$(SPLIT_PLAN)")
EXCLUDE_SOURCE_ARTIFACT ?=
EXCLUDE_SOURCE_ARTIFACT_FLAG = $(if $(EXCLUDE_SOURCE_ARTIFACT),--exclude-source-artifact "$(EXCLUDE_SOURCE_ARTIFACT)")
SOURCE_MANIFEST ?= manifests/source-manifest.audit.jsonl
SMOKE_REPORT = $(GITCTX_DATA_DIR)/artifacts/smoke/source-diffs.smoke.report.json
SMOKE_JSONL = $(GITCTX_DATA_DIR)/artifacts/smoke/source-diffs.smoke.jsonl
PILOT_REPORT = $(GITCTX_DATA_DIR)/artifacts/$(PILOT_ARTIFACT)/source-diffs.$(PILOT_ARTIFACT).report.json
PILOT_JSONL = $(GITCTX_DATA_DIR)/artifacts/$(PILOT_ARTIFACT)/source-diffs.$(PILOT_ARTIFACT).jsonl
REVIEWER ?= reviewer@example.com
REVIEW_TIMESTAMP ?= TBD
WRITE ?= 0
WRITE_FLAG = $(if $(filter 1 true yes,$(WRITE)),--write)
OLLAMA_NUM_CTX ?= 8192
OLLAMA_NUM_PREDICT ?= 1024
OLLAMA_PROGRESS_EVERY ?= 25
OLLAMA_REQUEST_TIMEOUT ?= 300
ALLOW_GENERATION_FAILURES ?= 0
ALLOW_MISSING_LABELS ?= 0
ALLOW_GENERATION_FAILURES_FLAG = $(if $(filter 1 true yes,$(ALLOW_GENERATION_FAILURES)),--allow-failures)
ALLOW_MISSING_LABELS_FLAG = $(if $(filter 1 true yes,$(ALLOW_MISSING_LABELS)),--allow-missing)
TRAIN_VERSION ?= v0
MERGE_INPUT_ARTIFACTS ?=
MERGE_INPUT_ARTIFACT_FLAGS = $(foreach artifact,$(MERGE_INPUT_ARTIFACTS),--input-artifact "$(artifact)")
PROTOTYPE_MODEL_VERSION ?= path-type-v0
NEURAL_MODEL_VERSION ?= tiny-softmax-v0
NEURAL_EPOCHS ?= 25
NEURAL_LEARNING_RATE ?= 0.35
NEURAL_L2 ?= 0.0001
PROOF_SOURCE_MANIFEST ?= manifests/source-manifest.$(PILOT_ARTIFACT).jsonl
PROOF_SPLIT_PLAN ?= manifests/split-plan.$(PILOT_ARTIFACT).json
PROOF_WRITE ?= 1
PROOF_WRITE_FLAG = $(if $(filter 1 true yes,$(PROOF_WRITE)),--write)

.PHONY: data-dir smoke smoke-check smoke-finalize pilot-source pilot-source-check pilot-source-finalize pilot-review-template pilot-review-policy pilot-review-check smoke-review-template smoke-review-check teacher-inputs teacher-input-check pilot-teacher-source-check pilot-teacher-inputs pilot-teacher-input-check teacher-generate teacher-generate-check pilot-teacher-generate pilot-teacher-generate-check generated-review-template generated-review-check pilot-generated-review-template pilot-generated-review-policy pilot-generated-review-check pilot-train-artifact pilot-train-artifact-check merge-train-artifact artifact-eval-baseline artifact-split-inspection proof-readiness training-smoke-train training-smoke-eval training-smoke neural-smoke-train neural-smoke-eval neural-smoke split-readiness pilot-eval-baseline test fixture-eval

data-dir:
	mkdir -p "$(GITCTX_DATA_DIR)"

smoke: data-dir
	PYTHONPATH=src $(PYTHON) -m gitctx.worker_smoke \
		--manifest "$(SMOKE_MANIFEST)" \
		--data-dir "$(GITCTX_DATA_DIR)" \
		--records "$(SMOKE_RECORDS)" \
		--artifact-name smoke \
		$(SPLIT_PLAN_FLAG) \
		$(EXCLUDE_SOURCE_ARTIFACT_FLAG)

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
		--artifact-name "$(PILOT_ARTIFACT)" \
		$(SPLIT_PLAN_FLAG) \
		$(EXCLUDE_SOURCE_ARTIFACT_FLAG)

pilot-source-check:
	$(PYTHON) -m json.tool "$(PILOT_REPORT)"
	wc -l "$(PILOT_JSONL)"

pilot-source-normalize:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" normalize-source --artifact-name "$(PILOT_ARTIFACT)" --manifest "$(SMOKE_MANIFEST)" $(if $(SPLIT_PLAN),--split-plan "$(SPLIT_PLAN)")

pilot-source-validate:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" validate-source --artifact-name "$(PILOT_ARTIFACT)" --manifest "$(SMOKE_MANIFEST)"

pilot-source-finalize: pilot-source-normalize pilot-source-validate smoke-checksum

pilot-review-template:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" create-source-review-template --artifact-name "$(PILOT_ARTIFACT)" --reviewer "$(REVIEWER)"

pilot-review-policy:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" apply-source-review-policy --artifact-name "$(PILOT_ARTIFACT)" --reviewer "$(REVIEWER)" --write

pilot-review-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" validate-source-review --artifact-name "$(PILOT_ARTIFACT)"
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

pilot-teacher-source-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.teacher_inputs --data-dir "$(GITCTX_DATA_DIR)" validate-source-cache --artifact-name "$(PILOT_ARTIFACT)"

pilot-teacher-inputs:
	PYTHONPATH=src $(PYTHON) -m gitctx.teacher_inputs --data-dir "$(GITCTX_DATA_DIR)" create --artifact-name "$(PILOT_ARTIFACT)"

pilot-teacher-input-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.teacher_inputs --data-dir "$(GITCTX_DATA_DIR)" validate --artifact-name "$(PILOT_ARTIFACT)"
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

teacher-generate:
	PYTHONPATH=src $(PYTHON) -m gitctx.ollama_generate --data-dir "$(GITCTX_DATA_DIR)" generate-smoke \
		--num-ctx "$(OLLAMA_NUM_CTX)" \
		--num-predict "$(OLLAMA_NUM_PREDICT)" \
		--progress-every "$(OLLAMA_PROGRESS_EVERY)" \
		--request-timeout "$(OLLAMA_REQUEST_TIMEOUT)" \
		$(ALLOW_GENERATION_FAILURES_FLAG)

teacher-generate-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.ollama_generate --data-dir "$(GITCTX_DATA_DIR)" validate-smoke \
		$(ALLOW_MISSING_LABELS_FLAG)
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

pilot-teacher-generate:
	PYTHONPATH=src $(PYTHON) -m gitctx.ollama_generate --data-dir "$(GITCTX_DATA_DIR)" generate --artifact-name "$(PILOT_ARTIFACT)" \
		--num-ctx "$(OLLAMA_NUM_CTX)" \
		--num-predict "$(OLLAMA_NUM_PREDICT)" \
		--progress-every "$(OLLAMA_PROGRESS_EVERY)" \
		--request-timeout "$(OLLAMA_REQUEST_TIMEOUT)" \
		$(ALLOW_GENERATION_FAILURES_FLAG)

pilot-teacher-generate-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.ollama_generate --data-dir "$(GITCTX_DATA_DIR)" validate --artifact-name "$(PILOT_ARTIFACT)" \
		$(ALLOW_MISSING_LABELS_FLAG)
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

generated-review-template:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" create-generated-label-review-template --reviewer "$(REVIEWER)"

generated-review-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" validate-generated-label-review
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

pilot-generated-review-template:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" create-named-generated-label-review-template --artifact-name "$(PILOT_ARTIFACT)" --reviewer "$(REVIEWER)"

pilot-generated-review-policy:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" apply-generated-label-review-policy \
		--artifact-name "$(PILOT_ARTIFACT)" \
		--reviewer "$(REVIEWER)" \
		--review-timestamp "$(REVIEW_TIMESTAMP)" \
		$(WRITE_FLAG)

pilot-generated-review-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" validate-named-generated-label-review --artifact-name "$(PILOT_ARTIFACT)"
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

pilot-train-artifact:
	PYTHONPATH=src $(PYTHON) -m gitctx.train_artifacts --data-dir "$(GITCTX_DATA_DIR)" create \
		--artifact-name "$(PILOT_ARTIFACT)" \
		--version "$(TRAIN_VERSION)"

pilot-train-artifact-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.train_artifacts --data-dir "$(GITCTX_DATA_DIR)" validate \
		--artifact-name "$(PILOT_ARTIFACT)" \
		--version "$(TRAIN_VERSION)"
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

merge-train-artifact:
	PYTHONPATH=src $(PYTHON) -m gitctx.train_artifacts --data-dir "$(GITCTX_DATA_DIR)" merge-inputs \
		--artifact-name "$(PILOT_ARTIFACT)" \
		--version "$(TRAIN_VERSION)" \
		$(MERGE_INPUT_ARTIFACT_FLAGS)
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

artifact-eval-baseline:
	PYTHONPATH=src $(PYTHON) -m gitctx.artifact_eval --data-dir "$(GITCTX_DATA_DIR)" evaluate \
		--artifact-name "$(PILOT_ARTIFACT)" \
		--version "$(TRAIN_VERSION)"
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

artifact-split-inspection:
	PYTHONPATH=src $(PYTHON) -m gitctx.artifact_eval --data-dir "$(GITCTX_DATA_DIR)" inspect-split \
		--artifact-name "$(PILOT_ARTIFACT)" \
		--split "$(DATA_SPLIT)" \
		--version "$(TRAIN_VERSION)"
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

proof-readiness:
	PYTHONPATH=src $(PYTHON) -m gitctx.proof_readiness --data-dir "$(GITCTX_DATA_DIR)" evaluate \
		--artifact-name "$(PILOT_ARTIFACT)" \
		--version "$(TRAIN_VERSION)" \
		--source-manifest "$(PROOF_SOURCE_MANIFEST)" \
		--split-plan "$(PROOF_SPLIT_PLAN)" \
		$(PROOF_WRITE_FLAG)
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

training-smoke-train:
	PYTHONPATH=src $(PYTHON) -m gitctx.prototype_model --data-dir "$(GITCTX_DATA_DIR)" train \
		--artifact-name "$(PILOT_ARTIFACT)" \
		--artifact-version "$(TRAIN_VERSION)" \
		--model-version "$(PROTOTYPE_MODEL_VERSION)" \
		--train-split DEV
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

training-smoke-eval:
	PYTHONPATH=src $(PYTHON) -m gitctx.prototype_model --data-dir "$(GITCTX_DATA_DIR)" evaluate \
		--artifact-name "$(PILOT_ARTIFACT)" \
		--artifact-version "$(TRAIN_VERSION)" \
		--model-version "$(PROTOTYPE_MODEL_VERSION)" \
		--split REPORT
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

training-smoke: training-smoke-train training-smoke-eval

neural-smoke-train:
	PYTHONPATH=src $(PYTHON) -m gitctx.tiny_neural --data-dir "$(GITCTX_DATA_DIR)" train \
		--artifact-name "$(PILOT_ARTIFACT)" \
		--artifact-version "$(TRAIN_VERSION)" \
		--model-version "$(NEURAL_MODEL_VERSION)" \
		--train-split DEV \
		--epochs "$(NEURAL_EPOCHS)" \
		--learning-rate "$(NEURAL_LEARNING_RATE)" \
		--l2 "$(NEURAL_L2)"
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

neural-smoke-eval:
	PYTHONPATH=src $(PYTHON) -m gitctx.tiny_neural --data-dir "$(GITCTX_DATA_DIR)" evaluate \
		--artifact-name "$(PILOT_ARTIFACT)" \
		--artifact-version "$(TRAIN_VERSION)" \
		--model-version "$(NEURAL_MODEL_VERSION)" \
		--split REPORT
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

neural-smoke: neural-smoke-train neural-smoke-eval

split-readiness:
	PYTHONPATH=src $(PYTHON) -m gitctx.split_readiness \
		--source-manifest "$(SOURCE_MANIFEST)" \
		--split-plan "$(SPLIT_PLAN)"

pilot-eval-baseline: artifact-eval-baseline

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests

fixture-eval:
	PYTHONPATH=src $(PYTHON) -m gitctx.eval fixtures/dev/commit_message_cases.jsonl

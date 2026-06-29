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
ALLOWED_DATA_SPLITS ?=
ALLOWED_DATA_SPLIT_FLAGS = $(foreach split,$(ALLOWED_DATA_SPLITS),--allowed-data-split "$(split)")
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
GCTX1_PROOF_ARTIFACT ?= gctx1-strict
GCTX1_PROOF_CONFIG ?= configs/gctx1-proof-model.v0.json
GCTX1_PROOF_SOURCE_MANIFEST ?= $(GITCTX_DATA_DIR)/manifests/source-manifest.gctx1.jsonl
GCTX1_PROOF_SPLIT_PLAN ?= $(GITCTX_DATA_DIR)/manifests/split-plan.gctx1.json
GCTX1_PROOF_HANDOFF = $(GITCTX_DATA_DIR)/artifacts/train-runs/gctx1-proof-model.v0.handoff.json
GCTX1_PROOF_RUN_ID ?= gctx1-proof-model.v0.dry-run
GCTX1_PROOF_TRAIN_REPORT = $(GITCTX_DATA_DIR)/artifacts/train-runs/$(GCTX1_PROOF_RUN_ID).report.json
GCTX1_PROOF_TRAIN_CHECKPOINT = $(GITCTX_DATA_DIR)/artifacts/train-runs/$(GCTX1_PROOF_RUN_ID).checkpoint.json
GCTX1_PROOF_SEQUENCE_PLAN = $(GITCTX_DATA_DIR)/artifacts/train-runs/$(GCTX1_PROOF_RUN_ID).sequence-plan.jsonl
GCTX1_PROOF_SEQUENCE_METADATA = $(GITCTX_DATA_DIR)/artifacts/train-runs/$(GCTX1_PROOF_RUN_ID).sequence-metadata.jsonl
GCTX1_PROOF_SEQUENCE_REPORT = $(GITCTX_DATA_DIR)/artifacts/train-runs/$(GCTX1_PROOF_RUN_ID).sequence-materialization.report.json
GCTX1_PROOF_SFT_SMOKE_REPORT = $(GITCTX_DATA_DIR)/artifacts/train-runs/$(GCTX1_PROOF_RUN_ID).sft-smoke.report.json
GCTX1_PROOF_SFT_SMOKE_CHECKPOINT = $(GITCTX_DATA_DIR)/artifacts/train-runs/$(GCTX1_PROOF_RUN_ID).sft-smoke.checkpoint.json
GCTX1_PROOF_SFT_SMOKE_MAX_RECORDS ?= 64
GCTX1_PROOF_SFT_SMOKE_MODEL_BUCKETS ?= 256
GCTX1_PROOF_SFT_SMOKE_LR_UNITS ?= 3
GCTX1_PROOF_SFT_SMOKE_STOP_AFTER ?=
GCTX1_PROOF_SFT_SMOKE_STOP_FLAG = $(if $(GCTX1_PROOF_SFT_SMOKE_STOP_AFTER),--stop-after-records "$(GCTX1_PROOF_SFT_SMOKE_STOP_AFTER)")
GCTX1_PROOF_SFT_SMOKE_RESUME ?= 0
GCTX1_PROOF_SFT_SMOKE_RESUME_FLAG = $(if $(filter 1 true yes,$(GCTX1_PROOF_SFT_SMOKE_RESUME)),--resume)
GCTX1_PROOF_TRAINER_JOB = $(GITCTX_DATA_DIR)/artifacts/train-runs/$(GCTX1_PROOF_RUN_ID).trainer-job.json
GCTX1_PROOF_TRAINER_SEED ?= 17
GCTX1_MAX_RAW_RECORD_TOKENS ?= 65536
GCTX1_LONG_RECORD_SAMPLE_LIMIT ?= 20
GCTX1_TOKENIZER_VERSION ?= regex-diff-v0
GCTX1_TOKENIZER_VOCAB_SIZE ?= 32000
GCTX1_TOKENIZER_MIN_FREQUENCY ?= 2
GCTX1_TOKENIZER_ARTIFACT = $(GITCTX_DATA_DIR)/artifacts/tokenizers/$(GCTX1_TOKENIZER_VERSION).$(GCTX1_PROOF_ARTIFACT).v0.json
GCTX1_TOKENIZER_REPORT = $(GITCTX_DATA_DIR)/artifacts/tokenizers/$(GCTX1_TOKENIZER_VERSION).$(GCTX1_PROOF_ARTIFACT).v0.report.json
GCTX1_PROTOTYPE_REPORT = $(GITCTX_DATA_DIR)/artifacts/eval/path-type-v0.$(GCTX1_PROOF_ARTIFACT).v0.report.report.json
GCTX1_NEURAL_REPORT = $(GITCTX_DATA_DIR)/artifacts/eval/tiny-softmax-v0.$(GCTX1_PROOF_ARTIFACT).v0.report.report.json

.PHONY: data-dir smoke smoke-check smoke-finalize pilot-source pilot-source-check pilot-source-finalize pilot-review-template pilot-review-policy pilot-review-check smoke-review-template smoke-review-check teacher-inputs teacher-input-check pilot-teacher-source-check pilot-teacher-inputs pilot-teacher-input-check teacher-generate teacher-generate-check pilot-teacher-generate pilot-teacher-generate-check generated-review-template generated-review-check pilot-generated-review-template pilot-generated-review-policy pilot-generated-review-check pilot-train-artifact pilot-train-artifact-check merge-train-artifact artifact-eval-baseline artifact-split-inspection proof-readiness training-smoke-train training-smoke-eval training-smoke neural-smoke-train neural-smoke-eval neural-smoke split-readiness pilot-eval-baseline test fixture-eval
.PHONY: gctx1-tokenizer gctx1-tokenizer-check gctx1-proof-config-check gctx1-proof-readiness gctx1-proof-handoff gctx1-proof-handoff-check gctx1-proof-train-dry-run gctx1-proof-train-dry-run-check gctx1-proof-sequences gctx1-proof-sequences-check gctx1-proof-sft-smoke gctx1-proof-sft-smoke-check gctx1-proof-trainer-job gctx1-proof-trainer-job-check gctx1-proof-smoke gctx1-proof-smoke-check

data-dir:
	mkdir -p "$(GITCTX_DATA_DIR)"

smoke: data-dir
	PYTHONPATH=src $(PYTHON) -m gitctx.worker_smoke \
		--manifest "$(SMOKE_MANIFEST)" \
		--data-dir "$(GITCTX_DATA_DIR)" \
		--records "$(SMOKE_RECORDS)" \
		--artifact-name smoke \
		$(SPLIT_PLAN_FLAG) \
		$(EXCLUDE_SOURCE_ARTIFACT_FLAG) \
		$(ALLOWED_DATA_SPLIT_FLAGS)

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
		$(EXCLUDE_SOURCE_ARTIFACT_FLAG) \
		$(ALLOWED_DATA_SPLIT_FLAGS)

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

gctx1-proof-config-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.proof_model_config validate "$(GCTX1_PROOF_CONFIG)"

gctx1-proof-readiness:
	$(MAKE) proof-readiness \
		GITCTX_DATA_DIR="$(GITCTX_DATA_DIR)" \
		PILOT_ARTIFACT="$(GCTX1_PROOF_ARTIFACT)" \
		PROOF_SOURCE_MANIFEST="$(GCTX1_PROOF_SOURCE_MANIFEST)" \
		PROOF_SPLIT_PLAN="$(GCTX1_PROOF_SPLIT_PLAN)"

gctx1-tokenizer:
	PYTHONPATH=src $(PYTHON) -m gitctx.proof_tokenizer --data-dir "$(GITCTX_DATA_DIR)" build \
		--artifact-name "$(GCTX1_PROOF_ARTIFACT)" \
		--artifact-version "$(TRAIN_VERSION)" \
		--tokenizer-version "$(GCTX1_TOKENIZER_VERSION)" \
		--vocab-size "$(GCTX1_TOKENIZER_VOCAB_SIZE)" \
		--min-frequency "$(GCTX1_TOKENIZER_MIN_FREQUENCY)"
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

gctx1-tokenizer-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.proof_tokenizer --data-dir "$(GITCTX_DATA_DIR)" validate \
		--artifact-name "$(GCTX1_PROOF_ARTIFACT)" \
		--artifact-version "$(TRAIN_VERSION)" \
		--tokenizer-version "$(GCTX1_TOKENIZER_VERSION)"
	$(PYTHON) -m json.tool "$(GCTX1_TOKENIZER_REPORT)"

gctx1-proof-handoff:
	PYTHONPATH=src $(PYTHON) -m gitctx.proof_handoff --data-dir "$(GITCTX_DATA_DIR)" create \
		--config "$(GCTX1_PROOF_CONFIG)" \
		--source-manifest "$(GCTX1_PROOF_SOURCE_MANIFEST)" \
		--split-plan "$(GCTX1_PROOF_SPLIT_PLAN)" \
		--tokenizer "$(GCTX1_TOKENIZER_ARTIFACT)" \
		--write \
		--fail-on-blocked
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

gctx1-proof-handoff-check:
	$(PYTHON) -m json.tool "$(GCTX1_PROOF_HANDOFF)"

gctx1-proof-train-dry-run:
	PYTHONPATH=src $(PYTHON) -m gitctx.proof_train --data-dir "$(GITCTX_DATA_DIR)" dry-run \
		--handoff "$(GCTX1_PROOF_HANDOFF)" \
		--run-id "$(GCTX1_PROOF_RUN_ID)" \
		--max-raw-record-tokens "$(GCTX1_MAX_RAW_RECORD_TOKENS)" \
		--long-record-sample-limit "$(GCTX1_LONG_RECORD_SAMPLE_LIMIT)" \
		--write \
		--fail-on-blocked
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

gctx1-proof-train-dry-run-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.proof_train --data-dir "$(GITCTX_DATA_DIR)" validate \
		--run-id "$(GCTX1_PROOF_RUN_ID)"
	$(PYTHON) -m json.tool "$(GCTX1_PROOF_TRAIN_REPORT)"
	$(PYTHON) -m json.tool "$(GCTX1_PROOF_TRAIN_CHECKPOINT)"
	wc -l "$(GCTX1_PROOF_SEQUENCE_PLAN)"

gctx1-proof-sequences:
	PYTHONPATH=src $(PYTHON) -m gitctx.proof_sequences --data-dir "$(GITCTX_DATA_DIR)" build \
		--handoff "$(GCTX1_PROOF_HANDOFF)" \
		--run-id "$(GCTX1_PROOF_RUN_ID)" \
		--write \
		--fail-on-blocked
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

gctx1-proof-sequences-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.proof_sequences --data-dir "$(GITCTX_DATA_DIR)" validate \
		--run-id "$(GCTX1_PROOF_RUN_ID)"
	$(PYTHON) -m json.tool "$(GCTX1_PROOF_SEQUENCE_REPORT)"
	wc -l "$(GCTX1_PROOF_SEQUENCE_METADATA)"

gctx1-proof-sft-smoke:
	PYTHONPATH=src $(PYTHON) -m gitctx.proof_sft_smoke --data-dir "$(GITCTX_DATA_DIR)" train \
		--handoff "$(GCTX1_PROOF_HANDOFF)" \
		--run-id "$(GCTX1_PROOF_RUN_ID)" \
		--max-records "$(GCTX1_PROOF_SFT_SMOKE_MAX_RECORDS)" \
		--model-buckets "$(GCTX1_PROOF_SFT_SMOKE_MODEL_BUCKETS)" \
		--learning-rate-units "$(GCTX1_PROOF_SFT_SMOKE_LR_UNITS)" \
		$(GCTX1_PROOF_SFT_SMOKE_STOP_FLAG) \
		$(GCTX1_PROOF_SFT_SMOKE_RESUME_FLAG) \
		--write \
		--fail-on-blocked
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

gctx1-proof-sft-smoke-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.proof_sft_smoke --data-dir "$(GITCTX_DATA_DIR)" validate \
		--run-id "$(GCTX1_PROOF_RUN_ID)"
	$(PYTHON) -m json.tool "$(GCTX1_PROOF_SFT_SMOKE_REPORT)"
	$(PYTHON) -m json.tool "$(GCTX1_PROOF_SFT_SMOKE_CHECKPOINT)"

gctx1-proof-trainer-job:
	PYTHONPATH=src $(PYTHON) -m gitctx.proof_train_job --data-dir "$(GITCTX_DATA_DIR)" create \
		--handoff "$(GCTX1_PROOF_HANDOFF)" \
		--run-id "$(GCTX1_PROOF_RUN_ID)" \
		--seed "$(GCTX1_PROOF_TRAINER_SEED)" \
		--write \
		--fail-on-blocked
	PYTHONPATH=src $(PYTHON) -m gitctx.data_artifacts --data-dir "$(GITCTX_DATA_DIR)" write-checksums

gctx1-proof-trainer-job-check:
	PYTHONPATH=src $(PYTHON) -m gitctx.proof_train_job --data-dir "$(GITCTX_DATA_DIR)" validate \
		--run-id "$(GCTX1_PROOF_RUN_ID)"
	$(PYTHON) -m json.tool "$(GCTX1_PROOF_TRAINER_JOB)"

gctx1-proof-smoke:
	$(MAKE) training-smoke \
		GITCTX_DATA_DIR="$(GITCTX_DATA_DIR)" \
		PILOT_ARTIFACT="$(GCTX1_PROOF_ARTIFACT)"
	$(MAKE) neural-smoke \
		GITCTX_DATA_DIR="$(GITCTX_DATA_DIR)" \
		PILOT_ARTIFACT="$(GCTX1_PROOF_ARTIFACT)"

gctx1-proof-smoke-check:
	$(PYTHON) -m json.tool "$(GCTX1_PROTOTYPE_REPORT)"
	$(PYTHON) -m json.tool "$(GCTX1_NEURAL_REPORT)"

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

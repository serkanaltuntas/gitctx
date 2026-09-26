#!/usr/bin/env bash
# Run one complete DEV epoch, then evaluate its frozen checkpoint on REPORT.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: run-proof-job.sh DATA_DIR RUN_ID [cpu|cuda|mps]' >&2
  exit 2
fi
proof_data_dir=$(realpath "$1")
proof_run_id=$2
proof_device=${3:-cuda}
proof_repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$proof_repo_dir"
proof_uv=${UV_BIN:-uv}
proof_resume=()
if [[ -f "$proof_data_dir/artifacts/train-runs/$proof_run_id/checkpoints/latest.json" ]]; then
  proof_resume=(--resume)
fi
"$proof_uv" run --locked python -u -m gitctx.proof_lm_train --data-dir "$proof_data_dir" train \
  --run-id "$proof_run_id" --device "$proof_device" --checkpoint-every 100 \
  --write --fail-on-blocked "${proof_resume[@]}"
"$proof_uv" run --locked python -m gitctx.proof_lm_train --data-dir "$proof_data_dir" validate \
  --run-id "$proof_run_id"
"$proof_uv" run --locked python -u -m gitctx.proof_lm_eval --data-dir "$proof_data_dir" \
  --run-id "$proof_run_id" --device "$proof_device" --max-new-tokens 256

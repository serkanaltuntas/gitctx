#!/usr/bin/env bash
# Run and validate both frozen DEV arms sequentially; resume after interruption.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: run-order-ablation.sh DATA_DIR EXPERIMENT_ID' >&2
  exit 2
fi
ablation_data_dir=$(realpath "$1")
ablation_experiment=$2
ablation_repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ablation_repo_dir"
ablation_uv=${UV_BIN:-uv}
for ablation_arm in ordered shuffled; do
  ablation_resume=()
  if [[ -f "$ablation_data_dir/artifacts/train-runs/$ablation_experiment.$ablation_arm/checkpoints/latest.json" ]]; then
    ablation_resume=(--resume)
  fi
  if [[ ! -f "$ablation_data_dir/artifacts/train-runs/$ablation_experiment.$ablation_arm/checkpoints/final.json" ]]; then
    "$ablation_uv" run --locked --no-sync python -u -m gitctx.proof_order_ablation train \
      --data-dir "$ablation_data_dir" --experiment-id "$ablation_experiment" \
      --arm "$ablation_arm" "${ablation_resume[@]}"
  fi
  "$ablation_uv" run --locked --no-sync python -m gitctx.proof_lm_train \
    --data-dir "$ablation_data_dir" validate --run-id "$ablation_experiment.$ablation_arm"
  if [[ ! -f "$ablation_data_dir/artifacts/train-runs/$ablation_experiment/$ablation_arm.validation.json" ]]; then
    "$ablation_uv" run --locked --no-sync python -u -m gitctx.proof_order_ablation evaluate \
      --data-dir "$ablation_data_dir" --experiment-id "$ablation_experiment" --arm "$ablation_arm"
  fi
done
"$ablation_uv" run --locked --no-sync python -m gitctx.proof_order_ablation compare \
  --data-dir "$ablation_data_dir" --experiment-id "$ablation_experiment"

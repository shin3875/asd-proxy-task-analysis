#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${ASD_DATASET_ROOT:-./asd_dataset}"
SAVE_ROOT="${PRETRAIN_SAVE_ROOT:-./pretrain_eval_results}"
LINEAR_EPOCHS="${LINEAR_EPOCHS:-200}"
LINEAR_BATCH_SIZE="${LINEAR_BATCH_SIZE:-64}"
LINEAR_HALF_SPLIT="${LINEAR_HALF_SPLIT:-per_section}"
REQUIRE_BEATS="${REQUIRE_BEATS:-0}"
REQUIRE_FULL_PRETRAIN_GRID="${REQUIRE_FULL_PRETRAIN_GRID:-0}"

MAP_ARGS=()
if [[ -n "${PRETRAIN_MAP_CSV:-}" ]]; then
  MAP_ARGS=(--pretrain_map_csv "${PRETRAIN_MAP_CSV}")
fi

run_pretrain() {
  local save_name="$1"
  shift
  "${PYTHON_BIN}" evaluate_pretrain.py \
    --data_dir "${DATA_ROOT}" \
    --save_dir "${SAVE_ROOT}/${save_name}" \
    --linear_epochs "${LINEAR_EPOCHS}" \
    --linear_batch_size "${LINEAR_BATCH_SIZE}" \
    --linear_half_split "${LINEAR_HALF_SPLIT}" \
    "${MAP_ARGS[@]}" \
    "$@"
}

run_beat_if_available() {
  local save_name="$1"
  local checkpoint="$2"
  local model_name="$3"

  if [[ -f "${checkpoint}" ]]; then
    run_pretrain "${save_name}" \
      --models beat \
      --beat_checkpoint "${checkpoint}" \
      --beat_model_name "${model_name}"
  elif [[ "${REQUIRE_BEATS}" == "1" || "${REQUIRE_FULL_PRETRAIN_GRID}" == "1" ]]; then
    echo "Missing BEAT checkpoint: ${checkpoint}" >&2
    exit 1
  else
    echo "Skip ${model_name}: checkpoint not found at ${checkpoint}. Set REQUIRE_BEATS=1 to fail instead." >&2
  fi
}

run_pretrain eat_base \
  --models eat \
  --eat_model_id "${EAT_BASE_MODEL_ID:-worstchan/EAT-base_epoch30_pretrain}"

run_pretrain eat_large \
  --models eat \
  --eat_model_id "${EAT_LARGE_MODEL_ID:-worstchan/EAT-large_epoch20_pretrain}"

for ced in tiny mini small base; do
  run_pretrain "ced_${ced}" \
    --models ced \
    --ced_hf_model "${CED_HF_PREFIX:-mispeech/ced}-${ced}" \
    --ced_model_name "CED-${ced}"
done

run_beat_if_available \
  beats_iter3 \
  "${BEATS_ITER3_CHECKPOINT:-./beats/BEATs_iter3.pt}" \
  "BEATs_iter3"

run_beat_if_available \
  beats_iter3_plus \
  "${BEATS_ITER3_PLUS_CHECKPOINT:-./beats/BEATs_iter3_plus_AS2M.pt}" \
  "BEATs_iter3+"

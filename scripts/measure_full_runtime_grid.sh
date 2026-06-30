#!/usr/bin/env bash
set -euo pipefail

# Full-grid runtime measurement helper.
# Default DRY_RUN=1 prints commands only. Set DRY_RUN=0 to run full measurements.

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
DEVICE="${DEVICE:-cuda:0}"
RUNTIME_ROOT="${RUNTIME_ROOT:-logs/runtime/full_grid}"
TASK_MODEL_ROOT="${TASK_MODEL_ROOT:-${RUNTIME_ROOT}/models/task_specific}"
SHARED_MODEL_ROOT="${SHARED_MODEL_ROOT:-${RUNTIME_ROOT}/models/shared}"
PRETRAIN_SAVE_ROOT="${PRETRAIN_SAVE_ROOT:-${RUNTIME_ROOT}/eval/pretrained}"
TASK_EVAL_ROOT="${TASK_EVAL_ROOT:-${RUNTIME_ROOT}/eval/task_specific}"
SHARED_EVAL_ROOT="${SHARED_EVAL_ROOT:-${RUNTIME_ROOT}/eval/shared}"

EPOCHS="${EPOCHS:-1}"
LINEAR_EPOCHS="${LINEAR_EPOCHS:-1}"
BATCH_SIZE_AE="${BATCH_SIZE_AE:-128}"
BATCH_SIZE_CLASSIFIER="${BATCH_SIZE_CLASSIFIER:-64}"
BATCH_SIZE_CONTRASTIVE="${BATCH_SIZE_CONTRASTIVE:-96}"
BATCH_SIZE_SEPARATION="${BATCH_SIZE_SEPARATION:-12}"
BATCH_SIZE_SHARED="${BATCH_SIZE_SHARED:-32}"
N_CPU="${N_CPU:-32}"
N_CPU_SEPARATION="${N_CPU_SEPARATION:-12}"
LINEAR_BATCH_SIZE="${LINEAR_BATCH_SIZE:-64}"
LINEAR_HALF_SPLIT="${LINEAR_HALF_SPLIT:-per_section}"
MATRIX_LOG_MODE="${MATRIX_LOG_MODE:-raw}"
SEGMENT_FRAMES="${SEGMENT_FRAMES:-313}"
TARGET_CLASSES="${TARGET_CLASSES:-bearing fan gearbox pump slider ToyCar ToyConveyor ToyTrain valve}"
CE_TARGET_CLASS="${CE_TARGET_CLASS:-pump}"
CONTRASTIVE_TARGET_CLASS="${CONTRASTIVE_TARGET_CLASS:-ToyCar}"
ARC_MARGINS="${ARC_MARGINS:-0.5}"
SNR_LIST="${SNR_LIST:--5 -4 -3 -2 -1 0 1 2 3 4 5}"

RUN_TASK_TRAIN="${RUN_TASK_TRAIN:-1}"
RUN_SHARED_TRAIN="${RUN_SHARED_TRAIN:-1}"
RUN_PRETRAIN_EVAL="${RUN_PRETRAIN_EVAL:-1}"
RUN_TASK_EVAL="${RUN_TASK_EVAL:-0}"
RUN_SHARED_EVAL="${RUN_SHARED_EVAL:-0}"
REQUIRE_FULL_PRETRAIN_GRID="${REQUIRE_FULL_PRETRAIN_GRID:-0}"

: "${ASD_DATASET_ROOT:?Set ASD_DATASET_ROOT to the external ASD wav dataset root.}"
: "${ASD_LOGMEL_ROOT:?Set ASD_LOGMEL_ROOT to the external logmel feature root.}"

read -r -a TARGET_CLASS_ARGS <<< "${TARGET_CLASSES}"
read -r -a ARC_MARGIN_ARGS <<< "${ARC_MARGINS}"
read -r -a SNR_ARGS <<< "${SNR_LIST}"

mkdir -p \
  "${RUNTIME_ROOT}/logs" \
  "${TASK_MODEL_ROOT}" \
  "${SHARED_MODEL_ROOT}" \
  "${PRETRAIN_SAVE_ROOT}" \
  "${TASK_EVAL_ROOT}" \
  "${SHARED_EVAL_ROOT}"

if [[ "${DRY_RUN}" == "0" ]]; then
  "${PYTHON_BIN}" - <<'PY' | tee "${RUNTIME_ROOT}/logs/runtime_context.txt"
import os
import torch

print("CUDA_DEVICE_ORDER:", os.environ.get("CUDA_DEVICE_ORDER"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"cuda_device_{idx}:", torch.cuda.get_device_name(idx))
PY
fi

print_cmd() {
  printf ' %q' "$@"
  printf '\n'
}

run_timed() {
  local name="$1"
  shift

  echo "[RUN] ${name}"
  if [[ "${SKIP_DONE}" == "1" && -f "${RUNTIME_ROOT}/logs/${name}.time" ]] \
    && grep -qx "exit_code=0" "${RUNTIME_ROOT}/logs/${name}.time"; then
    echo "[SKIP] ${name} already has exit_code=0. Set SKIP_DONE=0 to rerun."
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN]"
    print_cmd "$@"
    return 0
  fi

  set +e
  /usr/bin/time -f "elapsed_sec=%e\nuser_sec=%U\nsys_sec=%S\nmax_rss_kb=%M" \
    -o "${RUNTIME_ROOT}/logs/${name}.time" \
    "$@" 2>&1 | tee "${RUNTIME_ROOT}/logs/${name}.log"
  local status=${PIPESTATUS[0]}
  set -e

  echo "exit_code=${status}" >> "${RUNTIME_ROOT}/logs/${name}.time"
  if [[ "${status}" -ne 0 ]]; then
    echo "[FAIL] ${name} exit_code=${status}" >&2
    return "${status}"
  fi
  echo "[OK] ${name}"
}

run_beat_if_available() {
  local save_name="$1"
  local checkpoint="$2"
  local model_name="$3"

  if [[ -f "${checkpoint}" ]]; then
    run_timed "eval_pretrained_${save_name}" \
      "${PYTHON_BIN}" evaluate_pretrain.py \
        --data_dir "${ASD_DATASET_ROOT}" \
        --save_dir "${PRETRAIN_SAVE_ROOT}/${save_name}" \
        --models beat \
        --beat_checkpoint "${checkpoint}" \
        --beat_model_name "${model_name}" \
        --linear_epochs "${LINEAR_EPOCHS}" \
        --linear_batch_size "${LINEAR_BATCH_SIZE}" \
        --linear_half_split "${LINEAR_HALF_SPLIT}"
  elif [[ "${REQUIRE_FULL_PRETRAIN_GRID}" == "1" ]]; then
    echo "Missing BEAT checkpoint: ${checkpoint}" >&2
    exit 1
  else
    echo "Skip ${model_name}: checkpoint not found at ${checkpoint}." >&2
  fi
}

if [[ "${RUN_TASK_TRAIN}" == "1" ]]; then
  for latent in 4 8 16; do
    for hidden in 64 128 256; do
      run_timed "train_ae_comp${latent}_lin${hidden}" \
        "${PYTHON_BIN}" train_ae.py \
          --target_dir "${ASD_LOGMEL_ROOT}" \
          --save_model_dir "${TASK_MODEL_ROOT}/ae/comp${latent}_lin${hidden}" \
          --target_classes "${TARGET_CLASS_ARGS[@]}" \
          --matrix_log_mode "${MATRIX_LOG_MODE}" \
          --epochs "${EPOCHS}" \
          --batch_size "${BATCH_SIZE_AE}" \
          --n_cpu "${N_CPU}" \
          --latent_dims "${latent}" \
          --hidden_dims "${hidden}" \
          --save_interval 0 \
          --device "${DEVICE}"
    done
  done

  for resnet in resnet18 resnet34 resnet50 resnet101 resnet152; do
    run_timed "train_ce_${resnet}" \
      "${PYTHON_BIN}" train_classifier.py \
        --target_dir "${ASD_DATASET_ROOT}" \
        --save_model_dir "${TASK_MODEL_ROOT}/classifier_ce/${resnet}" \
        --target_class "${CE_TARGET_CLASS}" \
        --mode ce \
        --resnets "${resnet}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE_CLASSIFIER}" \
        --n_cpu "${N_CPU}" \
        --save_interval 0 \
        --device "${DEVICE}"

    run_timed "train_arcface_${resnet}_m0.5" \
      "${PYTHON_BIN}" train_classifier.py \
        --target_dir "${ASD_DATASET_ROOT}" \
        --save_model_dir "${TASK_MODEL_ROOT}/classifier_arcface/${resnet}" \
        --target_classes "${TARGET_CLASS_ARGS[@]}" \
        --mode arcface \
        --resnets "${resnet}" \
        --margins "${ARC_MARGIN_ARGS[@]}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE_CLASSIFIER}" \
        --n_cpu "${N_CPU}" \
        --save_interval 0 \
        --device "${DEVICE}"

    run_timed "train_simclr_${resnet}" \
      "${PYTHON_BIN}" train_contrastive.py \
        --target_dir "${ASD_LOGMEL_ROOT}" \
        --save_model_dir "${TASK_MODEL_ROOT}/simclr/${resnet}" \
        --target_class "${CONTRASTIVE_TARGET_CLASS}" \
        --mode simclr \
        --resnets "${resnet}" \
        --matrix_log_mode "${MATRIX_LOG_MODE}" \
        --segment_frames "${SEGMENT_FRAMES}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE_CONTRASTIVE}" \
        --n_cpu "${N_CPU}" \
        --save_interval 0 \
        --device "${DEVICE}"

    run_timed "train_simsiam_${resnet}" \
      "${PYTHON_BIN}" train_contrastive.py \
        --target_dir "${ASD_LOGMEL_ROOT}" \
        --save_model_dir "${TASK_MODEL_ROOT}/simsiam/${resnet}" \
        --target_class "${CONTRASTIVE_TARGET_CLASS}" \
        --mode simsiam \
        --resnets "${resnet}" \
        --matrix_log_mode "${MATRIX_LOG_MODE}" \
        --segment_frames "${SEGMENT_FRAMES}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE_CONTRASTIVE}" \
        --n_cpu "${N_CPU}" \
        --save_interval 0 \
        --device "${DEVICE}"
  done

  for cb in 0 1 2 4; do
    for channels in 64 128; do
      run_timed "train_sep_cb${cb}_ch${channels}" \
        "${PYTHON_BIN}" train_separation.py \
          --target_dir "${ASD_DATASET_ROOT}" \
          --save_model_dir "${TASK_MODEL_ROOT}/separation/cb${cb}_ch${channels}" \
          --target_classes "${TARGET_CLASS_ARGS[@]}" \
          --channels "${channels}" \
          --cbs "${cb}" \
          --snr_list "${SNR_ARGS[@]}" \
          --epochs "${EPOCHS}" \
          --batch_size "${BATCH_SIZE_SEPARATION}" \
          --n_cpu "${N_CPU_SEPARATION}" \
          --save_interval 0 \
          --device "${DEVICE}"
    done
  done
fi

if [[ "${RUN_SHARED_TRAIN}" == "1" ]]; then
  for lite in 0 1 2 3 4; do
    for mode in ae sep ce arcface simclr simsiam; do
      run_timed "train_shared_${mode}_lite${lite}" \
        "${PYTHON_BIN}" train_shared_backbone.py \
          --target_dir "${ASD_LOGMEL_ROOT}" \
          --save_model_dir "${SHARED_MODEL_ROOT}/${mode}/lite${lite}" \
          --mode "${mode}" \
          --lite_indices "${lite}" \
          --target_classes "${TARGET_CLASS_ARGS[@]}" \
          --matrix_log_mode "${MATRIX_LOG_MODE}" \
          --epochs "${EPOCHS}" \
          --linear_epochs "${LINEAR_EPOCHS}" \
          --batch_size "${BATCH_SIZE_SHARED}" \
          --n_cpu "${N_CPU}" \
          --segment_frames "${SEGMENT_FRAMES}" \
          --margins "${ARC_MARGIN_ARGS[@]}" \
          --save_interval 0 \
          --no_pretrained \
          --device "${DEVICE}" \
          --test_name "full_runtime"
    done
  done
fi

if [[ "${RUN_PRETRAIN_EVAL}" == "1" ]]; then
  run_timed "eval_pretrained_eat_base" \
    "${PYTHON_BIN}" evaluate_pretrain.py \
      --data_dir "${ASD_DATASET_ROOT}" \
      --save_dir "${PRETRAIN_SAVE_ROOT}/eat_base" \
      --models eat \
      --eat_model_id "${EAT_BASE_MODEL_ID:-worstchan/EAT-base_epoch30_pretrain}" \
      --linear_epochs "${LINEAR_EPOCHS}" \
      --linear_batch_size "${LINEAR_BATCH_SIZE}" \
      --linear_half_split "${LINEAR_HALF_SPLIT}"

  run_timed "eval_pretrained_eat_large" \
    "${PYTHON_BIN}" evaluate_pretrain.py \
      --data_dir "${ASD_DATASET_ROOT}" \
      --save_dir "${PRETRAIN_SAVE_ROOT}/eat_large" \
      --models eat \
      --eat_model_id "${EAT_LARGE_MODEL_ID:-worstchan/EAT-large_epoch20_pretrain}" \
      --linear_epochs "${LINEAR_EPOCHS}" \
      --linear_batch_size "${LINEAR_BATCH_SIZE}" \
      --linear_half_split "${LINEAR_HALF_SPLIT}"

  for ced in tiny mini small base; do
    run_timed "eval_pretrained_ced_${ced}" \
      "${PYTHON_BIN}" evaluate_pretrain.py \
        --data_dir "${ASD_DATASET_ROOT}" \
        --save_dir "${PRETRAIN_SAVE_ROOT}/ced_${ced}" \
        --models ced \
        --ced_hf_model "${CED_HF_PREFIX:-mispeech/ced}-${ced}" \
        --ced_model_name "CED-${ced}" \
        --linear_epochs "${LINEAR_EPOCHS}" \
        --linear_batch_size "${LINEAR_BATCH_SIZE}" \
        --linear_half_split "${LINEAR_HALF_SPLIT}"
  done

  run_beat_if_available \
    beats_iter3 \
    "${BEATS_ITER3_CHECKPOINT:-./beats/BEATs_iter3.pt}" \
    "BEATs_iter3"

  run_beat_if_available \
    beats_iter3_plus \
    "${BEATS_ITER3_PLUS_CHECKPOINT:-./beats/BEATs_iter3_plus_AS2M.pt}" \
    "BEATs_iter3+"
fi

if [[ "${RUN_TASK_EVAL}" == "1" ]]; then
  run_timed "eval_task_specific_full_grid" \
    "${PYTHON_BIN}" evaluate_proxy_asd.py \
      --data_dir "${ASD_DATASET_ROOT}" \
      --model_root "${TASK_MODEL_ROOT}" \
      --save_dir "${TASK_EVAL_ROOT}" \
      --model_type auto \
      --linear_epochs "${LINEAR_EPOCHS}" \
      --linear_batch_size "${LINEAR_BATCH_SIZE}" \
      --linear_half_split "${LINEAR_HALF_SPLIT}" \
      --sep_snr -5 0 5 \
      --sep_proxy_k 1000 \
      --no_save_roc \
      --device "${DEVICE}" \
      --continue_on_error
fi

if [[ "${RUN_SHARED_EVAL}" == "1" ]]; then
  run_timed "eval_shared_full_grid" \
    "${PYTHON_BIN}" evaluate_shared_backbone.py \
      --data_dir "${ASD_LOGMEL_ROOT}" \
      --model_root "${SHARED_MODEL_ROOT}" \
      --save_dir "${SHARED_EVAL_ROOT}" \
      --matrix_log_mode "${MATRIX_LOG_MODE}" \
      --linear_epochs "${LINEAR_EPOCHS}" \
      --linear_batch_size "${LINEAR_BATCH_SIZE}" \
      --linear_half_split "${LINEAR_HALF_SPLIT}" \
      --snr_db -5 0 5 \
      --device "${DEVICE}" \
      --no_save_roc
fi

if [[ "${DRY_RUN}" == "0" ]]; then
  grep -H "elapsed_sec" "${RUNTIME_ROOT}/logs/"*.time \
    | tee "${RUNTIME_ROOT}/runtime_summary.txt"
else
  echo "DRY_RUN=1; no commands were executed. Set DRY_RUN=0 to measure runtime."
fi

#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
LOGMEL_ROOT="${ASD_LOGMEL_ROOT:-./asd_dataset_logmel}"
SAVE_ROOT="${AE_SAVE_ROOT:-./saved_exp/ae}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
N_CPU="${N_CPU:-32}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
TARGET_CLASSES="${TARGET_CLASSES:-bearing fan gearbox pump slider ToyCar ToyConveyor ToyTrain valve}"
MATRIX_LOG_MODE="${MATRIX_LOG_MODE:-auto}"

read -r -a TARGET_CLASS_ARGS <<< "${TARGET_CLASSES}"

"${PYTHON_BIN}" train_ae.py \
  --target_dir "${LOGMEL_ROOT}" \
  --save_model_dir "${SAVE_ROOT}" \
  --target_classes "${TARGET_CLASS_ARGS[@]}" \
  --matrix_log_mode "${MATRIX_LOG_MODE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --n_cpu "${N_CPU}" \
  --latent_dims 4 8 16 \
  --hidden_dims 64 128 256 \
  --save_interval "${SAVE_INTERVAL}"

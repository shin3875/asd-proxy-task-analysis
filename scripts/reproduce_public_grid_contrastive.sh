#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
LOGMEL_ROOT="${ASD_LOGMEL_ROOT:-./asd_dataset_logmel}"
SAVE_ROOT="${CONTRASTIVE_SAVE_ROOT:-./saved_exp/contrastive}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-96}"
N_CPU="${N_CPU:-32}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
TARGET_CLASS="${TARGET_CLASS:-ToyCar}"
CONTRASTIVE_MODES="${CONTRASTIVE_MODES:-simclr simsiam}"
RESNETS="${RESNETS:-resnet18 resnet34 resnet50 resnet101 resnet152}"
MATRIX_LOG_MODE="${MATRIX_LOG_MODE:-auto}"
SEGMENT_FRAMES="${SEGMENT_FRAMES:-313}"

read -r -a RESNET_ARGS <<< "${RESNETS}"

for mode in ${CONTRASTIVE_MODES}; do
  "${PYTHON_BIN}" train_contrastive.py \
    --target_dir "${LOGMEL_ROOT}" \
    --save_model_dir "${SAVE_ROOT}/${mode}" \
    --target_class "${TARGET_CLASS}" \
    --mode "${mode}" \
    --resnets "${RESNET_ARGS[@]}" \
    --matrix_log_mode "${MATRIX_LOG_MODE}" \
    --segment_frames "${SEGMENT_FRAMES}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --n_cpu "${N_CPU}" \
    --save_interval "${SAVE_INTERVAL}"
done

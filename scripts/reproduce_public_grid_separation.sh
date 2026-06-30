#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${ASD_DATASET_ROOT:-./asd_dataset}"
SAVE_ROOT="${SEPARATION_SAVE_ROOT:-./saved_exp/separation}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-12}"
N_CPU="${N_CPU:-12}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
TARGET_CLASSES="${TARGET_CLASSES:-bearing fan gearbox pump slider ToyCar ToyConveyor ToyTrain valve}"
CHANNELS="${CHANNELS:-64 128}"
CBS="${CBS:-0 1 2 4}"
SNR_LIST="${SNR_LIST:--5 -4 -3 -2 -1 0 1 2 3 4 5}"

read -r -a TARGET_CLASS_ARGS <<< "${TARGET_CLASSES}"
read -r -a CHANNEL_ARGS <<< "${CHANNELS}"
read -r -a CB_ARGS <<< "${CBS}"
read -r -a SNR_ARGS <<< "${SNR_LIST}"

"${PYTHON_BIN}" train_separation.py \
  --target_dir "${DATA_ROOT}" \
  --save_model_dir "${SAVE_ROOT}" \
  --target_classes "${TARGET_CLASS_ARGS[@]}" \
  --channels "${CHANNEL_ARGS[@]}" \
  --cbs "${CB_ARGS[@]}" \
  --snr_list "${SNR_ARGS[@]}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --n_cpu "${N_CPU}" \
  --save_interval "${SAVE_INTERVAL}"

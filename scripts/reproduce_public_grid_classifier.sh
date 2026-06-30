#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${ASD_DATASET_ROOT:-./asd_dataset}"
SAVE_ROOT="${CLASSIFIER_SAVE_ROOT:-./saved_exp/classifier}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_CPU="${N_CPU:-32}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
TARGET_CLASSES="${TARGET_CLASSES:-bearing fan gearbox pump slider ToyCar ToyConveyor ToyTrain valve}"
CE_TARGET_CLASS="${CE_TARGET_CLASS:-pump}"
CLASSIFIER_MODES="${CLASSIFIER_MODES:-ce arcface}"
RESNETS="${RESNETS:-resnet18 resnet34 resnet50 resnet101 resnet152}"
MARGINS="${MARGINS:-0.5}"

read -r -a TARGET_CLASS_ARGS <<< "${TARGET_CLASSES}"
read -r -a RESNET_ARGS <<< "${RESNETS}"
read -r -a MARGIN_ARGS <<< "${MARGINS}"

for mode in ${CLASSIFIER_MODES}; do
  if [[ "${mode}" == "ce" ]]; then
    "${PYTHON_BIN}" train_classifier.py \
      --target_dir "${DATA_ROOT}" \
      --save_model_dir "${SAVE_ROOT}/${mode}" \
      --target_class "${CE_TARGET_CLASS}" \
      --mode ce \
      --resnets "${RESNET_ARGS[@]}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --n_cpu "${N_CPU}" \
      --save_interval "${SAVE_INTERVAL}"
  elif [[ "${mode}" == "arcface" ]]; then
    "${PYTHON_BIN}" train_classifier.py \
      --target_dir "${DATA_ROOT}" \
      --save_model_dir "${SAVE_ROOT}/${mode}" \
      --target_classes "${TARGET_CLASS_ARGS[@]}" \
      --mode arcface \
      --resnets "${RESNET_ARGS[@]}" \
      --margins "${MARGIN_ARGS[@]}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --n_cpu "${N_CPU}" \
      --save_interval "${SAVE_INTERVAL}"
  else
    echo "Unsupported CLASSIFIER_MODES entry: ${mode}" >&2
    exit 2
  fi
done

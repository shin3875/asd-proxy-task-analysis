#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Folder-level proxy + ASD evaluator.

Purpose
-------
1. Discover legacy proxy checkpoints from a file or directory.
2. Compute ASD scores using the legacy metric definitions:
   - Mahalanobis-distance AUC / pAUC
   - Linear-probe AUC / pAUC
     * leave-one-section-out
     * all-section train/test
     * half split
3. Compute task-specific proxy metrics for AE, classification, contrastive, and
   separation checkpoints where applicable.
4. Optionally save projection coordinates and plots.
5. Support devices whose section count differs from the original three-section
   layout.

Example
-------
python evaluate_proxy_asd.py \
  --data_dir ./asd_dataset \
  --model_root ./saved_exp \
  --save_dir ./batch_eval_out \
  --devices pump ToyConveyor bearing fan gearbox slider ToyCar ToyTrain valve

Notes
-----
- Run from the repository root so local model and utility modules are importable.
- Unknown checkpoint types are skipped unless `--model_type` is used to force an
  adapter.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

try:
    import torchaudio.transforms as T
except Exception:  # pragma: no cover - handled at runtime
    T = None

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    umap = None

try:
    from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
except Exception:  # pragma: no cover - handled at runtime
    ScaleInvariantSignalDistortionRatio = None


def get_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


DEFAULT_DEVICES = [
    "pump", "ToyConveyor", "bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"
]


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class ModelSpec:
    path: Path
    model_type: str
    model_id: str
    target_device: Optional[str]
    eval_devices: List[str]
    arch: Optional[str] = None
    comp_feat: Optional[int] = None
    lin_feat: Optional[int] = None
    channel_size: Optional[int] = None
    cb: Optional[int] = None
    batch_size_from_name: Optional[int] = None
    epoch_from_name: Optional[int] = None
    loss_from_name: Optional[float] = None
    unsup_mode: Optional[str] = None
    raw_name: str = ""


@dataclass
class FileRecord:
    path: str
    file_name: str
    target_device: str
    domain: str
    condition: str
    section: str
    class_label: str


@dataclass
class DataSplits:
    train_target: List[str]
    train_source: List[str]
    test_target: List[str]
    test_source: List[str]
    supplemental: List[str]
    other_train: List[str]
    other_test: List[str]
    class_list: List[str]

    @property
    def train_all(self) -> List[str]:
        # Legacy datalist_maker returned np.append(train_target, train_source).
        # Keep the same concatenation order; covariance is order-invariant, but
        # downstream half-split probes must see legacy ordering.
        return list(self.train_target) + list(self.train_source)

    @property
    def test_all(self) -> List[str]:
        # Legacy mahala_score_calc received np.append(test_target, test_source).
        # The order matters for the half-split linear probe.
        return list(self.test_target) + list(self.test_source)


@dataclass
class FeatureOutput:
    # Row-level feature for covariance estimation.
    cov_features: torch.Tensor
    # File-level feature for linear probes and projections. Shape: (1, D).
    file_feature: torch.Tensor
    # Row-level feature used for per-file Mahalanobis scoring.
    mah_features: torch.Tensor
    # Optional task-specific proxy values, e.g. AE losses or classifier outputs.
    proxy: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CovStats:
    train_mu: torch.Tensor
    train_cov_inv: torch.Tensor
    train_features: torch.Tensor
    source_mu: Optional[torch.Tensor] = None
    source_cov_inv: Optional[torch.Tensor] = None
    target_mu: Optional[torch.Tensor] = None
    target_cov_inv: Optional[torch.Tensor] = None
    train_proxy_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def safe_name(value: str) -> str:
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.\-]+", "_", value)
    return value.strip("_") or "unnamed"


def section_sort_key(section: str) -> Tuple[int, str]:
    m = re.search(r"section_(\d+)", section)
    if m:
        return int(m.group(1)), section
    return 10**9, section


def parse_section_from_name(path_or_name: str) -> str:
    base = os.path.basename(str(path_or_name))
    m = re.search(r"section_(\d+)", base)
    if m:
        return f"section_{m.group(1).zfill(2)}"
    # 기존 코드의 fallback은 section_00/01 외 나머지를 section_02로 처리했다.
    return "section_02"


def infer_domain(path_or_name: str) -> str:
    text = str(path_or_name).lower()
    if "source" in text:
        return "source"
    if "target" in text:
        return "target"
    return "unknown"


def infer_condition(path_or_name: str) -> str:
    return "anomaly" if "anomaly" in os.path.basename(str(path_or_name)).lower() else "normal"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy_1d(value: Sequence[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    return arr.reshape(-1)


def safe_auc(scores: Sequence[float], labels: Sequence[int], *, max_fpr: Optional[float] = None) -> float:
    scores_arr = to_numpy_1d(scores)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(scores_arr) != len(labels_arr):
        return float("nan")
    finite = np.isfinite(scores_arr)
    scores_arr = scores_arr[finite]
    labels_arr = labels_arr[finite]
    if len(scores_arr) == 0 or len(labels_arr) == 0 or len(np.unique(labels_arr)) < 2:
        return float("nan")
    try:
        if max_fpr is None:
            return float(roc_auc_score(y_score=scores_arr, y_true=labels_arr))
        return float(roc_auc_score(y_score=scores_arr, y_true=labels_arr, max_fpr=max_fpr))
    except Exception:
        return float("nan")


def save_roc(scores: Sequence[float], labels: Sequence[int], path: Path, title: str) -> Dict[str, float]:
    auc = safe_auc(scores, labels)
    pauc = safe_auc(scores, labels, max_fpr=0.1)
    scores_arr = to_numpy_1d(scores)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(scores_arr) != len(labels_arr):
        return {"auc": auc, "pauc": pauc}
    finite = np.isfinite(scores_arr)
    scores_arr = scores_arr[finite]
    labels_arr = labels_arr[finite]
    if len(scores_arr) == 0 or len(np.unique(labels_arr)) < 2:
        return {"auc": auc, "pauc": pauc}

    plt = get_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_score=scores_arr, y_true=labels_arr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title}\nAUC {auc:.6f}, pAUC {pauc:.6f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return {"auc": auc, "pauc": pauc}


def db_calc(x: np.ndarray) -> float:
    rms = np.sqrt(np.sum(x * x) / max(len(x), 1))
    return float(20 * np.log10(max(rms, 1e-12) / 1) + 3)


def manual_si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate.reshape(estimate.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + eps
    projection = torch.sum(estimate * target, dim=1, keepdim=True) * target / target_energy
    noise = estimate - projection
    ratio = (torch.sum(projection ** 2, dim=1) + eps) / (torch.sum(noise ** 2, dim=1) + eps)
    return 10 * torch.log10(ratio + eps)


def load_checkpoint_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(str(path), map_location=device)

    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")

    out: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if not torch.is_tensor(v):
            continue
        clean_k = k[7:] if k.startswith("module.") else k
        out[clean_k] = v
    if len(out) == 0:
        raise RuntimeError(f"No tensor state_dict entries found: {path}")
    return out


def infer_num_class_from_state_dict(state_dict: Dict[str, torch.Tensor], default_num_class: int) -> int:
    candidates: List[Tuple[str, int]] = []
    for k, v in state_dict.items():
        low = k.lower()
        if v.ndim == 2 and any(tok in low for tok in ["fc", "classifier", "head", "linear", "out"]):
            out_dim = int(v.shape[0])
            if 1 < out_dim <= 200:
                candidates.append((k, out_dim))
    if candidates:
        return candidates[-1][1]
    return int(default_num_class)


def load_state_dict_compat(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool,
    allow_partial_load: bool,
    context: str = "model",
) -> None:
    """Load a checkpoint and make non-strict mismatches visible.

    PyTorch returns missing/unexpected keys when strict=False.  The previous
    implementation ignored that return value, which made separation checkpoints
    look loaded even when the selected TSCNet variant did not match the CB count.
    """
    if not allow_partial_load:
        incompatible = model.load_state_dict(state_dict, strict=strict)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if not strict and (missing or unexpected):
            logging.warning(
                "%s load completed with strict=False but state_dict mismatches were found. "
                "missing=%d unexpected=%d first_missing=%s first_unexpected=%s",
                context,
                len(missing),
                len(unexpected),
                missing[:10],
                unexpected[:10],
            )
        return

    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for k, v in state_dict.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        logging.warning("%s partial load skipped %d shape/key-mismatched tensors. First skipped keys: %s", context, len(skipped), skipped[:10])
    if missing:
        logging.warning("%s partial load missing keys: %s", context, list(missing)[:10])
    if unexpected:
        logging.warning("%s partial load unexpected keys: %s", context, list(unexpected)[:10])


def extract_class_label(path: str, data_dir: str, class_list: Sequence[str]) -> str:
    parts = Path(path).parts
    class_map = {c.lower(): c for c in class_list}
    for part in parts:
        hit = class_map.get(part.lower())
        if hit is not None:
            return hit
    # fallback: data_dir 바로 아래 directory
    try:
        rel = Path(path).resolve().relative_to(Path(data_dir).resolve())
        if len(rel.parts) > 0:
            return rel.parts[0]
    except Exception:
        pass
    return "unknown"


def belongs_to_device(path: str, target_device: str) -> bool:
    target_norm = _normalize_machine_name(target_device)
    return any(_normalize_machine_name(part) == target_norm for part in Path(path).parts)


def relative_path_parts(root: Path, path: str) -> List[str]:
    path_obj = Path(path)
    try:
        rel = path_obj.resolve().relative_to(root.resolve())
        return [part.lower() for part in rel.parts]
    except Exception:
        return [part.lower() for part in path_obj.parts]


def get_data_splits(data_dir: str, target_device: str) -> DataSplits:
    data_root = Path(data_dir)
    dirs = sorted([p for p in data_root.glob("*") if p.is_dir()])
    class_list = [p.name for p in dirs]

    all_wavs = sorted([str(p) for p in data_root.rglob("*.wav")])
    train_target: List[str] = []
    train_source: List[str] = []
    test_target: List[str] = []
    test_source: List[str] = []
    supplemental: List[str] = []
    other_train: List[str] = []
    other_test: List[str] = []

    for wav in all_wavs:
        rel_parts = relative_path_parts(data_root, wav)
        base_low = Path(wav).name.lower()
        is_target_device = belongs_to_device(wav, target_device)
        is_aug = "aug" in rel_parts
        is_test = "test" in rel_parts or ("test" in base_low and "train" not in base_low)
        is_train = "train" in rel_parts or ("train" in base_low and not is_test)
        is_supp = "supplemental" in rel_parts or "supplemental" in base_low
        domain = infer_domain(wav)

        if is_target_device:
            if is_supp:
                supplemental.append(wav)
            if is_test:
                if domain == "source":
                    test_source.append(wav)
                elif domain == "target":
                    test_target.append(wav)
                else:
                    # domain token이 없는 경우 target 쪽으로 둔다.
                    test_target.append(wav)
            elif is_train and not is_aug:
                if domain == "source":
                    train_source.append(wav)
                elif domain == "target":
                    train_target.append(wav)
                else:
                    train_target.append(wav)
        else:
            if is_train and not is_aug:
                other_train.append(wav)
            elif is_test and not is_aug:
                other_test.append(wav)

    return DataSplits(
        train_target=sorted(train_target),
        train_source=sorted(train_source),
        test_target=sorted(test_target),
        test_source=sorted(test_source),
        supplemental=sorted(supplemental),
        other_train=sorted(other_train),
        other_test=sorted(other_test),
        class_list=class_list,
    )


def make_records(paths: Sequence[str], target_device: str, data_dir: str, class_list: Sequence[str]) -> List[FileRecord]:
    records: List[FileRecord] = []
    for p in paths:
        records.append(FileRecord(
            path=str(p),
            file_name=os.path.basename(str(p)),
            target_device=target_device,
            domain=infer_domain(str(p)),
            condition=infer_condition(str(p)),
            section=parse_section_from_name(str(p)),
            class_label=extract_class_label(str(p), data_dir, class_list),
        ))
    return records


def covariance_inverse(features: torch.Tensor, *, mu_override: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    if features.shape[0] < 2:
        raise RuntimeError(f"Need at least 2 feature rows for covariance, got {tuple(features.shape)}")
    mu = features.mean(dim=0) if mu_override is None else mu_override
    dev = features - mu
    cov = torch.einsum("bi,bj->ij", dev, dev) / max(features.shape[0] - 1, 1)
    cov_inv = torch.linalg.pinv(cov)
    return mu, cov_inv


def covariance_inverse_with_separate_score_mu(
    features: torch.Tensor,
    *,
    score_mu: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a scoring mean with covariance centered at the empirical mean.

    The uploaded AE/ResNet/unsup evaluators compute source/target covariance
    from centered features, but store source_mu/target_mu as source_dev.mean()
    and target_dev.mean(), which are approximately zero. This helper preserves
    that behavior: covariance uses the empirical mean; Mahalanobis scoring uses
    the supplied score_mu.
    """
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    if features.shape[0] < 2:
        raise RuntimeError(f"Need at least 2 feature rows for covariance, got {tuple(features.shape)}")
    cov_mu = features.mean(dim=0)
    dev = features - cov_mu
    cov = torch.einsum("bi,bj->ij", dev, dev) / max(features.shape[0] - 1, 1)
    cov_inv = torch.linalg.pinv(cov)
    return score_mu, cov_inv


def mahalanobis_rows(features: torch.Tensor, mu: torch.Tensor, cov_inv: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    centered = features - mu
    return ((centered @ cov_inv) * centered).sum(dim=1)


# -----------------------------------------------------------------------------
# Model detection
# -----------------------------------------------------------------------------

def detect_model_type(name: str) -> str:
    low = name.lower()
    if any(tok in low for tok in ["simclr", "simsiam", "moco", "byol", "barlow", "vicreg", "supcon", "ssl", "unsup", "contrast"]):
        return "unsup"
    if (re.search(r"(?:^|_)\d+cb(?:_|$)", low) or re.search(r"(?:^|_)\d+ch(?:_|$)", low)) and ("sep" in low or "snr" in low):
        return "separation"
    if re.search(r"comp\d+lin\d+", low) or re.search(r"l\d+h\d+", low) or re.search(r"(?:^|_)ae[0-9.\-e]+", low):
        return "ae"
    if any(tok in low for tok in ["resnet", "r18", "r34", "r50", "arcface", "crossentropy", "_ce_"]):
        return "classification"
    return "unknown"


def infer_target_device_from_name(path: Path, devices: Sequence[str]) -> Optional[str]:
    """Infer target device/class from filename or directory components.

    Earlier versions only matched filename tokens such as pump_* or *_pump_* and
    path snippets like /pump_.  This missed common layouts such as
    ./models/ae/pump/checkpoint.pth or ./shared/ae/bearing/... .  The path-part
    check below keeps the old filename behavior while also supporting directory
    names that are exactly machine-class names.
    """
    stem = path.stem
    text = str(path)
    path_parts = [p.lower() for p in path.parts]

    for dev in sorted(devices, key=len, reverse=True):
        dev_low = dev.lower()
        # New robust path-component match: .../<device>/...
        if dev_low in path_parts:
            return dev

        # Legacy filename/path-token behavior.
        if stem == dev or stem.startswith(dev + "_") or f"_{dev}_" in stem or f"/{dev}_" in text:
            return dev

    # weaker fallback for filenames like R18_SimClr_revadd_pump_...
    low = stem.lower()
    for dev in sorted(devices, key=len, reverse=True):
        if f"_{dev.lower()}_" in f"_{low}_":
            return dev
    return None


def parse_resnet_arch(name: str, default_arch: str) -> str:
    """Infer ResNet depth from a checkpoint filename/path.

    Important: older versions of this evaluator forgot ResNet152.  When a
    ResNet152 CE/ArcFace checkpoint was evaluated with the default
    --resnet=resnet18, torch reported many size-mismatch errors such as
    layer*.conv1.weight 1x1 vs 3x3 and fc 2048 vs 512.
    """
    low = name.lower()
    patterns = [
        (r"resnet\s*152|resnet152|r152", "resnet152"),
        (r"resnet\s*101|resnet101|r101", "resnet101"),
        (r"resnet\s*50|resnet50|r50", "resnet50"),
        (r"resnet\s*34|resnet34|r34", "resnet34"),
        (r"resnet\s*18|resnet18|r18", "resnet18"),
    ]
    for pat, arch in patterns:
        if re.search(pat, low):
            return arch
    return default_arch


def _state_dict_layer_block_counts(state_dict: Dict[str, torch.Tensor]) -> Dict[int, int]:
    """Return estimated number of blocks per ResNet layer from checkpoint keys."""
    counts: Dict[int, int] = {}
    # Accept keys such as model.layer3.22.conv1.weight, layer3.22.conv1.weight,
    # backbone.model.layer3.22.conv1.weight, etc.
    pat = re.compile(r"(?:^|\.)layer([1-4])\.(\d+)\.")
    for key in state_dict.keys():
        m = pat.search(key)
        if not m:
            continue
        layer = int(m.group(1))
        block = int(m.group(2))
        counts[layer] = max(counts.get(layer, 0), block + 1)
    return counts


def infer_resnet_arch_from_state_dict(state_dict: Dict[str, torch.Tensor], fallback_arch: str) -> str:
    """Infer ResNet depth from fc input dimension and layer block counts.

    This is a safety net for CE/ArcFace/unsup checkpoints whose filename does
    not contain an explicit architecture token.  Filename parsing still has
    priority in build_model_specs(), but this function prevents accidental
    loading of a 50/101/152 checkpoint into ResNet18.
    """
    fc_in: Optional[int] = None
    for key, value in state_dict.items():
        low = key.lower()
        if value.ndim == 2 and (low.endswith("fc.weight") or ".fc.weight" in low):
            fc_in = int(value.shape[1])
            break
    counts = _state_dict_layer_block_counts(state_dict)

    if fc_in == 2048:
        # Bottleneck ResNets.  torchvision/custom ResNet block counts are:
        # 50=[3,4,6,3], 101=[3,4,23,3], 152=[3,8,36,3].
        l2 = counts.get(2, 0)
        l3 = counts.get(3, 0)
        if l3 >= 30 or l2 >= 8:
            return "resnet152"
        if l3 >= 20:
            return "resnet101"
        return "resnet50"

    if fc_in == 512:
        # BasicBlock ResNets.  34=[3,4,6,3], 18=[2,2,2,2].
        if counts.get(3, 0) >= 6 or counts.get(2, 0) >= 4:
            return "resnet34"
        return "resnet18"

    return fallback_arch


def parse_unsup_mode(name: str, default_mode: str = "auto") -> str:
    low = name.lower()
    if "simsiam" in low:
        return "simsiam"
    if "simclr" in low or "sim_clr" in low:
        return "simclr"
    if default_mode and default_mode != "auto":
        return default_mode
    return "simclr"


def count_compatible_keys(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> int:
    model_state = model.state_dict()
    return sum(
        1
        for key, value in state_dict.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    )


def parse_ae_dims(name: str, default_comp: int, default_lin: int) -> Tuple[int, int]:
    low = name.lower()
    m = re.search(r"comp(\d+)lin(\d+)", low)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"l(\d+)h(\d+)", low)
    if m:
        return int(m.group(1)), int(m.group(2))
    return default_comp, default_lin


def parse_int_after(pattern: str, name: str) -> Optional[int]:
    m = re.search(pattern, name.lower())
    return int(m.group(1)) if m else None


def parse_last_int_after(pattern: str, name: str) -> Optional[int]:
    matches = re.findall(pattern, name.lower())
    if not matches:
        return None
    last = matches[-1]
    if isinstance(last, tuple):
        last = last[0]
    return int(last)


def infer_sep_cb_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    """Infer TSCNet_Cont CB count from checkpoint keys.

    TSCNet_Cont(cb=N) owns trainable blocks named TSCB_1 ... TSCB_N.
    For cb=0 there are no TSCB_* keys.  A missing return means the checkpoint
    does not expose enough information and the filename/default should be used.
    """
    max_cb = -1
    has_tscnet_keys = False
    pat = re.compile(r"(?:^|\.)TSCB_(\d+)(?:\.|$)")
    for key in state_dict.keys():
        if any(token in key for token in ["dense_encoder", "dense_R", "dense_P", "mask_decoder", "complex_decoder"]):
            has_tscnet_keys = True
        m = pat.search(key)
        if m:
            max_cb = max(max_cb, int(m.group(1)))
    if max_cb >= 0:
        return max_cb
    if has_tscnet_keys:
        return 0
    return None


def resolve_sep_cb(spec: ModelSpec, args: argparse.Namespace, state_dict: Dict[str, torch.Tensor]) -> int:
    """Resolve the CB count used to instantiate TSCNet_Cont.

    Checkpoint tensor keys are authoritative when they expose TSCB block counts.
    Filename metadata is still used for old state_dicts without enough keys.
    """
    cb_from_name = spec.cb
    cb_from_state = infer_sep_cb_from_state_dict(state_dict)
    if cb_from_state is not None:
        cb = int(cb_from_state)
        if cb_from_name is not None and cb_from_name != cb:
            logging.warning(
                "Separation CB mismatch between filename and checkpoint keys for %s: filename=%dcb, state_dict=%dcb. "
                "Using state_dict value.",
                spec.path,
                cb_from_name,
                cb_from_state,
            )
        return cb
    if cb_from_name is not None:
        return int(cb_from_name)
    return int(args.sep_default_cb)


def parse_float_after(pattern: str, name: str) -> Optional[float]:
    m = re.search(pattern, name.lower())
    if not m:
        return None
    try:
        return float(m.group(1).rstrip("."))
    except Exception:
        return None


def build_model_specs(args: argparse.Namespace) -> List[ModelSpec]:
    model_root = Path(args.model_root)
    files = sorted(model_root.rglob(args.model_glob)) if model_root.is_dir() else [model_root]
    specs: List[ModelSpec] = []

    for path in files:
        if path.suffix.lower() != ".pth":
            continue
        raw_name = path.name
        model_type = args.model_type if args.model_type != "auto" else detect_model_type(raw_name)
        if model_type == "unknown":
            logging.warning("Skip unknown model type: %s", path)
            continue

        target_device = infer_target_device_from_name(path, args.devices)
        if args.target_device is not None:
            target_device = args.target_device

        if target_device is not None:
            eval_devices = [target_device]
        elif model_type in ["classification", "unsup"]:
            eval_devices = list(args.devices)
        else:
            logging.warning("Skip model without target device in filename/path: %s", path)
            continue

        eval_devices = [d for d in eval_devices if d in args.devices]
        if not eval_devices:
            continue

        comp_feat, lin_feat = parse_ae_dims(raw_name, args.ae_default_comp_feat, args.ae_default_lin_feat)
        channel_size = parse_int_after(r"(\d+)ch", raw_name) or args.sep_default_channel
        cb = parse_last_int_after(r"(\d+)cb", raw_name)
        batch_size_from_name = parse_int_after(r"batch(\d+)", raw_name)
        epoch_from_name = parse_int_after(r"epoch(\d+)", raw_name)
        loss_from_name = parse_float_after(r"loss([\-+0-9.eE]+)", raw_name)
        arch = parse_resnet_arch(raw_name, args.resnet)
        unsup_mode = parse_unsup_mode(raw_name, args.unsup_mode) if model_type == "unsup" else None

        specs.append(ModelSpec(
            path=path,
            model_type=model_type,
            model_id=safe_name(path.stem),
            target_device=target_device,
            eval_devices=eval_devices,
            arch=arch,
            comp_feat=comp_feat,
            lin_feat=lin_feat,
            channel_size=channel_size,
            cb=cb,
            batch_size_from_name=batch_size_from_name,
            epoch_from_name=epoch_from_name,
            loss_from_name=loss_from_name,
            unsup_mode=unsup_mode,
            raw_name=raw_name,
        ))

    return specs


# -----------------------------------------------------------------------------
# Adapters
# -----------------------------------------------------------------------------

class BaseAdapter:
    has_domain_mahalanobis: bool = False
    supports_classification_proxy: bool = False

    def __init__(self, model: torch.nn.Module, spec: ModelSpec, args: argparse.Namespace, device: torch.device, class_list: Sequence[str]):
        self.model = model
        self.spec = spec
        self.args = args
        self.device = device
        self.class_list = list(class_list)
        self.model.eval()

    def extract(self, file_path: str) -> FeatureOutput:
        raise NotImplementedError

    def classification_logits(self, file_path: str) -> Optional[torch.Tensor]:
        return None


class AEAdapter(BaseAdapter):
    has_domain_mahalanobis = True

    def ae_feature(self, file_path: str) -> np.ndarray:
        ori_signal, sr = librosa.load(file_path, sr=None, mono=True)
        n_frames = int(self.args.ae_n_frames)
        dims = 128 * n_frames
        mel = librosa.feature.melspectrogram(
            y=ori_signal,
            sr=sr,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0,
            fmin=0.0,
        )
        log_mel = 10.0 * np.log10(np.maximum(mel, sys.float_info.epsilon))
        n_vectors = len(log_mel[0, :]) - n_frames + 1
        if n_vectors < 1:
            return np.empty((0, dims), dtype=np.float32)
        vectors = np.zeros((n_vectors, dims), dtype=np.float32)
        for t in range(n_frames):
            vectors[:, 128 * t:128 * (t + 1)] = log_mel[:, t:t + n_vectors].T
        return vectors

    def extract(self, file_path: str) -> FeatureOutput:
        vectors = self.ae_feature(file_path)
        if vectors.shape[0] == 0:
            raise RuntimeError(f"Too short audio for AE feature extraction: {file_path}")
        in_feature = torch.as_tensor(vectors, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = self.model(in_feature)
            recons = output[0] if isinstance(output, (tuple, list)) else output
        l1 = float(F.l1_loss(recons, in_feature).detach().cpu().item())
        l2 = float(F.mse_loss(recons, in_feature).detach().cpu().item())
        residual = (recons - in_feature).reshape(-1, 128).detach()
        file_feature = residual.mean(dim=0, keepdim=True).detach()
        return FeatureOutput(
            cov_features=residual,
            file_feature=file_feature,
            mah_features=residual,
            proxy={"ae_l1": l1, "ae_l2": l2},
        )


class SepAdapter(BaseAdapter):
    has_domain_mahalanobis = False

    def __init__(self, model: torch.nn.Module, spec: ModelSpec, args: argparse.Namespace, device: torch.device, class_list: Sequence[str]):
        super().__init__(model, spec, args, device, class_list)
        self.window = torch.hamming_window(400, device=self.device)
        self.segment_len = int(args.sep_segment_len)
        self.n_segments = int(args.sep_n_segments)

    def prepare_audio(self, wav: np.ndarray) -> np.ndarray:
        required = self.segment_len * self.n_segments
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if len(wav) < required:
            wav = np.pad(wav, (0, required - len(wav)))
        return wav[:required]

    def segment_feature(self, signal: np.ndarray) -> torch.Tensor:
        signal_t = torch.as_tensor(signal, dtype=torch.float32, device=self.device)
        spec = torch.stft(
            signal_t,
            400,
            200,
            window=self.window,
            onesided=True,
            return_complex=True,
        ).unsqueeze(dim=1)
        feat = torch.cat([spec.real, spec.imag], dim=1)
        # original: feature_maker(...).unsqueeze(0).swapaxes(1, 2)
        return feat.unsqueeze(dim=0).swapaxes(1, 2)

    def wav_convert(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(dim=0)
        spec = torch.complex(tensor[0, :, :], tensor[1, :, :])
        istft_kwargs = {
            "n_fft": 400,
            "hop_length": 200,
            "window": self.window,
            "onesided": True,
        }
        if self.args.sep_istft_fixed_length:
            istft_kwargs["length"] = self.segment_len
        wav = torch.istft(spec, **istft_kwargs)
        return wav

    def extract_from_wav(self, wav: np.ndarray) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        wav = self.prepare_audio(wav)
        cont_features: List[torch.Tensor] = []
        recons_outs: List[torch.Tensor] = []
        for i in range(self.n_segments):
            seg = wav[i * self.segment_len:(i + 1) * self.segment_len]
            feat = self.segment_feature(seg)
            with torch.no_grad():
                output = self.model(feat)
                if not isinstance(output, (tuple, list)) or len(output) < 2:
                    raise RuntimeError("Separation model must return (reconstruction, feature).")
                recons, cont = output[0], output[1]
            cont_features.append(cont.detach())
            recons_outs.append(recons.detach())
        full_feature = torch.cat(cont_features, dim=1).detach()
        return full_feature, recons_outs

    def extract(self, file_path: str) -> FeatureOutput:
        wav, _ = librosa.load(file_path, sr=None, mono=True)
        full_feature, _ = self.extract_from_wav(wav)
        return FeatureOutput(
            cov_features=full_feature,
            file_feature=full_feature,
            mah_features=full_feature,
            proxy={},
        )

    def reconstruct_wave(self, mix_wav: np.ndarray) -> torch.Tensor:
        _, recons_outs = self.extract_from_wav(mix_wav)
        wavs = [self.wav_convert(x) for x in recons_outs]
        return torch.cat(wavs, dim=0)


class ResNetAdapter(BaseAdapter):
    has_domain_mahalanobis = True
    supports_classification_proxy = True

    def __init__(self, model: torch.nn.Module, spec: ModelSpec, args: argparse.Namespace, device: torch.device, class_list: Sequence[str]):
        super().__init__(model, spec, args, device, class_list)
        if T is None:
            raise RuntimeError("torchaudio is required for ResNet/classification/unsup evaluation.")
        self.mel_conv = T.MelSpectrogram(
            sample_rate=args.resnet_sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0,
        )
        self.db_conv = T.AmplitudeToDB()

    def mel_feature(self, file_path: str) -> torch.Tensor:
        wav, _ = librosa.load(file_path, sr=None, mono=True)
        wav_t = torch.as_tensor(wav, dtype=torch.float32).unsqueeze(0)
        mel = self.db_conv(self.mel_conv(wav_t)).unsqueeze(dim=0)
        return mel.to(self.device)

    def forward_pair(self, file_path: str) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        in_feature = self.mel_feature(file_path)
        with torch.no_grad():
            try:
                output = self.model(in_feature)
            except TypeError:
                # 일부 contrastive/unsupervised wrapper는 forward(x_i, x_j)를 요구한다.
                # 평가 시에는 동일 view를 두 번 넣고 embedding 쪽 output만 사용한다.
                output = self.model(in_feature, in_feature)
        logits: Optional[torch.Tensor]
        feat: torch.Tensor
        if isinstance(output, (tuple, list)):
            if self.spec.model_type == "unsup" and len(output) >= 4:
                # common pattern: (_, z_i, _, z_j)
                logits, feat = None, output[1]
            elif self.spec.model_type == "unsup" and len(output) >= 2 and output[1].ndim >= 2:
                logits, feat = None, output[1]
            elif len(output) >= 2:
                logits, feat = output[0], output[1]
            elif len(output) == 1:
                logits, feat = None, output[0]
            else:
                raise RuntimeError("Empty model output.")
        else:
            logits, feat = None, output
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        elif feat.dim() > 2:
            feat = feat.reshape(feat.shape[0], -1)
        return logits, feat.detach()

    def extract(self, file_path: str) -> FeatureOutput:
        logits, feat = self.forward_pair(file_path)
        proxy: Dict[str, Any] = {}
        if logits is not None:
            proxy["has_logits"] = True
        return FeatureOutput(
            cov_features=feat,
            file_feature=feat,
            mah_features=feat,
            proxy=proxy,
        )

    def classification_logits(self, file_path: str) -> Optional[torch.Tensor]:
        logits, _ = self.forward_pair(file_path)
        if logits is None:
            return None
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return logits.detach()


class UnsupAdapter(BaseAdapter):
    """Adapter matching the uploaded unsup_eval.py evaluation path.

    The SimCLR/SimSiam checkpoint is loaded as a full contrastive wrapper, then
    the wrapper's backbone is used for ASD and proxy feature extraction.
    Input features follow the latest unsup_eval.py: librosa log-mel with shape
    (1, 1, 128, T), not the torchaudio classification feature path.
    """

    has_domain_mahalanobis = True
    supports_classification_proxy = False

    def feature_tensor(self, file_path: str) -> torch.Tensor:
        ori_signal, sr = librosa.load(file_path, sr=None, mono=True)
        mel = librosa.feature.melspectrogram(
            y=ori_signal,
            sr=sr,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0,
            fmin=0.0,
        )
        log_mel = 10.0 * np.log10(np.maximum(mel, sys.float_info.epsilon))
        log_mel = np.expand_dims(log_mel, 0)
        log_mel = np.expand_dims(log_mel, 0)
        return torch.as_tensor(log_mel, dtype=torch.float32, device=self.device)

    def embedding(self, file_path: str) -> torch.Tensor:
        in_feature = self.feature_tensor(file_path)
        with torch.no_grad():
            out = self.model(in_feature)
        if isinstance(out, (tuple, list)):
            # Defensive fallback if a wrapper rather than backbone is passed.
            if len(out) >= 4 and torch.is_tensor(out[1]):
                feat = out[1]
            elif len(out) >= 2 and torch.is_tensor(out[1]):
                feat = out[1]
            elif len(out) >= 1 and torch.is_tensor(out[0]):
                feat = out[0]
            else:
                raise RuntimeError("Unsupported unsup model output format.")
        else:
            feat = out
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        elif feat.dim() > 2:
            feat = feat.reshape(feat.shape[0], -1)
        if feat.shape[0] != 1:
            feat = feat.mean(dim=0, keepdim=True)
        return feat.detach()

    def extract(self, file_path: str) -> FeatureOutput:
        feat = self.embedding(file_path)
        return FeatureOutput(
            cov_features=feat,
            file_feature=feat,
            mah_features=feat,
            proxy={},
        )


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_model_for_spec(spec: ModelSpec, args: argparse.Namespace, device: torch.device, class_list: Sequence[str]) -> torch.nn.Module:
    state_dict = load_checkpoint_state_dict(spec.path, device)

    if spec.model_type == "ae":
        from ae_baseline_utils import AENet  # type: ignore
        model = AENet(
            input_dim=128 * int(args.ae_n_frames),
            block_size=128,
            lin_feat=int(spec.lin_feat or args.ae_default_lin_feat),
            comp_feat=int(spec.comp_feat or args.ae_default_comp_feat),
        )
        load_state_dict_compat(model, state_dict, strict=args.strict_load, allow_partial_load=args.allow_partial_load)

    elif spec.model_type == "separation":
        channel = int(spec.channel_size or args.sep_default_channel)
        cb = resolve_sep_cb(spec, args, state_dict)

        if cb < 0 or cb > 4:
            raise RuntimeError(f"Unsupported separation CB count for {spec.path}: cb={cb}. Expected 0, 1, 2, 3, or 4.")

        # Legacy separation evaluation used TSCNet_Cont(num_channel=..., cb=...).
        # Do not use sep_unsup_util.TSCNet here: that class is hard-coded to two
        # TSCB blocks and has a different feature_out layout from TSCNet_Cont(cb=2).
        if cb == 0 and args.sep_use_noncb_for_0cb:
            from models.generator_comp import TSCNetnonCB  # type: ignore
            model = TSCNetnonCB(num_channel=channel)
            sep_context = f"separation TSCNetnonCB channel={channel} filename_cb={cb}"
            sep_strict = args.strict_load or args.sep_strict_load
        else:
            from models.generator_comp import TSCNet_Cont  # type: ignore
            model = TSCNet_Cont(num_channel=channel, cb=cb)
            sep_context = f"separation TSCNet_Cont channel={channel} cb={cb}"
            sep_strict = args.strict_load or args.sep_strict_load

        logging.info("Load %s from %s", sep_context, spec.path)
        load_state_dict_compat(
            model,
            state_dict,
            strict=sep_strict,
            allow_partial_load=args.allow_partial_load,
            context=sep_context,
        )

    elif spec.model_type == "classification":
        from models.resnet_oth import ResNet  # type: ignore
        if args.num_class > 0:
            num_class = int(args.num_class)
        else:
            num_class = infer_num_class_from_state_dict(state_dict, default_num_class=max(len(class_list), 1))

        parsed_arch = spec.arch or args.resnet
        arch = infer_resnet_arch_from_state_dict(state_dict, parsed_arch)
        if arch != parsed_arch:
            logging.info(
                "Override classification ResNet arch from %s to %s based on checkpoint tensor shapes: %s",
                parsed_arch, arch, spec.path,
            )
        model = ResNet(num_class=num_class, resnet_type=arch)
        load_state_dict_compat(model, state_dict, strict=args.strict_load, allow_partial_load=args.allow_partial_load)

    elif spec.model_type == "unsup":
        from models.resnet_oth import ResNetBack, SimCLR, SimSiam  # type: ignore
        parsed_arch = spec.arch or args.resnet
        arch = infer_resnet_arch_from_state_dict(state_dict, parsed_arch)
        if arch != parsed_arch:
            logging.info(
                "Override unsup ResNet arch from %s to %s based on checkpoint tensor shapes: %s",
                parsed_arch, arch, spec.path,
            )
        mode = spec.unsup_mode or parse_unsup_mode(spec.raw_name, args.unsup_mode)
        backbone = ResNetBack(resnet_type=arch)
        if mode == "simsiam":
            if arch in ["resnet18", "resnet34"]:
                wrapper = SimSiam(backbone, proj_hidden_dim=512, proj_out_dim=512)
            else:
                wrapper = SimSiam(backbone, proj_hidden_dim=2048, proj_out_dim=2048)
        else:
            if arch in ["resnet18", "resnet34"]:
                wrapper = SimCLR(backbone, proj_hidden_dim=512)
            else:
                wrapper = SimCLR(backbone, proj_hidden_dim=2048)

        allow_partial = args.allow_partial_load or args.unsup_partial_load
        wrapper_matches = count_compatible_keys(wrapper, state_dict)
        backbone_matches = count_compatible_keys(wrapper.backbone, state_dict)
        if backbone_matches > wrapper_matches:
            # Handles checkpoints saved from model.backbone directly.
            load_state_dict_compat(
                wrapper.backbone,
                state_dict,
                strict=args.strict_load and not allow_partial,
                allow_partial_load=allow_partial,
            )
        else:
            # Handles the standard uploaded unsup_eval.py path: full SimCLR/SimSiam wrapper checkpoint.
            load_state_dict_compat(
                wrapper,
                state_dict,
                strict=args.strict_load and not allow_partial,
                allow_partial_load=allow_partial,
            )
        # Match latest unsup_eval.py: after loading the full contrastive wrapper,
        # evaluate model.backbone for ASD and Alignment/Uniformity embeddings.
        model = wrapper.backbone

    else:
        raise RuntimeError(f"Unsupported model_type={spec.model_type}")

    model.to(device)
    model.eval()
    return model


def make_adapter(model: torch.nn.Module, spec: ModelSpec, args: argparse.Namespace, device: torch.device, class_list: Sequence[str]) -> BaseAdapter:
    if spec.model_type == "ae":
        return AEAdapter(model, spec, args, device, class_list)
    if spec.model_type == "separation":
        return SepAdapter(model, spec, args, device, class_list)
    if spec.model_type == "classification":
        return ResNetAdapter(model, spec, args, device, class_list)
    if spec.model_type == "unsup":
        return UnsupAdapter(model, spec, args, device, class_list)
    raise RuntimeError(f"Unsupported model_type={spec.model_type}")


# -----------------------------------------------------------------------------
# Covariance, ASD scoring, linear probes
# -----------------------------------------------------------------------------

def compute_cov_stats(adapter: BaseAdapter, train_records: Sequence[FileRecord], args: argparse.Namespace) -> CovStats:
    all_cov: List[torch.Tensor] = []
    source_cov: List[torch.Tensor] = []
    target_cov: List[torch.Tensor] = []
    proxy_rows: List[Dict[str, Any]] = []

    for idx, rec in enumerate(train_records):
        fout = adapter.extract(rec.path)
        cov_feat = fout.cov_features.detach()
        all_cov.append(cov_feat)
        if rec.domain == "source":
            source_cov.append(cov_feat)
        elif rec.domain == "target":
            target_cov.append(cov_feat)
        else:
            target_cov.append(cov_feat)
        row = {
            "split": "train",
            "file_name": rec.file_name,
            "path": rec.path,
            "target_device": rec.target_device,
            "domain": rec.domain,
            "condition": rec.condition,
            "section": rec.section,
            **fout.proxy,
        }
        proxy_rows.append(row)
        if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
            logging.info("  train feature %d/%d", idx + 1, len(train_records))

    train_features = torch.cat(all_cov, dim=0)
    train_mu, train_cov_inv = covariance_inverse(train_features)

    source_mu = source_cov_inv = target_mu = target_cov_inv = None
    if adapter.has_domain_mahalanobis and len(source_cov) >= 1 and len(target_cov) >= 1:
        source_features = torch.cat(source_cov, dim=0)
        target_features = torch.cat(target_cov, dim=0)
        if source_features.shape[0] >= 2:
            if args.fix_domain_mu:
                source_mu, source_cov_inv = covariance_inverse(source_features)
            else:
                # Legacy compatibility: covariance centered by empirical domain mean,
                # but scoring mean is source_dev.mean() ~= 0 in the uploaded code.
                zero_mu = torch.zeros(source_features.shape[1], device=source_features.device, dtype=source_features.dtype)
                source_mu, source_cov_inv = covariance_inverse_with_separate_score_mu(source_features, score_mu=zero_mu)
        if target_features.shape[0] >= 2:
            if args.fix_domain_mu:
                target_mu, target_cov_inv = covariance_inverse(target_features)
            else:
                zero_mu = torch.zeros(target_features.shape[1], device=target_features.device, dtype=target_features.dtype)
                target_mu, target_cov_inv = covariance_inverse_with_separate_score_mu(target_features, score_mu=zero_mu)

    return CovStats(
        train_mu=train_mu,
        train_cov_inv=train_cov_inv,
        train_features=train_features.detach(),
        source_mu=source_mu,
        source_cov_inv=source_cov_inv,
        target_mu=target_mu,
        target_cov_inv=target_cov_inv,
        train_proxy_df=pd.DataFrame(proxy_rows),
    )


def linear_oracle_eval(
    id_test_features: torch.Tensor,
    ood_test_features: torch.Tensor,
    id_train_features: torch.Tensor,
    ood_train_features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_dim = int(id_train_features.shape[1])
    linear_classifier = torch.nn.Linear(feature_dim, 2).to(device)
    linear_classifier.train()
    optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    features = torch.cat([id_train_features.detach(), ood_train_features.detach()], dim=0)
    labels = torch.cat([
        torch.zeros(len(id_train_features), dtype=torch.long),
        torch.ones(len(ood_train_features), dtype=torch.long),
    ], dim=0)
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_model_state = copy.deepcopy(linear_classifier.state_dict())
    best_accuracy = -1.0
    for _epoch in range(epochs):
        correct = 0
        for inputs, y in loader:
            inputs = inputs.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = linear_classifier(inputs)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            correct += int((outputs.argmax(1) == y).sum().item())
        accuracy = correct / max(len(dataset), 1)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = copy.deepcopy(linear_classifier.state_dict())

    linear_classifier.load_state_dict(best_model_state)
    linear_classifier.eval()
    with torch.no_grad():
        id_logits = linear_classifier(id_test_features.to(device)).detach().cpu()
        ood_logits = linear_classifier(ood_test_features.to(device)).detach().cpu()
    id_scores = torch.softmax(id_logits, dim=1).numpy()[:, 1]
    ood_scores = torch.softmax(ood_logits, dim=1).numpy()[:, 1]
    return id_scores, ood_scores


def concat_feature_items(items: Sequence[Tuple[FileRecord, torch.Tensor]]) -> torch.Tensor:
    if not items:
        raise RuntimeError("No feature items to concatenate.")
    return torch.cat([x[1] for x in items], dim=0)


def run_linear_probes(
    section_items: Dict[str, Dict[str, List[Tuple[FileRecord, torch.Tensor]]]],
    args: argparse.Namespace,
    device: torch.device,
    model_id: str,
    target_device: str,
    out_dir: Path,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    metrics: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []
    sections = sorted(section_items.keys(), key=section_sort_key)
    valid_sections = [s for s in sections if section_items[s].get("normal") and section_items[s].get("anomaly")]
    if len(valid_sections) < 2:
        logging.warning("Not enough sections for linear probes: %s", valid_sections)
        return metrics, pd.DataFrame(rows)

    # 1) leave-one-section-out: 기존 3fold를 n-section으로 일반화
    loso_normal_scores: List[float] = []
    loso_anomaly_scores: List[float] = []
    for sec in valid_sections:
        test_n_items = section_items[sec]["normal"]
        test_a_items = section_items[sec]["anomaly"]
        train_n_items = [item for s in valid_sections if s != sec for item in section_items[s]["normal"]]
        train_a_items = [item for s in valid_sections if s != sec for item in section_items[s]["anomaly"]]
        if not train_n_items or not train_a_items:
            continue

        score_n, score_a = linear_oracle_eval(
            concat_feature_items(test_n_items),
            concat_feature_items(test_a_items),
            concat_feature_items(train_n_items),
            concat_feature_items(train_a_items),
            device=device,
            batch_size=args.linear_batch_size,
            epochs=args.linear_epochs,
            lr=args.linear_lr,
        )
        loso_normal_scores.extend(score_n.tolist())
        loso_anomaly_scores.extend(score_a.tolist())

        fold_scores = np.concatenate([score_n, score_a])
        fold_labels = np.concatenate([np.zeros(len(score_n)), np.ones(len(score_a))])
        metrics[f"linear_loso_auc_{sec}"] = safe_auc(fold_scores, fold_labels)
        metrics[f"linear_loso_pauc_{sec}"] = safe_auc(fold_scores, fold_labels, max_fpr=0.1)

        for rec, score in zip([x[0] for x in test_n_items], score_n):
            rows.append({"model_id": model_id, "target_device": target_device, "probe": "linear_loso", "fold": sec,
                         "file_name": rec.file_name, "path": rec.path, "condition": rec.condition, "domain": rec.domain,
                         "section": rec.section, "score": float(score)})
        for rec, score in zip([x[0] for x in test_a_items], score_a):
            rows.append({"model_id": model_id, "target_device": target_device, "probe": "linear_loso", "fold": sec,
                         "file_name": rec.file_name, "path": rec.path, "condition": rec.condition, "domain": rec.domain,
                         "section": rec.section, "score": float(score)})

    loso_scores = np.concatenate([np.asarray(loso_normal_scores), np.asarray(loso_anomaly_scores)])
    loso_labels = np.concatenate([np.zeros(len(loso_normal_scores)), np.ones(len(loso_anomaly_scores))])
    metrics["linear_loso_auc"] = safe_auc(loso_scores, loso_labels)
    metrics["linear_loso_pauc"] = safe_auc(loso_scores, loso_labels, max_fpr=0.1)
    if args.save_roc:
        save_roc(loso_scores, loso_labels, out_dir / "roc" / f"{model_id}_{target_device}_linear_loso_ROC.png",
                 f"Linear probe leave-one-section-out ({len(valid_sections)}fold) {target_device} {model_id}")

    # 2) all-section train/test: 기존 score_all_nor/score_all_ano 구성
    all_n_items = [item for s in valid_sections for item in section_items[s]["normal"]]
    all_a_items = [item for s in valid_sections for item in section_items[s]["anomaly"]]
    if all_n_items and all_a_items:
        score_n, score_a = linear_oracle_eval(
            concat_feature_items(all_n_items),
            concat_feature_items(all_a_items),
            concat_feature_items(all_n_items),
            concat_feature_items(all_a_items),
            device=device,
            batch_size=args.linear_batch_size,
            epochs=args.linear_epochs,
            lr=args.linear_lr,
        )
        all_scores = np.concatenate([score_n, score_a])
        all_labels = np.concatenate([np.zeros(len(score_n)), np.ones(len(score_a))])
        metrics["linear_all_auc"] = safe_auc(all_scores, all_labels)
        metrics["linear_all_pauc"] = safe_auc(all_scores, all_labels, max_fpr=0.1)
        if args.save_roc:
            save_roc(all_scores, all_labels, out_dir / "roc" / f"{model_id}_{target_device}_linear_all_ROC.png",
                     f"Linear probe all-section {target_device} {model_id}")
        for rec, score in zip([x[0] for x in all_n_items], score_n):
            rows.append({"model_id": model_id, "target_device": target_device, "probe": "linear_all", "fold": "all",
                         "file_name": rec.file_name, "path": rec.path, "condition": rec.condition, "domain": rec.domain,
                         "section": rec.section, "score": float(score)})
        for rec, score in zip([x[0] for x in all_a_items], score_a):
            rows.append({"model_id": model_id, "target_device": target_device, "probe": "linear_all", "fold": "all",
                         "file_name": rec.file_name, "path": rec.path, "condition": rec.condition, "domain": rec.domain,
                         "section": rec.section, "score": float(score)})

    # 3) half split.
    # Legacy code used half = int(len(feature_section_00_anomaly) / 2)
    # and applied that same index to every section/condition.  For 4 sections,
    # the default keeps the same rule using the first sorted section as reference.
    half_test_n: List[Tuple[FileRecord, torch.Tensor]] = []
    half_test_a: List[Tuple[FileRecord, torch.Tensor]] = []
    half_train_n: List[Tuple[FileRecord, torch.Tensor]] = []
    half_train_a: List[Tuple[FileRecord, torch.Tensor]] = []
    reference_half = int(len(section_items[valid_sections[0]]["anomaly"]) / 2)
    for sec in valid_sections:
        n_items = section_items[sec]["normal"]
        a_items = section_items[sec]["anomaly"]
        if args.linear_half_split == "legacy":
            hn = reference_half
            ha = reference_half
        else:
            hn = int(len(n_items) / 2)
            ha = int(len(a_items) / 2)
        if hn <= 0 or ha <= 0 or len(n_items) - hn < 1 or len(a_items) - ha < 1:
            continue
        # legacy: first half is the test argument, second half is the train argument.
        half_test_n.extend(n_items[:hn])
        half_test_a.extend(a_items[:ha])
        half_train_n.extend(n_items[hn:])
        half_train_a.extend(a_items[ha:])

    if half_test_n and half_test_a and half_train_n and half_train_a:
        score_n, score_a = linear_oracle_eval(
            concat_feature_items(half_test_n),
            concat_feature_items(half_test_a),
            concat_feature_items(half_train_n),
            concat_feature_items(half_train_a),
            device=device,
            batch_size=args.linear_batch_size,
            epochs=args.linear_epochs,
            lr=args.linear_lr,
        )
        half_scores = np.concatenate([score_n, score_a])
        half_labels = np.concatenate([np.zeros(len(score_n)), np.ones(len(score_a))])
        metrics["linear_half_auc"] = safe_auc(half_scores, half_labels)
        metrics["linear_half_pauc"] = safe_auc(half_scores, half_labels, max_fpr=0.1)
        if args.save_roc:
            save_roc(half_scores, half_labels, out_dir / "roc" / f"{model_id}_{target_device}_linear_half_ROC.png",
                     f"Linear probe half split {target_device} {model_id}")
        for rec, score in zip([x[0] for x in half_test_n], score_n):
            rows.append({"model_id": model_id, "target_device": target_device, "probe": "linear_half", "fold": "half",
                         "file_name": rec.file_name, "path": rec.path, "condition": rec.condition, "domain": rec.domain,
                         "section": rec.section, "score": float(score)})
        for rec, score in zip([x[0] for x in half_test_a], score_a):
            rows.append({"model_id": model_id, "target_device": target_device, "probe": "linear_half", "fold": "half",
                         "file_name": rec.file_name, "path": rec.path, "condition": rec.condition, "domain": rec.domain,
                         "section": rec.section, "score": float(score)})

    return metrics, pd.DataFrame(rows)


def save_projection_outputs(
    feature_items: Sequence[Tuple[FileRecord, torch.Tensor]],
    args: argparse.Namespace,
    model_id: str,
    target_device: str,
    out_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if len(feature_items) < 3:
        return pd.DataFrame(rows)

    records = [x[0] for x in feature_items]
    features = torch.cat([x[1] for x in feature_items], dim=0).detach().cpu().numpy()
    n_samples = features.shape[0]
    if n_samples < 3:
        return pd.DataFrame(rows)

    projection_dir = out_dir / "projection"
    projection_dir.mkdir(parents=True, exist_ok=True)

    tsne_xy = np.full((n_samples, 2), np.nan, dtype=np.float64)
    umap_xy = np.full((n_samples, 2), np.nan, dtype=np.float64)

    projection_random_state = args.seed if args.seed >= 0 else None
    if not args.skip_tsne and n_samples >= 4:
        perplexity = min(args.tsne_perplexity, max(2, (n_samples - 1) // 3))
        if perplexity < n_samples:
            tsne_xy = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=projection_random_state,
            ).fit_transform(features)

    if not args.skip_umap and umap is not None and n_samples >= 3:
        n_neighbors = min(args.umap_neighbors, max(2, n_samples - 1))
        umap_xy = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=args.umap_min_dist,
            random_state=projection_random_state,
        ).fit_transform(features)

    for i, rec in enumerate(records):
        rows.append({
            "model_id": model_id,
            "target_device": target_device,
            "file_name": rec.file_name,
            "path": rec.path,
            "condition": rec.condition,
            "domain": rec.domain,
            "section": rec.section,
            "tsne_x": float(tsne_xy[i, 0]) if np.isfinite(tsne_xy[i, 0]) else np.nan,
            "tsne_y": float(tsne_xy[i, 1]) if np.isfinite(tsne_xy[i, 1]) else np.nan,
            "umap_x": float(umap_xy[i, 0]) if np.isfinite(umap_xy[i, 0]) else np.nan,
            "umap_y": float(umap_xy[i, 1]) if np.isfinite(umap_xy[i, 1]) else np.nan,
        })

    df = pd.DataFrame(rows)
    coord_path = projection_dir / f"{model_id}_{target_device}_projection_coordinates.csv"
    df.to_csv(coord_path, index=False)

    np.savez_compressed(
        projection_dir / f"{model_id}_{target_device}_projection_features.npz",
        features=features,
        tsne=tsne_xy,
        umap=umap_xy,
        file_name=np.asarray([r.file_name for r in records]),
        condition=np.asarray([r.condition for r in records]),
        domain=np.asarray([r.domain for r in records]),
        section=np.asarray([r.section for r in records]),
    )

    if args.save_projection_plots:
        plt = get_pyplot()
        for method, xcol, ycol in [("umap", "umap_x", "umap_y"), ("tsne", "tsne_x", "tsne_y")]:
            if df[[xcol, ycol]].isna().all().all():
                continue
            # condition plot
            plt.figure(figsize=(8, 7))
            for cond, marker in [("normal", "o"), ("anomaly", "^")]:
                sub = df[df["condition"] == cond]
                if not sub.empty:
                    plt.scatter(sub[xcol], sub[ycol], label=cond, marker=marker, s=18)
            plt.title(f"{target_device} {model_id} {method.upper()} by condition")
            plt.xlabel(f"{method}_x")
            plt.ylabel(f"{method}_y")
            plt.legend()
            plt.tight_layout()
            plt.savefig(projection_dir / f"{model_id}_{target_device}_{method}_condition.png")
            plt.close()

            # section-condition plot
            plt.figure(figsize=(9, 8))
            for (sec, cond), sub in df.groupby(["section", "condition"]):
                marker = "o" if cond == "normal" else "^"
                plt.scatter(sub[xcol], sub[ycol], label=f"{sec} {cond}", marker=marker, s=18)
            plt.title(f"{target_device} {model_id} {method.upper()} by section")
            plt.xlabel(f"{method}_x")
            plt.ylabel(f"{method}_y")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(projection_dir / f"{model_id}_{target_device}_{method}_section.png")
            plt.close()

    return df


def evaluate_asd(
    adapter: BaseAdapter,
    stats: CovStats,
    test_records: Sequence[FileRecord],
    args: argparse.Namespace,
    model_id: str,
    target_device: str,
    out_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_file_rows: List[Dict[str, Any]] = []
    section_items: Dict[str, Dict[str, List[Tuple[FileRecord, torch.Tensor]]]] = {}
    projection_items: List[Tuple[FileRecord, torch.Tensor]] = []

    for idx, rec in enumerate(test_records):
        fout = adapter.extract(rec.path)
        train_scores = mahalanobis_rows(fout.mah_features.detach(), stats.train_mu, stats.train_cov_inv)
        mah_train = float(train_scores.mean().detach().cpu().item())
        mah_domain_min = float("nan")
        mah_source = float("nan")
        mah_target = float("nan")
        if adapter.has_domain_mahalanobis and stats.source_mu is not None and stats.source_cov_inv is not None:
            mah_source = float(mahalanobis_rows(fout.mah_features.detach(), stats.source_mu, stats.source_cov_inv).mean().detach().cpu().item())
        if adapter.has_domain_mahalanobis and stats.target_mu is not None and stats.target_cov_inv is not None:
            mah_target = float(mahalanobis_rows(fout.mah_features.detach(), stats.target_mu, stats.target_cov_inv).mean().detach().cpu().item())
        if np.isfinite(mah_source) and np.isfinite(mah_target):
            mah_domain_min = float(min(mah_source, mah_target))
        elif np.isfinite(mah_source):
            mah_domain_min = mah_source
        elif np.isfinite(mah_target):
            mah_domain_min = mah_target

        row = {
            "model_id": model_id,
            "model_name": adapter.spec.raw_name,
            "model_type": adapter.spec.model_type,
            "target_device": target_device,
            "file_name": rec.file_name,
            "path": rec.path,
            "condition": rec.condition,
            "domain": rec.domain,
            "section": rec.section,
            "class_label": rec.class_label,
            "mah_train": mah_train,
            "mah_source": mah_source,
            "mah_target": mah_target,
            "mah_domain_min": mah_domain_min,
            **fout.proxy,
        }
        per_file_rows.append(row)
        section_items.setdefault(rec.section, {"normal": [], "anomaly": []})
        section_items[rec.section][rec.condition].append((rec, fout.file_feature.detach()))
        projection_items.append((rec, fout.file_feature.detach()))

        if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
            logging.info("  test feature %d/%d", idx + 1, len(test_records))

    per_file_df = pd.DataFrame(per_file_rows)
    labels = (per_file_df["condition"] == "anomaly").astype(int).values

    summary: Dict[str, Any] = {
        "mah_train_auc": safe_auc(per_file_df["mah_train"].values, labels),
        "mah_train_pauc": safe_auc(per_file_df["mah_train"].values, labels, max_fpr=0.1),
        "n_sections": len(section_items),
        "sections": ";".join(sorted(section_items.keys(), key=section_sort_key)),
        "n_test": len(per_file_df),
        "n_test_normal": int((per_file_df["condition"] == "normal").sum()),
        "n_test_anomaly": int((per_file_df["condition"] == "anomaly").sum()),
    }

    if adapter.has_domain_mahalanobis and per_file_df["mah_domain_min"].notna().any():
        summary["mah_domain_min_auc"] = safe_auc(per_file_df["mah_domain_min"].values, labels)
        summary["mah_domain_min_pauc"] = safe_auc(per_file_df["mah_domain_min"].values, labels, max_fpr=0.1)

    if args.save_roc:
        save_roc(per_file_df["mah_train"].values, labels, out_dir / "roc" / f"{model_id}_{target_device}_mah_train_ROC.png",
                 f"Mahalanobis train-cov {target_device} {model_id}")
        if adapter.has_domain_mahalanobis and per_file_df["mah_domain_min"].notna().any():
            save_roc(per_file_df["mah_domain_min"].values, labels,
                     out_dir / "roc" / f"{model_id}_{target_device}_mah_domain_min_ROC.png",
                     f"Mahalanobis source/target-min {target_device} {model_id}")

    linear_metrics, linear_df = run_linear_probes(
        section_items,
        args,
        adapter.device,
        model_id,
        target_device,
        out_dir,
    )
    summary.update(linear_metrics)

    projection_df = pd.DataFrame()
    if not args.skip_projection:
        # Legacy projection concatenated section normals first, then section anomalies.
        # Preserve that ordering for coordinate CSV/NPZ and plots.
        projection_items_legacy_order: List[Tuple[FileRecord, torch.Tensor]] = []
        ordered_sections = sorted(section_items.keys(), key=section_sort_key)
        for sec in ordered_sections:
            projection_items_legacy_order.extend(section_items[sec].get("normal", []))
        for sec in ordered_sections:
            projection_items_legacy_order.extend(section_items[sec].get("anomaly", []))
        projection_df = save_projection_outputs(projection_items_legacy_order, args, model_id, target_device, out_dir)

    return summary, per_file_df, linear_df, projection_df


# -----------------------------------------------------------------------------
# Proxy metrics
# -----------------------------------------------------------------------------

def classification_proxy_benchmark(
    adapter: BaseAdapter,
    target_records: Sequence[FileRecord],
    other_records: Sequence[FileRecord],
    args: argparse.Namespace,
    model_id: str,
    target_device: str,
    out_dir: Path,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if not adapter.supports_classification_proxy:
        return {}, pd.DataFrame()
    if adapter.spec.model_type == "unsup" and not args.unsup_eval_classification_proxy:
        return {}, pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    def predict_one(rec: FileRecord, split_group: str) -> None:
        logits = adapter.classification_logits(rec.path)
        if logits is None:
            return
        pred_idx = int(logits.detach().cpu().numpy().argmax(axis=1)[0])
        if 0 <= pred_idx < len(adapter.class_list):
            pred_label = adapter.class_list[pred_idx]
        else:
            pred_label = f"class_{pred_idx}"
        rows.append({
            "model_id": model_id,
            "target_device": target_device,
            "split_group": split_group,
            "file_name": rec.file_name,
            "path": rec.path,
            "label": rec.class_label,
            "pred": pred_label,
            "domain": rec.domain,
            "condition": rec.condition,
            "section": rec.section,
        })

    for rec in target_records:
        predict_one(rec, "target_device_test")
    for rec in other_records:
        predict_one(rec, "other_device_test")

    df = pd.DataFrame(rows)
    if df.empty:
        return {}, df

    metrics: Dict[str, float] = {}

    def add_f1(metric_name: str, sub: pd.DataFrame, average: str = "micro") -> None:
        if sub.empty:
            metrics[metric_name] = float("nan")
        else:
            metrics[metric_name] = float(f1_score(sub["label"].tolist(), sub["pred"].tolist(), average=average))

    target_df = df[df["split_group"] == "target_device_test"]
    other_df = df[df["split_group"] == "other_device_test"]
    add_f1("clf_domain_target_micro_f1", target_df[target_df["domain"] == "target"])
    add_f1("clf_domain_source_micro_f1", target_df[target_df["domain"] == "source"])

    true_normal_df = target_df[target_df["condition"] == "normal"]
    true_anomaly_df = target_df[target_df["condition"] == "anomaly"]
    if args.legacy_clf_condition_swap:
        # Uploaded clf_benchmark appended normal files into ano_label and anomaly
        # files into nor_label, then saved nor_score under condition_normal and
        # ano_score under condition_anomaly. Keep default metric names comparable
        # with those CSVs, while also exposing corrected *_true_* metrics below.
        add_f1("clf_condition_normal_micro_f1", true_anomaly_df)
        add_f1("clf_condition_anomaly_micro_f1", true_normal_df)
    else:
        add_f1("clf_condition_normal_micro_f1", true_normal_df)
        add_f1("clf_condition_anomaly_micro_f1", true_anomaly_df)
    add_f1("clf_condition_normal_true_micro_f1", true_normal_df)
    add_f1("clf_condition_anomaly_true_micro_f1", true_anomaly_df)

    add_f1("clf_target_total_micro_f1", target_df)
    add_f1("clf_all_micro_f1", pd.concat([target_df, other_df], axis=0))
    add_f1("clf_all_macro_f1", pd.concat([target_df, other_df], axis=0), average="macro")

    proxy_dir = out_dir / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for metric_name, metric_score in metrics.items():
        metric_rows.append({
            "model_id": model_id,
            "target_device": target_device,
            "proxy_type": "classification_proxy",
            "metric_name": metric_name,
            "metric_score": metric_score,
            "n_target_device_test": int(len(target_df)),
            "n_other_device_test": int(len(other_df)),
        })
    metric_df = pd.DataFrame(metric_rows)

    pred_df = df.copy()
    pred_df.insert(0, "row_type", "prediction")
    if not metric_df.empty:
        metric_df.insert(0, "row_type", "metric")

    df.to_csv(proxy_dir / f"{model_id}_{target_device}_classification_proxy_predictions.csv", index=False)
    if not metric_df.empty:
        metric_df.to_csv(proxy_dir / f"{model_id}_{target_device}_classification_proxy_metrics.csv", index=False)

    detail_df = pd.concat([metric_df, pred_df], axis=0, ignore_index=True, sort=False) if not metric_df.empty else pred_df
    return metrics, detail_df


def separation_proxy_benchmark(
    adapter: BaseAdapter,
    splits: DataSplits,
    args: argparse.Namespace,
    model_id: str,
    target_device: str,
    out_dir: Path,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if not isinstance(adapter, SepAdapter):
        return {}, pd.DataFrame()
    if not splits.other_train:
        logging.warning("Skip separation proxy because other_train is empty for %s", target_device)
        return {}, pd.DataFrame()

    si_sdr_metric = None
    if ScaleInvariantSignalDistortionRatio is not None:
        try:
            si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(adapter.device)
        except Exception:
            si_sdr_metric = None

    def get_eval_list(env: str) -> List[str]:
        if env == "train_data":
            return splits.train_all
        if env == "test_target":
            return splits.test_target
        if env == "test_source":
            return splits.test_source
        if env == "test_normal":
            return [p for p in splits.test_all if infer_condition(p) == "normal"]
        if env == "test_anomaly":
            return [p for p in splits.test_all if infer_condition(p) == "anomaly"]
        raise ValueError(env)

    rows: List[Dict[str, Any]] = []
    envs = args.sep_proxy_sets
    for env in envs:
        target_list = get_eval_list(env)
        if not target_list:
            continue
        sample_count = min(args.sep_proxy_k, len(target_list)) if args.sep_proxy_no_replace else args.sep_proxy_k
        if args.sep_proxy_no_replace:
            sampled_targets = random.sample(target_list, k=sample_count)
        else:
            sampled_targets = random.choices(target_list, k=sample_count)
        for snr in args.sep_snr:
            for file_path in sampled_targets:
                target_wav, _ = librosa.load(file_path, sr=None, mono=True)
                noise_wav, _ = librosa.load(random.choice(splits.other_train), sr=None, mono=True)
                target_wav = adapter.prepare_audio(target_wav)
                noise_wav = adapter.prepare_audio(noise_wav)

                sig_db = db_calc(target_wav)
                noise_db = db_calc(noise_wav)
                corr_db = (sig_db - noise_db) - float(snr)
                scale = 1 / (10 ** (corr_db / 20))
                noise_wav = noise_wav / scale
                mix = target_wav + noise_wav
                max_abs = float(np.max(np.abs(mix)))
                if max_abs > 1.0:
                    mix = mix / max_abs
                    target_wav = target_wav / max_abs
                    noise_wav = noise_wav / max_abs

                with torch.no_grad():
                    recons_cat = adapter.reconstruct_wave(mix).unsqueeze(0).to(adapter.device)
                target_t = torch.as_tensor(target_wav, dtype=torch.float32, device=adapter.device).unsqueeze(0)
                min_len = min(recons_cat.shape[-1], target_t.shape[-1])
                recons_cat = recons_cat[..., :min_len]
                target_t = target_t[..., :min_len]
                if si_sdr_metric is not None:
                    sisdr = si_sdr_metric(recons_cat, target_t)
                    sisdr_value = float(sisdr.detach().cpu().reshape(-1)[0].item())
                else:
                    sisdr_value = float(manual_si_sdr(recons_cat, target_t).detach().cpu().reshape(-1)[0].item())
                rows.append({
                    "model_id": model_id,
                    "target_device": target_device,
                    "env": env,
                    "snr": float(snr),
                    "file_name": os.path.basename(file_path),
                    "path": file_path,
                    "si_sdr": sisdr_value,
                })

    df = pd.DataFrame(rows)
    metrics: Dict[str, float] = {}
    if not df.empty:
        metrics["sep_si_sdr_mean"] = float(df["si_sdr"].mean())
        for (env, snr), sub in df.groupby(["env", "snr"]):
            metrics[f"sep_si_sdr_mean_{env}_snr{snr:g}"] = float(sub["si_sdr"].mean())
        proxy_dir = out_dir / "proxy"
        proxy_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(proxy_dir / f"{model_id}_{target_device}_separation_proxy_si_sdr.csv", index=False)
    return metrics, df




def _normalize_machine_name(name: str) -> str:
    name = str(name).lower().replace("_", "").replace("-", "")
    if name in {"toyconveyor", "toyconveyer"}:
        return "toyconveyer"
    return name


def _path_matches_target_scope(path: Path, target_device: Optional[str], scope: str) -> bool:
    """Return whether an augmentation path belongs to the requested target scope."""
    if scope != "target" or target_device is None:
        return True
    target_norm = _normalize_machine_name(target_device)
    return any(_normalize_machine_name(part) == target_norm for part in path.parts)


def _is_class_dataset_dir(path: Path) -> bool:
    """Return True for a class directory containing train/test/aug/ta subfolders."""
    if not path.is_dir():
        return False
    return any((path / subdir).is_dir() for subdir in ("train", "test", "aug", "ta"))


def _iter_machine_dirs(base_path: Path, target_device: Optional[str], scope: str) -> List[Path]:
    """Return class/machine directories under an augmentation root.

    Supported layouts:
      1) <base>/<class>/{train,test,aug}/*.wav
      2) <base>/<class>/ta/*.wav
      3) <base> itself is a class directory with {train,test,aug}/*.wav
    """
    if not base_path.is_dir():
        return []

    dirs: List[Path] = []
    if _is_class_dataset_dir(base_path):
        dirs.append(base_path)
    dirs.extend(sorted([p for p in base_path.iterdir() if p.is_dir()]))

    deduped: List[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        try:
            key = d.resolve()
        except Exception:
            key = d
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)

    if scope == "target" and target_device is not None:
        target_norm = _normalize_machine_name(target_device)
        filtered = [d for d in deduped if _normalize_machine_name(d.name) == target_norm]
        # If --data_dir itself is a single class root, keep it as an explicit
        # target root even when its directory name does not encode target_device.
        if not filtered and _is_class_dataset_dir(base_path):
            return [base_path]
        deduped = filtered
    return deduped


def get_unsup_aug_base_dir(args: argparse.Namespace) -> str:
    """Base directory used for SimCLR/SimSiam Alignment/Uniformity pairs.

    unsup_eval.py uses args.data_dir directly.  Keep that as the default, but
    allow a separate augmentation root when the ASD wav root and ta/aug root are
    stored separately.
    """
    aug_dir = getattr(args, "unsup_aug_dir", None)
    return str(aug_dir) if aug_dir else str(args.data_dir)


def strip_stored_aug_prefix(aug_name: str, train_names: Sequence[str]) -> Optional[str]:
    """Return original train filename for stored augmentation files.

    Supported examples:
      fm0_section_00_source_train_normal_0000_vel_22.wav
      ts0.8_section_00_source_train_normal_0000_vel_22.wav
      <any_prefix>_section_00_source_train_normal_0000_vel_22.wav

    The most reliable rule is suffix matching against known train filenames.
    The section_ fallback is kept for datasets where train files were moved or
    the train directory is incomplete.
    """
    for train_name in train_names:
        if aug_name.endswith(train_name) and aug_name != train_name:
            return train_name
    idx = aug_name.find("section_")
    if idx >= 0:
        return aug_name[idx:]
    return None


def find_ta_wav_files(
    base_dir: str,
    *,
    target_device: Optional[str] = None,
    scope: str = "all",
    recursive: bool = False,
) -> List[List[str]]:
    """Find legacy ta/*.wav view-major augmentation lists.

    unsup_eval.py discovers files as ``<base>/<machine>/ta/*.wav`` and then
    treats each machine's list as view-major blocks.  This implementation first
    applies that exact one-level rule.  If no files are found and recursive=True,
    it falls back to any nested ``*/ta/*.wav`` folders, which covers datasets
    where machine directories are nested one level deeper than the legacy script.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        logging.warning("Unsupervised augmentation base directory does not exist: %s", base_dir)
        return []

    nested_file_list: List[List[str]] = []
    seen_ta_dirs: set[Path] = set()

    # Legacy unsup_eval.py behavior: only immediate child machine dirs.
    for dir_path in _iter_machine_dirs(base_path, target_device, scope):
        ta_dir = dir_path / "ta"
        if not ta_dir.is_dir():
            continue
        wav_files = sorted([str(p.resolve()) for p in ta_dir.glob("*.wav")])
        if wav_files:
            nested_file_list.append(wav_files)
            seen_ta_dirs.add(ta_dir.resolve())

    if nested_file_list or not recursive:
        return nested_file_list

    # Robust fallback for nested layouts such as <base>/<split>/<machine>/ta/*.wav.
    for ta_dir in sorted([p for p in base_path.rglob("ta") if p.is_dir()]):
        try:
            resolved_ta = ta_dir.resolve()
        except Exception:
            resolved_ta = ta_dir
        if resolved_ta in seen_ta_dirs:
            continue
        if not _path_matches_target_scope(ta_dir.parent, target_device, scope):
            continue
        wav_files = sorted([str(p.resolve()) for p in ta_dir.glob("*.wav")])
        if wav_files:
            nested_file_list.append(wav_files)
            seen_ta_dirs.add(resolved_ta)

    return nested_file_list


def find_class_aug_wav_files(
    base_dir: str,
    *,
    target_device: Optional[str] = None,
    scope: str = "all",
    recursive: bool = False,
) -> List[Tuple[str, List[str]]]:
    """Find ``<class>/aug/*.wav`` view-major augmentation lists.

    Current project datasets may not contain legacy ``ta`` folders and instead
    use this layout::

        <base>/<class>/train/*.wav
        <base>/<class>/test/*.wav
        <base>/<class>/aug/*.wav

    The returned object follows the same sampling semantics as
    ``find_ta_wav_files``: each class contributes one sorted list, and
    ``unsup_proxy_benchmark`` interprets it as view-major blocks with
    ``--unsup_aug_views`` views per sample.  The tuple key is used only for
    logging and pair-detail CSVs.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        logging.warning("Unsupervised augmentation base directory does not exist: %s", base_dir)
        return []

    nested_file_list: List[Tuple[str, List[str]]] = []
    seen_aug_dirs: set[Path] = set()

    # Expected one-level layout: <base>/<class>/aug/*.wav.
    for dir_path in _iter_machine_dirs(base_path, target_device, scope):
        aug_dir = dir_path / "aug"
        if not aug_dir.is_dir():
            continue
        wav_files = sorted([str(p.resolve()) for p in aug_dir.glob("*.wav")])
        if wav_files:
            nested_file_list.append((dir_path.name, wav_files))
            try:
                seen_aug_dirs.add(aug_dir.resolve())
            except Exception:
                seen_aug_dirs.add(aug_dir)

    if nested_file_list or not recursive:
        return nested_file_list

    # Fallback for nested variants while preserving the same parent-class rule.
    for aug_dir in sorted([p for p in base_path.rglob("aug") if p.is_dir()]):
        try:
            resolved_aug = aug_dir.resolve()
        except Exception:
            resolved_aug = aug_dir
        if resolved_aug in seen_aug_dirs:
            continue
        if not _path_matches_target_scope(aug_dir.parent, target_device, scope):
            continue
        wav_files = sorted([str(p.resolve()) for p in aug_dir.glob("*.wav")])
        if wav_files:
            try:
                group_key = str(aug_dir.parent.resolve().relative_to(base_path.resolve()))
            except Exception:
                group_key = aug_dir.parent.name
            nested_file_list.append((group_key, wav_files))
            seen_aug_dirs.add(resolved_aug)

    return nested_file_list


def find_stored_aug_wav_groups(
    base_dir: str,
    *,
    target_device: Optional[str] = None,
    scope: str = "all",
    recursive: bool = False,
) -> Dict[str, List[str]]:
    """Build positive-view groups from stored ``aug/*.wav`` files.

    Dataset layout:
        <base>/<machine>/train/*.wav
        <base>/<machine>/aug/<aug_prefix>_<original_train_filename>.wav

    The one-level layout is tried first.  If no groups are formed and
    recursive=True, nested ``*/aug`` folders are also inspected.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        logging.warning("Unsupervised augmentation base directory does not exist: %s", base_dir)
        return {}

    def collect_from_machine_dirs(machine_dirs: Sequence[Path]) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for machine_dir in machine_dirs:
            if not _path_matches_target_scope(machine_dir, target_device, scope):
                continue
            train_dir = machine_dir / "train"
            aug_dir = machine_dir / "aug"
            if not aug_dir.is_dir():
                continue
            train_names = [p.name for p in sorted(train_dir.glob("*.wav"))] if train_dir.is_dir() else []
            for aug_path in sorted(aug_dir.glob("*.wav")):
                orig_name = strip_stored_aug_prefix(aug_path.name, train_names)
                if orig_name is None:
                    continue
                key = f"{machine_dir.name}/{orig_name}"
                groups.setdefault(key, []).append(str(aug_path.resolve()))
        return {k: sorted(v) for k, v in groups.items() if len(v) >= 2}

    groups = collect_from_machine_dirs(_iter_machine_dirs(base_path, target_device, scope))
    if groups or not recursive:
        return groups

    nested_machine_dirs: List[Path] = []
    seen: set[Path] = set()
    for aug_dir in sorted([p for p in base_path.rglob("aug") if p.is_dir()]):
        machine_dir = aug_dir.parent
        try:
            resolved = machine_dir.resolve()
        except Exception:
            resolved = machine_dir
        if resolved in seen:
            continue
        seen.add(resolved)
        nested_machine_dirs.append(machine_dir)
    return collect_from_machine_dirs(nested_machine_dirs)


def resolve_unsup_aug_source(base_dir: str, args: argparse.Namespace, target_device: str) -> str:
    source = getattr(args, "unsup_aug_source", "auto")
    recursive_ta = bool(getattr(args, "unsup_ta_recursive", True))
    recursive_class_aug = bool(getattr(args, "unsup_class_aug_recursive", True))
    recursive_aug = bool(getattr(args, "unsup_stored_aug_recursive", True))
    if source != "auto":
        return source
    # Current layout first: <class>/aug/*.wav view-major blocks.
    if find_class_aug_wav_files(base_dir, target_device=target_device, scope=args.unsup_alignment_scope, recursive=recursive_class_aug):
        return "class_aug_wav"
    # Legacy unsup_eval.py layout.
    if find_ta_wav_files(base_dir, target_device=target_device, scope=args.unsup_alignment_scope, recursive=recursive_ta):
        return "ta_wav"
    # Fallback when augmented filenames can be grouped by original train filename.
    if find_stored_aug_wav_groups(base_dir, target_device=target_device, scope=args.unsup_alignment_scope, recursive=recursive_aug):
        return "stored_aug_wav"
    return "none"

def _write_unsup_proxy_metric_csv(
    out_dir: Path,
    model_id: str,
    target_device: str,
    metric_rows: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    metric_df = pd.DataFrame(list(metric_rows))
    proxy_dir = out_dir / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    metric_df.to_csv(proxy_dir / f"{model_id}_{target_device}_unsup_alignment_uniformity.csv", index=False)
    return metric_df


def _unsup_proxy_empty_result(
    *,
    model_id: str,
    target_device: str,
    out_dir: Path,
    aug_source: str,
    status: str,
    message: str,
    n_groups: int = 0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Return explicit NaN metrics instead of silently dropping proxy columns."""
    logging.warning("[%s/%s] unsup Alignment/Uniformity unavailable: %s", model_id, target_device, message)
    metrics: Dict[str, Any] = {
        "unsup_alignment": float("nan"),
        "unsup_uniformity": float("nan"),
        "unsup_proxy_pairs": 0.0,
        "unsup_proxy_groups": float(n_groups),
        "unsup_aug_source_used": aug_source,
        "unsup_proxy_status": status,
    }
    metric_rows = [
        {
            "row_type": "metric",
            "model_id": model_id,
            "target_device": target_device,
            "proxy_type": "unsup_alignment_uniformity",
            "metric_name": "alignment",
            "metric_score": float("nan"),
            "n_pairs": 0,
            "n_groups": int(n_groups),
            "aug_source": aug_source,
            "status": status,
            "message": message,
        },
        {
            "row_type": "metric",
            "model_id": model_id,
            "target_device": target_device,
            "proxy_type": "unsup_alignment_uniformity",
            "metric_name": "uniformity",
            "metric_score": float("nan"),
            "n_pairs": 0,
            "n_groups": int(n_groups),
            "aug_source": aug_source,
            "status": status,
            "message": message,
        },
    ]
    return metrics, _write_unsup_proxy_metric_csv(out_dir, model_id, target_device, metric_rows)


def unsup_proxy_benchmark(
    adapter: BaseAdapter,
    args: argparse.Namespace,
    model_id: str,
    target_device: str,
    out_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if not isinstance(adapter, UnsupAdapter):
        return {}, pd.DataFrame()
    if args.skip_unsup_alignment_uniformity:
        return {}, pd.DataFrame()

    aug_base_dir = get_unsup_aug_base_dir(args)
    recursive_ta = bool(getattr(args, "unsup_ta_recursive", True))
    recursive_class_aug = bool(getattr(args, "unsup_class_aug_recursive", True))
    recursive_aug = bool(getattr(args, "unsup_stored_aug_recursive", True))
    aug_source = resolve_unsup_aug_source(aug_base_dir, args, target_device)

    if aug_source == "ta_wav":
        aug_lists = find_ta_wav_files(
            aug_base_dir,
            target_device=target_device,
            scope=args.unsup_alignment_scope,
            recursive=recursive_ta,
        )
        if not aug_lists:
            return _unsup_proxy_empty_result(
                model_id=model_id,
                target_device=target_device,
                out_dir=out_dir,
                aug_source=aug_source,
                status="no_ta_wav_files",
                message=f"No ta/*.wav files were found under {aug_base_dir}",
            )
    elif aug_source == "class_aug_wav":
        aug_lists = find_class_aug_wav_files(
            aug_base_dir,
            target_device=target_device,
            scope=args.unsup_alignment_scope,
            recursive=recursive_class_aug,
        )
        if not aug_lists:
            return _unsup_proxy_empty_result(
                model_id=model_id,
                target_device=target_device,
                out_dir=out_dir,
                aug_source=aug_source,
                status="no_class_aug_wav_files",
                message=f"No <class>/aug/*.wav files were found under {aug_base_dir}",
            )
    elif aug_source == "stored_aug_wav":
        aug_groups = find_stored_aug_wav_groups(
            aug_base_dir,
            target_device=target_device,
            scope=args.unsup_alignment_scope,
            recursive=recursive_aug,
        )
        if not aug_groups:
            return _unsup_proxy_empty_result(
                model_id=model_id,
                target_device=target_device,
                out_dir=out_dir,
                aug_source=aug_source,
                status="no_aug_wav_groups",
                message=f"No aug/*.wav groups with at least two views were found under {aug_base_dir}",
            )
        aug_lists = list(aug_groups.items())
    else:
        return _unsup_proxy_empty_result(
            model_id=model_id,
            target_device=target_device,
            out_dir=out_dir,
            aug_source="none",
            status="no_augmentation_source",
            message=f"No supported <class>/aug/*.wav, ta/*.wav, or grouped aug/*.wav source was found under {aug_base_dir}",
        )

    all_z_i: List[torch.Tensor] = []
    all_z_j: List[torch.Tensor] = []
    all_z: List[torch.Tensor] = []
    pair_rows: List[Dict[str, Any]] = []
    n_views = int(args.unsup_aug_views)
    if n_views < 2:
        raise RuntimeError(f"--unsup_aug_views must be >= 2, got {n_views}")

    view_major_sources = {"ta_wav", "class_aug_wav"}
    iterable = list(enumerate(aug_lists))

    for group_idx, group_obj in iterable:
        if aug_source in view_major_sources:
            if aug_source == "class_aug_wav":
                group_key, ev_list = group_obj
            else:
                group_key = f"ta_group_{group_idx}"
                ev_list = group_obj
            ev_len = int(len(ev_list) / n_views)
            if ev_len <= 0:
                logging.warning(
                    "[%s/%s] skip %s group %d because len(group)=%d is smaller than --unsup_aug_views=%d",
                    model_id, target_device, aug_source, group_idx, len(ev_list), n_views,
                )
                continue
            if len(ev_list) % n_views != 0:
                logging.warning(
                    "[%s/%s] %s group %d has %d files, not divisible by --unsup_aug_views=%d; trailing files are ignored by floor division.",
                    model_id, target_device, aug_source, group_idx, len(ev_list), n_views,
                )
            pair_count = ev_len if args.unsup_alignment_pairs_per_group <= 0 else min(int(args.unsup_alignment_pairs_per_group), ev_len)
            for ev_idx in range(pair_count):
                # Same positive-pair sampling as unsup_eval.py:
                # choose two augmented views, then choose one sample index in that view-major block.
                aug_int = random.sample(range(0, n_views), 2)
                sample_int = random.randrange(ev_len)
                load0 = ev_list[int(aug_int[0] * ev_len + sample_int)]
                load1 = ev_list[int(aug_int[1] * ev_len + sample_int)]

                z_i = adapter.embedding(load0)
                z_j = adapter.embedding(load1)
                all_z_i.append(z_i)
                all_z_j.append(z_j)
                # Match unsup_eval.py uniformity behavior: use one side only.
                all_z.append(z_i)
                if args.unsup_save_pair_details:
                    z_i_norm = F.normalize(z_i, p=2, dim=1)
                    z_j_norm = F.normalize(z_j, p=2, dim=1)
                    pair_alignment = float(((z_i_norm - z_j_norm) ** 2).sum(dim=1).mean().detach().cpu().item())
                    pair_rows.append({
                        "model_id": model_id,
                        "target_device": target_device,
                        "aug_source": aug_source,
                        "group_key": group_key,
                        "group_idx": group_idx,
                        "ev_idx": ev_idx,
                        "view_i": int(aug_int[0]),
                        "view_j": int(aug_int[1]),
                        "sample_idx": int(sample_int),
                        "pair_alignment": pair_alignment,
                        "path_i": load0,
                        "path_j": load1,
                    })
        else:
            group_key, ev_list = group_obj
            ev_list = sorted(ev_list)
            if len(ev_list) < 2:
                continue
            if args.unsup_alignment_pairs_per_group <= 0:
                # Stored augmentation fallback: one positive pair per original sample.
                pair_count = 1
            else:
                pair_count = int(args.unsup_alignment_pairs_per_group)
            for ev_idx in range(pair_count):
                load0, load1 = random.sample(ev_list, 2)
                z_i = adapter.embedding(load0)
                z_j = adapter.embedding(load1)
                all_z_i.append(z_i)
                all_z_j.append(z_j)
                all_z.append(z_i)
                if args.unsup_save_pair_details:
                    z_i_norm = F.normalize(z_i, p=2, dim=1)
                    z_j_norm = F.normalize(z_j, p=2, dim=1)
                    pair_alignment = float(((z_i_norm - z_j_norm) ** 2).sum(dim=1).mean().detach().cpu().item())
                    pair_rows.append({
                        "model_id": model_id,
                        "target_device": target_device,
                        "aug_source": aug_source,
                        "group_key": group_key,
                        "group_idx": group_idx,
                        "ev_idx": ev_idx,
                        "view_i": -1,
                        "view_j": -1,
                        "sample_idx": -1,
                        "pair_alignment": pair_alignment,
                        "path_i": load0,
                        "path_j": load1,
                    })

    if not all_z_i:
        return _unsup_proxy_empty_result(
            model_id=model_id,
            target_device=target_device,
            out_dir=out_dir,
            aug_source=aug_source,
            status="no_valid_pairs",
            message=f"Augmentation source {aug_source} was found under {aug_base_dir}, but no valid positive pairs were formed.",
            n_groups=len(aug_lists),
        )

    z_i_total = torch.cat(all_z_i, dim=0)
    z_j_total = torch.cat(all_z_j, dim=0)
    z_total = torch.cat(all_z, dim=0)

    z_i_total = F.normalize(z_i_total, p=2, dim=1)
    z_j_total = F.normalize(z_j_total, p=2, dim=1)
    z_total = F.normalize(z_total, p=2, dim=1)

    alignment = float(((z_i_total - z_j_total) ** 2).sum(dim=1).mean().detach().cpu().item())
    if z_total.shape[0] >= 2:
        sq_pdist = torch.pdist(z_total, p=2).pow(2)
        uniformity = float(sq_pdist.mul(-2).exp().mean().log().detach().cpu().item())
    else:
        uniformity = float("nan")

    metrics: Dict[str, Any] = {
        "unsup_alignment": alignment,
        "unsup_uniformity": uniformity,
        "unsup_proxy_pairs": float(z_i_total.shape[0]),
        "unsup_proxy_groups": float(len(aug_lists)),
        "unsup_aug_source_used": aug_source,
        "unsup_proxy_status": "ok" if np.isfinite(alignment) and np.isfinite(uniformity) else "partial",
    }

    metric_rows = [
        {
            "row_type": "metric",
            "model_id": model_id,
            "target_device": target_device,
            "proxy_type": "unsup_alignment_uniformity",
            "metric_name": "alignment",
            "metric_score": alignment,
            "n_pairs": int(z_i_total.shape[0]),
            "n_groups": int(len(aug_lists)),
            "aug_source": aug_source,
            "status": metrics["unsup_proxy_status"],
            "message": "",
        },
        {
            "row_type": "metric",
            "model_id": model_id,
            "target_device": target_device,
            "proxy_type": "unsup_alignment_uniformity",
            "metric_name": "uniformity",
            "metric_score": uniformity,
            "n_pairs": int(z_i_total.shape[0]),
            "n_groups": int(len(aug_lists)),
            "aug_source": aug_source,
            "status": metrics["unsup_proxy_status"],
            "message": "",
        },
    ]

    metric_df = _write_unsup_proxy_metric_csv(out_dir, model_id, target_device, metric_rows)
    if pair_rows:
        proxy_dir = out_dir / "proxy"
        proxy_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pair_rows).to_csv(proxy_dir / f"{model_id}_{target_device}_unsup_alignment_uniformity_pairs.csv", index=False)

    logging.info(
        "[%s/%s] unsup Alignment/Uniformity: source=%s groups=%d pairs=%d alignment=%.6f uniformity=%.6f",
        model_id,
        target_device,
        aug_source,
        len(aug_lists),
        int(z_i_total.shape[0]),
        alignment,
        uniformity,
    )

    return metrics, metric_df

def summarize_ae_proxy(train_proxy_df: pd.DataFrame, test_per_file_df: pd.DataFrame) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for split_name, df in [("train", train_proxy_df), ("test", test_per_file_df)]:
        if df is None or df.empty:
            continue
        for col in ["ae_l1", "ae_l2"]:
            if col in df.columns:
                metrics[f"{col}_{split_name}_mean"] = float(pd.to_numeric(df[col], errors="coerce").mean())
                if "domain" in df.columns:
                    for domain, sub in df.groupby("domain"):
                        metrics[f"{col}_{split_name}_{domain}_mean"] = float(pd.to_numeric(sub[col], errors="coerce").mean())
                if "condition" in df.columns:
                    for condition, sub in df.groupby("condition"):
                        metrics[f"{col}_{split_name}_{condition}_mean"] = float(pd.to_numeric(sub[col], errors="coerce").mean())
    return metrics



def proxy_summary_available(model_type: str, proxy_summary: Dict[str, Any]) -> bool:
    """Whether a proxy metric was actually computed, not merely attempted."""
    if not proxy_summary:
        return False
    if model_type == "unsup":
        return bool(
            np.isfinite(pd.to_numeric(proxy_summary.get("unsup_alignment"), errors="coerce"))
            and np.isfinite(pd.to_numeric(proxy_summary.get("unsup_uniformity"), errors="coerce"))
        )
    if model_type == "separation":
        return bool(np.isfinite(pd.to_numeric(proxy_summary.get("sep_si_sdr_mean"), errors="coerce")))
    if model_type == "classification":
        return any(
            np.isfinite(pd.to_numeric(value, errors="coerce"))
            for key, value in proxy_summary.items()
            if key.startswith("clf_")
        )
    if model_type == "ae":
        return any(
            np.isfinite(pd.to_numeric(value, errors="coerce"))
            for key, value in proxy_summary.items()
            if key.startswith("ae_")
        )
    return bool(proxy_summary)


# -----------------------------------------------------------------------------
# Main evaluation loop
# -----------------------------------------------------------------------------

def evaluate_model_on_device(
    model: torch.nn.Module,
    spec: ModelSpec,
    target_device: str,
    args: argparse.Namespace,
    torch_device: torch.device,
    run_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = get_data_splits(args.data_dir, target_device)
    if len(splits.train_all) == 0:
        raise RuntimeError(f"No train files for target_device={target_device}")
    if len(splits.test_all) == 0:
        raise RuntimeError(f"No test files for target_device={target_device}")

    model_device_dir = run_dir / "per_model" / spec.model_id / target_device
    model_device_dir.mkdir(parents=True, exist_ok=True)

    adapter = make_adapter(model, spec, args, torch_device, splits.class_list)
    train_records = make_records(splits.train_all, target_device, args.data_dir, splits.class_list)
    test_records = make_records(splits.test_all, target_device, args.data_dir, splits.class_list)
    other_test_records = make_records(splits.other_test, target_device, args.data_dir, splits.class_list)

    logging.info("[%s/%s] train=%d test=%d", spec.model_id, target_device, len(train_records), len(test_records))

    stats = compute_cov_stats(adapter, train_records, args)
    stats.train_proxy_df.insert(0, "model_id", spec.model_id)
    stats.train_proxy_df.insert(1, "model_name", spec.raw_name)
    stats.train_proxy_df.insert(2, "model_type", spec.model_type)
    stats.train_proxy_df.to_csv(model_device_dir / f"{spec.model_id}_{target_device}_train_proxy.csv", index=False)

    asd_summary, per_file_df, linear_df, projection_df = evaluate_asd(
        adapter,
        stats,
        test_records,
        args,
        spec.model_id,
        target_device,
        model_device_dir,
    )
    per_file_df.to_csv(model_device_dir / f"{spec.model_id}_{target_device}_per_file_scores.csv", index=False)
    if not linear_df.empty:
        linear_df.to_csv(model_device_dir / f"{spec.model_id}_{target_device}_linear_probe_scores.csv", index=False)

    proxy_summary: Dict[str, Any] = {}
    clf_proxy_df = pd.DataFrame()
    sep_proxy_df = pd.DataFrame()
    if not args.skip_proxy:
        if spec.model_type == "ae":
            proxy_summary.update(summarize_ae_proxy(stats.train_proxy_df, per_file_df))
        elif spec.model_type == "separation":
            sep_metrics, sep_proxy_df = separation_proxy_benchmark(adapter, splits, args, spec.model_id, target_device, model_device_dir)
            proxy_summary.update(sep_metrics)
        elif spec.model_type == "classification":
            clf_metrics, clf_proxy_df = classification_proxy_benchmark(
                adapter,
                test_records,
                other_test_records,
                args,
                spec.model_id,
                target_device,
                model_device_dir,
            )
            proxy_summary.update(clf_metrics)
        elif spec.model_type == "unsup":
            unsup_metrics, unsup_proxy_df = unsup_proxy_benchmark(
                adapter,
                args,
                spec.model_id,
                target_device,
                model_device_dir,
            )
            proxy_summary.update(unsup_metrics)
            clf_proxy_df = pd.concat([clf_proxy_df, unsup_proxy_df], axis=0, ignore_index=True)

    summary: Dict[str, Any] = {
        "model_id": spec.model_id,
        "model_name": spec.raw_name,
        "model_path": str(spec.path),
        "model_type": spec.model_type,
        "target_device": target_device,
        "arch": spec.arch,
        "comp_feat": spec.comp_feat,
        "lin_feat": spec.lin_feat,
        "channel_size": spec.channel_size,
        "cb": spec.cb,
        "batch_size_from_name": spec.batch_size_from_name,
        "epoch_from_name": spec.epoch_from_name,
        "loss_from_name": spec.loss_from_name,
        "unsup_mode": spec.unsup_mode,
        "n_train": len(train_records),
        "n_train_source": len(splits.train_source),
        "n_train_target": len(splits.train_target),
        "n_test_source": len(splits.test_source),
        "n_test_target": len(splits.test_target),
        "proxy_available": proxy_summary_available(spec.model_type, proxy_summary),
        **asd_summary,
        **proxy_summary,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(model_device_dir / f"{spec.model_id}_{target_device}_summary.csv", index=False)

    return summary, per_file_df, linear_df, projection_df, stats.train_proxy_df, pd.concat([clf_proxy_df, sep_proxy_df], axis=0, ignore_index=True)



def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logging.warning("Could not read existing CSV for resume: %s (%r)", path, exc)
        return pd.DataFrame()
    return df


def existing_result_paths(run_dir: Path, spec: ModelSpec, target_device: str) -> Dict[str, Path]:
    model_device_dir = run_dir / "per_model" / spec.model_id / target_device
    prefix = f"{spec.model_id}_{target_device}"
    return {
        "dir": model_device_dir,
        "summary": model_device_dir / f"{prefix}_summary.csv",
        "per_file": model_device_dir / f"{prefix}_per_file_scores.csv",
        "linear": model_device_dir / f"{prefix}_linear_probe_scores.csv",
        "train_proxy": model_device_dir / f"{prefix}_train_proxy.csv",
        "projection": model_device_dir / "projection" / f"{prefix}_projection_coordinates.csv",
    }


def required_proxy_columns_for_resume(spec: ModelSpec, args: argparse.Namespace) -> List[str]:
    """Columns that must exist in a completed summary when --resume is used.

    Older patched runs could write a summary before proxy metrics were supported
    or after a proxy metric was skipped.  Without this check, --resume treats the
    stale per-model summary as complete and never recomputes Alignment/Uniformity
    or classification F1.
    """
    if getattr(args, "skip_proxy", False):
        return []
    if not getattr(args, "require_proxy_metrics_on_resume", True):
        return []
    if spec.model_type == "unsup":
        return ["unsup_alignment", "unsup_uniformity"]
    if spec.model_type == "classification":
        return ["clf_target_total_micro_f1", "clf_all_micro_f1", "clf_all_macro_f1"]
    return []


def is_completed_result(run_dir: Path, spec: ModelSpec, target_device: str, args: argparse.Namespace) -> bool:
    """Return True only for a completed model-device evaluation.

    In addition to checking the per-model summary exists, this v3 patch verifies
    that proxy metrics expected for classification and unsupervised checkpoints
    are present.  This prevents --resume from skipping older results that lack
    SimCLR/SimSiam Alignment/Uniformity or CE/ArcFace F1 metrics.
    """
    summary_path = existing_result_paths(run_dir, spec, target_device)["summary"]
    if not summary_path.exists():
        return False
    try:
        summary_df = pd.read_csv(summary_path)
    except Exception:
        return False
    if summary_df.empty:
        return False

    required_cols = required_proxy_columns_for_resume(spec, args)
    if required_cols:
        missing = [c for c in required_cols if c not in summary_df.columns]
        if missing:
            logging.info(
                "[%s/%s] resume: completed summary exists but required proxy columns are missing: %s; recomputing.",
                spec.model_id, target_device, missing,
            )
            return False
        for col in required_cols:
            values = pd.to_numeric(summary_df[col], errors="coerce")
            if values.isna().all():
                # New unsup proxy code writes explicit NaN alignment/uniformity
                # with a non-empty status when augmentation files are genuinely
                # unavailable.  Treat that as a completed attempted evaluation
                # so --resume does not recompute the same missing-source case
                # forever.  Older summaries without this status still recompute.
                if spec.model_type == "unsup" and "unsup_proxy_status" in summary_df.columns:
                    status_values = summary_df["unsup_proxy_status"].astype(str).str.strip()
                    if status_values.replace({"nan": "", "None": ""}).ne("").any():
                        continue
                logging.info(
                    "[%s/%s] resume: completed summary exists but required proxy column %s is NaN; recomputing.",
                    spec.model_id, target_device, col,
                )
                return False

        # Also require the per-model proxy metric CSV.  This ensures global
        # proxy_detail_scores.csv can be rebuilt under --resume, not only
        # results_summary.csv.
        proxy_dir = existing_result_paths(run_dir, spec, target_device)["dir"] / "proxy"
        if spec.model_type == "unsup":
            required_files = list(proxy_dir.glob("*unsup_alignment_uniformity.csv"))
            if not required_files:
                logging.info(
                    "[%s/%s] resume: unsup proxy metrics are present in summary but proxy CSV is missing; recomputing.",
                    spec.model_id, target_device,
                )
                return False
        elif spec.model_type == "classification":
            required_files = list(proxy_dir.glob("*classification_proxy_metrics.csv"))
            if not required_files:
                logging.info(
                    "[%s/%s] resume: classification F1 metrics are present in summary but proxy metric CSV is missing; recomputing.",
                    spec.model_id, target_device,
                )
                return False

    return True


def load_existing_result(
    run_dir: Path,
    spec: ModelSpec,
    target_device: str,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a completed model-device result so global CSVs can be regenerated.

    This is used by --resume. It prevents completed evaluations from being
    recomputed while still rebuilding results_summary.csv, results_long.csv,
    per_file_scores.csv, linear_probe_scores.csv, projection_coordinates.csv,
    train_proxy_scores.csv, and proxy_detail_scores.csv from existing per-model
    files.
    """
    paths = existing_result_paths(run_dir, spec, target_device)
    summary_df = _read_csv_if_exists(paths["summary"])
    if summary_df.empty:
        raise RuntimeError(f"Cannot resume because completed summary is missing or empty: {paths['summary']}")
    summary = summary_df.iloc[0].to_dict()

    per_file_df = _read_csv_if_exists(paths["per_file"])
    linear_df = _read_csv_if_exists(paths["linear"])
    projection_df = _read_csv_if_exists(paths["projection"])
    train_proxy_df = _read_csv_if_exists(paths["train_proxy"])

    proxy_dir = paths["dir"] / "proxy"
    proxy_dfs: List[pd.DataFrame] = []
    if proxy_dir.exists():
        # Pair-detail CSVs are optional diagnostics and were not included in the
        # original global proxy_detail_scores.csv aggregation.
        for csv_path in sorted(proxy_dir.glob("*.csv")):
            if csv_path.name.endswith("_pairs.csv"):
                continue
            df = _read_csv_if_exists(csv_path)
            if not df.empty:
                proxy_dfs.append(df)
    proxy_df = pd.concat(proxy_dfs, axis=0, ignore_index=True, sort=False) if proxy_dfs else pd.DataFrame()

    return summary, per_file_df, linear_df, projection_df, train_proxy_df, proxy_df


def write_global_outputs(
    run_dir: Path,
    summary_rows: Sequence[Dict[str, Any]],
    per_file_dfs: Sequence[pd.DataFrame],
    linear_dfs: Sequence[pd.DataFrame],
    projection_dfs: Sequence[pd.DataFrame],
    train_proxy_dfs: Sequence[pd.DataFrame],
    proxy_dfs: Sequence[pd.DataFrame],
    error_rows: Sequence[Dict[str, Any]],
) -> None:
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(run_dir / "results_summary.csv", index=False)
        id_cols = ["model_id", "model_name", "model_type", "target_device", "arch", "comp_feat", "lin_feat", "channel_size", "cb", "unsup_mode"]
        metric_cols = [c for c in summary_df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(summary_df[c])]
        long_rows: List[Dict[str, Any]] = []
        for _, row in summary_df.iterrows():
            for metric in metric_cols:
                long_rows.append({
                    "model_id": row.get("model_id"),
                    "model_name": row.get("model_name"),
                    "model_type": row.get("model_type"),
                    "target_device": row.get("target_device"),
                    "metric_name": metric,
                    "metric_value": row.get(metric),
                })
        pd.DataFrame(long_rows).to_csv(run_dir / "results_long.csv", index=False)
    if per_file_dfs:
        non_empty = [df for df in per_file_dfs if not df.empty]
        if non_empty:
            pd.concat(non_empty, axis=0, ignore_index=True).to_csv(run_dir / "per_file_scores.csv", index=False)
    if linear_dfs:
        non_empty = [df for df in linear_dfs if not df.empty]
        if non_empty:
            pd.concat(non_empty, axis=0, ignore_index=True).to_csv(run_dir / "linear_probe_scores.csv", index=False)
    if projection_dfs:
        non_empty = [df for df in projection_dfs if not df.empty]
        if non_empty:
            pd.concat(non_empty, axis=0, ignore_index=True).to_csv(run_dir / "projection_coordinates.csv", index=False)
    if train_proxy_dfs:
        non_empty = [df for df in train_proxy_dfs if not df.empty]
        if non_empty:
            pd.concat(non_empty, axis=0, ignore_index=True).to_csv(run_dir / "train_proxy_scores.csv", index=False)
    if proxy_dfs:
        non_empty = [df for df in proxy_dfs if not df.empty]
        if non_empty:
            pd.concat(non_empty, axis=0, ignore_index=True).to_csv(run_dir / "proxy_detail_scores.csv", index=False)
    error_path = run_dir / "errors.csv"
    if error_rows:
        pd.DataFrame(error_rows).to_csv(error_path, index=False)
    elif error_path.exists():
        # Avoid carrying a stale errors.csv from an earlier failed run after a clean rerun.
        error_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, default="./asd_dataset", help="DCASE-style dataset root")
    parser.add_argument("--model_root", type=str, required=True, help=".pth file or folder containing .pth files")
    parser.add_argument("--model_glob", type=str, default="*.pth", help="pattern used by Path.rglob under model_root")
    parser.add_argument("--save_dir", type=str, default="./batch_eval_results", help="output directory")
    parser.add_argument("--devices", nargs="+", default=DEFAULT_DEVICES, help="target devices/classes to evaluate")
    parser.add_argument("--target_device", type=str, default=None, help="force all models to this target device")
    parser.add_argument("--model_type", choices=["auto", "ae", "separation", "classification", "unsup"], default="auto")
    parser.add_argument("--device", type=str, default="cuda", help="torch device string, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="global RNG seed; use -1 to leave RNG unseeded")
    parser.add_argument("--continue_on_error", action="store_true", help="continue remaining models when one model fails")
    parser.add_argument("--resume", action="store_true",
                        help="skip completed model-device evaluations and rebuild global CSVs from per_model outputs")
    parser.add_argument("--overwrite_completed", action="store_true",
                        help="with --resume, recompute completed model-device evaluations instead of skipping them")
    parser.add_argument("--progress_every", type=int, default=0, help="log feature extraction progress every N files; 0 disables")

    # loading / architecture defaults
    parser.add_argument("--strict_load", action="store_true", help="use strict state_dict loading")
    parser.add_argument("--allow_partial_load", action="store_true", help="drop shape-mismatched checkpoint tensors")
    parser.add_argument("--unsup_partial_load", action="store_true", default=True, help="allow partial load for unsup checkpoints")
    parser.add_argument("--no_unsup_partial_load", dest="unsup_partial_load", action="store_false",
                        help="disable default partial loading for unsupervised checkpoints")
    parser.add_argument("--num_class", type=int, default=-1, help="ResNet num_class; <=0 infers from checkpoint/class list")
    parser.add_argument("--resnet", type=str, default="resnet18", help="default ResNet architecture when filename has no arch token")
    parser.add_argument("--resnet_sample_rate", type=int, default=16000)
    parser.add_argument("--unsup_mode", choices=["auto", "simclr", "simsiam"], default="auto",
                        help="unsupervised checkpoint mode; auto infers from filename")

    # AE defaults
    parser.add_argument("--ae_n_frames", type=int, default=5)
    parser.add_argument("--ae_default_comp_feat", type=int, default=16)
    parser.add_argument("--ae_default_lin_feat", type=int, default=128)

    # Separation defaults
    parser.add_argument("--sep_default_channel", type=int, default=128)
    parser.add_argument("--sep_default_cb", type=int, default=4,
                        help="fallback CB count for separation checkpoints whose filename has no <n>cb token")
    parser.add_argument("--sep_segment_len", type=int, default=32000)
    parser.add_argument("--sep_n_segments", type=int, default=5)
    parser.add_argument("--sep_strict_load", action="store_true", default=True,
                        help="strictly load separation checkpoints so CB/class mismatches fail instead of silently evaluating a partially loaded model")
    parser.add_argument("--no_sep_strict_load", dest="sep_strict_load", action="store_false",
                        help="allow non-strict separation checkpoint loading; mismatches are still logged")
    parser.add_argument("--sep_use_noncb_for_0cb", action="store_true", help="use TSCNetnonCB for filenames containing 0cb instead of TSCNet_Cont(cb=0)")
    parser.add_argument("--sep_istft_fixed_length", action="store_true",
                        help="set istft length=sep_segment_len; default omits length like uploaded sep code")

    # ASD scoring
    parser.add_argument("--fix_domain_mu", action="store_true", help="use actual source/target means; default keeps legacy near-zero domain mean")
    parser.add_argument("--linear_epochs", type=int, default=200)
    parser.add_argument("--linear_batch_size", type=int, default=64)
    parser.add_argument("--linear_lr", type=float, default=1e-3)
    parser.add_argument("--linear_half_split", choices=["legacy", "per_section"], default="legacy",
                        help="legacy uses section_00 anomaly half length for every section; per_section splits each section separately")
    parser.add_argument("--save_roc", action="store_true", default=True)
    parser.add_argument("--no_save_roc", dest="save_roc", action="store_false")

    # Projection
    parser.add_argument("--skip_projection", action="store_true")
    parser.add_argument("--save_projection_plots", action="store_true", default=True)
    parser.add_argument("--no_projection_plots", dest="save_projection_plots", action="store_false")
    parser.add_argument("--skip_tsne", action="store_true")
    parser.add_argument("--skip_umap", action="store_true")
    parser.add_argument("--tsne_perplexity", type=int, default=30)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.25)

    # Proxy
    parser.add_argument("--skip_proxy", action="store_true")
    parser.add_argument("--require_proxy_metrics_on_resume", action="store_true", default=True,
                        help="with --resume, recompute classification/unsup results if expected proxy metrics are absent from an existing summary")
    parser.add_argument("--no_require_proxy_metrics_on_resume", dest="require_proxy_metrics_on_resume", action="store_false",
                        help="allow --resume to skip completed summaries even if classification/unsup proxy metrics are missing")
    parser.add_argument("--legacy_clf_condition_swap", action="store_true", default=True,
                        help="match uploaded clf_benchmark condition_normal/condition_anomaly naming bug")
    parser.add_argument("--no_legacy_clf_condition_swap", dest="legacy_clf_condition_swap", action="store_false",
                        help="use semantically corrected condition_normal/condition_anomaly metric names")
    parser.add_argument("--unsup_eval_classification_proxy", action="store_true",
                        help="deprecated for latest unsup path; kept for CLI compatibility")
    parser.add_argument("--skip_unsup_alignment_uniformity", action="store_true",
                        help="skip unsupervised Alignment/Uniformity proxy metrics")
    parser.add_argument("--unsup_aug_dir", type=str, default=None,
                        help="optional augmentation root for SimCLR/SimSiam Alignment/Uniformity; default uses --data_dir like unsup_eval.py")
    parser.add_argument("--unsup_ta_recursive", action="store_true", default=True,
                        help="when legacy <machine>/ta/*.wav is not found, also search nested */ta/*.wav folders")
    parser.add_argument("--no_unsup_ta_recursive", dest="unsup_ta_recursive", action="store_false",
                        help="disable nested */ta/*.wav fallback and use only unsup_eval.py's one-level discovery")
    parser.add_argument("--unsup_class_aug_recursive", action="store_true", default=True,
                        help="when <class>/aug/*.wav is not found directly, also search nested */aug/*.wav as view-major class augmentation lists")
    parser.add_argument("--no_unsup_class_aug_recursive", dest="unsup_class_aug_recursive", action="store_false",
                        help="disable nested class/aug/*.wav fallback")
    parser.add_argument("--unsup_stored_aug_recursive", action="store_true", default=True,
                        help="when <machine>/aug/*.wav suffix-grouped pairs are not found, also search nested */aug/*.wav folders")
    parser.add_argument("--no_unsup_stored_aug_recursive", dest="unsup_stored_aug_recursive", action="store_false",
                        help="disable nested */aug/*.wav suffix-grouped fallback")
    parser.add_argument("--unsup_aug_source", choices=["auto", "class_aug_wav", "ta_wav", "stored_aug_wav"], default="auto",
                        help="augmentation source for unsup Alignment/Uniformity. auto uses <class>/aug/*.wav first, then ta/*.wav, then filename-grouped stored aug/*.wav")
    parser.add_argument("--unsup_aug_views", type=int, default=33,
                        help="number of augmented views per sample in view-major ta_wav or class_aug_wav layouts")
    parser.add_argument("--unsup_alignment_scope", choices=["all", "target"], default="all",
                        help="all uses all machine classes; target uses only the current target device/class")
    parser.add_argument("--unsup_alignment_pairs_per_group", type=int, default=-1,
                        help="pairs sampled per group; for ta_wav/class_aug_wav <=0 uses ev_len like legacy evaluator, for stored_aug_wav <=0 uses one pair per original sample")
    parser.add_argument("--unsup_save_pair_details", action="store_true",
                        help="save sampled augmentation-pair details for Alignment/Uniformity evaluation")
    parser.add_argument("--sep_snr", nargs="+", type=float, default=[-5.0, 0.0, 5.0])
    parser.add_argument("--sep_proxy_k", type=int, default=100)
    parser.add_argument("--sep_proxy_no_replace", action="store_true", help="sample separation proxy targets without replacement")
    parser.add_argument("--sep_proxy_sets", nargs="+", default=["train_data", "test_target", "test_source", "test_normal", "test_anomaly"],
                        choices=["train_data", "test_target", "test_source", "test_normal", "test_anomaly"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed >= 0:
        set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable. Falling back to CPU.")
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(args.device)

    run_dir = Path(args.save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    specs = build_model_specs(args)
    if not specs:
        raise RuntimeError(f"No model specs found under {args.model_root} with glob={args.model_glob}")

    logging.info("Found %d model(s) to evaluate.", len(specs))

    summary_rows: List[Dict[str, Any]] = []
    per_file_dfs: List[pd.DataFrame] = []
    linear_dfs: List[pd.DataFrame] = []
    projection_dfs: List[pd.DataFrame] = []
    train_proxy_dfs: List[pd.DataFrame] = []
    proxy_dfs: List[pd.DataFrame] = []
    error_rows: List[Dict[str, Any]] = []

    for spec_idx, spec in enumerate(specs, start=1):
        logging.info("[%d/%d] Model: %s (%s)", spec_idx, len(specs), spec.path, spec.model_type)

        pending_devices: List[str] = []
        for target_device in spec.eval_devices:
            if args.resume and not args.overwrite_completed and is_completed_result(run_dir, spec, target_device, args):
                try:
                    result = load_existing_result(run_dir, spec, target_device)
                    summary, per_file_df, linear_df, projection_df, train_proxy_df, proxy_df = result
                    summary_rows.append(summary)
                    per_file_dfs.append(per_file_df)
                    linear_dfs.append(linear_df)
                    projection_dfs.append(projection_df)
                    train_proxy_dfs.append(train_proxy_df)
                    proxy_dfs.append(proxy_df)
                    logging.info("[%s/%s] resume: completed result found; skipping evaluation.", spec.model_id, target_device)
                    write_global_outputs(run_dir, summary_rows, per_file_dfs, linear_dfs, projection_dfs, train_proxy_dfs, proxy_dfs, error_rows)
                except Exception as exc:
                    logging.warning("[%s/%s] resume load failed; recomputing. Error: %r", spec.model_id, target_device, exc)
                    pending_devices.append(target_device)
            else:
                pending_devices.append(target_device)

        if not pending_devices:
            logging.info("[%s] all requested devices were already completed; model load skipped.", spec.model_id)
            continue

        try:
            # class_list는 checkpoint num_class infer fallback에만 필요하므로 첫 pending eval device 기준으로 조회한다.
            logging.info("[%d/%d] Loading model: %s (%s)", spec_idx, len(specs), spec.path, spec.model_type)
            first_splits = get_data_splits(args.data_dir, pending_devices[0])
            model = load_model_for_spec(spec, args, torch_device, first_splits.class_list)

            for target_device in pending_devices:
                try:
                    result = evaluate_model_on_device(model, spec, target_device, args, torch_device, run_dir)
                    summary, per_file_df, linear_df, projection_df, train_proxy_df, proxy_df = result
                    summary_rows.append(summary)
                    per_file_dfs.append(per_file_df)
                    linear_dfs.append(linear_df)
                    projection_dfs.append(projection_df)
                    train_proxy_dfs.append(train_proxy_df)
                    proxy_dfs.append(proxy_df)
                    write_global_outputs(run_dir, summary_rows, per_file_dfs, linear_dfs, projection_dfs, train_proxy_dfs, proxy_dfs, error_rows)
                except Exception as exc:
                    logging.exception("Failed evaluating model %s on target_device=%s", spec.path, target_device)
                    error_rows.append({
                        "model_path": str(spec.path),
                        "model_id": spec.model_id,
                        "model_type": spec.model_type,
                        "target_device": target_device,
                        "error": repr(exc),
                    })
                    write_global_outputs(run_dir, summary_rows, per_file_dfs, linear_dfs, projection_dfs, train_proxy_dfs, proxy_dfs, error_rows)
                    if not args.continue_on_error:
                        raise

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            # Model loading failures are model-level; evaluation failures inside the
            # device loop above are already recorded per target_device.
            logging.exception("Failed evaluating model %s", spec.path)
            if not any(row.get("model_path") == str(spec.path) and row.get("error") == repr(exc) for row in error_rows):
                error_rows.append({
                    "model_path": str(spec.path),
                    "model_id": spec.model_id,
                    "model_type": spec.model_type,
                    "target_device": ";".join(pending_devices),
                    "error": repr(exc),
                })
            write_global_outputs(run_dir, summary_rows, per_file_dfs, linear_dfs, projection_dfs, train_proxy_dfs, proxy_dfs, error_rows)
            if not args.continue_on_error:
                raise

    write_global_outputs(run_dir, summary_rows, per_file_dfs, linear_dfs, projection_dfs, train_proxy_dfs, proxy_dfs, error_rows)
    logging.info("Done. Summary: %s", run_dir / "results_summary.csv")


if __name__ == "__main__":
    main()

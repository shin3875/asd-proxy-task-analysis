#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared-backbone EfficientNet-Lite ASD evaluator (batched fast path).

This script is a shared-backbone adaptation of the existing folder-level proxy
ASD evaluator. The ASD scoring path is intentionally kept close to the existing
code: Mahalanobis AUC/pAUC, linear-probe AUC/pAUC, and optional projection
outputs. The task-specific legacy adapters are replaced by one unified adapter
that evaluates the shared embedding z = model(x, task="encode")["z"].

Expected checkpoint layout produced by train_shared_backbone.py:

    <shared_root>/<task>/<target>/<backbone>/*.pth
    <shared_root>/arcface/__all__/<backbone>/margin_<m>/*.pth

Examples:
    ./saved_proxy_lite/ae/bearing/tf_efficientnet_lite0.in1k/*.pth
    ./saved_proxy_lite/simclr/fan/tf_efficientnet_lite2.in1k/*.pth
    ./saved_proxy_lite/arcface/__all__/tf_efficientnet_lite3.in1k/margin_0.5/*.pth
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE

try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None


def get_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# -----------------------------------------------------------------------------
# Project imports with local fallbacks
# -----------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

try:
    from shared_backbone.proxy_shared_backbone_lite import SharedBackboneLiteProxyNet  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - fallback for colocated file
    if exc.name not in {"shared_backbone", "shared_backbone.proxy_shared_backbone_lite"}:
        raise
    from proxy_shared_backbone_lite import SharedBackboneLiteProxyNet  # type: ignore

try:
    from shared_backbone.proxy_audio_training_utils import (  # type: ignore
        canonical_machine_name,
        coerce_logmel_matrix,
        crop_or_pad_time,
        make_feature_domain_mixture,
        make_local_frame_targets,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - fallback for colocated file
    if exc.name not in {"shared_backbone", "shared_backbone.proxy_audio_training_utils"}:
        raise
    from proxy_audio_training_utils import (  # type: ignore
        canonical_machine_name,
        coerce_logmel_matrix,
        crop_or_pad_time,
        make_feature_domain_mixture,
        make_local_frame_targets,
    )

TASKS = ["ae", "sep_direct", "ce", "arcface", "simclr", "simsiam"]
UNSUP_TASKS = {"simclr", "simsiam"}
CLASSIFICATION_TASKS = {"ce", "arcface"}

# Path metadata and deterministic positive-pair plans are independent of model
# weights. Cache them for the lifetime of one evaluator process.
_NON_AUG_FILE_CACHE: Dict[Tuple[str, str], List[str]] = {}
_DATA_SPLIT_CACHE: Dict[Tuple[str, str, str], "DataSplits"] = {}


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class SharedModelSpec:
    path: Path
    task: str
    model_id: str
    target_device: Optional[str]
    eval_devices: List[str]
    backbone_name: str
    phase: str
    margin: Optional[float] = None
    feature_index: Optional[int] = None
    segment_frames_from_name: Optional[int] = None
    frame_stack: Optional[int] = None
    batch_size_from_name: Optional[int] = None
    epoch_from_name: Optional[int] = None
    loss_from_name: Optional[float] = None
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
    data_ext: str

    @property
    def train_all(self) -> List[str]:
        return list(self.train_target) + list(self.train_source)

    @property
    def test_all(self) -> List[str]:
        return list(self.test_target) + list(self.test_source)


@dataclass
class FeatureOutput:
    cov_features: torch.Tensor
    file_feature: torch.Tensor
    mah_features: torch.Tensor
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


@dataclass
class LoadedCheckpoint:
    state_dict: Dict[str, torch.Tensor]
    payload: Dict[str, Any]
    arcface_state_dict: Optional[Dict[str, torch.Tensor]] = None


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.\-]+", "_", text)
    return text.strip("_") or "unnamed"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def section_sort_key(section: str) -> Tuple[int, str]:
    m = re.search(r"section_(\d+)", str(section))
    if m:
        return int(m.group(1)), str(section)
    return 10 ** 9, str(section)


def parse_section_from_name(path_or_name: str) -> str:
    base = os.path.basename(str(path_or_name))
    m = re.search(r"section_(\d+)", base)
    if m:
        return f"section_{m.group(1).zfill(2)}"
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


def to_numpy_1d(value: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    return arr.reshape(-1)


def safe_auc(scores: Sequence[float] | np.ndarray | pd.Series, labels: Sequence[int] | np.ndarray | pd.Series, *, max_fpr: Optional[float] = None) -> float:
    """AUC/pAUC with shape and finite-value guards.

    Some optional score columns, especially source/target-domain Mahalanobis
    variants, can contain NaN when a domain-specific covariance is unavailable.
    Filtering finite rows prevents a partially missing auxiliary score from
    invalidating the entire evaluation table.
    """
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
            return float(roc_auc_score(y_true=labels_arr, y_score=scores_arr))
        return float(roc_auc_score(y_true=labels_arr, y_score=scores_arr, max_fpr=max_fpr))
    except Exception:
        return float("nan")


def save_roc(scores: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray, path: Path, title: str) -> Dict[str, float]:
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
    fpr, tpr, _ = roc_curve(y_true=labels_arr, y_score=scores_arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = get_pyplot()
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


def extract_class_label(path: str, data_dir: str, class_list: Sequence[str]) -> str:
    class_map = {c.lower(): c for c in class_list}
    for part in Path(path).parts:
        hit = class_map.get(part.lower())
        if hit is not None:
            return hit
    try:
        rel = Path(path).resolve().relative_to(Path(data_dir).resolve())
        if rel.parts:
            return rel.parts[0]
    except Exception:
        pass
    return "unknown"


def belongs_to_device(path: str, target_device: str) -> bool:
    target_canon = canonical_machine_name(target_device)
    for part in Path(path).parts:
        if canonical_machine_name(part) == target_canon:
            return True
    return False


def discover_devices(data_dir: str | Path) -> List[str]:
    root = Path(data_dir)
    return sorted([p.name for p in root.glob("*") if p.is_dir()])


# -----------------------------------------------------------------------------
# Checkpoint / model-spec discovery
# -----------------------------------------------------------------------------

def parse_int_after(pattern: str, name: str) -> Optional[int]:
    m = re.search(pattern, name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_float_after(pattern: str, name: str) -> Optional[float]:
    m = re.search(pattern, name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def infer_task_from_path(path: Path, model_root: Path, cli_task: str) -> str:
    if cli_task != "auto":
        return "sep_direct" if cli_task == "sep" else cli_task
    candidates = list(path.parts)
    for part in candidates:
        low = part.lower()
        if low in TASKS:
            return low
        if low == "sep":
            return "sep_direct"
    stem = path.stem.lower()
    for t in TASKS:
        if re.search(rf"(?:^|_){re.escape(t)}(?:_|$)", stem):
            return t
    if "sep" in stem:
        return "sep_direct"
    raise RuntimeError(f"Cannot infer task from path. Pass --task explicitly: {path}")


def parse_backbone_from_text(text: str) -> Optional[str]:
    # accepts tf_efficientnet_lite0, tf_efficientnet_lite0.in1k, sanitized names, etc.
    m = re.search(r"tf[_\-]efficientnet[_\-]lite([0-4])(?:\.in1k)?", text, flags=re.IGNORECASE)
    if not m:
        return None
    idx = m.group(1)
    if re.search(rf"tf[_\-]efficientnet[_\-]lite{idx}\.in1k", text, flags=re.IGNORECASE):
        return f"tf_efficientnet_lite{idx}.in1k"
    return f"tf_efficientnet_lite{idx}.in1k"


def infer_phase(task: str, path: Path, payload: Optional[Dict[str, Any]] = None) -> str:
    if payload and isinstance(payload.get("phase"), str):
        return str(payload["phase"])
    stem = path.stem.lower()
    if task == "arcface":
        if re.search(r"(?:^|_)linear(?:_|$)", stem):
            return "linear"
        if re.search(r"(?:^|_)arcface(?:_|$)", stem):
            return "arcface"
    return task


def infer_target_and_backbone(path: Path, model_root: Path, task: str, devices: Sequence[str]) -> Tuple[Optional[str], str]:
    try:
        rel_parts = path.resolve().relative_to(model_root.resolve()).parts
    except Exception:
        rel_parts = path.parts

    # Case A: model_root is ./shared and relative path starts with task.
    if len(rel_parts) >= 4 and rel_parts[0].lower() in TASKS + ["sep"]:
        target = rel_parts[1]
        backbone = rel_parts[2]
        if task == "arcface" and len(rel_parts) >= 5 and rel_parts[3].startswith("margin_"):
            backbone = rel_parts[2]
        return target, parse_backbone_from_text(backbone) or backbone

    # Case B: model_root is ./shared/<task>; relative path starts with target.
    if len(rel_parts) >= 3:
        target = rel_parts[0]
        backbone = rel_parts[1]
        if target == "arcface" and len(rel_parts) >= 4:
            # Defensive fallback for accidental ./shared root.
            target = rel_parts[1]
            backbone = rel_parts[2]
        return target, parse_backbone_from_text(backbone) or backbone

    # Fallback: infer target from any path component, backbone from full path/stem.
    text = str(path)
    backbone = parse_backbone_from_text(text) or "tf_efficientnet_lite0.in1k"
    target = None
    for dev in sorted(devices, key=len, reverse=True):
        if belongs_to_device(str(path), dev):
            target = dev
            break
    if "__all__" in path.parts:
        target = "__all__"
    return target, backbone


def filter_devices_by_requested(devices: Sequence[str], requested: Optional[Sequence[str]]) -> List[str]:
    if requested is None:
        return list(devices)
    requested_set = {str(x) for x in requested}
    requested_canon = {canonical_machine_name(x) for x in requested}
    return [d for d in devices if d in requested_set or canonical_machine_name(d) in requested_canon]


def build_model_specs(args: argparse.Namespace, devices: Sequence[str]) -> List[SharedModelSpec]:
    model_root = Path(args.model_root)
    if not model_root.exists():
        raise FileNotFoundError(f"model_root does not exist: {model_root}")
    paths = sorted(model_root.rglob(args.model_glob))
    specs: List[SharedModelSpec] = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".pth":
            continue
        raw_name = path.stem
        if args.best_only and "best" not in raw_name.lower():
            continue
        task = infer_task_from_path(path, model_root, args.task)
        target_device, backbone_name = infer_target_and_backbone(path, model_root, task, devices)
        phase = infer_phase(task, path)

        if task == "arcface" and args.arcface_phase != "all" and phase != args.arcface_phase:
            continue

        margin = None
        for part in path.parts:
            if part.startswith("margin_"):
                margin = parse_float_after(r"margin_([\-+0-9.eE]+)", part)
        if margin is None:
            margin = parse_float_after(r"(?:^|_)m([0-9.]+)(?:_|$)", raw_name)

        feature_index = parse_int_after(r"fi(\d+)", raw_name)
        segment_frames = parse_int_after(r"seg(\d+)", raw_name)
        frame_stack = parse_int_after(r"fs(\d+)", raw_name)
        batch_size = parse_int_after(r"batch(\d+)", raw_name)
        epoch = parse_int_after(r"epoch(\d+)", raw_name)
        loss = parse_float_after(r"loss([\-+0-9.eE]+)", raw_name)

        if target_device == "__all__":
            eval_devices = list(devices)
        elif target_device is None:
            eval_devices = list(devices) if task in CLASSIFICATION_TASKS else []
        else:
            # Map canonical/sanitized target folder back to the actual data-dir name if possible.
            matched = None
            for dev in devices:
                if dev == target_device or canonical_machine_name(dev) == canonical_machine_name(target_device):
                    matched = dev
                    break
            eval_devices = [matched or target_device]

        eval_devices = filter_devices_by_requested(eval_devices, args.eval_devices)
        if not eval_devices:
            logging.warning("Skip checkpoint without eval devices: %s", path)
            continue

        specs.append(SharedModelSpec(
            path=path,
            task=task,
            model_id=safe_name(path.stem),
            target_device=target_device,
            eval_devices=eval_devices,
            backbone_name=backbone_name,
            phase=phase,
            margin=margin,
            feature_index=feature_index,
            segment_frames_from_name=segment_frames,
            frame_stack=frame_stack,
            batch_size_from_name=batch_size,
            epoch_from_name=epoch,
            loss_from_name=loss,
            raw_name=raw_name,
        ))
    return specs


# -----------------------------------------------------------------------------
# Feature loading and data splits
# -----------------------------------------------------------------------------

def _list_non_aug_files(data_root: Path, ext: str) -> List[str]:
    """List evaluation files while pruning ``aug`` directories.

    The legacy implementation recursively listed every file, including the very
    large stored-augmentation tree, and discarded those paths afterwards. This
    function returns the identical non-augmentation set without entering aug
    directories and caches it for all checkpoints in the same run.
    """
    root_key = str(data_root.resolve())
    key = (root_key, str(ext).lower())
    cached = _NON_AUG_FILE_CACHE.get(key)
    if cached is not None:
        return cached

    suffix = f".{str(ext).lower().lstrip('.')}"
    paths: List[str] = []
    t0 = time.perf_counter()
    for dirpath, dirnames, filenames in os.walk(root_key, topdown=True):
        # The old evaluator ultimately removed every path containing an aug
        # component, so pruning the directory is set-equivalent.
        dirnames[:] = [d for d in dirnames if d.lower() != "aug"]
        for filename in filenames:
            if filename.lower().endswith(suffix):
                paths.append(str(Path(dirpath) / filename))
    paths.sort()
    _NON_AUG_FILE_CACHE[key] = paths
    logging.info(
        "Indexed %d non-augmentation .%s files in %.2fs (cached for this run)",
        len(paths), ext, time.perf_counter() - t0,
    )
    return paths


def resolve_data_ext(args: argparse.Namespace) -> str:
    if args.data_ext != "auto":
        return args.data_ext
    root = Path(args.data_dir)
    if _list_non_aug_files(root, "npy"):
        return "npy"
    if _list_non_aug_files(root, "wav"):
        return "wav"
    raise FileNotFoundError(f"No non-augmentation .npy or .wav files found under {args.data_dir}")


def is_split_file(path: Path, split: str) -> bool:
    return any(part.lower() == split.lower() for part in path.parts)


def is_aug_file(path: Path) -> bool:
    return any(part.lower() == "aug" for part in path.parts)


def get_data_splits(data_dir: str, target_device: str, args: argparse.Namespace) -> DataSplits:
    data_root = Path(data_dir)
    ext = resolve_data_ext(args)
    cache_key = (str(data_root.resolve()), canonical_machine_name(target_device), ext)
    cached = _DATA_SPLIT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    class_list = discover_devices(data_root)
    all_files = _list_non_aug_files(data_root, ext)

    train_target: List[str] = []
    train_source: List[str] = []
    test_target: List[str] = []
    test_source: List[str] = []
    supplemental: List[str] = []
    other_train: List[str] = []
    other_test: List[str] = []

    for fp in all_files:
        p = Path(fp)
        # Defensive parity with the legacy implementation. The cached index has
        # already pruned these paths.
        if is_aug_file(p):
            continue
        is_target = belongs_to_device(fp, target_device)
        is_train = is_split_file(p, "train") and not is_split_file(p, "test")
        is_test = is_split_file(p, "test")
        is_supp = any(part.lower() == "supplemental" for part in p.parts)
        domain = infer_domain(fp)

        if is_target:
            if is_supp:
                supplemental.append(fp)
            if is_test:
                if domain == "source":
                    test_source.append(fp)
                elif domain == "target":
                    test_target.append(fp)
                else:
                    test_target.append(fp)
            elif is_train:
                if domain == "source":
                    train_source.append(fp)
                elif domain == "target":
                    train_target.append(fp)
                else:
                    train_target.append(fp)
        else:
            if is_train:
                other_train.append(fp)
            elif is_test:
                other_test.append(fp)

    result = DataSplits(
        train_target=sorted(train_target),
        train_source=sorted(train_source),
        test_target=sorted(test_target),
        test_source=sorted(test_source),
        supplemental=sorted(supplemental),
        other_train=sorted(other_train),
        other_test=sorted(other_test),
        class_list=class_list,
        data_ext=ext,
    )
    _DATA_SPLIT_CACHE[cache_key] = result
    return result

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


def center_crop_or_pad(matrix: np.ndarray, segment_frames: int) -> np.ndarray:
    return crop_or_pad_time(matrix, segment_frames=segment_frames, random_crop=False)


def load_feature_matrix(path: str, args: argparse.Namespace) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
        matrix = coerce_logmel_matrix(
            arr,
            npy_path=str(p),
            n_mels=args.n_mels,
            n_frame=args.frame_stack,
            matrix_log_mode=args.matrix_log_mode,
        )
    elif p.suffix.lower() == ".wav":
        if librosa is None:
            raise RuntimeError("librosa is required for wav evaluation but is not installed.")
        wav, sr = librosa.load(str(p), sr=None, mono=True)
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            power=2.0,
            fmin=0.0,
        )
        matrix = 10.0 * np.log10(np.maximum(mel, sys.float_info.epsilon)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported feature file: {path}")

    if int(args.eval_segment_frames) > 0:
        matrix = center_crop_or_pad(matrix, segment_frames=int(args.eval_segment_frames))
    return np.asarray(matrix, dtype=np.float32)


def matrix_to_tensor(matrix: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(matrix, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)


class _FeaturePathDataset(torch.utils.data.Dataset):
    """CPU-side feature loader used by the batched evaluation fast path.

    It applies exactly the same ``load_feature_matrix`` function as the original
    file-at-a-time evaluator. Only I/O and model invocation are parallelized.
    """

    def __init__(self, paths: Sequence[str], args: argparse.Namespace):
        self.paths = [str(x) for x in paths]
        self.args = args

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        matrix = load_feature_matrix(self.paths[index], self.args)
        # [F,T] -> [1,F,T], still on CPU.
        return int(index), torch.from_numpy(np.asarray(matrix, dtype=np.float32)).unsqueeze(0)


def _shape_group_collate(batch: Sequence[Tuple[int, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Group a loader batch by tensor shape without padding.

    Full-clip files may theoretically have different lengths. Padding would alter
    global average pooling and therefore the metric. Grouping by exact shape keeps
    the numerical evaluation definition unchanged.
    """
    grouped: Dict[Tuple[int, ...], List[Tuple[int, torch.Tensor]]] = defaultdict(list)
    for index, tensor in batch:
        grouped[tuple(tensor.shape)].append((int(index), tensor))
    output: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for items in grouped.values():
        indices = torch.tensor([x[0] for x in items], dtype=torch.long)
        tensors = torch.stack([x[1] for x in items], dim=0)
        output.append((indices, tensors))
    return output


def _make_feature_loader(paths: Sequence[str], args: argparse.Namespace, batch_size: int) -> torch.utils.data.DataLoader:
    workers = max(int(args.eval_num_workers), 0)
    kwargs: Dict[str, Any] = {
        "dataset": _FeaturePathDataset(paths, args),
        "batch_size": max(int(batch_size), 1),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": bool(args.eval_pin_memory),
        "collate_fn": _shape_group_collate,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(args.eval_persistent_workers)
        kwargs["prefetch_factor"] = max(int(args.eval_prefetch_factor), 1)
    return torch.utils.data.DataLoader(**kwargs)


# -----------------------------------------------------------------------------
# Checkpoint/model loading
# -----------------------------------------------------------------------------

class ArcMarginProduct(nn.Module):
    """ArcFace head compatible with train_shared_backbone.py."""

    def __init__(self, in_features: int, out_features: int, scale: float = 30.0, margin: float = 0.5, easy_margin: bool = False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.scale = float(scale)
        self.margin = float(margin)
        self.easy_margin = bool(easy_margin)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp_min(1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


def load_checkpoint(path: Path, device: torch.device) -> LoadedCheckpoint:
    try:
        obj = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(str(path), map_location=device)

    payload: Dict[str, Any] = obj if isinstance(obj, dict) else {}
    arcface_state = None
    if isinstance(payload.get("arcface_state_dict"), dict):
        arcface_state = payload["arcface_state_dict"]

    state_obj = obj
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                state_obj = obj[key]
                break

    if not isinstance(state_obj, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")

    state_dict: Dict[str, torch.Tensor] = {}
    for k, v in state_obj.items():
        if torch.is_tensor(v):
            clean_k = k[7:] if k.startswith("module.") else k
            state_dict[clean_k] = v
    if not state_dict:
        raise RuntimeError(f"No tensor state_dict entries found: {path}")
    return LoadedCheckpoint(state_dict=state_dict, payload=payload, arcface_state_dict=arcface_state)


def infer_num_classes(state_dict: Dict[str, torch.Tensor], payload: Dict[str, Any], class_list: Sequence[str]) -> int:
    if isinstance(payload.get("class_to_idx"), dict) and payload["class_to_idx"]:
        return len(payload["class_to_idx"])
    for key in ["clf_head.weight", "module.clf_head.weight"]:
        if key in state_dict and state_dict[key].ndim == 2:
            return int(state_dict[key].shape[0])
    return max(len(class_list), 1)


def infer_model_hparams(spec: SharedModelSpec, ckpt: LoadedCheckpoint, args: argparse.Namespace, class_list: Sequence[str]) -> Dict[str, Any]:
    sd = ckpt.state_dict
    payload_args = ckpt.payload.get("args") if isinstance(ckpt.payload.get("args"), dict) else {}

    def arg_or_payload(name: str, default: Any) -> Any:
        if getattr(args, f"override_{name}", None) is not None:
            return getattr(args, f"override_{name}")
        if args.use_checkpoint_args and name in payload_args:
            return payload_args[name]
        return default

    # Prefer explicit CLI override > full-checkpoint args > filename-derived value > CLI default.
    # This avoids accidental mismatch when a checkpoint was saved with full metadata.
    frame_stack = int(arg_or_payload("frame_stack", spec.frame_stack or args.frame_stack))
    n_mels = int(arg_or_payload("n_mels", args.n_mels))

    # Infer n_mels from token-head output dimension if possible.
    for key in ["ae_head.net.2.weight", "sep_direct_head.net.2.weight"]:
        if key in sd and sd[key].ndim == 2:
            out_dim = int(sd[key].shape[0])
            if frame_stack > 0 and out_dim % frame_stack == 0:
                n_mels = out_dim // frame_stack
                break

    h = {
        "backbone_name": ckpt.payload.get("backbone_name") or spec.backbone_name,
        "pretrained": False,
        "in_chans": 1,
        "n_freq": n_mels,
        "frame_stack": frame_stack,
        "num_classes": infer_num_classes(sd, ckpt.payload, class_list),
        "feature_index": int(arg_or_payload("feature_index", spec.feature_index or args.feature_index)),
        "token_time_mode": str(arg_or_payload("token_time_mode", args.token_time_mode)),
        "token_hidden_dim": int(arg_or_payload("token_hidden_dim", args.token_hidden_dim)),
        "projection_hidden_dim": int(arg_or_payload("projection_hidden_dim", args.projection_hidden_dim)),
        "projection_dim": int(arg_or_payload("projection_dim", args.projection_dim)),
        "simsiam_pred_hidden_dim": int(arg_or_payload("simsiam_pred_hidden_dim", args.simsiam_pred_hidden_dim)),
        "normalize_projection": str(arg_or_payload("normalize_projection", args.normalize_projection)).lower() in {"true", "1", "yes", "y"} if isinstance(arg_or_payload("normalize_projection", args.normalize_projection), str) else bool(arg_or_payload("normalize_projection", args.normalize_projection)),
    }

    # State-dict shape overrides keep evaluation robust when checkpoint args are unavailable.
    if "ae_head.net.0.weight" in sd:
        h["token_hidden_dim"] = int(sd["ae_head.net.0.weight"].shape[0])
    elif "sep_direct_head.net.0.weight" in sd:
        h["token_hidden_dim"] = int(sd["sep_direct_head.net.0.weight"].shape[0])
    if "projection_head.net.0.weight" in sd:
        h["projection_hidden_dim"] = int(sd["projection_head.net.0.weight"].shape[0])
    if "projection_head.net.2.weight" in sd:
        h["projection_dim"] = int(sd["projection_head.net.2.weight"].shape[0])
    if "simsiam_predictor.net.0.weight" in sd:
        h["simsiam_pred_hidden_dim"] = int(sd["simsiam_predictor.net.0.weight"].shape[0])
    return h


def load_state_dict_compat(model: nn.Module, state_dict: Dict[str, torch.Tensor], *, strict: bool, allow_partial_load: bool) -> None:
    if not allow_partial_load:
        model.load_state_dict(state_dict, strict=strict)
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
        logging.warning("Partial load skipped %d tensors. First skipped keys: %s", len(skipped), skipped[:10])
    if missing:
        logging.warning("Partial load missing keys: %s", list(missing)[:10])
    if unexpected:
        logging.warning("Partial load unexpected keys: %s", list(unexpected)[:10])


def build_and_load_model(spec: SharedModelSpec, args: argparse.Namespace, device: torch.device, class_list: Sequence[str]) -> Tuple[SharedBackboneLiteProxyNet, Optional[ArcMarginProduct], LoadedCheckpoint]:
    # Load tensors on CPU first. Mapping the checkpoint directly to CUDA and then
    # loading it into a CPU model causes an unnecessary GPU->CPU->GPU round trip.
    # This changes neither parameters nor evaluation logic.
    ckpt = load_checkpoint(spec.path, torch.device("cpu"))
    hparams = infer_model_hparams(spec, ckpt, args, class_list)
    model = SharedBackboneLiteProxyNet(**hparams)
    load_state_dict_compat(model, ckpt.state_dict, strict=args.strict_load, allow_partial_load=args.allow_partial_load)
    model.to(device)
    model.eval()

    arcface_head: Optional[ArcMarginProduct] = None
    if ckpt.arcface_state_dict is not None:
        out_features = infer_num_classes(ckpt.state_dict, ckpt.payload, class_list)
        in_features = int(getattr(model, "feat_dim"))
        scale = float(ckpt.payload.get("arcface_scale", args.arcface_scale))
        margin = float(ckpt.payload.get("margin", spec.margin if spec.margin is not None else args.arcface_margin))
        arcface_head = ArcMarginProduct(in_features=in_features, out_features=out_features, scale=scale, margin=margin).to(device)
        load_state_dict_compat(arcface_head, ckpt.arcface_state_dict, strict=False, allow_partial_load=True)
        arcface_head.eval()

    return model, arcface_head, ckpt


# -----------------------------------------------------------------------------
# Unified shared-backbone adapter
# -----------------------------------------------------------------------------

class SharedLiteAdapter:
    has_domain_mahalanobis = True

    def __init__(
        self,
        model: SharedBackboneLiteProxyNet,
        spec: SharedModelSpec,
        args: argparse.Namespace,
        device: torch.device,
        class_list: Sequence[str],
        arcface_head: Optional[ArcMarginProduct] = None,
        checkpoint_class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.model = model
        self.spec = spec
        self.args = args
        self.device = device
        self.class_list = list(class_list)
        self.class_to_idx = {canonical_machine_name(c): i for i, c in enumerate(self.class_list)}
        self.idx_to_class = {i: c for i, c in enumerate(self.class_list)}
        if checkpoint_class_to_idx:
            # Training stores canonical class names. Map them back to actual eval directory names when possible.
            canonical_to_actual = {canonical_machine_name(c): c for c in self.class_list}
            self.idx_to_class = {}
            for name, idx in checkpoint_class_to_idx.items():
                canon = canonical_machine_name(name)
                self.idx_to_class[int(idx)] = canonical_to_actual.get(canon, str(name))
        self.arcface_head = arcface_head
        self.model.eval()
        if self.arcface_head is not None:
            self.arcface_head.eval()

    @property
    def supports_classification_proxy(self) -> bool:
        if self.spec.task == "ce":
            return True
        if self.spec.task == "arcface" and self.spec.phase == "linear":
            return True
        if self.spec.task == "arcface" and self.arcface_head is not None:
            return True
        return False

    @property
    def classifier_out_features(self) -> int:
        if self.spec.task == "arcface" and self.spec.phase == "arcface" and self.arcface_head is not None:
            return int(self.arcface_head.out_features)
        return int(getattr(self.model.clf_head, "out_features", 0))

    @property
    def classification_label_mode(self) -> str:
        # Training supports machine-label and domain-label classification.
        # With state_dict-only checkpoints, label_mode is not stored, so infer from
        # classifier dimensionality. In this project machine classification is 9-way,
        # while domain classification is source/target 2-way.
        out_features = self.classifier_out_features
        if out_features == len(self.class_list):
            return "machine"
        if out_features == 2:
            return "domain"
        return "unknown"

    def feature_tensor(self, file_path: str) -> torch.Tensor:
        matrix = load_feature_matrix(file_path, self.args)
        return matrix_to_tensor(matrix, self.device)

    @torch.no_grad()
    def embedding(self, file_path: str) -> torch.Tensor:
        x = self.feature_tensor(file_path)
        out = self.model(x, task="encode")
        z = out["z"]
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return z.detach()

    @torch.inference_mode()
    def embeddings(self, paths: Sequence[str], *, batch_size: Optional[int] = None) -> torch.Tensor:
        """Extract embeddings in batches while preserving input order exactly.

        No padding, resampling, mixed precision, or model change is introduced.
        Files with different time lengths are separated into exact-shape groups
        inside each loader batch.
        """
        paths = [str(x) for x in paths]
        if not paths:
            return torch.empty((0, int(getattr(self.model, "feat_dim", 0))), device=self.device)
        bs = int(batch_size or self.args.eval_batch_size)
        started = time.perf_counter()
        logging.info(
            "[%s] batched embedding files=%d batch=%d workers=%d",
            self.spec.model_id, len(paths), bs, int(self.args.eval_num_workers),
        )
        loader = _make_feature_loader(paths, self.args, batch_size=bs)
        z_chunks: List[torch.Tensor] = []
        index_chunks: List[torch.Tensor] = []
        non_blocking = bool(self.args.eval_pin_memory and self.device.type == "cuda")
        for shape_groups in loader:
            for indices, x_cpu in shape_groups:
                x = x_cpu.to(self.device, non_blocking=non_blocking)
                z = self.model(x, task="encode")["z"]
                if z.dim() == 1:
                    z = z.unsqueeze(0)
                z_chunks.append(z.detach())
                index_chunks.append(indices)
        z_cat = torch.cat(z_chunks, dim=0)
        idx_cat = torch.cat(index_chunks, dim=0).to(self.device, non_blocking=non_blocking)
        order = torch.argsort(idx_cat)
        result = z_cat.index_select(0, order)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        logging.info(
            "[%s] batched embedding completed files=%d in %.2fs",
            self.spec.model_id, len(paths), time.perf_counter() - started,
        )
        return result

    @torch.no_grad()
    def ae_proxy(self, file_path: str) -> Dict[str, float]:
        x = self.feature_tensor(file_path)
        pred = self.model(x, task="ae")["ae_pred"]
        target = make_local_frame_targets(x, token_steps=pred.shape[1], frame_stack=self.model.frame_stack)
        return {
            "ae_l1": float(F.l1_loss(pred, target).detach().cpu().item()),
            "ae_l2": float(F.mse_loss(pred, target).detach().cpu().item()),
        }

    @torch.no_grad()
    def extract(self, file_path: str) -> FeatureOutput:
        z = self.embedding(file_path)
        proxy: Dict[str, Any] = {}
        if self.spec.task == "ae" and not self.args.skip_reconstruction_proxy:
            try:
                proxy.update(self.ae_proxy(file_path))
            except Exception as e:
                logging.warning("AE proxy metric failed for %s: %s", file_path, e)
        if self.supports_classification_proxy:
            proxy["has_logits"] = True
        return FeatureOutput(cov_features=z, file_feature=z, mah_features=z, proxy=proxy)

    @torch.no_grad()
    def classification_logits(self, file_path: str) -> Optional[torch.Tensor]:
        if not self.supports_classification_proxy:
            return None
        x = self.feature_tensor(file_path)
        if self.spec.task == "arcface" and self.spec.phase == "arcface" and self.arcface_head is not None:
            feat = self.model(x, task="encode")["z"]
            logits = self.arcface_head(feat, labels=None)
        else:
            logits = self.model(x, task="ce")["logits"]
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return logits.detach()


# -----------------------------------------------------------------------------
# Covariance and ASD scoring
# -----------------------------------------------------------------------------

def covariance_inverse(features: torch.Tensor, *, mu_override: Optional[torch.Tensor] = None, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    if features.shape[0] < 2:
        raise RuntimeError(f"Need at least 2 rows for covariance, got {tuple(features.shape)}")
    mu = features.mean(dim=0) if mu_override is None else mu_override
    dev = features - mu
    cov = torch.einsum("bi,bj->ij", dev, dev) / max(features.shape[0] - 1, 1)
    eye = torch.eye(cov.shape[0], device=features.device, dtype=features.dtype)
    cov = cov + eye * eps
    try:
        inv = torch.linalg.inv(cov)
    except Exception:
        inv = torch.linalg.pinv(cov)
    return mu.detach(), inv.detach()


def covariance_inverse_with_separate_score_mu(features: torch.Tensor, score_mu: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    _, inv = covariance_inverse(features, eps=eps)
    return score_mu.detach(), inv.detach()


def mahalanobis_rows(rows: torch.Tensor, mu: torch.Tensor, cov_inv: torch.Tensor) -> torch.Tensor:
    if rows.ndim != 2:
        rows = rows.reshape(rows.shape[0], -1)
    delta = rows - mu
    return torch.einsum("bi,ij,bj->b", delta, cov_inv, delta)


def compute_cov_stats(adapter: SharedLiteAdapter, train_records: Sequence[FileRecord], args: argparse.Namespace) -> CovStats:
    proxy_rows: List[Dict[str, Any]] = []

    # AE reconstruction metrics require the task head and are kept on the original
    # per-file path. All other tasks use the exact same embedding operation in a
    # batched form, which removes thousands of batch-size-one GPU launches.
    use_batched = int(args.eval_batch_size) > 1 and not (
        adapter.spec.task == "ae" and not args.skip_reconstruction_proxy
    )

    if use_batched:
        train_features = adapter.embeddings(
            [rec.path for rec in train_records],
            batch_size=int(args.eval_batch_size),
        )
        source_indices: List[int] = []
        target_indices: List[int] = []
        for idx, rec in enumerate(train_records):
            if rec.domain == "source":
                source_indices.append(idx)
            else:
                target_indices.append(idx)
            proxy_rows.append({
                "split": "train",
                "file_name": rec.file_name,
                "path": rec.path,
                "target_device": rec.target_device,
                "domain": rec.domain,
                "condition": rec.condition,
                "section": rec.section,
            })
        source_features = train_features[source_indices] if source_indices else None
        target_features = train_features[target_indices] if target_indices else None
    else:
        all_cov: List[torch.Tensor] = []
        source_cov: List[torch.Tensor] = []
        target_cov: List[torch.Tensor] = []
        for idx, rec in enumerate(train_records):
            fout = adapter.extract(rec.path)
            cov_feat = fout.cov_features.detach()
            all_cov.append(cov_feat)
            if rec.domain == "source":
                source_cov.append(cov_feat)
            else:
                target_cov.append(cov_feat)
            proxy_rows.append({
                "split": "train",
                "file_name": rec.file_name,
                "path": rec.path,
                "target_device": rec.target_device,
                "domain": rec.domain,
                "condition": rec.condition,
                "section": rec.section,
                **fout.proxy,
            })
            if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
                logging.info("  train feature %d/%d", idx + 1, len(train_records))
        train_features = torch.cat(all_cov, dim=0)
        source_features = torch.cat(source_cov, dim=0) if source_cov else None
        target_features = torch.cat(target_cov, dim=0) if target_cov else None

    train_mu, train_cov_inv = covariance_inverse(train_features, eps=args.cov_eps)

    source_mu = source_cov_inv = target_mu = target_cov_inv = None
    if source_features is not None and target_features is not None:
        if source_features.shape[0] >= 2:
            if args.fix_domain_mu:
                source_mu, source_cov_inv = covariance_inverse(source_features, eps=args.cov_eps)
            else:
                zero_mu = torch.zeros(source_features.shape[1], device=source_features.device, dtype=source_features.dtype)
                source_mu, source_cov_inv = covariance_inverse_with_separate_score_mu(source_features, zero_mu, eps=args.cov_eps)
        if target_features.shape[0] >= 2:
            if args.fix_domain_mu:
                target_mu, target_cov_inv = covariance_inverse(target_features, eps=args.cov_eps)
            else:
                zero_mu = torch.zeros(target_features.shape[1], device=target_features.device, dtype=target_features.dtype)
                target_mu, target_cov_inv = covariance_inverse_with_separate_score_mu(target_features, zero_mu, eps=args.cov_eps)

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


def evaluate_asd_scores(
    adapter: SharedLiteAdapter,
    test_records: Sequence[FileRecord],
    stats: CovStats,
    args: argparse.Namespace,
    model_id: str,
    target_device: str,
    out_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Dict[str, List[Tuple[FileRecord, torch.Tensor]]]]]:
    per_file_rows: List[Dict[str, Any]] = []
    section_items: Dict[str, Dict[str, List[Tuple[FileRecord, torch.Tensor]]]] = {}

    use_batched = int(args.eval_batch_size) > 1 and not (
        adapter.spec.task == "ae" and not args.skip_reconstruction_proxy
    )

    if use_batched:
        test_features = adapter.embeddings(
            [rec.path for rec in test_records],
            batch_size=int(args.eval_batch_size),
        )
        score_train_all = mahalanobis_rows(test_features, stats.train_mu, stats.train_cov_inv).detach().cpu().numpy()
        score_source_all: Optional[np.ndarray] = None
        score_target_all: Optional[np.ndarray] = None
        if stats.source_mu is not None and stats.source_cov_inv is not None:
            score_source_all = mahalanobis_rows(test_features, stats.source_mu, stats.source_cov_inv).detach().cpu().numpy()
        if stats.target_mu is not None and stats.target_cov_inv is not None:
            score_target_all = mahalanobis_rows(test_features, stats.target_mu, stats.target_cov_inv).detach().cpu().numpy()

        for idx, rec in enumerate(test_records):
            file_feat = test_features[idx : idx + 1].detach()
            score_train = float(score_train_all[idx])
            score_source = float(score_source_all[idx]) if score_source_all is not None else float("nan")
            score_target = float(score_target_all[idx]) if score_target_all is not None else float("nan")
            if np.isfinite(score_source) and np.isfinite(score_target):
                score_domain_min = float(min(score_source, score_target))
            elif np.isfinite(score_source):
                score_domain_min = score_source
            elif np.isfinite(score_target):
                score_domain_min = score_target
            else:
                score_domain_min = float("nan")
            per_file_rows.append({
                "model_id": model_id,
                "target_device": target_device,
                "file_name": rec.file_name,
                "path": rec.path,
                "domain": rec.domain,
                "condition": rec.condition,
                "section": rec.section,
                "class_label": rec.class_label,
                "mah_train": score_train,
                "mah_source": score_source,
                "mah_target": score_target,
                "mah_domain_min": score_domain_min,
            })
            section_items.setdefault(rec.section, {"normal": [], "anomaly": []})
            section_items[rec.section].setdefault(rec.condition, []).append((rec, file_feat))
    else:
        for idx, rec in enumerate(test_records):
            fout = adapter.extract(rec.path)
            mah_feat = fout.mah_features.detach()
            file_feat = fout.file_feature.detach()

            score_train = float(mahalanobis_rows(mah_feat, stats.train_mu, stats.train_cov_inv).mean().detach().cpu().item())
            score_source = score_target = score_domain_min = float("nan")
            if stats.source_mu is not None and stats.source_cov_inv is not None:
                score_source = float(mahalanobis_rows(mah_feat, stats.source_mu, stats.source_cov_inv).mean().detach().cpu().item())
            if stats.target_mu is not None and stats.target_cov_inv is not None:
                score_target = float(mahalanobis_rows(mah_feat, stats.target_mu, stats.target_cov_inv).mean().detach().cpu().item())
            if np.isfinite(score_source) and np.isfinite(score_target):
                score_domain_min = float(min(score_source, score_target))
            elif np.isfinite(score_source):
                score_domain_min = float(score_source)
            elif np.isfinite(score_target):
                score_domain_min = float(score_target)

            row = {
                "model_id": model_id,
                "target_device": target_device,
                "file_name": rec.file_name,
                "path": rec.path,
                "domain": rec.domain,
                "condition": rec.condition,
                "section": rec.section,
                "class_label": rec.class_label,
                "mah_train": score_train,
                "mah_source": score_source,
                "mah_target": score_target,
                "mah_domain_min": score_domain_min,
                **fout.proxy,
            }
            per_file_rows.append(row)
            section_items.setdefault(rec.section, {"normal": [], "anomaly": []})
            section_items[rec.section].setdefault(rec.condition, []).append((rec, file_feat))
            if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
                logging.info("  test feature %d/%d", idx + 1, len(test_records))

    per_file_df = pd.DataFrame(per_file_rows)
    labels = (per_file_df["condition"] == "anomaly").astype(int).values if not per_file_df.empty else np.asarray([])
    summary: Dict[str, Any] = {
        "mah_train_auc": safe_auc(per_file_df["mah_train"].values, labels) if not per_file_df.empty else float("nan"),
        "mah_train_pauc": safe_auc(per_file_df["mah_train"].values, labels, max_fpr=0.1) if not per_file_df.empty else float("nan"),
        "n_sections": len(section_items),
        "sections": ";".join(sorted(section_items.keys(), key=section_sort_key)),
        "n_test": len(per_file_df),
        "n_test_normal": int((per_file_df["condition"] == "normal").sum()) if not per_file_df.empty else 0,
        "n_test_anomaly": int((per_file_df["condition"] == "anomaly").sum()) if not per_file_df.empty else 0,
    }
    if not per_file_df.empty and per_file_df["mah_domain_min"].notna().any():
        summary["mah_domain_min_auc"] = safe_auc(per_file_df["mah_domain_min"].values, labels)
        summary["mah_domain_min_pauc"] = safe_auc(per_file_df["mah_domain_min"].values, labels, max_fpr=0.1)

    if args.save_roc and not per_file_df.empty:
        save_roc(per_file_df["mah_train"].values, labels, out_dir / "roc" / f"{model_id}_{target_device}_mah_train_ROC.png", f"Mahalanobis train-cov {target_device} {model_id}")
        if per_file_df["mah_domain_min"].notna().any():
            save_roc(per_file_df["mah_domain_min"].values, labels, out_dir / "roc" / f"{model_id}_{target_device}_mah_domain_min_ROC.png", f"Mahalanobis source/target-min {target_device} {model_id}")

    return summary, per_file_df, section_items


# -----------------------------------------------------------------------------
# Linear probes and projection outputs
# -----------------------------------------------------------------------------

def linear_oracle_eval(
    id_test_features: torch.Tensor,
    ood_test_features: torch.Tensor,
    id_train_features: torch.Tensor,
    ood_train_features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if id_train_features.numel() == 0 or ood_train_features.numel() == 0:
        return np.asarray([]), np.asarray([])
    features = torch.cat([id_train_features, ood_train_features], dim=0).float()
    labels = torch.cat([
        torch.zeros(id_train_features.shape[0], dtype=torch.long),
        torch.ones(ood_train_features.shape[0], dtype=torch.long),
    ], dim=0)
    features = features.reshape(features.shape[0], -1)
    input_dim = int(features.shape[1])
    linear_classifier = nn.Linear(input_dim, 2).to(device)
    optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_model_state = copy.deepcopy(linear_classifier.state_dict())
    best_accuracy = -1.0
    for _ in range(int(epochs)):
        correct = 0
        for inputs, y in loader:
            inputs = inputs.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
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
        id_logits = linear_classifier(id_test_features.reshape(id_test_features.shape[0], -1).float().to(device)).detach().cpu()
        ood_logits = linear_classifier(ood_test_features.reshape(ood_test_features.shape[0], -1).float().to(device)).detach().cpu()
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
            rows.append({"model_id": model_id, "target_device": target_device, "probe": "linear_loso", "fold": sec, "file_name": rec.file_name, "path": rec.path, "condition": rec.condition, "domain": rec.domain, "section": rec.section, "score": float(score)})
        for rec, score in zip([x[0] for x in test_a_items], score_a):
            rows.append({"model_id": model_id, "target_device": target_device, "probe": "linear_loso", "fold": sec, "file_name": rec.file_name, "path": rec.path, "condition": rec.condition, "domain": rec.domain, "section": rec.section, "score": float(score)})

    loso_scores = np.concatenate([np.asarray(loso_normal_scores), np.asarray(loso_anomaly_scores)]) if (loso_normal_scores or loso_anomaly_scores) else np.asarray([])
    loso_labels = np.concatenate([np.zeros(len(loso_normal_scores)), np.ones(len(loso_anomaly_scores))]) if len(loso_scores) else np.asarray([])
    metrics["linear_loso_auc"] = safe_auc(loso_scores, loso_labels)
    metrics["linear_loso_pauc"] = safe_auc(loso_scores, loso_labels, max_fpr=0.1)
    if args.save_roc and len(loso_scores):
        save_roc(loso_scores, loso_labels, out_dir / "roc" / f"{model_id}_{target_device}_linear_loso_ROC.png", f"Linear probe leave-one-section-out ({len(valid_sections)}fold) {target_device} {model_id}")

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
            save_roc(all_scores, all_labels, out_dir / "roc" / f"{model_id}_{target_device}_linear_all_ROC.png", f"Linear probe all-section {target_device} {model_id}")
        for rec, score in zip([x[0] for x in all_n_items], score_n):
            rows.append({"model_id": model_id, "target_device": target_device, "probe": "linear_all", "fold": "all", "file_name": rec.file_name, "path": rec.path, "condition": rec.condition, "domain": rec.domain, "section": rec.section, "score": float(score)})
        for rec, score in zip([x[0] for x in all_a_items], score_a):
            rows.append({"model_id": model_id, "target_device": target_device, "probe": "linear_all", "fold": "all", "file_name": rec.file_name, "path": rec.path, "condition": rec.condition, "domain": rec.domain, "section": rec.section, "score": float(score)})

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
            save_roc(half_scores, half_labels, out_dir / "roc" / f"{model_id}_{target_device}_linear_half_ROC.png", f"Linear probe half split {target_device} {model_id}")
        for rec, score in zip([x[0] for x in half_test_n], score_n):
            rows.append({"model_id": model_id, "target_device": target_device, "probe": "linear_half", "fold": "half", "file_name": rec.file_name, "path": rec.path, "condition": rec.condition, "domain": rec.domain, "section": rec.section, "score": float(score)})
        for rec, score in zip([x[0] for x in half_test_a], score_a):
            rows.append({"model_id": model_id, "target_device": target_device, "probe": "linear_half", "fold": "half", "file_name": rec.file_name, "path": rec.path, "condition": rec.condition, "domain": rec.domain, "section": rec.section, "score": float(score)})

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
        perplexity = min(args.tsne_perplexity, max(2, n_samples // 3), n_samples - 1)
        try:
            tsne_xy = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=projection_random_state,
                init="pca",
                learning_rate="auto",
            ).fit_transform(features)
        except Exception as e:
            logging.warning("t-SNE failed for %s/%s: %s", model_id, target_device, e)
    if not args.skip_umap and umap is not None and n_samples >= 4:
        try:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(args.umap_neighbors, n_samples - 1),
                min_dist=args.umap_min_dist,
                random_state=projection_random_state,
                n_jobs=1,  # random_state already forces deterministic single-worker UMAP
            )
            umap_xy = reducer.fit_transform(features)
        except Exception as e:
            logging.warning("UMAP failed for %s/%s: %s", model_id, target_device, e)

    for i, rec in enumerate(records):
        rows.append({
            "model_id": model_id,
            "target_device": target_device,
            "file_name": rec.file_name,
            "path": rec.path,
            "condition": rec.condition,
            "domain": rec.domain,
            "section": rec.section,
            "class_label": rec.class_label,
            "tsne_x": float(tsne_xy[i, 0]),
            "tsne_y": float(tsne_xy[i, 1]),
            "umap_x": float(umap_xy[i, 0]),
            "umap_y": float(umap_xy[i, 1]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(projection_dir / f"{model_id}_{target_device}_projection_coordinates.csv", index=False)
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


# -----------------------------------------------------------------------------
# Proxy metrics
# -----------------------------------------------------------------------------

def summarize_proxy_columns(prefix: str, train_proxy_df: pd.DataFrame, test_per_file_df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for split_name, df in [("train", train_proxy_df), ("test", test_per_file_df)]:
        if df is None or df.empty:
            continue
        for col in columns:
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            metrics[f"{prefix}_{col}_{split_name}_mean"] = float(values.mean())
            if "domain" in df.columns:
                for domain, sub in df.groupby("domain"):
                    metrics[f"{prefix}_{col}_{split_name}_{domain}_mean"] = float(pd.to_numeric(sub[col], errors="coerce").mean())
            if "condition" in df.columns:
                for condition, sub in df.groupby("condition"):
                    metrics[f"{prefix}_{col}_{split_name}_{condition}_mean"] = float(pd.to_numeric(sub[col], errors="coerce").mean())
    return metrics


def infer_classification_proxy_label_mode(adapter: SharedLiteAdapter, args: argparse.Namespace) -> str:
    mode = str(args.classification_label_mode).lower()
    if mode in {"machine", "domain"}:
        return mode
    # State-dict-only checkpoints do not store label_mode.  In the current 9-class
    # experiments, a two-output classifier means source/target domain classification;
    # otherwise it is machine classification.
    out_features = None
    if adapter.spec.task == "arcface" and adapter.spec.phase == "arcface" and adapter.arcface_head is not None:
        out_features = int(adapter.arcface_head.out_features)
    elif hasattr(adapter.model, "clf_head") and hasattr(adapter.model.clf_head, "out_features"):
        out_features = int(adapter.model.clf_head.out_features)
    if out_features == 2:
        return "domain"
    return "machine"


def classification_proxy_benchmark(
    adapter: SharedLiteAdapter,
    target_records: Sequence[FileRecord],
    other_records: Sequence[FileRecord],
    args: argparse.Namespace,
    model_id: str,
    target_device: str,
    out_dir: Path,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate the trained CE/ArcFace classifier head on all class test data.

    Important semantics:
      * The ASD metrics are still computed from the shared embedding z.
      * This proxy metric evaluates the classifier head itself.
      * For machine-label CE/ArcFace, the intended proxy metric is the global
        multi-class classification result over all machine classes, not only the
        current target device subset.

    The evaluator is still called once per target_device because the ASD path is
    target-specific.  Therefore global classifier metrics are repeated across
    target_device rows for a given checkpoint.  This is expected and useful for
    joining with target-specific ASD rows.  Per-class metrics are stored in
    proxy_detail_scores.csv to make class-wise failures explicit.
    """
    if not adapter.supports_classification_proxy:
        return {}, pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    label_mode = infer_classification_proxy_label_mode(adapter, args)
    idx_to_class = adapter.idx_to_class
    idx_to_domain = {0: "source", 1: "target"}

    if label_mode == "domain":
        label_order = ["source", "target"]
    elif label_mode == "machine":
        # Preserve classifier index order.  This must match training class_to_idx.
        label_order = [idx_to_class.get(i, f"class_{i}") for i in range(adapter.classifier_out_features)]
    else:
        label_order = [f"class_{i}" for i in range(adapter.classifier_out_features)]

    # Use all non-target test files unless explicitly limited.  Keeping this option
    # allows quick debugging, but final classifier proxy evaluation should use all.
    max_other = int(getattr(args, "classification_proxy_max_other", 0))
    if max_other <= 0:
        selected_other_records = list(other_records)
    else:
        selected_other_records = list(other_records)[:max_other]

    def predict_one(rec: FileRecord, split_group: str) -> None:
        logits = adapter.classification_logits(rec.path)
        if logits is None:
            return
        pred_idx = int(logits.detach().cpu().numpy().argmax(axis=1)[0])
        if label_mode == "domain":
            true_label = rec.domain
            pred_label = idx_to_domain.get(pred_idx, f"domain_{pred_idx}")
        elif label_mode == "machine":
            true_label = rec.class_label
            pred_label = idx_to_class.get(pred_idx, f"class_{pred_idx}")
        else:
            true_label = rec.class_label
            pred_label = f"class_{pred_idx}"
        rows.append({
            "row_type": "prediction",
            "model_id": model_id,
            "target_device": target_device,
            "proxy_type": "classification_proxy_prediction",
            "split_group": split_group,
            "classification_label_mode": label_mode,
            "file_name": rec.file_name,
            "path": rec.path,
            "true_label": true_label,
            "pred_label": pred_label,
            "true_class": rec.class_label,
            "pred_class": pred_label if label_mode == "machine" else "",
            "true_domain": rec.domain,
            "pred_domain": pred_label if label_mode == "domain" else "",
            "pred_idx": pred_idx,
            "domain": rec.domain,
            "condition": rec.condition,
            "section": rec.section,
        })

    for rec in target_records:
        predict_one(rec, "target_device_test")
    for rec in selected_other_records:
        predict_one(rec, "other_device_test")

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        return {}, pred_df

    def add_metric_block(metrics: Dict[str, float], df: pd.DataFrame, prefix: str, *, labels: Optional[Sequence[str]] = None) -> None:
        if df.empty:
            return
        true = df["true_label"].astype(str).values
        pred = df["pred_label"].astype(str).values
        metrics[f"{prefix}_accuracy"] = float(np.mean(true == pred))
        metrics[f"{prefix}_micro_f1"] = float(f1_score(true, pred, average="micro", labels=labels, zero_division=0))
        metrics[f"{prefix}_macro_f1"] = float(f1_score(true, pred, average="macro", labels=labels, zero_division=0))
        metrics[f"{prefix}_weighted_f1"] = float(f1_score(true, pred, average="weighted", labels=labels, zero_division=0))
        metrics[f"{prefix}_n"] = float(len(df))

    label_mode_value = str(pred_df["classification_label_mode"].iloc[0]) if "classification_label_mode" in pred_df.columns else "unknown"
    metrics: Dict[str, float] = {
        "clf_proxy_label_mode": label_mode_value,
        "clf_global_n_classes": float(len(label_order)),
        "clf_other_limited": float(1 if (max_other > 0 and len(other_records) > len(selected_other_records)) else 0),
    }

    # Primary classifier proxy: global all-class evaluation.
    add_metric_block(metrics, pred_df, "clf_global", labels=label_order)

    # Backward-compatible names used by previous evaluator versions.
    metrics["clf_all_accuracy"] = metrics.get("clf_global_accuracy", float("nan"))
    metrics["clf_all_micro_f1"] = metrics.get("clf_global_micro_f1", float("nan"))
    metrics["clf_all_macro_f1"] = metrics.get("clf_global_macro_f1", float("nan"))
    metrics["clf_all_weighted_f1"] = metrics.get("clf_global_weighted_f1", float("nan"))
    metrics["clf_all_n"] = metrics.get("clf_global_n", float("nan"))

    # Secondary diagnostics.  These are not the primary multi-class proxy metric.
    target_df = pred_df[pred_df["split_group"] == "target_device_test"]
    other_df = pred_df[pred_df["split_group"] == "other_device_test"]
    add_metric_block(metrics, target_df, "clf_target_total", labels=label_order)
    add_metric_block(metrics, other_df, "clf_other", labels=label_order)

    # Backward-compatible proxy aliases.
    metrics["clf_proxy_accuracy"] = metrics.get("clf_global_accuracy", float("nan"))
    metrics["clf_proxy_micro_f1"] = metrics.get("clf_global_micro_f1", float("nan"))
    metrics["clf_proxy_macro_f1"] = metrics.get("clf_global_macro_f1", float("nan"))
    metrics["clf_proxy_weighted_f1"] = metrics.get("clf_global_weighted_f1", float("nan"))
    metrics["clf_proxy_n"] = metrics.get("clf_global_n", float("nan"))

    true_all = pred_df["true_label"].astype(str).values
    pred_all = pred_df["pred_label"].astype(str).values

    precision, recall, f1, support = precision_recall_fscore_support(
        true_all,
        pred_all,
        labels=list(label_order),
        zero_division=0,
    )

    per_class_rows: List[Dict[str, Any]] = []
    per_class_lookup: Dict[str, Dict[str, float]] = {}
    for label, p_val, r_val, f_val, s_val in zip(label_order, precision, recall, f1, support):
        per_class_lookup[str(label)] = {
            "precision": float(p_val),
            "recall": float(r_val),
            "f1": float(f_val),
            "support": float(s_val),
        }
        for metric_name, metric_score in [
            ("precision", p_val),
            ("recall", r_val),
            ("f1", f_val),
            ("support", s_val),
        ]:
            per_class_rows.append({
                "row_type": "per_class_metric",
                "model_id": model_id,
                "target_device": target_device,
                "proxy_type": "classification_proxy_per_class",
                "classification_label_mode": label_mode_value,
                "class_label": str(label),
                "metric_name": str(metric_name),
                "metric_score": float(metric_score),
            })

    # Target-device class-wise metrics from the global confusion table.  These are
    # easier to interpret than target-only macro-F1, which can be unstable because
    # the true labels contain only one class.
    target_label = None
    for label in label_order:
        if canonical_machine_name(label) == canonical_machine_name(target_device):
            target_label = str(label)
            break
    if target_label is not None and target_label in per_class_lookup:
        metrics["clf_target_class_precision"] = per_class_lookup[target_label]["precision"]
        metrics["clf_target_class_recall"] = per_class_lookup[target_label]["recall"]
        metrics["clf_target_class_f1"] = per_class_lookup[target_label]["f1"]
        metrics["clf_target_class_support"] = per_class_lookup[target_label]["support"]

    cm = confusion_matrix(true_all, pred_all, labels=list(label_order))
    confusion_rows: List[Dict[str, Any]] = []
    for i, true_label in enumerate(label_order):
        for j, pred_label in enumerate(label_order):
            count = int(cm[i, j])
            if count == 0:
                continue
            confusion_rows.append({
                "row_type": "confusion",
                "model_id": model_id,
                "target_device": target_device,
                "proxy_type": "classification_proxy_confusion",
                "classification_label_mode": label_mode_value,
                "true_label": str(true_label),
                "pred_label": str(pred_label),
                "count": count,
            })

    metric_rows: List[Dict[str, Any]] = []
    for name, value in metrics.items():
        if name == "clf_proxy_label_mode":
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            metric_rows.append({
                "row_type": "metric",
                "model_id": model_id,
                "target_device": target_device,
                "proxy_type": "classification_proxy",
                "metric_name": name,
                "metric_score": float(value),
                "classification_label_mode": label_mode_value,
            })
    metric_df = pd.DataFrame(metric_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    proxy_dir = out_dir / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(proxy_dir / f"{model_id}_{target_device}_classification_proxy_predictions.csv", index=False)
    if not metric_df.empty:
        metric_df.to_csv(proxy_dir / f"{model_id}_{target_device}_classification_proxy_metrics.csv", index=False)
    if not per_class_df.empty:
        per_class_df.to_csv(proxy_dir / f"{model_id}_{target_device}_classification_proxy_per_class.csv", index=False)
    if not confusion_df.empty:
        confusion_df.to_csv(proxy_dir / f"{model_id}_{target_device}_classification_proxy_confusion.csv", index=False)

    detail_frames = [df for df in [metric_df, per_class_df, confusion_df, pred_df] if not df.empty]
    detail_df = pd.concat(detail_frames, ignore_index=True, sort=False) if detail_frames else pd.DataFrame()
    return metrics, detail_df

def _snr_metric_tag(snr_db: float) -> str:
    """Return a filesystem/column-safe tag, e.g. -5 -> m5, 0 -> 0, 2.5 -> p2p5."""
    value = float(snr_db)
    if value == 0.0:
        return "0"
    prefix = "m" if value < 0.0 else "p"
    magnitude = f"{abs(value):g}".replace(".", "p")
    return f"{prefix}{magnitude}"


def _as_snr_list(value: Any) -> List[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        values = [float(v) for v in value]
    else:
        values = [float(value)]
    if not values:
        raise ValueError("At least one --snr_db value is required.")
    # Preserve CLI order while removing exact duplicates.
    unique: List[float] = []
    for v in values:
        if v not in unique:
            unique.append(v)
    return unique


def sep_direct_feature_proxy_benchmark(
    adapter: SharedLiteAdapter,
    target_paths: Sequence[str],
    nontarget_paths: Sequence[str],
    args: argparse.Namespace,
    model_id: str,
    target_device: str,
    out_dir: Path,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if adapter.spec.task != "sep_direct" or args.skip_sep_feature_proxy:
        return {}, pd.DataFrame()
    if not target_paths or not nontarget_paths:
        return {}, pd.DataFrame()

    snr_values = _as_snr_list(args.snr_db)
    rows: List[Dict[str, Any]] = []
    rng = random.Random(args.seed)
    n_eval = min(len(target_paths), int(args.sep_proxy_max_pairs)) if args.sep_proxy_max_pairs > 0 else len(target_paths)
    chosen_targets = list(target_paths)[:n_eval]
    nontarget_pool = list(nontarget_paths)

    # Draw each target/non-target pair once, then evaluate the same pair at every SNR.
    # This makes the mean across SNRs a controlled comparison rather than a pairing artifact.
    for i, t_path in enumerate(chosen_targets):
        nt_path = rng.choice(nontarget_pool)
        try:
            x_t = matrix_to_tensor(load_feature_matrix(t_path, args), adapter.device)
            x_nt = matrix_to_tensor(load_feature_matrix(nt_path, args), adapter.device)
            if x_t.shape != x_nt.shape:
                min_t = min(x_t.shape[-1], x_nt.shape[-1])
                x_t = x_t[..., :min_t]
                x_nt = x_nt[..., :min_t]

            for snr_db in snr_values:
                x_mix, x_nt_scaled, _ = make_feature_domain_mixture(
                    x_t,
                    x_nt,
                    snr_db=float(snr_db),
                    feature_scale=args.feature_scale,
                )
                with torch.no_grad():
                    pred = adapter.model(x_mix, task="sep_direct")["sep_pred"]
                    target = make_local_frame_targets(
                        x_nt_scaled,
                        token_steps=pred.shape[1],
                        frame_stack=adapter.model.frame_stack,
                    )
                    l1 = float(F.l1_loss(pred, target).detach().cpu().item())
                    l2 = float(F.mse_loss(pred, target).detach().cpu().item())

                rows.append({
                    "row_type": "pair_metric",
                    "model_id": model_id,
                    "target_device": target_device,
                    "pair_index": i,
                    "snr_db": float(snr_db),
                    "target_path": t_path,
                    "nontarget_path": nt_path,
                    "sep_direct_l1": l1,
                    "sep_direct_l2": l2,
                })
        except Exception as e:
            logging.warning("SEP direct proxy failed for pair %s / %s: %s", t_path, nt_path, e)

    df = pd.DataFrame(rows)
    if df.empty:
        return {}, df

    # Because every valid pair is evaluated at every SNR, these overall means are
    # exactly the unweighted mean of the per-SNR means. Existing column names are
    # retained for backward compatibility.
    metrics: Dict[str, float] = {
        "sep_direct_l1_mean": float(df["sep_direct_l1"].mean()),
        "sep_direct_l2_mean": float(df["sep_direct_l2"].mean()),
        "sep_direct_l1_snr_mean": float(df.groupby("snr_db")["sep_direct_l1"].mean().mean()),
        "sep_direct_l2_snr_mean": float(df.groupby("snr_db")["sep_direct_l2"].mean().mean()),
        "sep_direct_proxy_pairs": float(df["pair_index"].nunique()),
        "sep_direct_proxy_evaluations": float(len(df)),
        "sep_direct_snr_count": float(df["snr_db"].nunique()),
    }

    for snr_db in snr_values:
        sub = df[np.isclose(df["snr_db"].astype(float), float(snr_db))]
        if sub.empty:
            continue
        tag = _snr_metric_tag(float(snr_db))
        metrics[f"sep_direct_l1_snr_{tag}_mean"] = float(sub["sep_direct_l1"].mean())
        metrics[f"sep_direct_l2_snr_{tag}_mean"] = float(sub["sep_direct_l2"].mean())
        metrics[f"sep_direct_pairs_snr_{tag}"] = float(len(sub))

    proxy_dir = out_dir / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(proxy_dir / f"{model_id}_{target_device}_sep_direct_feature_proxy.csv", index=False)

    snr_summary_rows: List[Dict[str, Any]] = []
    for snr_db in snr_values:
        sub = df[np.isclose(df["snr_db"].astype(float), float(snr_db))]
        if sub.empty:
            continue
        snr_summary_rows.extend([
            {
                "row_type": "metric",
                "model_id": model_id,
                "target_device": target_device,
                "proxy_type": "sep_direct_feature_proxy",
                "snr_db": float(snr_db),
                "metric_name": "sep_direct_l1_mean",
                "metric_score": float(sub["sep_direct_l1"].mean()),
            },
            {
                "row_type": "metric",
                "model_id": model_id,
                "target_device": target_device,
                "proxy_type": "sep_direct_feature_proxy",
                "snr_db": float(snr_db),
                "metric_name": "sep_direct_l2_mean",
                "metric_score": float(sub["sep_direct_l2"].mean()),
            },
        ])
    snr_summary_rows.extend([
        {
            "row_type": "metric",
            "model_id": model_id,
            "target_device": target_device,
            "proxy_type": "sep_direct_feature_proxy",
            "snr_db": "mean",
            "metric_name": "sep_direct_l1_snr_mean",
            "metric_score": metrics["sep_direct_l1_snr_mean"],
        },
        {
            "row_type": "metric",
            "model_id": model_id,
            "target_device": target_device,
            "proxy_type": "sep_direct_feature_proxy",
            "snr_db": "mean",
            "metric_name": "sep_direct_l2_snr_mean",
            "metric_score": metrics["sep_direct_l2_snr_mean"],
        },
    ])
    snr_summary_df = pd.DataFrame(snr_summary_rows)
    snr_summary_df.to_csv(
        proxy_dir / f"{model_id}_{target_device}_sep_direct_feature_proxy_metrics.csv",
        index=False,
    )

    detail_df = pd.concat([snr_summary_df, df], ignore_index=True, sort=False)
    return metrics, detail_df


# The selected augmentation pairs do not depend on model weights. Reuse them
# across the 45 checkpoints instead of rescanning millions of filenames each time.
_UNSUP_PAIR_CACHE: Dict[Tuple[Any, ...], Tuple[List[Tuple[int, str, int, str, str]], int]] = {}


def build_stored_aug_groups(data_dir: str, target_device: str, scope: str = "all") -> Dict[str, List[str]]:
    """Build positive-view groups from stored <machine>/aug/*.npy files."""
    root = Path(data_dir)
    if scope == "all":
        target_dirs = [p for p in root.glob("*") if p.is_dir()]
    elif scope == "target":
        target_dirs = [p for p in root.glob("*") if p.is_dir() and (p.name == target_device or canonical_machine_name(p.name) == canonical_machine_name(target_device))]
    else:
        raise ValueError(f"Unsupported unsupervised alignment scope: {scope}")

    groups: Dict[str, List[str]] = {}
    for d in sorted(target_dirs):
        train_dir = d / "train"
        aug_dir = d / "aug"
        if not train_dir.is_dir() or not aug_dir.is_dir():
            continue
        train_files = sorted(train_dir.glob("*.npy"))
        train_names = [p.name for p in train_files]
        train_name_set = set(train_names)
        train_name_lengths = sorted({len(x) for x in train_name_set}, reverse=True)
        train_by_name = {p.name: str(p) for p in train_files}

        for aug in sorted(aug_dir.glob("*.npy")):
            orig_name: Optional[str] = None
            for length in train_name_lengths:
                if length >= len(aug.name):
                    continue
                candidate = aug.name[-length:]
                if candidate in train_name_set:
                    orig_name = candidate
                    break
            if orig_name is None:
                idx = aug.name.find("section_")
                if idx >= 0:
                    orig_name = aug.name[idx:]
            if orig_name is None:
                continue
            key_base = train_by_name.get(orig_name, orig_name)
            key = f"{d.name}/{key_base}"
            groups.setdefault(key, []).append(str(aug))
    return {k: v for k, v in groups.items() if len(v) >= 1}


def build_ta_wav_groups(data_dir: str, target_device: str, n_views: int, scope: str = "all") -> Dict[str, List[str]]:
    # Preserve the layout assumption of the previous evaluator:
    # each <machine>/ta directory contains files ordered as
    #   view0_all_samples, view1_all_samples, ..., view{n_views-1}_all_samples.
    root = Path(data_dir)
    groups: Dict[str, List[str]] = {}
    if n_views < 2 or not root.is_dir():
        return groups
    for dir_path in sorted([p for p in root.iterdir() if p.is_dir()]):
        if scope == "target" and not (dir_path.name == target_device or canonical_machine_name(dir_path.name) == canonical_machine_name(target_device)):
            continue
        wav_files = sorted(str(p.resolve()) for p in (dir_path / "ta").glob("*.wav"))
        if len(wav_files) >= 2:
            groups[str(dir_path)] = wav_files
    return groups


def _prepare_unsup_pairs(
    args: argparse.Namespace,
    target_device: str,
) -> Tuple[List[Tuple[int, str, int, str, str]], int]:
    """Select the exact positive pairs once and cache them across checkpoints."""
    effective_target = "__all__" if args.unsup_alignment_scope == "all" else canonical_machine_name(target_device)
    cache_key: Tuple[Any, ...] = (
        str(Path(args.data_dir).resolve()),
        args.unsup_aug_source,
        args.unsup_alignment_scope,
        effective_target,
        args.unsup_pair_policy,
        int(args.unsup_aug_views),
        int(args.unsup_alignment_pairs_per_group),
        int(args.seed),
    )
    cached = _UNSUP_PAIR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    started = time.perf_counter()
    if args.unsup_aug_source == "stored_aug":
        groups = build_stored_aug_groups(args.data_dir, target_device, scope=args.unsup_alignment_scope)
    elif args.unsup_aug_source == "ta_wav":
        groups = build_ta_wav_groups(args.data_dir, target_device, args.unsup_aug_views, scope=args.unsup_alignment_scope)
    else:
        raise ValueError(args.unsup_aug_source)

    rng = random.Random(args.seed)
    pairs: List[Tuple[int, str, int, str, str]] = []
    for group_idx, (orig_key, aug_list_raw) in enumerate(sorted(groups.items())):
        aug_list = sorted(aug_list_raw)
        if args.unsup_aug_source == "ta_wav":
            n_views = int(args.unsup_aug_views)
            ev_len = int(len(aug_list) / max(n_views, 1))
            if ev_len <= 0:
                continue
            pair_count = ev_len if int(args.unsup_alignment_pairs_per_group) <= 0 else min(int(args.unsup_alignment_pairs_per_group), ev_len)
        else:
            pair_count = int(args.unsup_alignment_pairs_per_group)
            if pair_count <= 0:
                pair_count = max(1, min(len(aug_list), int(args.unsup_aug_views)))

        for ev_idx in range(pair_count):
            if args.unsup_aug_source == "ta_wav":
                aug_int = rng.sample(range(0, int(args.unsup_aug_views)), 2)
                sample_int = rng.randrange(ev_len)
                load0 = aug_list[int(aug_int[0] * ev_len + sample_int)]
                load1 = aug_list[int(aug_int[1] * ev_len + sample_int)]
            elif args.unsup_pair_policy == "original_aug" and Path(orig_key).is_file():
                load0 = str(orig_key)
                load1 = rng.choice(aug_list)
            else:
                if len(aug_list) < 2:
                    continue
                load0, load1 = rng.sample(aug_list, 2)
            pairs.append((group_idx, orig_key, ev_idx, str(load0), str(load1)))

    result = (pairs, len(groups))
    _UNSUP_PAIR_CACHE[cache_key] = result
    logging.info(
        "Prepared %d unsup pairs from %d groups in %.2fs; pair list will be reused across checkpoints",
        len(pairs), len(groups), time.perf_counter() - started,
    )
    return result


def _uniformity_exact(
    z: torch.Tensor,
    *,
    block_size: int,
    pdist_max: int,
) -> float:
    """Compute log(mean(exp(-2*||zi-zj||^2))) over all i<j.

    For small inputs this uses the original ``torch.pdist`` expression verbatim.
    For larger inputs it evaluates the same complete pair set in bounded-memory
    matrix blocks. This is not a sampled or approximate uniformity metric.
    """
    n = int(z.shape[0])
    if n < 2:
        return float("nan")
    if pdist_max > 0 and n <= int(pdist_max):
        sq_pdist = torch.pdist(z, p=2).pow(2)
        return float(sq_pdist.mul(-2).exp().mean().log().detach().cpu().item())

    bs = max(int(block_size), 1)
    # Match the legacy torch.pdist expression's input dtype while avoiding a
    # slow FP64 reduction on consumer GPUs.
    total = torch.zeros((), device=z.device, dtype=z.dtype)
    count = 0
    norms = (z * z).sum(dim=1)
    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)
        zi = z[i0:i1]
        ni = norms[i0:i1]
        for j0 in range(i0, n, bs):
            j1 = min(j0 + bs, n)
            zj = z[j0:j1]
            nj = norms[j0:j1]
            # Squared Euclidean distance, evaluated via GEMM for high GPU usage.
            d2 = ni[:, None] + nj[None, :] - 2.0 * (zi @ zj.T)
            d2 = d2.clamp_min_(0.0)
            values = torch.exp(-2.0 * d2)
            if i0 == j0:
                mask = torch.triu(torch.ones_like(values, dtype=torch.bool), diagonal=1)
                block_values = values.masked_select(mask)
                count += int(block_values.numel())
                total = total + block_values.sum()
            else:
                count += int(values.numel())
                total = total + values.sum()
    if count <= 0:
        return float("nan")
    return float(torch.log(total / float(count)).detach().cpu().item())


def unsup_alignment_uniformity_benchmark(
    adapter: SharedLiteAdapter,
    args: argparse.Namespace,
    model_id: str,
    target_device: str,
    out_dir: Path,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if adapter.spec.task not in UNSUP_TASKS or args.skip_unsup_proxy or args.unsup_aug_source == "off":
        return {}, pd.DataFrame()

    pairs, n_groups = _prepare_unsup_pairs(args, target_device)
    if not pairs:
        logging.warning("No unsup augmentation pairs found for %s using source=%s", target_device, args.unsup_aug_source)
        return {}, pd.DataFrame()

    started = time.perf_counter()
    # Deduplicate paths without changing the selected positive-pair list. Model eval
    # is deterministic, so repeated references to the same file have identical z.
    unique_paths: List[str] = []
    path_to_index: Dict[str, int] = {}
    for _, _, _, load0, load1 in pairs:
        for path in (load0, load1):
            if path not in path_to_index:
                path_to_index[path] = len(unique_paths)
                unique_paths.append(path)

    unsup_bs = int(args.unsup_batch_size) if int(args.unsup_batch_size) > 0 else int(args.eval_batch_size)
    unique_z = adapter.embeddings(unique_paths, batch_size=unsup_bs)
    index_i = torch.tensor([path_to_index[x[3]] for x in pairs], dtype=torch.long, device=adapter.device)
    index_j = torch.tensor([path_to_index[x[4]] for x in pairs], dtype=torch.long, device=adapter.device)
    z_i_total = F.normalize(unique_z.index_select(0, index_i), p=2, dim=1)
    z_j_total = F.normalize(unique_z.index_select(0, index_j), p=2, dim=1)

    if args.unsup_uniformity_pool == "legacy_i":
        z_total = z_i_total
    elif args.unsup_uniformity_pool == "both":
        z_total = torch.cat([z_i_total, z_j_total], dim=0)
    else:
        raise ValueError(args.unsup_uniformity_pool)

    pair_alignment_values = ((z_i_total - z_j_total) ** 2).sum(dim=1)
    alignment = float(pair_alignment_values.mean().detach().cpu().item())
    uniformity = _uniformity_exact(
        z_total,
        block_size=int(args.unsup_uniformity_block_size),
        pdist_max=int(args.unsup_uniformity_pdist_max),
    )

    pair_rows: List[Dict[str, Any]] = []
    if args.unsup_save_pair_details:
        pair_alignment_cpu = pair_alignment_values.detach().cpu().numpy()
        for pair_idx, (group_idx, orig_key, ev_idx, load0, load1) in enumerate(pairs):
            pair_rows.append({
                "model_id": model_id,
                "target_device": target_device,
                "group_idx": group_idx,
                "orig_key": orig_key,
                "ev_idx": ev_idx,
                "pair_alignment": float(pair_alignment_cpu[pair_idx]),
                "path_i": load0,
                "path_j": load1,
            })

    n_pairs = len(pairs)
    metrics: Dict[str, Any] = {
        "unsup_alignment": alignment,
        "unsup_uniformity": uniformity,
        "unsup_proxy_pairs": float(n_pairs),
        "unsup_proxy_embeddings": float(z_total.shape[0]),
        "unsup_proxy_groups": float(n_groups),
        "unsup_alignment_scope": args.unsup_alignment_scope,
        "unsup_uniformity_pool": args.unsup_uniformity_pool,
    }
    metric_df = pd.DataFrame([
        {"row_type": "metric", "model_id": model_id, "target_device": target_device, "proxy_type": "unsup_alignment_uniformity", "metric_name": "alignment", "metric_score": alignment, "n_pairs": int(n_pairs), "n_embeddings": int(z_total.shape[0]), "n_groups": int(n_groups), "scope": args.unsup_alignment_scope, "uniformity_pool": args.unsup_uniformity_pool},
        {"row_type": "metric", "model_id": model_id, "target_device": target_device, "proxy_type": "unsup_alignment_uniformity", "metric_name": "uniformity", "metric_score": uniformity, "n_pairs": int(n_pairs), "n_embeddings": int(z_total.shape[0]), "n_groups": int(n_groups), "scope": args.unsup_alignment_scope, "uniformity_pool": args.unsup_uniformity_pool},
    ])
    proxy_dir = out_dir / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    metric_df.to_csv(proxy_dir / f"{model_id}_{target_device}_unsup_alignment_uniformity.csv", index=False)
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(proxy_dir / f"{model_id}_{target_device}_unsup_alignment_uniformity_pairs.csv", index=False)
    logging.info(
        "[%s/%s] unsup proxy pairs=%d unique_files=%d groups=%d completed in %.2fs",
        model_id, target_device, n_pairs, len(unique_paths), n_groups, time.perf_counter() - started,
    )
    return metrics, metric_df


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------

def _sync_eval_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_model_on_device(
    model: SharedBackboneLiteProxyNet,
    arcface_head: Optional[ArcMarginProduct],
    spec: SharedModelSpec,
    target_device: str,
    args: argparse.Namespace,
    torch_device: torch.device,
    run_dir: Path,
    ckpt_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_started = time.perf_counter()
    stage_times: Dict[str, float] = {}

    stage_started = time.perf_counter()
    splits = get_data_splits(args.data_dir, target_device, args)
    stage_times["split_index"] = time.perf_counter() - stage_started
    if len(splits.train_all) == 0:
        raise RuntimeError(f"No train files for target_device={target_device}")
    if len(splits.test_all) == 0:
        raise RuntimeError(f"No test files for target_device={target_device}")

    model_device_dir = run_dir / "per_model" / spec.model_id / target_device
    model_device_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_class_to_idx = None
    if ckpt_payload is not None and isinstance(ckpt_payload.get("class_to_idx"), dict):
        checkpoint_class_to_idx = ckpt_payload["class_to_idx"]
    adapter = SharedLiteAdapter(
        model,
        spec,
        args,
        torch_device,
        splits.class_list,
        arcface_head=arcface_head,
        checkpoint_class_to_idx=checkpoint_class_to_idx,
    )
    train_records = make_records(splits.train_all, target_device, args.data_dir, splits.class_list)
    test_records = make_records(splits.test_all, target_device, args.data_dir, splits.class_list)
    other_test_records = make_records(splits.other_test, target_device, args.data_dir, splits.class_list)

    logging.info("[%s/%s] train=%d test=%d", spec.model_id, target_device, len(train_records), len(test_records))

    _sync_eval_device(torch_device)
    stage_started = time.perf_counter()
    stats = compute_cov_stats(adapter, train_records, args)
    _sync_eval_device(torch_device)
    stage_times["train_embedding_cov"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    summary, per_file_df, section_items = evaluate_asd_scores(adapter, test_records, stats, args, spec.model_id, target_device, model_device_dir)
    _sync_eval_device(torch_device)
    stage_times["test_embedding_asd"] = time.perf_counter() - stage_started

    linear_df = pd.DataFrame()
    if not args.skip_linear:
        stage_started = time.perf_counter()
        linear_metrics, linear_df = run_linear_probes(section_items, args, adapter.device, spec.model_id, target_device, model_device_dir)
        _sync_eval_device(torch_device)
        stage_times["linear_probes"] = time.perf_counter() - stage_started
        summary.update(linear_metrics)

    projection_df = pd.DataFrame()
    if not args.skip_projection:
        stage_started = time.perf_counter()
        projection_items: List[Tuple[FileRecord, torch.Tensor]] = []
        ordered_sections = sorted(section_items.keys(), key=section_sort_key)
        for sec in ordered_sections:
            projection_items.extend(section_items[sec].get("normal", []))
        for sec in ordered_sections:
            projection_items.extend(section_items[sec].get("anomaly", []))
        projection_df = save_projection_outputs(projection_items, args, spec.model_id, target_device, model_device_dir)
        stage_times["projection"] = time.perf_counter() - stage_started

    proxy_detail_frames: List[pd.DataFrame] = []
    # AE proxy summary from train/test rows.
    summary.update(summarize_proxy_columns("proxy", stats.train_proxy_df, per_file_df, ["ae_l1", "ae_l2"]))

    if not args.skip_classification_proxy:
        stage_started = time.perf_counter()
        clf_metrics, clf_df = classification_proxy_benchmark(adapter, test_records, other_test_records, args, spec.model_id, target_device, model_device_dir)
        _sync_eval_device(torch_device)
        stage_times["classification_proxy"] = time.perf_counter() - stage_started
        summary.update(clf_metrics)
        if not clf_df.empty:
            proxy_detail_frames.append(clf_df)

    stage_started = time.perf_counter()
    sep_metrics, sep_df = sep_direct_feature_proxy_benchmark(adapter, splits.train_all, splits.other_train, args, spec.model_id, target_device, model_device_dir)
    _sync_eval_device(torch_device)
    stage_times["separation_proxy"] = time.perf_counter() - stage_started
    summary.update(sep_metrics)
    if not sep_df.empty:
        proxy_detail_frames.append(sep_df)

    stage_started = time.perf_counter()
    unsup_metrics, unsup_df = unsup_alignment_uniformity_benchmark(adapter, args, spec.model_id, target_device, model_device_dir)
    _sync_eval_device(torch_device)
    stage_times["unsup_proxy"] = time.perf_counter() - stage_started
    summary.update(unsup_metrics)
    if not unsup_df.empty:
        proxy_detail_frames.append(unsup_df)

    stage_times["total"] = time.perf_counter() - total_started
    for stage_name, seconds in stage_times.items():
        summary[f"runtime_{stage_name}_sec"] = float(seconds)
    logging.info(
        "[%s/%s] runtime split=%.2fs train=%.2fs test=%.2fs linear=%.2fs projection=%.2fs unsup=%.2fs total=%.2fs",
        spec.model_id,
        target_device,
        stage_times.get("split_index", 0.0),
        stage_times.get("train_embedding_cov", 0.0),
        stage_times.get("test_embedding_asd", 0.0),
        stage_times.get("linear_probes", 0.0),
        stage_times.get("projection", 0.0),
        stage_times.get("unsup_proxy", 0.0),
        stage_times.get("total", 0.0),
    )

    proxy_detail_df = pd.concat(proxy_detail_frames, ignore_index=True, sort=False) if proxy_detail_frames else pd.DataFrame()
    return summary, per_file_df, linear_df, projection_df, stats.train_proxy_df, proxy_detail_df

def write_global_outputs(
    out_dir: Path,
    summary_rows: List[Dict[str, Any]],
    per_file_rows: List[pd.DataFrame],
    linear_rows: List[pd.DataFrame],
    projection_rows: List[pd.DataFrame],
    train_proxy_rows: List[pd.DataFrame],
    proxy_detail_rows: List[pd.DataFrame],
    error_rows: List[Dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "results_summary.csv", index=False)
    if per_file_rows:
        pd.concat(per_file_rows, ignore_index=True, sort=False).to_csv(out_dir / "per_file_scores.csv", index=False)
    if linear_rows:
        pd.concat(linear_rows, ignore_index=True, sort=False).to_csv(out_dir / "linear_probe_scores.csv", index=False)
    if projection_rows:
        pd.concat(projection_rows, ignore_index=True, sort=False).to_csv(out_dir / "projection_coordinates.csv", index=False)
    if train_proxy_rows:
        pd.concat(train_proxy_rows, ignore_index=True, sort=False).to_csv(out_dir / "train_proxy_scores.csv", index=False)
    if proxy_detail_rows:
        pd.concat(proxy_detail_rows, ignore_index=True, sort=False).to_csv(out_dir / "proxy_detail_scores.csv", index=False)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(out_dir / "errors.csv", index=False)
    else:
        stale_errors = out_dir / "errors.csv"
        if stale_errors.exists():
            stale_errors.unlink()


def read_existing_outputs(save_dir: Path) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame], List[Dict[str, Any]]]:
    """Load existing global CSV outputs for safe resume without dropping completed rows."""
    def read_df(name: str) -> pd.DataFrame:
        path = save_dir / name
        if not path.is_file():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as e:
            logging.warning("Could not read existing %s: %s", path, e)
            return pd.DataFrame()

    summary_df = read_df("results_summary.csv")
    per_file_df = read_df("per_file_scores.csv")
    linear_df = read_df("linear_probe_scores.csv")
    projection_df = read_df("projection_coordinates.csv")
    train_proxy_df = read_df("train_proxy_scores.csv")
    proxy_detail_df = read_df("proxy_detail_scores.csv")
    errors_df = read_df("errors.csv")
    return (
        summary_df.to_dict("records") if not summary_df.empty else [],
        [per_file_df] if not per_file_df.empty else [],
        [linear_df] if not linear_df.empty else [],
        [projection_df] if not projection_df.empty else [],
        [train_proxy_df] if not train_proxy_df.empty else [],
        [proxy_detail_df] if not proxy_detail_df.empty else [],
        errors_df.to_dict("records") if not errors_df.empty else [],
    )


def _find_summary_row(summary_rows: Sequence[Dict[str, Any]], model_id: str, target_device: str) -> Optional[Dict[str, Any]]:
    for row in summary_rows:
        if str(row.get("model_id")) == str(model_id) and str(row.get("target_device")) == str(target_device):
            return row
    return None


def _is_finite_metric(row: Dict[str, Any], name: str) -> bool:
    if name not in row:
        return False
    value = row.get(name)
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _concat_existing_proxy_detail(proxy_detail_rows: Sequence[pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in proxy_detail_rows if isinstance(df, pd.DataFrame) and not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _has_proxy_metric(proxy_detail_rows: Sequence[pd.DataFrame], model_id: str, target_device: str, proxy_type: str, metric_names: Sequence[str]) -> bool:
    df = _concat_existing_proxy_detail(proxy_detail_rows)
    if df.empty or "model_id" not in df.columns or "target_device" not in df.columns:
        return False
    sub = df[(df["model_id"].astype(str) == str(model_id)) & (df["target_device"].astype(str) == str(target_device))]
    if sub.empty:
        return False
    if "proxy_type" in sub.columns:
        sub = sub[sub["proxy_type"].astype(str) == str(proxy_type)]
    if sub.empty or "metric_name" not in sub.columns:
        return False
    found = set(sub["metric_name"].astype(str).tolist())
    return all(name in found for name in metric_names)


def _requires_classification_proxy(spec: SharedModelSpec, args: argparse.Namespace) -> bool:
    return (
        not bool(args.skip_classification_proxy)
        and (spec.task == "ce" or (spec.task == "arcface" and spec.phase == "linear"))
    )


def _classification_resume_label_mode(summary_row: Optional[Dict[str, Any]], args: argparse.Namespace) -> str:
    mode = str(args.classification_label_mode).lower()
    if mode in {"machine", "domain"}:
        return mode
    if summary_row is not None:
        value = str(summary_row.get("clf_proxy_label_mode", "")).lower()
        if value in {"machine", "domain"}:
            return value
    return mode


def _classification_proxy_metric_names(summary_row: Optional[Dict[str, Any]], args: argparse.Namespace) -> List[str]:
    metric_names = ["clf_global_micro_f1", "clf_global_macro_f1"]
    if _classification_resume_label_mode(summary_row, args) != "domain":
        metric_names.append("clf_target_class_recall")
    return metric_names


def required_summary_metrics_for_spec(
    spec: SharedModelSpec,
    args: argparse.Namespace,
    summary_row: Optional[Dict[str, Any]] = None,
) -> List[str]:
    required: List[str] = []
    if spec.task == "ae" and not args.skip_reconstruction_proxy:
        required.extend(["proxy_ae_l1_train_mean", "proxy_ae_l1_test_mean"])
    if spec.task == "sep_direct" and not args.skip_sep_feature_proxy:
        required.extend(["sep_direct_l1_mean", "sep_direct_l2_mean", "sep_direct_l1_snr_mean", "sep_direct_l2_snr_mean"])
        for snr_db in _as_snr_list(args.snr_db):
            tag = _snr_metric_tag(float(snr_db))
            required.extend([f"sep_direct_l1_snr_{tag}_mean", f"sep_direct_l2_snr_{tag}_mean"])
    if spec.task in UNSUP_TASKS and not args.skip_unsup_proxy and args.unsup_aug_source != "off":
        required.extend(["unsup_alignment", "unsup_uniformity"])
    if _requires_classification_proxy(spec, args):
        required.extend(_classification_proxy_metric_names(summary_row, args))
    return required


def existing_result_is_complete(
    summary_rows: Sequence[Dict[str, Any]],
    proxy_detail_rows: Sequence[pd.DataFrame],
    spec: SharedModelSpec,
    target_device: str,
    args: argparse.Namespace,
) -> bool:
    row = _find_summary_row(summary_rows, spec.model_id, target_device)
    if row is None:
        return False
    required = required_summary_metrics_for_spec(spec, args, row)
    for name in required:
        if not _is_finite_metric(row, name):
            return False

    # For proxy metrics that users inspect directly, require proxy_detail metric rows as well.
    if spec.task in UNSUP_TASKS and not args.skip_unsup_proxy and args.unsup_aug_source != "off":
        if not _has_proxy_metric(proxy_detail_rows, spec.model_id, target_device, "unsup_alignment_uniformity", ["alignment", "uniformity"]):
            return False
    if _requires_classification_proxy(spec, args):
        metric_names = _classification_proxy_metric_names(row, args)
        if not _has_proxy_metric(proxy_detail_rows, spec.model_id, target_device, "classification_proxy", metric_names):
            return False
    return True


def _filter_df_list_by_key(dfs: Sequence[pd.DataFrame], model_id: str, target_device: str) -> List[pd.DataFrame]:
    out: List[pd.DataFrame] = []
    for df in dfs:
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if "model_id" not in df.columns or "target_device" not in df.columns:
            out.append(df)
            continue
        keep = ~((df["model_id"].astype(str) == str(model_id)) & (df["target_device"].astype(str) == str(target_device)))
        filtered = df.loc[keep].copy()
        if not filtered.empty:
            out.append(filtered)
    return out


def drop_existing_rows_for_key(
    model_id: str,
    target_device: str,
    summary_rows: List[Dict[str, Any]],
    per_file_rows: List[pd.DataFrame],
    linear_rows: List[pd.DataFrame],
    projection_rows: List[pd.DataFrame],
    train_proxy_rows: List[pd.DataFrame],
    proxy_detail_rows: List[pd.DataFrame],
) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    summary_rows = [r for r in summary_rows if not (str(r.get("model_id")) == str(model_id) and str(r.get("target_device")) == str(target_device))]
    return (
        summary_rows,
        _filter_df_list_by_key(per_file_rows, model_id, target_device),
        _filter_df_list_by_key(linear_rows, model_id, target_device),
        _filter_df_list_by_key(projection_rows, model_id, target_device),
        _filter_df_list_by_key(train_proxy_rows, model_id, target_device),
        _filter_df_list_by_key(proxy_detail_rows, model_id, target_device),
    )


# -----------------------------------------------------------------------------
# Args / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model discovery
    parser.add_argument("--model_root", type=str, required=True, help="Task folder such as ./saved_proxy_lite/ae or root ./saved_proxy_lite")
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "sep", *TASKS])
    parser.add_argument("--model_glob", type=str, default="*best*.pth")
    parser.add_argument("--best_only", action="store_true", default=False)
    parser.add_argument("--arcface_phase", choices=["linear", "arcface", "all"], default="linear", help="ArcFace default is linear because it contains the fine-tuned clf_head.")

    # Data
    parser.add_argument("--data_dir", type=str, default="./asd_dataset_logmel")
    parser.add_argument("--data_ext", choices=["auto", "npy", "wav"], default="auto")
    parser.add_argument("--devices", nargs="+", default=None, help="If omitted, inferred from data_dir subdirectories.")
    parser.add_argument("--eval_devices", nargs="+", default=None, help="Optional subset of devices to evaluate.")
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--frame_stack", type=int, default=5)
    parser.add_argument(
        "--matrix_log_mode",
        type=str,
        default="auto",
        choices=["auto", "raw", "already_log", "log", "db", "none"],
        help=(
            "Feature-scale handling for .npy matrices. The shared-backbone logmel "
            "training script defaults to auto so raw mel-power and already-log/db "
            "features follow the same train/eval scale handling. Use explicit raw "
            "or already_log for archival reruns."
        ),
    )
    parser.add_argument("--feature_scale", type=str, default="db", choices=["db", "ln", "linear"])
    parser.add_argument("--eval_segment_frames", type=int, default=0, help="0 means evaluate full file. Positive value center-crops/pads to this length.")
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=512)

    # Model hparams, used if checkpoint args are unavailable.
    parser.add_argument("--use_checkpoint_args", action="store_true", default=True)
    parser.add_argument("--no_use_checkpoint_args", dest="use_checkpoint_args", action="store_false")
    parser.add_argument("--feature_index", type=int, default=3)
    parser.add_argument("--token_time_mode", type=str, default="upsample", choices=["native", "upsample"])
    parser.add_argument("--token_hidden_dim", type=int, default=128)
    parser.add_argument("--projection_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--simsiam_pred_hidden_dim", type=int, default=256)
    parser.add_argument("--normalize_projection", action="store_true")
    # Explicit overrides are rarely needed but useful for non-standard checkpoints.
    parser.add_argument("--override_feature_index", type=int, default=None)
    parser.add_argument("--override_frame_stack", type=int, default=None)
    parser.add_argument("--override_n_mels", type=int, default=None)
    parser.add_argument("--override_token_hidden_dim", type=int, default=None)
    parser.add_argument("--override_projection_hidden_dim", type=int, default=None)
    parser.add_argument("--override_projection_dim", type=int, default=None)
    parser.add_argument("--override_simsiam_pred_hidden_dim", type=int, default=None)
    parser.add_argument("--override_token_time_mode", type=str, default=None)
    parser.add_argument("--override_normalize_projection", choices=["true", "false", "1", "0"], default=None)

    # Loading
    # For result integrity, strict full-model loading is the default.
    # Use --allow_partial_load only for diagnostic recovery of non-standard checkpoints.
    parser.add_argument("--strict_load", action="store_true", default=True)
    parser.add_argument("--no_strict_load", dest="strict_load", action="store_false")
    parser.add_argument("--allow_partial_load", action="store_true", default=False)
    parser.add_argument("--no_allow_partial_load", dest="allow_partial_load", action="store_false")
    parser.add_argument("--arcface_scale", type=float, default=30.0)
    parser.add_argument("--arcface_margin", type=float, default=0.5)

    # ASD scoring
    parser.add_argument("--cov_eps", type=float, default=1e-5)
    # Shared-backbone embeddings should use the empirical source/target mean by default.
    # --zero_domain_mu restores the legacy zero-mean covariance behavior used for some residual-feature evaluations.
    parser.add_argument("--fix_domain_mu", action="store_true", default=True)
    parser.add_argument("--zero_domain_mu", dest="fix_domain_mu", action="store_false")
    parser.add_argument("--skip_linear", action="store_true")
    parser.add_argument("--linear_epochs", type=int, default=200)
    parser.add_argument("--linear_batch_size", type=int, default=64)
    parser.add_argument("--linear_lr", type=float, default=1e-3)
    parser.add_argument("--linear_half_split", choices=["legacy", "per_section"], default="legacy")
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

    # Proxy summaries
    parser.add_argument("--skip_reconstruction_proxy", action="store_true")
    parser.add_argument("--skip_sep_feature_proxy", action="store_true")
    parser.add_argument("--sep_proxy_max_pairs", type=int, default=500)
    parser.add_argument("--snr_db", type=float, nargs="+", default=[0.0], help="One or more feature-mixture SNRs in dB. Example: --snr_db -5 0 5. SEP proxy reports each SNR and their unweighted mean.")
    parser.add_argument("--skip_classification_proxy", action="store_true")
    parser.add_argument("--classification_proxy_max_other", type=int, default=100000)
    parser.add_argument("--classification_label_mode", choices=["auto", "machine", "domain"], default="auto", help="Proxy classifier label interpretation. auto treats 2-output checkpoints as domain classifiers and larger heads as machine classifiers.")

    # Unsupervised proxy: ASD still evaluates original train/test. This only controls alignment/uniformity.
    parser.add_argument("--skip_unsup_proxy", action="store_true")
    parser.add_argument("--unsup_aug_source", choices=["stored_aug", "ta_wav", "off"], default="stored_aug")
    parser.add_argument("--unsup_pair_policy", choices=["aug_aug", "original_aug"], default="aug_aug")
    parser.add_argument("--unsup_aug_views", type=int, default=33)
    parser.add_argument("--unsup_alignment_scope", choices=["all", "target"], default="all", help="Alignment/uniformity scope for stored_aug and ta_wav. all matches the previous evaluator default; target restricts to the evaluated machine.")
    parser.add_argument("--unsup_uniformity_pool", choices=["legacy_i", "both"], default="legacy_i", help="Uniformity feature pool. legacy_i uses one side of each pair to match the previous evaluator; both uses both stored/ta augmented views.")
    parser.add_argument("--unsup_alignment_pairs_per_group", type=int, default=1)
    parser.add_argument("--unsup_save_pair_details", action="store_true")

    # Batched feature extraction. These options change only execution scheduling,
    # not preprocessing, model outputs, sample selection, or metric definitions.
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Batch size for train/test embedding extraction. Use 1 for the legacy file-at-a-time path.")
    parser.add_argument("--unsup_batch_size", type=int, default=128, help="Batch size for stored-augmentation embedding extraction. <=0 reuses --eval_batch_size.")
    parser.add_argument("--eval_num_workers", type=int, default=8, help="CPU workers for npy/wav loading and preprocessing.")
    parser.add_argument("--eval_prefetch_factor", type=int, default=2)
    parser.add_argument("--eval_pin_memory", action="store_true", default=True)
    parser.add_argument("--no_eval_pin_memory", dest="eval_pin_memory", action="store_false")
    parser.add_argument("--eval_persistent_workers", action="store_true", default=False)
    parser.add_argument("--no_eval_persistent_workers", dest="eval_persistent_workers", action="store_false")
    parser.add_argument("--unsup_uniformity_block_size", type=int, default=2048, help="Bounded-memory block size for exact all-pairs uniformity when the pool is large.")
    parser.add_argument("--unsup_uniformity_pdist_max", type=int, default=8192, help="Use the original torch.pdist expression up to this many embeddings; larger pools use the exact blockwise all-pairs calculation.")

    # Runtime / saving
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite_completed", action="store_true", help="Recompute model-device results even if they are present in existing global CSVs.")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--write_every", type=int, default=5, help="Rewrite global CSVs every N completed model-device jobs. Use 0 to write only once at the end.")
    parser.add_argument("--empty_cuda_cache_each_model", action="store_true", help="Force torch.cuda.empty_cache() after every checkpoint. Disabled by default because allocator-cache reuse is faster.")

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "eval_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    if args.allow_partial_load:
        logging.warning("--allow_partial_load is enabled. This is diagnostic only and can invalidate final results if any tensors are skipped.")
    if args.matrix_log_mode == "auto":
        logging.warning("--matrix_log_mode auto is enabled. This matches the training default; use explicit raw or already_log for archival final reports.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(args.device)

    devices = args.devices if args.devices is not None else discover_devices(args.data_dir)
    devices = filter_devices_by_requested(devices, args.eval_devices)
    if not devices:
        raise RuntimeError("No devices resolved. Check --data_dir / --devices / --eval_devices.")

    specs = build_model_specs(args, devices)
    if not specs:
        raise RuntimeError(f"No checkpoints found under {args.model_root} with glob={args.model_glob}")
    logging.info("Resolved %d checkpoints", len(specs))

    if args.resume:
        (
            summary_rows,
            per_file_rows,
            linear_rows,
            projection_rows,
            train_proxy_rows,
            proxy_detail_rows,
            error_rows,
        ) = read_existing_outputs(save_dir)
    else:
        summary_rows: List[Dict[str, Any]] = []
        per_file_rows: List[pd.DataFrame] = []
        linear_rows: List[pd.DataFrame] = []
        projection_rows: List[pd.DataFrame] = []
        train_proxy_rows: List[pd.DataFrame] = []
        proxy_detail_rows: List[pd.DataFrame] = []
        error_rows: List[Dict[str, Any]] = []

    processed = 0

    for spec_idx, spec in enumerate(specs, start=1):
        logging.info("[%d/%d] Load model: %s", spec_idx, len(specs), spec.path)
        model = None
        arcface_head = None
        try:
            # Use class_list from data_dir for this load; actual device split may have same class list.
            class_list = discover_devices(args.data_dir)
            load_started = time.perf_counter()
            model, arcface_head, ckpt = build_and_load_model(spec, args, torch_device, class_list)
            logging.info("[%d/%d] Model loaded in %.2fs", spec_idx, len(specs), time.perf_counter() - load_started)
            for target_device in spec.eval_devices:
                if args.resume and not args.overwrite_completed and existing_result_is_complete(summary_rows, proxy_detail_rows, spec, target_device, args):
                    logging.info("Skip completed with required proxy metrics: %s / %s", spec.model_id, target_device)
                    continue
                if args.resume:
                    # If a previous row exists but required proxy metrics are absent, recompute and
                    # remove the stale rows from the in-memory global outputs before appending.
                    if _find_summary_row(summary_rows, spec.model_id, target_device) is not None:
                        logging.info("Recompute incomplete or overwritten result: %s / %s", spec.model_id, target_device)
                        summary_rows, per_file_rows, linear_rows, projection_rows, train_proxy_rows, proxy_detail_rows = drop_existing_rows_for_key(
                            spec.model_id, target_device, summary_rows, per_file_rows, linear_rows, projection_rows, train_proxy_rows, proxy_detail_rows
                        )
                try:
                    run_dir = save_dir
                    summary, per_file_df, linear_df, projection_df, train_proxy_df, proxy_detail_df = evaluate_model_on_device(
                        model=model,
                        arcface_head=arcface_head,
                        spec=spec,
                        target_device=target_device,
                        args=args,
                        torch_device=torch_device,
                        run_dir=run_dir,
                        ckpt_payload=ckpt.payload,
                    )
                    row = {
                        "model_id": spec.model_id,
                        "checkpoint_path": str(spec.path),
                        "task": spec.task,
                        "phase": spec.phase,
                        "target_device": target_device,
                        "trained_target": spec.target_device,
                        "backbone_name": spec.backbone_name,
                        "margin": spec.margin,
                        "feature_index": spec.feature_index,
                        "segment_frames_from_name": spec.segment_frames_from_name,
                        "frame_stack": spec.frame_stack,
                        "batch_size_from_name": spec.batch_size_from_name,
                        "epoch_from_name": spec.epoch_from_name,
                        "loss_from_name": spec.loss_from_name,
                        **summary,
                    }
                    summary_rows.append(row)
                    if not per_file_df.empty:
                        per_file_rows.append(per_file_df.assign(model_id=spec.model_id, task=spec.task, phase=spec.phase, backbone_name=spec.backbone_name))
                    if not linear_df.empty:
                        linear_rows.append(linear_df.assign(task=spec.task, phase=spec.phase, backbone_name=spec.backbone_name))
                    if not projection_df.empty:
                        projection_rows.append(projection_df.assign(task=spec.task, phase=spec.phase, backbone_name=spec.backbone_name))
                    if not train_proxy_df.empty:
                        train_proxy_rows.append(train_proxy_df.assign(model_id=spec.model_id, task=spec.task, phase=spec.phase, backbone_name=spec.backbone_name))
                    if not proxy_detail_df.empty:
                        proxy_detail_rows.append(proxy_detail_df.assign(task=spec.task, phase=spec.phase, backbone_name=spec.backbone_name))
                    processed += 1
                    if args.write_every > 0 and processed % args.write_every == 0:
                        write_global_outputs(save_dir, summary_rows, per_file_rows, linear_rows, projection_rows, train_proxy_rows, proxy_detail_rows, error_rows)
                except Exception as e:
                    logging.exception("Evaluation failed for %s / %s", spec.path, target_device)
                    error_rows.append({"checkpoint_path": str(spec.path), "model_id": spec.model_id, "task": spec.task, "phase": spec.phase, "target_device": target_device, "error": repr(e)})
                    if args.fail_fast:
                        raise
        except Exception as e:
            logging.exception("Model load failed: %s", spec.path)
            error_rows.append({"checkpoint_path": str(spec.path), "model_id": spec.model_id, "task": spec.task, "phase": spec.phase, "target_device": "__load__", "error": repr(e)})
            if args.fail_fast:
                raise
        finally:
            del model
            del arcface_head
            if torch_device.type == "cuda" and args.empty_cuda_cache_each_model:
                torch.cuda.empty_cache()

    write_global_outputs(save_dir, summary_rows, per_file_rows, linear_rows, projection_rows, train_proxy_rows, proxy_detail_rows, error_rows)
    logging.info("Finished. summary=%d, errors=%d, save_dir=%s", len(summary_rows), len(error_rows), save_dir)


if __name__ == "__main__":
    main(parse_args())

# proxy_audio_training_utils.py

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as Fnn
from torch.utils.data import Dataset, DataLoader


DEFAULT_N_MELS = 128
DEFAULT_N_FRAME = 5
DEFAULT_INPUT_DIM = DEFAULT_N_MELS * DEFAULT_N_FRAME


# -----------------------------------------------------------------------------
# Path / class utilities
# -----------------------------------------------------------------------------


def infer_domain(path: str | os.PathLike[str]) -> str:
    """Infer DCASE-style domain from filename."""
    name = os.path.basename(str(path)).lower()
    return "source" if "source" in name else "target"


def canonical_machine_name(name: str) -> str:
    n = str(name).lower().replace("_", "").replace("-", "")
    if n in {"toyconveyor", "toyconveyer"}:
        return "toyconveyer"
    return n


def is_target_machine(machine_dir_name: str, target: str) -> bool:
    if str(target).lower() in {"__all__", "all"}:
        return True
    return canonical_machine_name(machine_dir_name) == canonical_machine_name(target)


def sanitize_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def discover_machine_dirs(root_dir: str | os.PathLike[str]) -> List[Path]:
    root = Path(root_dir)
    return sorted(p for p in root.glob("*") if p.is_dir())


def discover_target_machine_dirs(root_dir: str | os.PathLike[str], target: str) -> List[Path]:
    if target == "__all__":
        return discover_machine_dirs(root_dir)
    return [p for p in discover_machine_dirs(root_dir) if is_target_machine(p.name, target)]


def glob_npys(machine_dir: Path, include_aug: bool = False, splits: Sequence[str] = ("train",)) -> List[str]:
    paths: List[str] = []
    for split in splits:
        split_dir = machine_dir / split
        if split_dir.is_dir():
            paths.extend(sorted(str(p) for p in split_dir.glob("*.npy")))
    if include_aug:
        aug_dir = machine_dir / "aug"
        if aug_dir.is_dir():
            paths.extend(sorted(str(p) for p in aug_dir.glob("*.npy")))
    return paths


def machine_name_from_path(path: str | os.PathLike[str]) -> str:
    """Expected layout: <root>/<machine>/<split>/<file>.npy."""
    p = Path(path)
    if len(p.parts) >= 3:
        return p.parent.parent.name
    return "unknown"


# -----------------------------------------------------------------------------
# Feature conversion utilities
# -----------------------------------------------------------------------------


def ae_log_from_raw_mel(mel: np.ndarray) -> np.ndarray:
    """DCASE AE-style log conversion: 10 * log10(power)."""
    return (10.0 * np.log10(np.maximum(mel, sys.float_info.epsilon))).astype(np.float32)


def maybe_convert_matrix_to_ae_log(matrix: np.ndarray, matrix_log_mode: str = "auto") -> np.ndarray:
    """Return matrix in DCASE AE-style dB/log scale.

    matrix_log_mode:
      - "auto": apply 10*log10 only when min >= 0.
      - "raw": always apply 10*log10(max(x, eps)).
      - "already_log": do not transform.
    """
    mode = str(matrix_log_mode).lower()
    matrix = np.asarray(matrix)

    if mode == "auto":
        finite = matrix[np.isfinite(matrix)]
        if finite.size and float(np.min(finite)) >= 0.0:
            return ae_log_from_raw_mel(matrix)
        return matrix.astype(np.float32, copy=False)

    if mode in {"raw", "raw_mel", "power"}:
        return ae_log_from_raw_mel(matrix)

    if mode in {"already_log", "log", "db", "none"}:
        return matrix.astype(np.float32, copy=False)

    raise ValueError(
        f"Unsupported matrix_log_mode={matrix_log_mode!r}. Use auto, raw, or already_log."
    )


def ae_vectors_to_matrix(vectors: np.ndarray, n_mels: int = DEFAULT_N_MELS, n_frame: int = DEFAULT_N_FRAME) -> np.ndarray:
    """Reconstruct an approximate [n_mels, T] matrix from DCASE AE vectors.

    If vectors were produced by consecutive frame stacking, this overlap-add
    reconstruction exactly recovers the original matrix except for floating-point noise.
    """
    vectors = np.asarray(vectors)
    input_dim = n_mels * n_frame
    if vectors.ndim != 2 or vectors.shape[1] != input_dim:
        raise ValueError(f"Expected vectors [N,{input_dim}], got {vectors.shape}")

    n_vectors = vectors.shape[0]
    total_frames = n_vectors + n_frame - 1
    acc = np.zeros((n_mels, total_frames), dtype=np.float32)
    cnt = np.zeros((1, total_frames), dtype=np.float32)

    for offset in range(n_frame):
        start = n_mels * offset
        end = n_mels * (offset + 1)
        acc[:, offset : offset + n_vectors] += vectors[:, start:end].T.astype(np.float32)
        cnt[:, offset : offset + n_vectors] += 1.0

    return acc / np.maximum(cnt, 1.0)


def coerce_logmel_matrix(
    arr: np.ndarray,
    *,
    npy_path: str,
    n_mels: int = DEFAULT_N_MELS,
    n_frame: int = DEFAULT_N_FRAME,
    matrix_log_mode: str = "auto",
) -> np.ndarray:
    """Convert supported npy layouts into [n_mels, T] matrix.

    Supported layouts:
      - [n_mels, T]
      - [T, n_mels]
      - [1, n_mels, T]
      - [N, n_mels*n_frame] DCASE AE vectors
      - 1D flattened DCASE AE vectors
    """
    arr = np.asarray(arr)
    input_dim = n_mels * n_frame

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 1:
        if arr.size % input_dim != 0:
            raise ValueError(
                f"{npy_path}: 1D npy has {arr.size} values, not divisible by input_dim={input_dim}."
            )
        vectors = arr.reshape(-1, input_dim).astype(np.float32, copy=False)
        matrix = ae_vectors_to_matrix(vectors, n_mels=n_mels, n_frame=n_frame)
        return maybe_convert_matrix_to_ae_log(matrix, matrix_log_mode=matrix_log_mode)

    if arr.ndim == 2:
        # Matrix layout [F, T].
        # Give this precedence because the CNN backbone expects file-wise matrices.
        # Note: [128, 640] is treated as a matrix, not AE vectors.
        if arr.shape[0] == n_mels:
            return maybe_convert_matrix_to_ae_log(arr, matrix_log_mode=matrix_log_mode)

        # Transposed matrix layout [T, F]
        if arr.shape[1] == n_mels:
            return maybe_convert_matrix_to_ae_log(arr.T, matrix_log_mode=matrix_log_mode)

        # DCASE AE vector layout [N, F*n_frame]
        if arr.shape[1] == input_dim:
            matrix = ae_vectors_to_matrix(arr.astype(np.float32, copy=False), n_mels=n_mels, n_frame=n_frame)
            return maybe_convert_matrix_to_ae_log(matrix, matrix_log_mode=matrix_log_mode)

    raise ValueError(
        f"{npy_path}: incompatible npy shape {arr.shape}. Expected one of "
        f"({n_mels},T), (T,{n_mels}), (1,{n_mels},T), (N,{input_dim}), or flattened AE vectors."
    )


def crop_or_pad_time(matrix: np.ndarray, segment_frames: int, random_crop: bool = True) -> np.ndarray:
    """Return [F, segment_frames] by crop/pad on time axis."""
    if segment_frames <= 0:
        return matrix.astype(np.float32, copy=False)

    matrix = np.asarray(matrix, dtype=np.float32)
    n_freq, n_frames = matrix.shape

    if n_frames == segment_frames:
        return matrix

    if n_frames > segment_frames:
        if random_crop:
            start = random.randint(0, n_frames - segment_frames)
        else:
            start = max((n_frames - segment_frames) // 2, 0)
        return matrix[:, start : start + segment_frames]

    pad = segment_frames - n_frames
    if n_frames == 0:
        return np.zeros((n_freq, segment_frames), dtype=np.float32)
    return np.pad(matrix, ((0, 0), (0, pad)), mode="edge").astype(np.float32, copy=False)


def load_logmel_tensor(
    npy_path: str,
    *,
    n_mels: int = DEFAULT_N_MELS,
    n_frame: int = DEFAULT_N_FRAME,
    segment_frames: int = 160,
    matrix_log_mode: str = "auto",
    random_crop: bool = True,
) -> torch.Tensor:
    arr = np.load(npy_path)
    matrix = coerce_logmel_matrix(
        arr,
        npy_path=npy_path,
        n_mels=n_mels,
        n_frame=n_frame,
        matrix_log_mode=matrix_log_mode,
    )
    matrix = crop_or_pad_time(matrix, segment_frames=segment_frames, random_crop=random_crop)
    # [F, T] -> [1, F, T]
    return torch.from_numpy(matrix).float().unsqueeze(0)


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------


class ProxySpectrogramDataset(Dataset):
    """File-wise spectrogram dataset for AE / CLF / SupCon / SimCLR."""

    def __init__(
        self,
        paths: Sequence[str],
        class_to_idx: Dict[str, int],
        *,
        n_mels: int = DEFAULT_N_MELS,
        n_frame: int = DEFAULT_N_FRAME,
        segment_frames: int = 160,
        matrix_log_mode: str = "auto",
        random_crop: bool = True,
    ):
        self.paths = [str(p) for p in paths]
        self.class_to_idx = class_to_idx
        self.n_mels = int(n_mels)
        self.n_frame = int(n_frame)
        self.segment_frames = int(segment_frames)
        self.matrix_log_mode = matrix_log_mode
        self.random_crop = bool(random_crop)

        if not self.paths:
            raise ValueError("ProxySpectrogramDataset received an empty path list.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.paths[idx]
        machine = machine_name_from_path(path)
        domain = infer_domain(path)

        logmel = load_logmel_tensor(
            path,
            n_mels=self.n_mels,
            n_frame=self.n_frame,
            segment_frames=self.segment_frames,
            matrix_log_mode=self.matrix_log_mode,
            random_crop=self.random_crop,
        )

        return {
            "logmel": logmel,
            "machine_label": torch.tensor(self.class_to_idx[canonical_machine_name(machine)], dtype=torch.long),
            "domain_label": torch.tensor(0 if domain == "source" else 1, dtype=torch.long),
            "path": path,
            "machine": machine,
            "domain": domain,
        }


class ProxySeparationPairDataset(Dataset):
    """Target / non-target pair dataset for feature-domain SEP."""

    def __init__(
        self,
        target_paths: Sequence[str],
        nontarget_paths: Sequence[str],
        class_to_idx: Dict[str, int],
        *,
        n_mels: int = DEFAULT_N_MELS,
        n_frame: int = DEFAULT_N_FRAME,
        segment_frames: int = 160,
        matrix_log_mode: str = "auto",
        random_crop: bool = True,
    ):
        self.target_paths = [str(p) for p in target_paths]
        self.nontarget_paths = [str(p) for p in nontarget_paths]
        self.class_to_idx = class_to_idx
        self.n_mels = int(n_mels)
        self.n_frame = int(n_frame)
        self.segment_frames = int(segment_frames)
        self.matrix_log_mode = matrix_log_mode
        self.random_crop = bool(random_crop)

        if not self.target_paths:
            raise ValueError("ProxySeparationPairDataset received an empty target path list.")
        if not self.nontarget_paths:
            raise ValueError("ProxySeparationPairDataset received an empty non-target path list.")

    def __len__(self) -> int:
        return len(self.target_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        target_path = self.target_paths[idx]
        nt_path = random.choice(self.nontarget_paths)

        target_machine = machine_name_from_path(target_path)
        nt_machine = machine_name_from_path(nt_path)

        target_logmel = load_logmel_tensor(
            target_path,
            n_mels=self.n_mels,
            n_frame=self.n_frame,
            segment_frames=self.segment_frames,
            matrix_log_mode=self.matrix_log_mode,
            random_crop=self.random_crop,
        )
        nontarget_logmel = load_logmel_tensor(
            nt_path,
            n_mels=self.n_mels,
            n_frame=self.n_frame,
            segment_frames=self.segment_frames,
            matrix_log_mode=self.matrix_log_mode,
            random_crop=self.random_crop,
        )

        return {
            "target_logmel": target_logmel,
            "nontarget_logmel": nontarget_logmel,
            "target_machine_label": torch.tensor(
                self.class_to_idx[canonical_machine_name(target_machine)], dtype=torch.long
            ),
            "nontarget_machine_label": torch.tensor(
                self.class_to_idx[canonical_machine_name(nt_machine)], dtype=torch.long
            ),
            "target_path": target_path,
            "nontarget_path": nt_path,
            "target_machine": target_machine,
            "nontarget_machine": nt_machine,
        }


# -----------------------------------------------------------------------------
# Stored augmentation pair dataset for SimCLR / SimSiam
# -----------------------------------------------------------------------------


def _extract_original_name_from_aug_filename(aug_name: str, original_names: set[str]) -> Optional[str]:
    """Infer the original train filename from a stored augmentation filename.

    Expected pattern:
        <aug_prefix>_<original_filename>.npy

    Example:
        fm0_section_00_source_train_normal_0000_vel_22.npy
        -> section_00_source_train_normal_0000_vel_22.npy

    The function is intentionally conservative: it returns a match only when the
    inferred original filename exists in the provided anchor filename set.
    """
    name = Path(aug_name).name

    # Fast path for DCASE-style filenames. The augmentation prefix is before
    # the first occurrence of "section_".
    section_pos = name.find("section_")
    if section_pos > 0:
        candidate = name[section_pos:]
        if candidate in original_names:
            return candidate

    # Fallback: augmented filename ends with "_<original_filename>".
    # This supports prefixes such as fm0, ts0.8, noise_snr3, etc.
    matches = [base for base in original_names if name.endswith("_" + base)]
    if matches:
        return max(matches, key=len)

    # Accept exact match only as a last-resort compatibility path. This is not
    # expected for files under aug/, but it makes the dataset robust to manually
    # copied original files.
    if name in original_names:
        return name

    return None


def build_stored_augmentation_index(
    anchor_paths: Sequence[str],
    aug_paths: Sequence[str],
) -> Tuple[Dict[Tuple[str, str], List[str]], List[str]]:
    """Map each train file to stored augmentation files under aug/.

    Keys are (canonical_machine_name, original_basename) to avoid collisions
    across different machine directories whose train filenames may be identical.
    """
    original_names_by_machine: Dict[str, set[str]] = {}
    for path in anchor_paths:
        machine = canonical_machine_name(machine_name_from_path(path))
        original_names_by_machine.setdefault(machine, set()).add(Path(path).name)

    index: Dict[Tuple[str, str], List[str]] = {}
    unmatched: List[str] = []

    for aug_path in aug_paths:
        machine = canonical_machine_name(machine_name_from_path(aug_path))
        original_names = original_names_by_machine.get(machine, set())
        original_name = _extract_original_name_from_aug_filename(Path(aug_path).name, original_names)
        if original_name is None:
            unmatched.append(str(aug_path))
            continue
        index.setdefault((machine, original_name), []).append(str(aug_path))

    for paths in index.values():
        paths.sort()
    return index, unmatched


class ProxyStoredAugmentPairDataset(Dataset):
    """Positive-pair dataset that uses precomputed files in <machine>/aug/.

    This dataset is intended for SimCLR and SimSiam in the shared-backbone
    experiment. It does not create augmentations online. Instead, each anchor
    train file is matched to stored augmentations whose filenames contain the
    original train filename after an augmentation prefix.

    Returned tensors:
        view0: [1, F, T]
        view1: [1, F, T]

    pair_policy:
        - "aug_aug": sample two stored augmentations of the same original file.
                     If only one augmentation exists, fallback to original_aug.
        - "original_aug": sample the original train file and one stored
                          augmentation of it.
    """

    def __init__(
        self,
        anchor_paths: Sequence[str],
        aug_paths: Sequence[str],
        class_to_idx: Dict[str, int],
        *,
        n_mels: int = DEFAULT_N_MELS,
        n_frame: int = DEFAULT_N_FRAME,
        segment_frames: int = 160,
        matrix_log_mode: str = "auto",
        random_crop: bool = True,
        pair_policy: str = "aug_aug",
    ):
        pair_policy = str(pair_policy).lower()
        if pair_policy not in {"aug_aug", "original_aug"}:
            raise ValueError("pair_policy must be 'aug_aug' or 'original_aug'.")

        self.anchor_paths = [str(p) for p in anchor_paths]
        self.aug_paths = [str(p) for p in aug_paths]
        self.class_to_idx = class_to_idx
        self.n_mels = int(n_mels)
        self.n_frame = int(n_frame)
        self.segment_frames = int(segment_frames)
        self.matrix_log_mode = matrix_log_mode
        self.random_crop = bool(random_crop)
        self.pair_policy = pair_policy

        if not self.anchor_paths:
            raise ValueError("ProxyStoredAugmentPairDataset received an empty train-anchor path list.")
        if not self.aug_paths:
            raise ValueError(
                "ProxyStoredAugmentPairDataset received an empty augmentation path list. "
                "Expected files under <target_dir>/<machine>/aug/*.npy."
            )

        aug_index, unmatched = build_stored_augmentation_index(self.anchor_paths, self.aug_paths)
        self.unmatched_aug_paths = unmatched

        self.samples: List[Tuple[str, List[str]]] = []
        for anchor_path in self.anchor_paths:
            machine = canonical_machine_name(machine_name_from_path(anchor_path))
            key = (machine, Path(anchor_path).name)
            matched_augs = aug_index.get(key, [])
            if matched_augs:
                self.samples.append((anchor_path, matched_augs))

        if not self.samples:
            example_anchor = Path(self.anchor_paths[0]).name if self.anchor_paths else "<none>"
            example_aug = Path(self.aug_paths[0]).name if self.aug_paths else "<none>"
            raise ValueError(
                "No stored augmentation files could be matched to train anchors. "
                f"Example anchor={example_anchor!r}, example aug={example_aug!r}. "
                "Expected augmented filenames like '<prefix>_<original_train_filename>.npy', "
                "e.g. fm0_section_00_source_train_normal_0000_vel_22.npy."
            )

        self.num_augmented_files = sum(len(augs) for _, augs in self.samples)
        self.num_single_aug_anchors = sum(1 for _, augs in self.samples if len(augs) == 1)

    def __len__(self) -> int:
        # One epoch iterates over original train anchors. The positive view is
        # sampled from the stored augmentation list at each load.
        return len(self.samples)

    def _sample_pair_paths(self, anchor_path: str, aug_paths: List[str]) -> Tuple[str, str]:
        if self.pair_policy == "original_aug":
            return anchor_path, random.choice(aug_paths)

        # Default: two stored augmentations of the same original. This matches
        # SimCLR/SimSiam's two-view assumption while using the precomputed aug set.
        if len(aug_paths) >= 2:
            view0, view1 = random.sample(aug_paths, k=2)
            return view0, view1

        # Degenerate but safe fallback. The user's dataset is expected to have
        # many augmentations per original, e.g. 33 variants, so this should rarely
        # be used unless the augmentation directory is incomplete.
        return anchor_path, aug_paths[0]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        anchor_path, aug_paths = self.samples[idx]
        view0_path, view1_path = self._sample_pair_paths(anchor_path, aug_paths)

        machine = machine_name_from_path(anchor_path)
        domain = infer_domain(anchor_path)
        machine_label = torch.tensor(self.class_to_idx[canonical_machine_name(machine)], dtype=torch.long)
        domain_label = torch.tensor(0 if domain == "source" else 1, dtype=torch.long)

        view0 = load_logmel_tensor(
            view0_path,
            n_mels=self.n_mels,
            n_frame=self.n_frame,
            segment_frames=self.segment_frames,
            matrix_log_mode=self.matrix_log_mode,
            random_crop=self.random_crop,
        )
        view1 = load_logmel_tensor(
            view1_path,
            n_mels=self.n_mels,
            n_frame=self.n_frame,
            segment_frames=self.segment_frames,
            matrix_log_mode=self.matrix_log_mode,
            random_crop=self.random_crop,
        )

        return {
            "view0": view0,
            "view1": view1,
            "machine_label": machine_label,
            "domain_label": domain_label,
            "anchor_path": anchor_path,
            "view0_path": view0_path,
            "view1_path": view1_path,
            "machine": machine,
            "domain": domain,
        }


def build_class_index(root_dir: str | os.PathLike[str]) -> Dict[str, int]:
    machine_dirs = discover_machine_dirs(root_dir)
    class_names = [canonical_machine_name(p.name) for p in machine_dirs]
    # preserve sorted order but remove duplicates after canonicalization
    unique = []
    for name in class_names:
        if name not in unique:
            unique.append(name)
    return {name: i for i, name in enumerate(unique)}


def build_proxy_loader(
    *,
    target_dir: str | os.PathLike[str],
    target_class: str,
    task: str,
    batch_size: int,
    n_cpu: int,
    n_mels: int = DEFAULT_N_MELS,
    n_frame: int = DEFAULT_N_FRAME,
    segment_frames: int = 160,
    matrix_log_mode: str = "auto",
    include_aug: bool = False,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = False,
    random_crop: bool = True,
    contrastive_pair_policy: str = "aug_aug",
) -> Tuple[DataLoader, Dict[str, int]]:
    """Build DataLoader for a proxy task.

    Expected npy root:
        <target_dir>/<machine>/train/*.npy
        <target_dir>/<machine>/aug/*.npy  # optional
    """
    root = Path(target_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"target_dir does not exist: {target_dir}")

    class_to_idx = build_class_index(root)
    all_dirs = discover_machine_dirs(root)
    target_dirs = discover_target_machine_dirs(root, target_class)

    if not target_dirs:
        available = [p.name for p in all_dirs]
        raise FileNotFoundError(
            f"No target machine directory matched target_class={target_class!r} under {target_dir!r}. "
            f"Available machines: {available}"
        )

    if task in {"sep", "sep_direct", "sep_mask"}:
        target_paths: List[str] = []
        for d in target_dirs:
            target_paths.extend(glob_npys(d, include_aug=include_aug, splits=("train",)))

        target_canon = {canonical_machine_name(d.name) for d in target_dirs}
        nontarget_dirs = [d for d in all_dirs if canonical_machine_name(d.name) not in target_canon]
        nontarget_paths: List[str] = []
        for d in nontarget_dirs:
            nontarget_paths.extend(glob_npys(d, include_aug=include_aug, splits=("train",)))

        dataset = ProxySeparationPairDataset(
            target_paths=target_paths,
            nontarget_paths=nontarget_paths,
            class_to_idx=class_to_idx,
            n_mels=n_mels,
            n_frame=n_frame,
            segment_frames=segment_frames,
            matrix_log_mode=matrix_log_mode,
            random_crop=random_crop,
        )
    elif task in {"simclr", "simsiam", "contrastive"}:
        # Contrastive learning uses stored augmentations in <machine>/aug/.
        # Do not include aug files as independent anchors; anchors come from train/.
        anchor_paths: List[str] = []
        aug_paths: List[str] = []
        for d in target_dirs:
            anchor_paths.extend(glob_npys(d, include_aug=False, splits=("train",)))
            aug_dir = d / "aug"
            if aug_dir.is_dir():
                aug_paths.extend(sorted(str(p) for p in aug_dir.glob("*.npy")))

        dataset = ProxyStoredAugmentPairDataset(
            anchor_paths=anchor_paths,
            aug_paths=aug_paths,
            class_to_idx=class_to_idx,
            n_mels=n_mels,
            n_frame=n_frame,
            segment_frames=segment_frames,
            matrix_log_mode=matrix_log_mode,
            random_crop=random_crop,
            pair_policy=contrastive_pair_policy,
        )
    else:
        paths: List[str] = []
        for d in target_dirs:
            paths.extend(glob_npys(d, include_aug=include_aug, splits=("train",)))

        dataset = ProxySpectrogramDataset(
            paths=paths,
            class_to_idx=class_to_idx,
            n_mels=n_mels,
            n_frame=n_frame,
            segment_frames=segment_frames,
            matrix_log_mode=matrix_log_mode,
            random_crop=random_crop,
        )

    effective_drop_last = drop_last and len(dataset) >= batch_size

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=effective_drop_last,
        num_workers=n_cpu,
        pin_memory=pin_memory,
    ), class_to_idx


# -----------------------------------------------------------------------------
# Loss / proxy-task utilities
# -----------------------------------------------------------------------------


def make_local_frame_targets(x: torch.Tensor, token_steps: int, frame_stack: int = DEFAULT_N_FRAME) -> torch.Tensor:
    """Make local frame-stack targets.

    Args:
        x: [B, 1, F, T]
        token_steps: number of output temporal tokens
        frame_stack: odd number, e.g. 5

    Returns:
        [B, token_steps, F * frame_stack]
    """
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"x should be [B, 1, F, T], got {tuple(x.shape)}")
    if frame_stack % 2 != 1:
        raise ValueError("frame_stack should be odd.")

    bsz, _, n_freq, n_frames = x.shape
    pad = frame_stack // 2
    x_2d = x.squeeze(1)  # [B, F, T]
    x_pad = Fnn.pad(x_2d, (pad, pad), mode="replicate")
    windows = x_pad.unfold(dimension=2, size=frame_stack, step=1)  # [B, F, T, K]
    targets = windows.permute(0, 2, 1, 3).contiguous().view(bsz, n_frames, n_freq * frame_stack)

    if token_steps != n_frames:
        targets = Fnn.interpolate(
            targets.transpose(1, 2),
            size=token_steps,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    return targets


def feature_to_linear(x: torch.Tensor, feature_scale: str = "db", eps: float = 1e-8) -> torch.Tensor:
    """Convert feature-domain tensor to non-negative linear power/magnitude proxy."""
    scale = str(feature_scale).lower()
    if scale in {"db", "log10", "10log10"}:
        return torch.pow(10.0, x / 10.0).clamp_min(eps)
    if scale in {"ln", "log", "natural_log"}:
        return torch.exp(x).clamp_min(eps)
    if scale in {"linear", "raw", "power"}:
        return x.clamp_min(eps)
    raise ValueError(f"Unsupported feature_scale={feature_scale!r}. Use db, ln, or linear.")


def linear_to_feature(x: torch.Tensor, feature_scale: str = "db", eps: float = 1e-8) -> torch.Tensor:
    scale = str(feature_scale).lower()
    x = x.clamp_min(eps)
    if scale in {"db", "log10", "10log10"}:
        return 10.0 * torch.log10(x)
    if scale in {"ln", "log", "natural_log"}:
        return torch.log(x)
    if scale in {"linear", "raw", "power"}:
        return x
    raise ValueError(f"Unsupported feature_scale={feature_scale!r}. Use db, ln, or linear.")


def make_feature_domain_mixture(
    target_feat: torch.Tensor,
    nontarget_feat: torch.Tensor,
    *,
    snr_db: Optional[float] = 0.0,
    alpha: Optional[torch.Tensor | float] = None,
    feature_scale: str = "db",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Feature-domain target + non-target mixture.

    For DCASE AE-style features, use feature_scale="db" because features are
    typically 10*log10(mel_power).

    Returns:
        mix_feat: [B,1,F,T]
        nontarget_scaled_feat: [B,1,F,T]
        alpha_tensor: [B,1,1,1] or broadcastable
    """
    if target_feat.shape != nontarget_feat.shape:
        raise ValueError(f"Shape mismatch: {target_feat.shape} vs {nontarget_feat.shape}")

    target_lin = feature_to_linear(target_feat, feature_scale=feature_scale, eps=eps)
    nt_lin = feature_to_linear(nontarget_feat, feature_scale=feature_scale, eps=eps)

    if alpha is None:
        if snr_db is None:
            alpha_tensor = torch.ones(
                (target_feat.shape[0], 1, 1, 1),
                device=target_feat.device,
                dtype=target_feat.dtype,
            )
        else:
            p_t = target_lin.mean(dim=(-2, -1), keepdim=True)
            p_nt = nt_lin.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
            ratio = 10.0 ** (float(snr_db) / 10.0)
            alpha_tensor = (p_t / (p_nt * ratio)).clamp_min(eps)
    else:
        if not torch.is_tensor(alpha):
            alpha_tensor = torch.tensor(alpha, device=target_feat.device, dtype=target_feat.dtype)
        else:
            alpha_tensor = alpha.to(device=target_feat.device, dtype=target_feat.dtype)
        if alpha_tensor.ndim == 0:
            alpha_tensor = alpha_tensor.view(1, 1, 1, 1)
        elif alpha_tensor.ndim == 1:
            alpha_tensor = alpha_tensor.view(-1, 1, 1, 1)

    nt_scaled = alpha_tensor * nt_lin
    mix_lin = target_lin + nt_scaled

    return (
        linear_to_feature(mix_lin, feature_scale=feature_scale, eps=eps),
        linear_to_feature(nt_scaled, feature_scale=feature_scale, eps=eps),
        alpha_tensor,
    )


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if z1.shape != z2.shape:
        raise ValueError(f"Shape mismatch: z1={z1.shape}, z2={z2.shape}")

    bsz = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = Fnn.normalize(z, dim=-1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(float("-inf"))
    labels = torch.arange(2 * bsz, device=z.device)
    labels = (labels + bsz) % (2 * bsz)
    return Fnn.cross_entropy(logits, labels)


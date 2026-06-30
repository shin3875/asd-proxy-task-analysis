"""AE baseline utilities compatible with file-wise 128-bin logmel npy files.

This module preserves the public API names used by the existing AE training code:
AENet, AEDataset_old, AEDataset, ae_dataset, loss_function_mahala, etc.

Supported npy layouts:
- (n_vectors, 640): already AE-compatible 5-frame vectors.
- (128, T): 128-bin mel/logmel matrix. Converted to (T - 5 + 1, 640).
- (T, 128): transposed matrix. Converted after transpose.
- (1, 128, T): batch-axis matrix. Converted after squeezing axis 0.

For matrix npy files, matrix_log_mode="auto" applies the old AE log conversion
10*log10(max(x, eps)) only when the matrix is non-negative, which is the usual
signature of raw mel-power files generated with --logmel-mode raw. If the matrix
already contains negative dB/log values, it is treated as already log-scaled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
from torch import nn


DEFAULT_N_MELS = 128
DEFAULT_N_FRAME = 5
DEFAULT_INPUT_DIM = DEFAULT_N_MELS * DEFAULT_N_FRAME


class AENet(nn.Module):
    def __init__(self, input_dim, block_size, lin_feat=128, comp_feat=8):
        super(AENet, self).__init__()
        self.input_dim = input_dim
        self.feat_dim = lin_feat

        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_all = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.feat_dim, comp_feat),
            nn.BatchNorm1d(comp_feat, momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(comp_feat, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.input_dim),
        )

    def forward(self, x):
        # reshape is safer than view when DataLoader/collate creates non-contiguous tensors.
        z = self.encoder(x.reshape(-1, self.input_dim))
        return self.decoder(z), z


class AEDataset_old(torch.utils.data.Dataset):
    """Original wav-on-the-fly feature extraction retained for compatibility."""

    def __init__(self, ori_list, n_frame=DEFAULT_N_FRAME):
        self.ori_list = list(ori_list)
        self.n_frame = int(n_frame)

    def __len__(self):
        return len(self.ori_list)

    def __getitem__(self, idx):
        wav_path = self.ori_list[idx]
        ori_signal, sr = librosa.load(wav_path, sr=None)
        domain = infer_domain(wav_path)
        mel = librosa.feature.melspectrogram(
            y=ori_signal,
            sr=sr,
            n_fft=1024,
            hop_length=512,
            n_mels=DEFAULT_N_MELS,
            power=2.0,
            fmin=0.0,
        )
        logmel = ae_log_from_raw_mel(mel)
        vectors = stack_consecutive_frames(logmel, self.n_frame)
        return vectors, domain


def infer_domain(path: str | os.PathLike[str]) -> str:
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


def ae_log_from_raw_mel(mel: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.maximum(mel, sys.float_info.epsilon))).astype(np.float32)


def maybe_convert_matrix_to_ae_log(matrix: np.ndarray, matrix_log_mode: str) -> np.ndarray:
    """Return a matrix in the same log scale as AEDataset_old.

    matrix_log_mode:
      - "auto": raw if min >= 0, otherwise already_log.
      - "raw": always apply 10*log10(max(x, eps)).
      - "already_log": do not transform.
    """
    mode = str(matrix_log_mode).lower()
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
        f"Unsupported matrix_log_mode={matrix_log_mode!r}. "
        "Use auto, raw, already_log, log, db, or none."
    )


def stack_consecutive_frames(mel: np.ndarray, n_frame: int = DEFAULT_N_FRAME) -> np.ndarray:
    mel = np.asarray(mel)
    if mel.ndim != 2:
        raise ValueError(f"Expected a 2D mel matrix, got shape {mel.shape}")
    n_mels, n_total_frames = mel.shape
    n_vectors = n_total_frames - n_frame + 1
    dims = n_mels * n_frame
    if n_vectors < 1:
        return np.empty((0, dims), dtype=np.float32)
    vectors = np.zeros((n_vectors, dims), dtype=np.float32)
    for t in range(n_frame):
        vectors[:, n_mels * t : n_mels * (t + 1)] = mel[:, t : t + n_vectors].T
    return vectors


def coerce_ae_vectors(
    arr: np.ndarray,
    *,
    npy_path: str,
    input_dim: int = DEFAULT_INPUT_DIM,
    n_mels: int = DEFAULT_N_MELS,
    n_frame: int = DEFAULT_N_FRAME,
    matrix_log_mode: str = "auto",
) -> np.ndarray:
    """Convert supported npy layouts into (n_vectors, input_dim)."""
    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 1:
        if arr.size % input_dim != 0:
            raise ValueError(f"{npy_path}: 1D npy has {arr.size} values, not divisible by input_dim={input_dim}.")
        return arr.reshape(-1, input_dim).astype(np.float32, copy=False)

    if arr.ndim == 2:
        # Already AE-compatible vector layout.
        if arr.shape[1] == input_dim:
            return arr.astype(np.float32, copy=False)

        # Matrix layout (n_mels, T) -> AE vectors.
        if arr.shape[0] == n_mels:
            matrix = maybe_convert_matrix_to_ae_log(arr, matrix_log_mode=matrix_log_mode)
            vectors = stack_consecutive_frames(matrix, n_frame=n_frame)
            if vectors.shape[1] != input_dim:
                raise ValueError(
                    f"{npy_path}: converted vector dim is {vectors.shape[1]}, expected {input_dim}. "
                    f"Check n_mels={n_mels}, n_frame={n_frame}."
                )
            return vectors.astype(np.float32, copy=False)

        # Transposed matrix layout (T, n_mels).
        if arr.shape[1] == n_mels:
            matrix = maybe_convert_matrix_to_ae_log(arr.T, matrix_log_mode=matrix_log_mode)
            vectors = stack_consecutive_frames(matrix, n_frame=n_frame)
            if vectors.shape[1] != input_dim:
                raise ValueError(
                    f"{npy_path}: converted vector dim is {vectors.shape[1]}, expected {input_dim}. "
                    f"Check n_mels={n_mels}, n_frame={n_frame}."
                )
            return vectors.astype(np.float32, copy=False)

    raise ValueError(
        f"{npy_path}: incompatible npy shape {arr.shape} for AE input_dim={input_dim}. "
        f"Expected one of: (N,{input_dim}), ({n_mels},T), (T,{n_mels}), or (1,{n_mels},T)."
    )


class AEDataset(torch.utils.data.Dataset):
    def __init__(self, ori_list, n_frame=DEFAULT_N_FRAME, n_mels=DEFAULT_N_MELS, input_dim=None, matrix_log_mode="auto"):
        self.ori_list = [str(p) for p in ori_list]
        self.n_frame = int(n_frame)
        self.n_mels = int(n_mels)
        self.input_dim = int(input_dim or self.n_mels * self.n_frame)
        self.matrix_log_mode = matrix_log_mode

    def __len__(self):
        return len(self.ori_list)

    def __getitem__(self, idx):
        npy_path = self.ori_list[idx]
        arr = np.load(npy_path)
        vectors = coerce_ae_vectors(
            arr,
            npy_path=npy_path,
            input_dim=self.input_dim,
            n_mels=self.n_mels,
            n_frame=self.n_frame,
            matrix_log_mode=self.matrix_log_mode,
        )
        return torch.from_numpy(vectors).float(), infer_domain(npy_path)


def _glob_npys(machine_dir: Path, split: str) -> list[str]:
    split_dir = machine_dir / split
    if not split_dir.is_dir():
        return []
    return sorted(str(p) for p in split_dir.glob("*.npy"))


def discover_target_machine_dirs(target_dir: str | os.PathLike[str], target: str) -> list[Path]:
    root = Path(target_dir)
    machine_dirs = sorted(p for p in root.glob("*") if p.is_dir())
    return [p for p in machine_dirs if is_target_machine(p.name, target)]


def ae_dataset(
    target_dir,
    batch_size,
    n_cpu,
    target,
    include_aug: bool = False,
    n_frame: int = DEFAULT_N_FRAME,
    n_mels: int = DEFAULT_N_MELS,
    input_dim: Optional[int] = None,
    matrix_log_mode: str = "auto",
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = False,
):
    """Return the AE training DataLoader.

    Expected npy root:
        <target_dir>/<machine>/train/*.npy
        <target_dir>/<machine>/aug/*.npy     # optional, used when include_aug=True
    """
    target_dirs = discover_target_machine_dirs(target_dir, target)
    if not target_dirs:
        available = [p.name for p in sorted(Path(target_dir).glob("*")) if p.is_dir()]
        raise FileNotFoundError(
            f"No target machine directory matched target={target!r} under {target_dir!r}. Available machines: {available}"
        )

    train_list: list[str] = []
    for machine_dir in target_dirs:
        train_list.extend(_glob_npys(machine_dir, "train"))
        if include_aug:
            train_list.extend(_glob_npys(machine_dir, "aug"))

    if not train_list:
        splits = {p.name: sorted(d.name for d in p.glob("*") if d.is_dir()) for p in target_dirs}
        raise FileNotFoundError(
            f"No .npy training files found for target={target!r} under {target_dir!r}. Checked: {splits}"
        )

    dataset = AEDataset(
        ori_list=train_list,
        n_frame=n_frame,
        n_mels=n_mels,
        input_dim=input_dim or n_mels * n_frame,
        matrix_log_mode=matrix_log_mode,
    )

    effective_drop_last = drop_last and len(dataset) >= batch_size

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=shuffle,
        sampler=None,
        drop_last=effective_drop_last,
        num_workers=n_cpu,
    )


def cov_v_diff(in_v):
    in_v_tmp = in_v.clone()
    mu = torch.mean(in_v_tmp.t(), 1)
    diff = torch.sub(in_v, mu)
    return diff, mu


def cov_v(diff, num):
    return torch.matmul(diff.T, diff) / num


def mahalanobis(u, v, cov_x, use_precision=False, reduction=True):
    num, _ = v.size()
    inv_cov = cov_x if use_precision else torch.inverse(cov_x)
    delta = torch.sub(u, v)
    m_loss = torch.matmul(torch.matmul(delta, inv_cov), delta.t())
    if reduction:
        return torch.sum(m_loss) / num
    return m_loss, num


def loss_function_mahala(
    recon_x,
    x,
    block_size=128,
    cov=None,
    is_source_list=None,
    is_target_list=None,
    update_cov=False,
    use_precision=False,
    reduction=True,
):
    if update_cov is False:
        loss = mahalanobis(
            recon_x.reshape(-1, block_size),
            x.reshape(-1, block_size),
            cov,
            use_precision,
            reduction=reduction,
        )
        return loss

    diff = x - recon_x
    cov_diff_source, _ = cov_v_diff(in_v=(diff[is_source_list]).reshape(-1, block_size))
    cov_diff_target = None
    is_calc_cov_target = any(is_target_list)
    if is_calc_cov_target:
        cov_diff_target, _ = cov_v_diff(in_v=(diff[is_target_list]).reshape(-1, block_size))
    loss = diff ** 2
    if reduction:
        loss = torch.mean(loss, dim=1)
    cov_diff_all, _ = cov_v_diff(in_v=diff.reshape(-1, block_size))
    return loss, cov_diff_source, cov_diff_target, cov_diff_all

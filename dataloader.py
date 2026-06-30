"""Unified dataloaders for public proxy-task training scripts."""

from __future__ import annotations

import glob
import os
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.utils.data
import torchaudio.transforms as T


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def db_calc(x):
    rms = np.sqrt(np.sum(x * x) / len(x))
    return 20 * np.log10(rms / 1) + 3


def _normal_path(path):
    return str(path).replace("\\", "/")


def _wav_files(directory):
    query = os.path.abspath("{dirs_in}/*.{ext}".format(dirs_in=directory, ext="wav"))
    return sorted(glob.glob(query))


def _relative_parts(root, path):
    root_path = Path(root).resolve()
    path_obj = Path(path).resolve()
    try:
        return [part.lower() for part in path_obj.relative_to(root_path).parts]
    except ValueError:
        return [part.lower() for part in path_obj.parts]


def _canonical_machine_name(name):
    name = str(name).lower().replace("_", "").replace("-", "")
    if name in {"toyconveyor", "toyconveyer"}:
        return "toyconveyer"
    return name


def _path_has_target(parts, target):
    target_canon = _canonical_machine_name(target)
    return any(_canonical_machine_name(part) == target_canon for part in parts)


def _npy_files(directory):
    query = os.path.abspath("{dirs_in}/*.{ext}".format(dirs_in=directory, ext="npy"))
    return sorted(glob.glob(query))


def _collect_train_aug_lists(target_dir, target):
    root = Path(target_dir)
    tr_query = os.path.abspath("{base}/*".format(base=target_dir))
    tr_dirs = sorted(glob.glob(tr_query))
    tr_dirs = [f for f in tr_dirs if os.path.isdir(f)]

    train_list = []
    aug_list = []
    train_list_oth = []
    aug_list_oth = []

    for split_parent in tr_dirs:
        dirs_in = glob.glob(split_parent + "/*")
        dirs_in = [f for f in dirs_in if os.path.isdir(f)]
        for data_dir in list(dirs_in):
            parts = _relative_parts(root, data_dir)
            if _path_has_target(parts, target):
                if "test" in parts:
                    continue
                if "aug" in parts:
                    aug_list = np.append(aug_list, _wav_files(data_dir))
                else:
                    train_list = np.append(train_list, _wav_files(data_dir))

    for split_parent in tr_dirs:
        dirs_in = glob.glob(split_parent + "/*")
        dirs_in = [f for f in dirs_in if os.path.isdir(f)]
        for data_dir in list(dirs_in):
            parts = _relative_parts(root, data_dir)
            if not _path_has_target(parts, target) and "test" not in parts:
                if "aug" in parts:
                    aug_list_oth = np.append(aug_list_oth, _wav_files(data_dir))
                else:
                    train_list_oth = np.append(train_list_oth, _wav_files(data_dir))

    return train_list, aug_list, train_list_oth, aug_list_oth


def _collect_npy_train_aug_lists(target_dir, target):
    root = Path(target_dir)
    tr_query = os.path.abspath("{base}/*".format(base=target_dir))
    tr_dirs = sorted(glob.glob(tr_query))
    tr_dirs = [f for f in tr_dirs if os.path.isdir(f)]

    train_list = []
    aug_list = []
    train_list_oth = []
    aug_list_oth = []

    for split_parent in tr_dirs:
        dirs_in = glob.glob(split_parent + "/*")
        dirs_in = [f for f in dirs_in if os.path.isdir(f)]
        for data_dir in list(dirs_in):
            parts = _relative_parts(root, data_dir)
            if _path_has_target(parts, target):
                if "test" in parts:
                    continue
                if "aug" in parts:
                    aug_list = np.append(aug_list, _npy_files(data_dir))
                else:
                    train_list = np.append(train_list, _npy_files(data_dir))

    for split_parent in tr_dirs:
        dirs_in = glob.glob(split_parent + "/*")
        dirs_in = [f for f in dirs_in if os.path.isdir(f)]
        for data_dir in list(dirs_in):
            parts = _relative_parts(root, data_dir)
            if not _path_has_target(parts, target) and "test" not in parts:
                if "aug" in parts:
                    aug_list_oth = np.append(aug_list_oth, _npy_files(data_dir))
                else:
                    train_list_oth = np.append(train_list_oth, _npy_files(data_dir))

    return train_list, aug_list, train_list_oth, aug_list_oth


def _group_by_class(paths, class_list):
    grouped = []
    for single_class in list(class_list):
        grouped.append([path for path in paths if _path_has_class(path, single_class)])
    return grouped


def _path_has_class(path, class_name):
    class_canon = _canonical_machine_name(class_name)
    return any(_canonical_machine_name(part) == class_canon for part in Path(path).parts)


def _class_from_path(path, class_list):
    norm_path = _normal_path(path)
    parts = [part for part in norm_path.split("/") if part]
    for class_name in class_list:
        class_canon = _canonical_machine_name(class_name)
        if any(_canonical_machine_name(part) == class_canon for part in parts):
            return class_name
    raise ValueError(f"Could not infer class from path: {path}")


def _crop_or_pad_signal(signal, cut_len):
    signal = np.asarray(signal)
    if len(signal) > cut_len:
        start = np.random.randint(0, len(signal) - cut_len + 1)
        signal = signal[start:start + cut_len]
    if len(signal) < cut_len:
        signal = np.pad(signal, (0, cut_len - len(signal)), mode="constant")
    return signal


def _crop_or_pad_pair(signal_a, signal_b, cut_len):
    signal_a = np.asarray(signal_a)
    signal_b = np.asarray(signal_b)
    if len(signal_a) > cut_len and len(signal_b) > cut_len:
        max_len = min(len(signal_a), len(signal_b)) - cut_len
        start = np.random.randint(0, max_len + 1)
        return signal_a[start:start + cut_len], signal_b[start:start + cut_len]
    return _crop_or_pad_signal(signal_a, cut_len), _crop_or_pad_signal(signal_b, cut_len)


def _raw_mel_to_log(matrix):
    return (10.0 * np.log10(np.maximum(matrix, sys.float_info.epsilon))).astype(np.float32)


def _maybe_convert_logmel_matrix(matrix, matrix_log_mode="auto"):
    mode = str(matrix_log_mode).lower()
    matrix = np.asarray(matrix, dtype=np.float32)

    if mode == "auto":
        finite = matrix[np.isfinite(matrix)]
        if finite.size and float(np.min(finite)) >= 0.0:
            return _raw_mel_to_log(matrix)
        return matrix.astype(np.float32, copy=False)
    if mode in {"raw", "raw_mel", "power"}:
        return _raw_mel_to_log(matrix)
    if mode in {"already_log", "log", "db", "none"}:
        return matrix.astype(np.float32, copy=False)

    raise ValueError(
        f"Unsupported matrix_log_mode={matrix_log_mode!r}. "
        "Use auto, raw, already_log, log, db, or none."
    )


def _crop_or_pad_logmel_frames(
    arr: np.ndarray,
    segment_frames: int,
    random_crop: bool = False,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    segment_frames = int(segment_frames)
    if segment_frames <= 0:
        return arr.astype(np.float32, copy=False)

    n_frames = int(arr.shape[1])
    if n_frames == segment_frames:
        return arr.astype(np.float32, copy=False)

    if n_frames > segment_frames:
        if random_crop:
            start = np.random.randint(0, n_frames - segment_frames + 1)
        else:
            start = (n_frames - segment_frames) // 2
        return arr[:, start:start + segment_frames].astype(np.float32, copy=False)

    finite = arr[np.isfinite(arr)]
    pad_value = float(np.min(finite)) if finite.size else 0.0
    pad_width = segment_frames - n_frames
    return np.pad(
        arr,
        ((0, 0), (0, pad_width)),
        mode="constant",
        constant_values=pad_value,
    ).astype(np.float32, copy=False)


def _load_logmel_npy(
    path,
    matrix_log_mode="auto",
    segment_frames: int = 313,
    random_crop: bool = False,
):
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2D logmel matrix, got shape {arr.shape}")
    if arr.shape[0] != 128 and arr.shape[1] == 128:
        arr = arr.T
    if arr.shape[0] != 128:
        raise ValueError(f"{path}: expected 128 mel bins, got shape {arr.shape}")
    arr = _maybe_convert_logmel_matrix(arr, matrix_log_mode=matrix_log_mode)
    arr = _crop_or_pad_logmel_frames(arr, segment_frames=segment_frames, random_crop=random_crop)
    return torch.from_numpy(arr).float().unsqueeze(dim=0)


def _split_parent_key(path, split_name):
    parts = [part.lower() for part in Path(path).parts]
    split_name = split_name.lower()
    if split_name in parts:
        split_idx = parts.index(split_name)
        if split_idx > 0:
            return parts[split_idx - 1]
    return None


def _build_aug_pairs(anchor_paths, aug_paths):
    anchors = [str(path) for path in anchor_paths]
    augs = [str(path) for path in aug_paths]
    anchor_entries = [
        (_split_parent_key(path, split_name="train"), os.path.basename(path), path)
        for path in anchors
    ]
    pairs = []
    grouped = {(split_key, name, path): [] for split_key, name, path in anchor_entries}

    for aug_path in augs:
        aug_key = _split_parent_key(aug_path, split_name="aug")
        aug_name = os.path.basename(aug_path)
        for split_key, anchor_name, anchor_path in anchor_entries:
            same_group = aug_key is None or split_key is None or aug_key == split_key
            if same_group and aug_name.endswith(anchor_name):
                grouped[(split_key, anchor_name, anchor_path)].append(aug_path)
                break

    for (_, _, anchor_path), matched in grouped.items():
        if matched:
            pairs.append((anchor_path, sorted(matched)))

    return pairs


class ContrastiveNPYDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples,
        matrix_log_mode="auto",
        segment_frames: int = 313,
        random_crop: bool = False,
    ):
        self.samples = list(samples)
        self.matrix_log_mode = matrix_log_mode
        self.segment_frames = int(segment_frames)
        self.random_crop = bool(random_crop)
        if not self.samples:
            raise ValueError("ContrastiveNPYDataset received no matched augmentation pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_path, aug_paths = self.samples[idx]
        if len(aug_paths) >= 2:
            view0_path, view1_path = random.sample(aug_paths, k=2)
        else:
            view0_path, view1_path = anchor_path, aug_paths[0]
        return (
            _load_logmel_npy(
                view0_path,
                matrix_log_mode=self.matrix_log_mode,
                segment_frames=self.segment_frames,
                random_crop=self.random_crop,
            ),
            _load_logmel_npy(
                view1_path,
                matrix_log_mode=self.matrix_log_mode,
                segment_frames=self.segment_frames,
                random_crop=self.random_crop,
            ),
        )


class ARCDataset(torch.utils.data.Dataset):
    def __init__(self, ori_list, aug_list, other_list, other_list_aug, min_len, class_list, snr_list):
        self.cut_len = int(min_len)
        self.ori_list = ori_list
        self.aug_list = aug_list
        self.oth_list = other_list
        self.otha_list = other_list_aug
        self.class_list = class_list
        self.snr_list = snr_list
        self.mel_spectrogram_transformer = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0,
        )
        self.logmel = T.AmplitudeToDB()

    def __len__(self):
        return len(self.ori_list)

    def __getitem__(self, idx):
        ori_line = random.choice(self.ori_list)
        ori_class = _class_from_path(ori_line, self.class_list)
        ori_idx = self.class_list.index(ori_class)

        ori_signal, _ = librosa.load(ori_line, sr=None)
        if len(ori_signal) < self.cut_len:
            pad_len = self.cut_len - len(ori_signal)
            ori_signal = np.pad(ori_signal, (0, pad_len), mode="constant")

        ori_signal = torch.FloatTensor(ori_signal).unsqueeze(dim=0)
        ori_mel = self.mel_spectrogram_transformer(ori_signal)
        ori_mel = self.logmel(ori_mel)

        return [ori_mel, ori_idx]


class SEPDataset(torch.utils.data.Dataset):
    def __init__(self, ori_list, aug_list, other_list, other_list_aug, min_len, class_list, snr_list):
        self.cut_len = int(min_len)
        self.ori_list = ori_list
        self.aug_list = aug_list
        self.oth_list = other_list
        self.otha_list = other_list_aug
        self.class_list = class_list
        self.snr_list = snr_list

    def __len__(self):
        return len(self.ori_list)

    def __getitem__(self, idx):
        ori_list = self.ori_list
        aug_list = self.aug_list

        if len(ori_list) == 0:
            raise RuntimeError("SEPDataset has no target train files.")
        if len(aug_list) == 0:
            raise RuntimeError("SEPDataset has no target augmentation files.")
        if len(self.class_list) == 0:
            raise RuntimeError("SEPDataset has no class list.")

        aug_sample = random.choice(aug_list)
        aug_line = aug_sample.split("section")[-1]
        ori_match = [k for k in ori_list if aug_line in k]
        if not ori_match:
            raise RuntimeError(f"Could not match augmentation to target train file: {aug_sample}")

        available_other = [
            (class_idx, self.oth_list[class_idx], self.otha_list[class_idx])
            for class_idx in range(min(len(self.oth_list), len(self.otha_list)))
            if len(self.oth_list[class_idx]) > 0 and len(self.otha_list[class_idx]) > 0
        ]
        if not available_other:
            raise RuntimeError("SEPDataset could not find non-target train/augmentation pairs.")

        class_idx, other_list, other_aug_list = random.choice(available_other)

        other_aug_line = random.choice(other_aug_list)
        other_aug_name = other_aug_line.split("section")[-1]
        other_match = [k for k in other_list if other_aug_name in k]
        if not other_match:
            raise RuntimeError(f"Could not match non-target augmentation to train file: {other_aug_line}")

        other_class = _class_from_path(other_match[0], self.class_list)
        ori_class = _class_from_path(ori_match[0], self.class_list)
        ori_idx = self.class_list.index(ori_class)
        oth_idx = self.class_list.index(other_class)

        ori_signal, _ = librosa.load(ori_match[0], sr=None)
        aug_signal, _ = librosa.load(aug_sample, sr=None)
        other_signal, _ = librosa.load(other_match[0], sr=None)
        other_aug_signal, _ = librosa.load(other_aug_line, sr=None)

        snr_target = random.choice(self.snr_list)
        temp_oth = other_signal

        ori_signal, temp_oth = _crop_or_pad_pair(ori_signal, temp_oth, self.cut_len)

        ori_db = db_calc(ori_signal)
        oth_db = db_calc(temp_oth)
        corr_db = (ori_db - oth_db) - snr_target
        sca = 1 / (10 ** (corr_db / 20))
        temp_oth = temp_oth / sca

        mix_sample = 32000
        start = 0
        mix_out = np.copy(ori_signal)
        mix_out[start:start + mix_sample] = mix_out[start:start + mix_sample] + temp_oth[start:start + mix_sample]
        temp_oth = temp_oth[start:start + mix_sample]

        if max(abs(mix_out)) > 1.0:
            divider = np.max(np.abs(mix_out))
            mix_out = mix_out / divider
            ori_signal = ori_signal / divider
            temp_oth = temp_oth / divider

        class_idx = [ori_idx, ori_idx, oth_idx, oth_idx]

        ori_signal, aug_signal = _crop_or_pad_pair(ori_signal, aug_signal, self.cut_len)
        other_signal, other_aug_signal = _crop_or_pad_pair(other_signal, other_aug_signal, self.cut_len)

        return [ori_signal, aug_signal, other_signal, other_aug_signal, mix_out, class_idx, temp_oth]


def class_data_sep(target_dir, batch_size, n_cpu, target, class_list, snr_list, input_len=80000):
    train_list, aug_list, train_list_oth, aug_list_oth = _collect_train_aug_lists(target_dir, target)
    if len(train_list) == 0:
        raise FileNotFoundError(f"No target train wavs found for target={target!r} under {target_dir!r}.")
    if len(aug_list) == 0:
        raise FileNotFoundError(f"No target augmentation wavs found for target={target!r} under {target_dir!r}.")
    if len(train_list_oth) == 0:
        raise FileNotFoundError(
            f"No non-target train wavs found for target={target!r}. "
            "Classifier/separation smoke tests require at least two device folders."
        )
    if len(aug_list_oth) == 0:
        raise FileNotFoundError(
            f"No non-target augmentation wavs found for target={target!r}. "
            "Separation smoke tests require at least two device folders with aug wavs."
        )

    class_aug_set = _group_by_class(aug_list_oth, class_list)
    class_oth_set = _group_by_class(train_list_oth, class_list)

    tar_min = min(train_list, key=os.path.getsize)
    non_min = min(train_list_oth, key=os.path.getsize)
    temp0, _ = librosa.load(tar_min, mono=False, sr=None)
    temp1, _ = librosa.load(non_min, mono=False, sr=None)
    shortest = min([len(temp0), len(temp1)])

    if shortest != 96000:
        print(f"Shortest file len {shortest}")

    cont_ds = SEPDataset(
        min_len=input_len,
        ori_list=train_list,
        aug_list=aug_list,
        other_list=class_oth_set,
        other_list_aug=class_aug_set,
        class_list=class_list,
        snr_list=snr_list,
    )

    return torch.utils.data.DataLoader(
        dataset=cont_ds,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=True,
        sampler=None,
        drop_last=len(cont_ds) >= batch_size,
        num_workers=n_cpu,
    )


def class_data_arcface(target_dir, batch_size, n_cpu, target, class_list, snr_list):
    train_list, aug_list, train_list_oth, aug_list_oth = _collect_train_aug_lists(target_dir, target)
    if len(train_list) == 0:
        raise FileNotFoundError(f"No target train wavs found for target={target!r} under {target_dir!r}.")
    if len(train_list_oth) == 0:
        raise FileNotFoundError(
            f"No non-target train wavs found for target={target!r}. "
            "Classifier/separation smoke tests require at least two device folders."
        )

    class_aug_set = _group_by_class(aug_list_oth, class_list)
    class_oth_set = _group_by_class(train_list_oth, class_list)

    tar_min = min(train_list, key=os.path.getsize)
    non_min = min(train_list_oth, key=os.path.getsize)
    librosa.load(tar_min, mono=False, sr=None)
    librosa.load(non_min, mono=False, sr=None)

    class_oth_set = list(filter(None, class_oth_set))
    class_aug_set = list(filter(None, class_aug_set))

    for oth_list in list(class_oth_set):
        train_list = np.append(train_list, oth_list)

    cont_ds = ARCDataset(
        min_len=160000,
        ori_list=train_list,
        aug_list=aug_list,
        other_list=class_oth_set,
        other_list_aug=class_aug_set,
        class_list=class_list,
        snr_list=snr_list,
    )

    return torch.utils.data.DataLoader(
        dataset=cont_ds,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=True,
        sampler=None,
        drop_last=len(cont_ds) >= batch_size,
        num_workers=n_cpu,
    )


def class_data_contrastive(
    target_dir,
    batch_size,
    n_cpu,
    target,
    class_list,
    snr_list,
    matrix_log_mode="auto",
    segment_frames: int = 313,
    random_crop: bool = False,
):
    train_list, aug_list, train_list_oth, aug_list_oth = _collect_npy_train_aug_lists(target_dir, target)

    for oth_train in _group_by_class(train_list_oth, class_list):
        train_list = np.append(train_list, oth_train)
    aug_list = np.append(aug_list, aug_list_oth)

    samples = _build_aug_pairs(train_list, aug_list)
    if not samples:
        class_aug_set = _group_by_class(aug_list, class_list)
        for aug_group in class_aug_set:
            if len(aug_group) >= 2:
                samples.append((aug_group[0], sorted(aug_group)))

    cont_ds = ContrastiveNPYDataset(
        samples,
        matrix_log_mode=matrix_log_mode,
        segment_frames=segment_frames,
        random_crop=random_crop,
    )

    return torch.utils.data.DataLoader(
        dataset=cont_ds,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=True,
        sampler=None,
        drop_last=len(cont_ds) >= batch_size,
        num_workers=n_cpu,
    )

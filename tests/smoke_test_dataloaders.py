#!/usr/bin/env python3
"""Toy-data smoke checks for public dataloader paths.

This script creates a tiny synthetic DCASE-style dataset in a temporary
directory. It does not read real ASD data.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dataloader
from ae_baseline_utils import ae_dataset


MACHINES = ["pump", "fan"]
SR = 16000
WAV_LEN = 40000
LOGMEL_SHAPE = (128, 64)
CONTRASTIVE_SEGMENT_FRAMES = 313


def write_wav(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, WAV_LEN / SR, WAV_LEN, endpoint=False)
    signal = 0.05 * np.sin(2 * np.pi * (220 + seed) * t)
    signal += 0.005 * rng.standard_normal(WAV_LEN)
    wavfile.write(path, SR, signal.astype(np.float32))


def write_npy(path: Path, seed: int, frames: int = LOGMEL_SHAPE[1]) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.random((LOGMEL_SHAPE[0], frames), dtype=np.float32)
    np.save(path, arr)


def create_toy_roots(work_dir: Path) -> tuple[Path, Path]:
    wav_root = work_dir / "toy_asd_dataset"
    npy_root = work_dir / "toy_asd_dataset_logmel"
    seed = 0

    for machine in MACHINES:
        for root in [wav_root, npy_root]:
            for split in ["train", "test", "aug"]:
                (root / machine / split).mkdir(parents=True, exist_ok=True)

        train_names = [
            "source_train_normal_section_00_00000000",
            "target_train_normal_section_01_00000000",
        ]
        for name in train_names:
            seed += 1
            write_wav(wav_root / machine / "train" / f"{name}.wav", seed)
            write_npy(npy_root / machine / "train" / f"{name}.npy", seed)
            for aug_idx in range(2):
                seed += 1
                aug_name = f"np{aug_idx}_{name}"
                write_wav(wav_root / machine / "aug" / f"{aug_name}.wav", seed)
                write_npy(
                    npy_root / machine / "aug" / f"{aug_name}.npy",
                    seed,
                    frames=48 + aug_idx * 72,
                )

        for section in range(3):
            for condition in ["normal", "anomaly"]:
                domain = "source" if section != 1 else "target"
                name = f"{domain}_test_{condition}_section_{section:02d}_00000000"
                seed += 1
                write_wav(wav_root / machine / "test" / f"{name}.wav", seed)
                write_npy(npy_root / machine / "test" / f"{name}.npy", seed)

    return wav_root, npy_root


def count_files(root: Path, machine: str, split: str, ext: str) -> int:
    return len(list((root / machine / split).glob(f"*.{ext}")))


def assert_file_counts(wav_root: Path, npy_root: Path) -> None:
    expected = {"train": 2, "aug": 4, "test": 6}
    for machine in MACHINES:
        for split, expected_count in expected.items():
            wav_count = count_files(wav_root, machine, split, "wav")
            npy_count = count_files(npy_root, machine, split, "npy")
            assert wav_count == expected_count, (machine, split, wav_count, expected_count)
            assert npy_count == expected_count, (machine, split, npy_count, expected_count)


def assert_ae_npy_shape(npy_root: Path) -> None:
    sample_paths = sorted(npy_root.glob("*/*/*.npy"))
    assert sample_paths, "no npy files generated"
    for path in sample_paths:
        arr = np.load(path)
        assert arr.shape[0] == LOGMEL_SHAPE[0], f"{path}: {arr.shape}"
        if path.parent.name in {"train", "test"}:
            assert arr.shape == LOGMEL_SHAPE, f"{path}: {arr.shape}"


def assert_ae_loader(npy_root: Path) -> None:
    loader = ae_dataset(
        str(npy_root),
        batch_size=2,
        n_cpu=0,
        target="pump",
        matrix_log_mode="auto",
    )
    assert len(loader) > 0, "AE DataLoader has no batches"
    batch = next(iter(loader))
    x = batch[0]
    assert x.ndim in {3, 4}, f"unexpected AE batch shape: {tuple(x.shape)}"
    assert np.isfinite(x.numpy()).all(), "AE batch contains non-finite values"


def assert_prepare_feature_shape(wav_root: Path, work_dir: Path) -> None:
    out_root = work_dir / "prepared_logmel"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "prepare_dataset_features.py"),
            "--root",
            str(wav_root),
            "--machines",
            "pump",
            "fan",
            "--skip-augment",
            "--exclude-aug",
            "--features",
            "logmel",
            "--logmel-mode",
            "raw",
            "--mel-n-fft",
            "1024",
            "--mel-hop",
            "512",
            "--feature-duration",
            "10.0",
            "--logmel-out",
            str(out_root),
            "--n-mels",
            "128",
            "--overwrite",
        ],
        check=True,
    )
    sample = next(out_root.glob("pump/train/*.npy"))
    arr = np.load(sample)
    assert arr.shape == (128, 313), f"{sample}: {arr.shape}"


def assert_contrastive_pairs(npy_root: Path) -> None:
    train_list, aug_list, other_train, other_aug = dataloader._collect_npy_train_aug_lists(str(npy_root), "pump")
    pairs = dataloader._build_aug_pairs(
        list(train_list) + list(other_train),
        list(aug_list) + list(other_aug),
    )
    assert len(pairs) == 4, f"expected 4 contrastive pairs, got {len(pairs)}"

    loader = dataloader.class_data_contrastive(
        str(npy_root),
        batch_size=4,
        n_cpu=0,
        target="pump",
        class_list=MACHINES,
        snr_list=[0],
        matrix_log_mode="auto",
    )
    assert len(loader) > 0, "contrastive DataLoader has no batches"
    view0, view1 = next(iter(loader))
    expected_shape = (1, LOGMEL_SHAPE[0], CONTRASTIVE_SEGMENT_FRAMES)
    assert tuple(view0.shape[1:]) == expected_shape, tuple(view0.shape)
    assert tuple(view1.shape[1:]) == expected_shape, tuple(view1.shape)
    assert np.isfinite(view0.numpy()).all(), "contrastive view0 contains non-finite values"
    assert np.isfinite(view1.numpy()).all(), "contrastive view1 contains non-finite values"
    assert float(view0.max()) <= 0.0, "raw mel-power was not converted to legacy log scale"


def assert_separation_pairs(wav_root: Path) -> None:
    loader = dataloader.class_data_sep(
        str(wav_root),
        batch_size=2,
        n_cpu=0,
        target="pump",
        class_list=MACHINES,
        snr_list=[0],
        input_len=32000,
    )
    assert len(loader) > 0, "separation DataLoader has no batches"
    batch = next(iter(loader))
    assert len(batch) == 7, f"unexpected separation batch fields: {len(batch)}"


def assert_classifier_loader(wav_root: Path) -> None:
    loader = dataloader.class_data_arcface(
        str(wav_root),
        batch_size=2,
        n_cpu=0,
        target="pump",
        class_list=MACHINES,
        snr_list=[0],
    )
    assert len(loader) > 0, "classifier DataLoader has no batches"
    batch = next(iter(loader))
    assert len(batch) == 2, f"unexpected classifier batch fields: {len(batch)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataloader smoke checks on synthetic toy data.")
    parser.add_argument("--work-dir", type=Path, default=None, help="optional directory for generated toy data")
    parser.add_argument("--keep", action="store_true", help="keep generated toy data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="dcase_dataloader_smoke_"))
        cleanup = not args.keep
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False

    try:
        wav_root, npy_root = create_toy_roots(work_dir)
        assert_file_counts(wav_root, npy_root)
        assert_ae_npy_shape(npy_root)
        assert_ae_loader(npy_root)
        assert_prepare_feature_shape(wav_root, work_dir)
        assert_contrastive_pairs(npy_root)
        assert_separation_pairs(wav_root)
        assert_classifier_loader(wav_root)
        print("dataloader smoke passed")
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

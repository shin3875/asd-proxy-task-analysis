#!/usr/bin/env python3
"""aug_maker.py-compatible 33x wav augmentation + file-wise npy conversion.

Expected input:
    ./asd_dataset/<machine>/train/*.wav
    ./asd_dataset/<machine>/test/*.wav

Generated augmentation wavs:
    ./asd_dataset/<machine>/aug/*.wav

Generated features, one npy per wav:
    ./asd_dataset_np/<machine>/train/<stem>.npy
    ./asd_dataset_np/<machine>/test/<stem>.npy
    ./asd_dataset_np/<machine>/aug/<aug_stem>.npy

    ./asd_dataset_logmel/<machine>/train/<stem>.npy
    ./asd_dataset_logmel/<machine>/test/<stem>.npy
    ./asd_dataset_logmel/<machine>/aug/<aug_stem>.npy

The augmentation naming and parameters match the uploaded aug_maker.py logic:
    np-3.0_, np-6.0_, ..., np-18.0_          6 negative pitch shifts
    pp3.0_, pp6.0_, ..., pp18.0_             6 positive pitch shifts
    wn15_, wn12_, wn9_, wn6_, wn3_           5 white-noise variants
    ts0.8_, ts0.85_, ts0.9_, ts0.95_         4 time-stretch variants
    tm0_..tm5_ and fm0_..fm5_                12 STFT-mask variants

Total: 33 augmented wavs per original train wav.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Iterator, Optional, Sequence

import librosa
import numpy as np
import soundfile as sf


RATE_N = [-3.0, -6.0, -9.0, -12.0, -15.0, -18.0]
RATE_P = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
SNR_LIST = [15, 12, 9, 6, 3]
TS_LIST = [0.8, 0.85, 0.9, 0.95]
AUG_PER_TRAIN = 33
DEFAULT_MACHINES = [
    "bearing",
    "fan",
    "gearbox",
    "pump",
    "slider",
    "ToyCar",
    "ToyConveyor",
    "ToyTrain",
    "valve",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_audio(path: Path, sr: int, duration: Optional[float] = None) -> tuple[np.ndarray, int]:
    y, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    y = y.astype(np.float32, copy=False)
    if duration is not None and duration > 0:
        target = int(round(duration * loaded_sr))
        if len(y) > target:
            y = y[:target]
        elif len(y) < target:
            y = librosa.util.fix_length(y, size=target)
    return y, loaded_sr


def write_wav(path: Path, y: np.ndarray, sr: int, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    ensure_dir(path.parent)
    sf.write(str(path), np.asarray(y), sr)
    return True


def time_masking(in_spec: np.ndarray) -> np.ndarray:
    """Apply legacy STFT-domain time masking in-place."""
    if in_spec.shape[1] <= 6:
        return in_spec
    time_list = np.arange(3, in_spec.shape[1] - 3, 1)
    if len(time_list) == 0:
        return in_spec
    for _ in range(10):
        time_bin = np.random.choice(time_list)
        in_spec[:, time_bin - 2] = in_spec[:, time_bin - 2] * 0.5
        in_spec[:, time_bin - 1] = in_spec[:, time_bin - 1] * 0.4
        in_spec[:, time_bin] = in_spec[:, time_bin] * 0.3
        in_spec[:, time_bin + 1] = in_spec[:, time_bin + 1] * 0.4
        in_spec[:, time_bin + 2] = in_spec[:, time_bin + 2] * 0.5
    return in_spec


def fr_masking(in_spec: np.ndarray) -> np.ndarray:
    """Apply legacy STFT-domain frequency masking in-place."""
    if in_spec.shape[0] <= 4:
        return in_spec
    freq_list = np.arange(2, in_spec.shape[0] - 2, 1)
    if len(freq_list) == 0:
        return in_spec
    for _ in range(7):
        freq_bin = np.random.choice(freq_list)
        in_spec[freq_bin - 1, :] = in_spec[freq_bin - 1, :] * 0.5
        in_spec[freq_bin, :] = in_spec[freq_bin, :] * 0.3
        in_spec[freq_bin + 1:, :] = in_spec[freq_bin + 1, :] * 0.5
    return in_spec


def istft_compat(spec: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    try:
        return librosa.istft(spec, n_fft=n_fft, hop_length=hop_length)
    except TypeError:
        return librosa.istft(spec, hop_length=hop_length)


def make_33_aug_for_file(wav_path: Path, aug_dir: Path, sr: int, n_fft: int, hop_length: int, overwrite: bool) -> int:
    loaded, loaded_sr = load_audio(wav_path, sr=sr, duration=None)
    file_name = wav_path.name
    written = 0

    # pitch shift: paired order, same as uploaded aug_maker.py
    for idx in range(len(RATE_N)):
        loaded_np = librosa.effects.pitch_shift(
            loaded, sr=loaded_sr, bins_per_octave=24, n_steps=RATE_N[idx]
        )
        loaded_pp = librosa.effects.pitch_shift(
            loaded, sr=loaded_sr, bins_per_octave=24, n_steps=RATE_P[idx]
        )
        written += int(write_wav(aug_dir / f"np{RATE_N[idx]}_{file_name}", loaded_np, loaded_sr, overwrite))
        written += int(write_wav(aug_dir / f"pp{RATE_P[idx]}_{file_name}", loaded_pp, loaded_sr, overwrite))

    # white noise
    for snr in SNR_LIST:
        signal_v = max(float(np.mean(np.abs(loaded))), np.finfo(np.float32).tiny)
        signal_db = 10.0 * np.log10(signal_v)
        noise_db = signal_db - snr
        noise_v = 10 ** (noise_db / 10)
        noise = np.random.normal(0, np.sqrt(noise_v), len(loaded))
        loaded_wn = loaded + noise
        if np.max(loaded_wn) > 1.0:
            loaded_wn = librosa.util.normalize(loaded_wn)
        written += int(write_wav(aug_dir / f"wn{snr}_{file_name}", loaded_wn, loaded_sr, overwrite))

    # time stretch: no crop/pad, same as uploaded aug_maker.py
    for tsr in TS_LIST:
        loaded_ts = librosa.effects.time_stretch(loaded, rate=tsr)
        written += int(write_wav(aug_dir / f"ts{tsr}_{file_name}", loaded_ts, loaded_sr, overwrite))

    # STFT-domain masking. Preserve original in-place/cumulative behavior.
    ori_feature = librosa.stft(loaded, n_fft=n_fft, hop_length=hop_length)
    for mask_idx in range(6):
        tm_feature = time_masking(ori_feature)
        fm_feature = fr_masking(ori_feature)

        loaded_tm = istft_compat(tm_feature, n_fft=n_fft, hop_length=hop_length)
        loaded_fm = istft_compat(fm_feature, n_fft=n_fft, hop_length=hop_length)

        written += int(write_wav(aug_dir / f"tm{mask_idx}_{file_name}", loaded_tm, loaded_sr, overwrite))
        written += int(write_wav(aug_dir / f"fm{mask_idx}_{file_name}", loaded_fm, loaded_sr, overwrite))

    return written


def expected_aug_names(file_name: str) -> list[str]:
    names: list[str] = []
    for idx in range(len(RATE_N)):
        names.append(f"np{RATE_N[idx]}_{file_name}")
        names.append(f"pp{RATE_P[idx]}_{file_name}")
    names.extend([f"wn{snr}_{file_name}" for snr in SNR_LIST])
    names.extend([f"ts{tsr}_{file_name}" for tsr in TS_LIST])
    for idx in range(6):
        names.append(f"tm{idx}_{file_name}")
        names.append(f"fm{idx}_{file_name}")
    return names


def clean_aug(root: Path, machines: Sequence[str], dry_run: bool) -> None:
    for machine in machines:
        aug_dir = root / machine / "aug"
        wavs = sorted(aug_dir.glob("*.wav")) if aug_dir.exists() else []
        print(f"[CLEAN-AUG] {machine}: {len(wavs)} wav files in {aug_dir}")
        if not dry_run:
            for wav in wavs:
                wav.unlink()


def clean_npy(out_roots: Sequence[Path], machines: Sequence[str], dry_run: bool) -> None:
    for out_root in out_roots:
        for machine in machines:
            machine_dir = out_root / machine
            count = len(list(machine_dir.rglob("*.npy"))) if machine_dir.exists() else 0
            print(f"[CLEAN-NPY] {machine_dir}: {count} npy files")
            if not dry_run and machine_dir.exists():
                shutil.rmtree(machine_dir)


def make_augmentations(root: Path, machines: Sequence[str], sr: int, n_fft: int, hop_length: int, overwrite: bool, dry_run: bool) -> None:
    for machine in machines:
        train_dir = root / machine / "train"
        aug_dir = root / machine / "aug"
        train_wavs = sorted(train_dir.glob("*.wav"))
        print(f"[AUG33] {machine}: train={len(train_wavs)}, expected_aug={len(train_wavs) * AUG_PER_TRAIN}, out={aug_dir}")
        if dry_run:
            continue
        ensure_dir(aug_dir)
        written = 0
        for idx, wav_path in enumerate(train_wavs, 1):
            written += make_33_aug_for_file(
                wav_path=wav_path,
                aug_dir=aug_dir,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                overwrite=overwrite,
            )
            if idx % 50 == 0 or idx == len(train_wavs):
                print(f"  {machine}: {idx}/{len(train_wavs)} train wavs processed, written/overwritten={written}")


def wav_label(path: Path, normal_label: int) -> int:
    name = path.name.lower()
    if "normal" in name and "anomaly" not in name:
        return normal_label
    if "anomaly" in name:
        return 1 - normal_label
    return -1


def iter_wavs(root: Path, machine: str, include_aug: bool) -> Iterator[tuple[str, Path]]:
    for split in ("train", "test"):
        split_dir = root / machine / split
        if split_dir.exists():
            for wav in sorted(split_dir.glob("*.wav")):
                yield split, wav
    if include_aug:
        aug_dir = root / machine / "aug"
        if aug_dir.exists():
            for wav in sorted(aug_dir.glob("*.wav")):
                yield "aug", wav


def extract_feature(
    wav_path: Path,
    feature: str,
    sr: int,
    duration: Optional[float],
    stft_n_fft: int,
    stft_hop: int,
    stft_format: str,
    mel_n_fft: int,
    mel_hop: int,
    n_mels: int,
    logmel_mode: str,
    add_batch_axis: bool,
) -> np.ndarray:
    y, loaded_sr = load_audio(wav_path, sr=sr, duration=duration)

    if feature == "stft":
        spec = librosa.stft(y, n_fft=stft_n_fft, hop_length=stft_hop)
        if stft_format == "magnitude":
            arr = np.abs(spec).astype(np.float32)
        elif stft_format == "power":
            arr = (np.abs(spec) ** 2).astype(np.float32)
        elif stft_format == "complex":
            arr = spec.astype(np.complex64)
        elif stft_format == "realimag":
            arr = np.stack([spec.real, spec.imag], axis=0).astype(np.float32)
            return arr
        else:
            raise ValueError(f"Unsupported stft_format: {stft_format}")
    elif feature == "logmel":
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=loaded_sr,
            n_fft=mel_n_fft,
            hop_length=mel_hop,
            n_mels=n_mels,
            power=2.0,
        )
        if logmel_mode == "raw":
            arr = mel.astype(np.float32)
        elif logmel_mode == "db":
            arr = (10.0 * np.log10(np.maximum(mel, sys.float_info.epsilon))).astype(np.float32)
        elif logmel_mode == "db_refmax":
            arr = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        elif logmel_mode == "log1p":
            arr = np.log1p(mel).astype(np.float32)
        else:
            raise ValueError(f"Unsupported logmel_mode: {logmel_mode}")
    else:
        raise ValueError(f"Unsupported feature: {feature}")

    if add_batch_axis:
        arr = arr[np.newaxis, ...]
    return arr


def convert_to_npy(
    root: Path,
    machines: Sequence[str],
    feature: str,
    out_root: Path,
    include_aug: bool,
    sr: int,
    duration: Optional[float],
    stft_n_fft: int,
    stft_hop: int,
    stft_format: str,
    mel_n_fft: int,
    mel_hop: int,
    n_mels: int,
    logmel_mode: str,
    add_batch_axis: bool,
    normal_label: int,
    overwrite: bool,
    dry_run: bool,
    write_manifest: bool,
) -> None:
    rows: list[dict[str, str | int]] = []
    total_seen = 0
    total_written = 0

    for machine in machines:
        items = list(iter_wavs(root, machine, include_aug=include_aug))
        counts = {"train": 0, "test": 0, "aug": 0}
        for split, _ in items:
            counts[split] += 1
        print(f"[NPY:{feature}] {machine}: train={counts['train']}, test={counts['test']}, aug={counts['aug']}, out={out_root / machine}")

        for idx, (split, wav_path) in enumerate(items, 1):
            out_path = out_root / wav_path.relative_to(root).with_suffix(".npy")
            rows.append(
                {
                    "machine": machine,
                    "split": split,
                    "is_augmented": int(split == "aug"),
                    "label": wav_label(wav_path, normal_label=normal_label),
                    "wav_path": wav_path.as_posix(),
                    "npy_path": out_path.as_posix(),
                }
            )
            total_seen += 1

            if dry_run:
                continue
            if out_path.exists() and not overwrite:
                continue

            ensure_dir(out_path.parent)
            arr = extract_feature(
                wav_path=wav_path,
                feature=feature,
                sr=sr,
                duration=duration,
                stft_n_fft=stft_n_fft,
                stft_hop=stft_hop,
                stft_format=stft_format,
                mel_n_fft=mel_n_fft,
                mel_hop=mel_hop,
                n_mels=n_mels,
                logmel_mode=logmel_mode,
                add_batch_axis=add_batch_axis,
            )
            np.save(out_path, arr)
            total_written += 1

            if idx % 500 == 0 or idx == len(items):
                print(f"  {machine}: {idx}/{len(items)} wavs converted")

    if write_manifest and not dry_run:
        ensure_dir(out_root)
        manifest_path = out_root / f"manifest_{feature}.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["machine", "split", "is_augmented", "label", "wav_path", "npy_path"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[MANIFEST:{feature}] {manifest_path} rows={len(rows)}")

    print(f"[SUMMARY:{feature}] wav_seen={total_seen}, npy_written_or_overwritten={total_written}, skipped_existing={0 if dry_run else total_seen - total_written}")


def _expected_npy_paths(root: Path, machine: str, include_aug: bool) -> set[Path]:
    return {wav.relative_to(root).with_suffix(".npy") for _, wav in iter_wavs(root, machine, include_aug=include_aug)}


def _actual_npy_paths(out_root: Path, machine: str, include_aug: bool) -> set[Path]:
    splits = ["train", "test"] + (["aug"] if include_aug else [])
    paths: set[Path] = set()
    for split in splits:
        split_dir = out_root / machine / split
        if split_dir.exists():
            paths.update(p.relative_to(out_root) for p in split_dir.glob("*.npy"))
    return paths


def _sample_actual_paths_by_split(actual_paths: set[Path], out_root: Path, sample_limit: int) -> list[Path]:
    limit = max(int(sample_limit), 0)
    if limit == 0:
        return []

    selected: list[Path] = []
    for split in ("train", "test", "aug"):
        split_paths = sorted(
            p for p in actual_paths
            if len(p.parts) >= 2 and p.parts[1].lower() == split
        )
        selected.extend(out_root / p for p in split_paths[:limit])
    return selected


def _logmel_shape(arr: np.ndarray) -> tuple[int, ...]:
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return tuple(arr.shape)


def verify_npy_details(
    root: Path,
    out_root: Path,
    machine: str,
    feature: str,
    include_aug: bool,
    verify_shape: bool,
    expected_logmel_frames: int,
    expected_n_mels: int,
    sample_limit: int,
) -> int:
    failures = 0
    expected_paths = _expected_npy_paths(root, machine, include_aug=include_aug)
    actual_paths = _actual_npy_paths(out_root, machine, include_aug=include_aug)

    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    if missing:
        failures += len(missing)
        print(f"    MISSING npy examples: {[p.as_posix() for p in missing[:5]]}")
    if extra:
        failures += len(extra)
        print(f"    EXTRA npy examples: {[p.as_posix() for p in extra[:5]]}")

    if not verify_shape:
        return failures

    sample_paths = _sample_actual_paths_by_split(actual_paths, out_root, sample_limit)
    for npy_path in sample_paths:
        try:
            arr = np.load(npy_path)
        except Exception as exc:
            failures += 1
            print(f"    LOAD FAILED {npy_path}: {exc}")
            continue

        if not np.isfinite(arr).all():
            failures += 1
            print(f"    NON-FINITE {npy_path}")

        if feature == "logmel":
            shape = _logmel_shape(arr)
            expected_shape = (int(expected_n_mels), int(expected_logmel_frames))
            if shape != expected_shape:
                failures += 1
                print(f"    SHAPE MISMATCH {npy_path}: {shape}, expected={expected_shape}")

    return failures


def verify(
    root: Path,
    out_roots: dict[str, Path],
    machines: Sequence[str],
    include_aug: bool,
    require_aug: bool,
    verify_shape: bool,
    expected_logmel_frames: int,
    expected_n_mels: int,
    sample_limit: int,
) -> tuple[int, int]:
    print("[VERIFY]")
    issues = 0
    fatal_aug_issues = 0
    for machine in machines:
        train_wavs = sorted((root / machine / "train").glob("*.wav"))
        test_wavs = sorted((root / machine / "test").glob("*.wav"))
        aug_dir = root / machine / "aug"
        aug_wavs = sorted(aug_dir.glob("*.wav")) if aug_dir.exists() else []
        expected_aug = len(train_wavs) * AUG_PER_TRAIN
        if require_aug:
            aug_status = "OK" if len(aug_wavs) == expected_aug else f"MISMATCH expected={expected_aug}"
            if len(aug_wavs) != expected_aug:
                fatal_aug_issues += 1
        else:
            aug_status = f"observed; expected_legacy33={expected_aug}" if aug_wavs else f"not required; expected_legacy33={expected_aug}"
        used_total = len(train_wavs) + len(test_wavs) + (len(aug_wavs) if include_aug else 0)
        print(f"  wav {machine}: train={len(train_wavs)}, test={len(test_wavs)}, aug={len(aug_wavs)} [{aug_status}] used_total={used_total}")

        if include_aug and (require_aug or aug_wavs):
            missing_examples: list[str] = []
            aug_name_set = {p.name for p in aug_wavs}
            for train_wav in train_wavs:
                missing = [name for name in expected_aug_names(train_wav.name) if name not in aug_name_set]
                if missing:
                    missing_examples.append(f"{train_wav.name}: {missing[:5]}")
                    if len(missing_examples) >= 3:
                        break
            if missing_examples:
                if require_aug:
                    fatal_aug_issues += 1
                print("  legacy33 filename check: MISSING examples")
                for item in missing_examples:
                    print(f"    {item}")
            else:
                print("  legacy33 filename check: OK")

        for feature, out_root in out_roots.items():
            counts = {}
            for split in ("train", "test", "aug"):
                d = out_root / machine / split
                counts[split] = len(list(d.glob("*.npy"))) if d.exists() else 0
            npy_total = counts["train"] + counts["test"] + (counts["aug"] if include_aug else 0)
            status = "OK" if npy_total == used_total else f"MISMATCH expected_total={used_total}"
            print(f"  {feature} {machine}: train={counts['train']}, test={counts['test']}, aug={counts['aug']}, total_used={npy_total} [{status}]")
            issues += verify_npy_details(
                root=root,
                out_root=out_root,
                machine=machine,
                feature=feature,
                include_aug=include_aug,
                verify_shape=verify_shape,
                expected_logmel_frames=expected_logmel_frames,
                expected_n_mels=expected_n_mels,
                sample_limit=sample_limit,
            )

    if issues:
        print(f"[VERIFY] reported {issues} npy issue(s).")
    return issues, fatal_aug_issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create legacy 33x train augmentations and mirrored file-wise npy features.")
    parser.add_argument("--root", default=os.environ.get("ASD_DATASET_ROOT", "./asd_dataset"), help="dataset root containing <machine>/train and <machine>/test")
    parser.add_argument("--machines", nargs="+", default=DEFAULT_MACHINES, help="machine folder names to process")

    parser.add_argument("--skip-augment", action="store_true", help="do not create aug wavs; only convert existing wavs to npy")
    parser.add_argument("--skip-npy", action="store_true", help="only create aug wavs; do not convert wavs to npy")
    parser.add_argument("--exclude-aug", action="store_true", help="do not convert <machine>/aug wavs to npy")
    parser.add_argument("--clean-aug", action="store_true", help="delete existing <machine>/aug/*.wav before augmentation")
    parser.add_argument("--clean-npy", action="store_true", help="delete selected machine npy output folders before conversion")

    parser.add_argument("--features", nargs="+", choices=["stft", "logmel"], default=["logmel"])
    parser.add_argument("--stft-out", default=os.environ.get("ASD_STFT_ROOT", "./asd_dataset_np"))
    parser.add_argument("--logmel-out", default=os.environ.get("ASD_LOGMEL_ROOT", "./asd_dataset_logmel"))

    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument(
        "--feature-duration",
        type=float,
        default=10.0,
        help="crop/pad wav before npy extraction; 10.0 matches DCASE 10 s inputs at 16 kHz and 0 keeps full length",
    )
    parser.add_argument("--aug-n-fft", type=int, default=512, help="n_fft used for legacy tm/fm wav augmentation")
    parser.add_argument("--aug-hop", type=int, default=128, help="hop_length used for legacy tm/fm wav augmentation")

    parser.add_argument("--stft-n-fft", type=int, default=512,
                        help="STFT npy export n_fft. Legacy waveform separation does not consume this output.")
    parser.add_argument("--stft-hop", type=int, default=128,
                        help="STFT npy export hop_length. Use 200 with --stft-format realimag for separation-like exploratory exports.")
    parser.add_argument("--stft-format", choices=["magnitude", "power", "complex", "realimag"], default="magnitude")
    parser.add_argument("--mel-n-fft", type=int, default=1024)
    parser.add_argument("--mel-hop", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument(
        "--logmel-mode",
        choices=["raw", "db", "db_refmax", "log1p"],
        default="raw",
        help=(
            "raw stores mel-power for matrix_log_mode=auto loaders; "
            "db stores absolute 10*log10(mel), matching legacy evaluators; "
            "db_refmax stores per-file max-normalized librosa.power_to_db output"
        ),
    )

    parser.add_argument("--add-batch-axis", action="store_true", help="save each feature as (1, F, T) instead of (F, T); realimag remains (2, F, T)")
    parser.add_argument("--normal-label", type=int, choices=[0, 1], default=1, help="normal label; anomaly uses 1-normal_label")
    parser.add_argument("--seed", type=int, default=42, help="global numpy seed for noise/masking, matching uploaded drafts")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing wav/npy outputs")
    parser.add_argument("--dry-run", action="store_true", help="print planned counts without writing files")
    parser.add_argument("--write-manifest", action="store_true", help="write manifest_<feature>.csv under each output root")
    parser.add_argument("--verify", action="store_true", help="verify wav/npy counts and legacy33 filename patterns")
    parser.add_argument("--require-aug", action="store_true", help="fail verification if legacy 33x augmentation files are missing")
    parser.add_argument("--verify-shape", action="store_true", help="also verify sampled npy shape and finite values")
    parser.add_argument("--expected-logmel-frames", type=int, default=313, help="expected logmel time frames when --verify-shape is set")
    parser.add_argument(
        "--verify-sample-limit",
        "--verify-sample-limit-per-split",
        dest="verify_sample_limit",
        type=int,
        default=50,
        help="maximum npy files per split/machine/feature to inspect for shape",
    )
    parser.add_argument("--strict-verify", action="store_true", help="fail if npy count, shape, or finite-value verification reports any issue")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    missing_dirs = []
    for machine in args.machines:
        if not (root / machine / "train").exists():
            missing_dirs.append(str(root / machine / "train"))
        if not (root / machine / "test").exists():
            missing_dirs.append(str(root / machine / "test"))
    if missing_dirs:
        raise FileNotFoundError("Missing expected directories: " + ", ".join(missing_dirs))

    np.random.seed(args.seed)

    out_roots: dict[str, Path] = {
        feature: Path(args.stft_out if feature == "stft" else args.logmel_out)
        for feature in args.features
    }

    if args.clean_aug:
        clean_aug(root=root, machines=args.machines, dry_run=args.dry_run)

    if args.clean_npy and not args.skip_npy:
        clean_npy(out_roots=list(out_roots.values()), machines=args.machines, dry_run=args.dry_run)

    if not args.skip_augment:
        make_augmentations(
            root=root,
            machines=args.machines,
            sr=args.sr,
            n_fft=args.aug_n_fft,
            hop_length=args.aug_hop,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    include_aug = not args.exclude_aug
    feature_duration = None if args.feature_duration == 0 else float(args.feature_duration)

    if not args.skip_npy:
        for feature, out_root in out_roots.items():
            convert_to_npy(
                root=root,
                machines=args.machines,
                feature=feature,
                out_root=out_root,
                include_aug=include_aug,
                sr=args.sr,
                duration=feature_duration,
                stft_n_fft=args.stft_n_fft,
                stft_hop=args.stft_hop,
                stft_format=args.stft_format,
                mel_n_fft=args.mel_n_fft,
                mel_hop=args.mel_hop,
                n_mels=args.n_mels,
                logmel_mode=args.logmel_mode,
                add_batch_axis=args.add_batch_axis,
                normal_label=args.normal_label,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                write_manifest=args.write_manifest,
            )

    if args.verify:
        issues, fatal_aug_issues = verify(
            root=root,
            out_roots=out_roots,
            machines=args.machines,
            include_aug=include_aug,
            require_aug=args.require_aug,
            verify_shape=args.verify_shape,
            expected_logmel_frames=args.expected_logmel_frames,
            expected_n_mels=args.n_mels,
            sample_limit=args.verify_sample_limit,
        )
        if args.require_aug and fatal_aug_issues:
            raise RuntimeError(f"Legacy augmentation verification failed with {fatal_aug_issues} issue(s).")
        if args.strict_verify and issues:
            raise RuntimeError(f"NPY verification failed with {issues} issue(s).")


if __name__ == "__main__":
    main()

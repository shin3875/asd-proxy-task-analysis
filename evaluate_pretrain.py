"""Evaluate BEAT, CED, and EAT pre-trained audio representations for ASD.

The legacy scripts in ``pre-train_backup/`` evaluated each pre-trained model in
separate files.  This public entry point keeps the common evaluation path in one
place:

1. collect train/test wav files for each target device,
2. extract one clip-level representation per wav,
3. fit covariance statistics on target-device train features,
4. score test normal/anomaly files with Mahalanobis distance,
5. train ASD linear probes unless --skip_linear is set,
6. write Mahalanobis and LP summary CSV files.

The script does not train proxy models; linear probes are ASD evaluation heads.
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import librosa
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


DEFAULT_DEVICES = ["pump", "ToyConveyor", "bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]
SECTION_RE = re.compile(r"section[_-]?(\d+)", re.IGNORECASE)
PRETRAIN_AS2M_MAP = {
    "eatlarge": 49.5,
    "eatbase": 48.9,
    "eatbaseepoch30pretrain": 48.9,
    "beatsiter3": 48.0,
    "beatsiter3plus": 48.6,
    "cedbase": 50.0,
    "cedmini": 49.0,
    "cedsmall": 49.6,
    "cedtiny": 48.1,
}


@dataclass(frozen=True)
class FileRecord:
    path: Path
    device: str
    split: str
    condition: str
    domain: str
    section: str


@dataclass
class FeatureExtractor:
    kind: str
    label: str
    device: torch.device
    model: object
    processor: Optional[object] = None

    def extract(self, wav: np.ndarray, sample_rate: int) -> torch.Tensor:
        if self.kind == "eat":
            return extract_eat(self.model, wav, sample_rate, self.device)
        if self.kind == "beat":
            return extract_beat(self.model, wav, self.device)
        if self.kind == "ced":
            return extract_ced(self.model, self.processor, wav, sample_rate, self.device)
        raise ValueError(f"Unsupported model kind: {self.kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BEAT/CED/EAT pre-trained representations on ASD data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", default=os.environ.get("ASD_DATASET_ROOT", "./asd_dataset"),
                        help="dataset root containing <device>/train and <device>/test")
    parser.add_argument("--save_dir", default="./pretrain_eval_results", help="directory for CSV/figure outputs")
    parser.add_argument("--models", nargs="+", default=["eat"], choices=["eat", "beat", "ced"],
                        help="one or more pre-trained models to evaluate")
    parser.add_argument("--devices", nargs="+", default=DEFAULT_DEVICES, help="target devices to evaluate")
    parser.add_argument("--target_device", default=None, help="single-device alias; overrides --devices when set")
    parser.add_argument("--sample_rate", type=int, default=16000, help="audio sample rate used for feature extraction")
    parser.add_argument("--seed", type=int, default=42, help="random seed for linear-probe training")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    parser.add_argument("--dry_run", action="store_true", help="only report per-device file counts; do not load models")
    parser.add_argument("--limit_train", type=int, default=None, help="optional train-file limit for smoke checks")
    parser.add_argument("--limit_test", type=int, default=None, help="optional test-file limit for smoke checks")
    parser.add_argument("--cov_eps", type=float, default=0.0,
                        help="optional diagonal covariance regularizer. Default 0 preserves the legacy pinv path.")
    parser.add_argument("--save_roc", action="store_true", help="save ROC curves in addition to CSV files")
    parser.add_argument("--skip_linear", action="store_true",
                        help="skip linear-probe ASD metrics and compute Mahalanobis metrics only")
    parser.add_argument("--linear_epochs", type=int, default=200, help="linear-probe training epochs")
    parser.add_argument("--linear_batch_size", type=int, default=64, help="linear-probe batch size")
    parser.add_argument("--linear_lr", type=float, default=1e-3, help="linear-probe learning rate")
    parser.add_argument("--linear_half_split", choices=["legacy", "per_section"], default="per_section",
                        help="split policy for in-domain half-split LP metrics")
    parser.add_argument("--pretrain_map_csv", default=None,
                        help="optional CSV with model,map or model,pretrain_map columns for proxy mAP values")

    parser.add_argument("--eat_model_id", default="worstchan/EAT-base_epoch30_pretrain",
                        help="Hugging Face model id for EAT")
    parser.add_argument("--ced_hf_model", default="mispeech/ced-small", help="Hugging Face model id for CED")
    parser.add_argument("--ced_model_name", default="CED-small", help="display label for CED outputs")
    parser.add_argument("--beat_checkpoint", default="./beats/BEATs_iter3.pt",
                        help="local BEATs checkpoint path")
    parser.add_argument("--beat_model_name", default="BEATs_iter3", help="display label for BEAT outputs")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                        help="allow Hugging Face remote model code, matching the legacy scripts")
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    return parser.parse_args()


def get_device(args: argparse.Namespace) -> torch.device:
    if args.cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_section(path: Path) -> str:
    match = SECTION_RE.search(path.name)
    if match is None:
        return "section_unknown"
    return f"section_{int(match.group(1)):02d}"


def infer_condition(path: Path) -> str:
    return "anomaly" if "anomaly" in path.name.lower() else "normal"


def infer_domain(path: Path) -> str:
    name = path.name.lower()
    if "source" in name:
        return "source"
    if "target" in name:
        return "target"
    return "unknown"


def collect_records(root: Path, device: str, split: str) -> list[FileRecord]:
    split_dir = root / device / split
    if not split_dir.is_dir():
        return []
    records = []
    for path in sorted(split_dir.glob("*.wav")):
        records.append(
            FileRecord(
                path=path,
                device=device,
                split=split,
                condition=infer_condition(path),
                domain=infer_domain(path),
                section=parse_section(path),
            )
        )
    return records


def collect_device_records(args: argparse.Namespace, device: str) -> tuple[list[FileRecord], list[FileRecord]]:
    root = Path(args.data_dir)
    train_records = collect_records(root, device, "train")
    test_records = collect_records(root, device, "test")
    if args.limit_train is not None:
        train_records = train_records[: args.limit_train]
    if args.limit_test is not None:
        test_records = test_records[: args.limit_test]
    return train_records, test_records


def load_extractor(model_kind: str, args: argparse.Namespace, device: torch.device) -> FeatureExtractor:
    if model_kind == "eat":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(args.eat_model_id, trust_remote_code=args.trust_remote_code)
        model.eval().to(device)
        return FeatureExtractor(kind="eat", label=Path(args.eat_model_id).name, device=device, model=model)

    if model_kind == "ced":
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        processor = AutoFeatureExtractor.from_pretrained(args.ced_hf_model, trust_remote_code=args.trust_remote_code)
        model = AutoModelForAudioClassification.from_pretrained(args.ced_hf_model, trust_remote_code=args.trust_remote_code)
        model.eval().to(device)
        return FeatureExtractor(kind="ced", label=args.ced_model_name, device=device, model=model, processor=processor)

    if model_kind == "beat":
        checkpoint_path = Path(args.beat_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "BEAT evaluation requires a local checkpoint. "
                "Pass --beat_checkpoint with the BEATs .pt file."
            )
        try:
            from beats.BEATs import BEATs, BEATsConfig
        except ImportError as exc:
            raise ImportError(
                "BEAT evaluation requires the upstream BEATs implementation "
                "to be importable as beats.BEATs."
            ) from exc

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = BEATs(BEATsConfig(checkpoint["cfg"]))
        model.load_state_dict(checkpoint["model"])
        model.eval().to(device)
        return FeatureExtractor(kind="beat", label=args.beat_model_name, device=device, model=model)

    raise ValueError(f"Unsupported model kind: {model_kind}")


def _as_single_row(feature: torch.Tensor) -> torch.Tensor:
    feature = feature.detach()
    if feature.dim() == 1:
        feature = feature.unsqueeze(0)
    elif feature.dim() > 2:
        feature = feature.reshape(feature.shape[0], -1)
    if feature.shape[0] != 1:
        feature = feature.mean(dim=0, keepdim=True)
    return feature.float()


def extract_eat(model, wav: np.ndarray, sample_rate: int, device: torch.device) -> torch.Tensor:
    import torchaudio

    waveform = torch.as_tensor(wav, dtype=torch.float32)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    ).unsqueeze(0)

    n_frames = fbank.shape[1]
    if n_frames < 1024:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, 1024 - n_frames))
    elif n_frames > 1024:
        fbank = fbank[:, :1024, :]

    fbank = (fbank - -4.268) / (4.569 * 2)
    fbank = fbank.unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.extract_features(fbank)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.dim() >= 3 and features.shape[1] > 1:
            features = features[:, 1:]
        features = features.mean(dim=1)
    return _as_single_row(features)


def extract_beat(model, wav: np.ndarray, device: torch.device) -> torch.Tensor:
    waveform = torch.as_tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)
    padding_mask = torch.zeros(1, waveform.shape[1], dtype=torch.bool, device=device)
    with torch.no_grad():
        features = model.extract_features(waveform, padding_mask=padding_mask)[0].mean(1)
    return _as_single_row(features)


def extract_ced(model, processor, wav: np.ndarray, sample_rate: int, device: torch.device) -> torch.Tensor:
    waveform = np.asarray(wav, dtype=np.float32)
    waveform = waveform - float(np.mean(waveform))
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # CED follows the legacy evaluator: classifier logits are the clip-level representation.
    return _as_single_row(outputs.logits)


def extract_features(extractor: FeatureExtractor, records: Sequence[FileRecord], args: argparse.Namespace) -> torch.Tensor:
    rows = []
    for index, record in enumerate(records, start=1):
        wav, _ = librosa.load(record.path, sr=args.sample_rate, mono=True)
        rows.append(extractor.extract(wav, args.sample_rate).to(extractor.device))
        if index % 50 == 0:
            logging.info("  extracted %d/%d files", index, len(records))
    if not rows:
        raise RuntimeError("No features were extracted.")
    return torch.cat(rows, dim=0)


def covariance_stats(features: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if features.shape[0] < 2:
        raise RuntimeError("At least two train files are required for covariance scoring.")
    mu = features.mean(dim=0)
    dev = features - mu
    cov = torch.einsum("bi,bj->ij", dev, dev) / (features.shape[0] - 1)
    if eps > 0:
        cov = cov + torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype) * eps
    return mu, torch.linalg.pinv(cov)


def mahalanobis_scores(features: torch.Tensor, mu: torch.Tensor, cov_inv: torch.Tensor) -> np.ndarray:
    dev = features - mu
    return ((dev @ cov_inv) * dev).sum(dim=1).detach().cpu().numpy()


def normalize_model_key(name: str) -> str:
    text = str(name).replace("+", "plus").lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def load_pretrain_map(path: Optional[str]) -> dict[str, float]:
    values = dict(PRETRAIN_AS2M_MAP)
    if not path:
        return values

    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row.get("model") or row.get("model_name") or row.get("model_id")
            map_value = row.get("pretrain_map") or row.get("mAP") or row.get("map")
            if not model_name or map_value in {None, ""}:
                continue
            try:
                values[normalize_model_key(model_name)] = float(map_value)
            except ValueError:
                logging.warning("Ignoring non-numeric pretrain mAP for %s: %s", model_name, map_value)
    return values


def lookup_pretrain_map(model_name: str, map_values: dict[str, float]) -> float:
    key = normalize_model_key(model_name)
    if key in map_values:
        return map_values[key]
    for alias, value in sorted(map_values.items(), key=lambda item: len(item[0]), reverse=True):
        if alias and alias in key:
            return value
    return math.nan


def metric_pair(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(scores) != len(labels):
        return math.nan, math.nan

    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return math.nan, math.nan

    try:
        auc = float(roc_auc_score(y_true=labels, y_score=scores))
        pauc = float(roc_auc_score(y_true=labels, y_score=scores, max_fpr=0.1))
        return auc, pauc
    except Exception:
        return math.nan, math.nan


def linear_oracle_eval(
    normal_test_features: torch.Tensor,
    anomaly_test_features: torch.Tensor,
    normal_train_features: torch.Tensor,
    anomaly_train_features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    feature_dim = int(normal_train_features.shape[1])
    classifier = torch.nn.Linear(feature_dim, 2).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    features = torch.cat([normal_train_features.detach(), anomaly_train_features.detach()], dim=0)
    labels = torch.cat(
        [
            torch.zeros(len(normal_train_features), dtype=torch.long),
            torch.ones(len(anomaly_train_features), dtype=torch.long),
        ],
        dim=0,
    )
    generator = None
    if seed >= 0:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    best_state = copy.deepcopy(classifier.state_dict())
    best_accuracy = -1.0
    classifier.train()
    for _ in range(max(int(epochs), 0)):
        correct = 0
        total = 0
        for inputs, target in loader:
            inputs = inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = classifier(inputs)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            correct += int((logits.argmax(dim=1) == target).sum().item())
            total += int(target.numel())
        accuracy = correct / max(total, 1)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(classifier.state_dict())

    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        normal_logits = classifier(normal_test_features.to(device)).detach().cpu()
        anomaly_logits = classifier(anomaly_test_features.to(device)).detach().cpu()
    normal_scores = torch.softmax(normal_logits, dim=1).numpy()[:, 1]
    anomaly_scores = torch.softmax(anomaly_logits, dim=1).numpy()[:, 1]
    return normal_scores, anomaly_scores


def concat_feature_items(items: Sequence[tuple[FileRecord, torch.Tensor]]) -> torch.Tensor:
    if not items:
        raise RuntimeError("No feature items to concatenate.")
    return torch.cat([item[1] for item in items], dim=0)


def section_sort_key(section: str) -> tuple[int, str]:
    match = SECTION_RE.search(str(section))
    if match is None:
        return (10_000, str(section))
    return (int(match.group(1)), str(section))


def run_linear_probes(
    records: Sequence[FileRecord],
    features: torch.Tensor,
    args: argparse.Namespace,
    model_label: str,
    device_name: str,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    section_items: dict[str, dict[str, list[tuple[FileRecord, torch.Tensor]]]] = {}
    for index, record in enumerate(records):
        condition = "anomaly" if record.condition == "anomaly" else "normal"
        section_items.setdefault(record.section, {}).setdefault(condition, []).append(
            (record, features[index : index + 1].detach())
        )

    metrics: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    sections = sorted(section_items.keys(), key=section_sort_key)
    valid_sections = [
        section for section in sections
        if section_items[section].get("normal") and section_items[section].get("anomaly")
    ]
    if len(valid_sections) < 2:
        logging.warning("[%s/%s] not enough sections for LP metrics: %s", model_label, device_name, valid_sections)
        return metrics, rows

    loso_normal_scores: list[float] = []
    loso_anomaly_scores: list[float] = []
    for section in valid_sections:
        test_normal = section_items[section]["normal"]
        test_anomaly = section_items[section]["anomaly"]
        train_normal = [item for sec in valid_sections if sec != section for item in section_items[sec]["normal"]]
        train_anomaly = [item for sec in valid_sections if sec != section for item in section_items[sec]["anomaly"]]
        if not train_normal or not train_anomaly:
            continue

        normal_scores, anomaly_scores = linear_oracle_eval(
            concat_feature_items(test_normal),
            concat_feature_items(test_anomaly),
            concat_feature_items(train_normal),
            concat_feature_items(train_anomaly),
            device=device,
            batch_size=args.linear_batch_size,
            epochs=args.linear_epochs,
            lr=args.linear_lr,
            seed=args.seed,
        )
        loso_normal_scores.extend(normal_scores.tolist())
        loso_anomaly_scores.extend(anomaly_scores.tolist())
        fold_scores = np.concatenate([normal_scores, anomaly_scores])
        fold_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
        metrics[f"linear_loso_auc_{section}"], metrics[f"linear_loso_pauc_{section}"] = metric_pair(fold_scores, fold_labels)

        for record, score in zip([item[0] for item in test_normal], normal_scores):
            rows.append(linear_score_row(model_label, device_name, "linear_loso", section, record, score))
        for record, score in zip([item[0] for item in test_anomaly], anomaly_scores):
            rows.append(linear_score_row(model_label, device_name, "linear_loso", section, record, score))

    loso_scores = np.concatenate([np.asarray(loso_normal_scores), np.asarray(loso_anomaly_scores)])
    loso_labels = np.concatenate([np.zeros(len(loso_normal_scores)), np.ones(len(loso_anomaly_scores))])
    metrics["linear_loso_auc"], metrics["linear_loso_pauc"] = metric_pair(loso_scores, loso_labels)

    all_normal = [item for section in valid_sections for item in section_items[section]["normal"]]
    all_anomaly = [item for section in valid_sections for item in section_items[section]["anomaly"]]
    if all_normal and all_anomaly:
        normal_scores, anomaly_scores = linear_oracle_eval(
            concat_feature_items(all_normal),
            concat_feature_items(all_anomaly),
            concat_feature_items(all_normal),
            concat_feature_items(all_anomaly),
            device=device,
            batch_size=args.linear_batch_size,
            epochs=args.linear_epochs,
            lr=args.linear_lr,
            seed=args.seed,
        )
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
        metrics["linear_all_auc"], metrics["linear_all_pauc"] = metric_pair(all_scores, all_labels)
        for record, score in zip([item[0] for item in all_normal], normal_scores):
            rows.append(linear_score_row(model_label, device_name, "linear_all", "all", record, score))
        for record, score in zip([item[0] for item in all_anomaly], anomaly_scores):
            rows.append(linear_score_row(model_label, device_name, "linear_all", "all", record, score))

    half_test_normal: list[tuple[FileRecord, torch.Tensor]] = []
    half_test_anomaly: list[tuple[FileRecord, torch.Tensor]] = []
    half_train_normal: list[tuple[FileRecord, torch.Tensor]] = []
    half_train_anomaly: list[tuple[FileRecord, torch.Tensor]] = []
    reference_half = int(len(section_items[valid_sections[0]]["anomaly"]) / 2)
    for section in valid_sections:
        normal_items = section_items[section]["normal"]
        anomaly_items = section_items[section]["anomaly"]
        if args.linear_half_split == "legacy":
            n_half = reference_half
            a_half = reference_half
        else:
            n_half = int(len(normal_items) / 2)
            a_half = int(len(anomaly_items) / 2)
        if n_half <= 0 or a_half <= 0 or len(normal_items) - n_half < 1 or len(anomaly_items) - a_half < 1:
            continue
        half_test_normal.extend(normal_items[:n_half])
        half_test_anomaly.extend(anomaly_items[:a_half])
        half_train_normal.extend(normal_items[n_half:])
        half_train_anomaly.extend(anomaly_items[a_half:])

    if half_test_normal and half_test_anomaly and half_train_normal and half_train_anomaly:
        normal_scores, anomaly_scores = linear_oracle_eval(
            concat_feature_items(half_test_normal),
            concat_feature_items(half_test_anomaly),
            concat_feature_items(half_train_normal),
            concat_feature_items(half_train_anomaly),
            device=device,
            batch_size=args.linear_batch_size,
            epochs=args.linear_epochs,
            lr=args.linear_lr,
            seed=args.seed,
        )
        half_scores = np.concatenate([normal_scores, anomaly_scores])
        half_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
        metrics["linear_half_auc"], metrics["linear_half_pauc"] = metric_pair(half_scores, half_labels)
        for record, score in zip([item[0] for item in half_test_normal], normal_scores):
            rows.append(linear_score_row(model_label, device_name, "linear_half", "half", record, score))
        for record, score in zip([item[0] for item in half_test_anomaly], anomaly_scores):
            rows.append(linear_score_row(model_label, device_name, "linear_half", "half", record, score))

    return metrics, rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_name(name: str) -> str:
    text = str(name).replace("+", "plus")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "model"


def linear_metric_fields() -> list[str]:
    return [
        "linear_all_auc",
        "linear_all_pauc",
        "linear_half_auc",
        "linear_half_pauc",
        "linear_loso_auc",
        "linear_loso_pauc",
    ]


def write_plotter_compatible_summary(save_root: Path, summary_rows: Sequence[dict], map_values: dict[str, float]) -> None:
    rows = []
    for row in summary_rows:
        if str(row.get("scope", "")).lower() != "overall" or str(row.get("group", "")).lower() != "all":
            continue
        model_name = str(row["model"])
        pretrain_map = lookup_pretrain_map(model_name, map_values)
        linear_values = {field: row.get(field, math.nan) for field in linear_metric_fields()}
        rows.append(
            {
                "model_id": safe_name(model_name),
                "model_name": model_name,
                "model_type": "pretrained",
                "task": "pretrained",
                "target_device": row["device"],
                "scope": row["scope"],
                "group": row["group"],
                "score_type": row["score_type"],
                "mah_train_auc": row["auc"],
                "mah_train_pauc": row["pauc"],
                **linear_values,
                "pretrain_map": pretrain_map,
                "mAP": pretrain_map,
                "pretrain_asd_auc": row["auc"],
                "pretrain_asd_pauc": row["pauc"],
            }
        )
    write_csv(
        save_root / "results_summary.csv",
        [
            "model_id",
            "model_name",
            "model_type",
            "task",
            "target_device",
            "scope",
            "group",
            "score_type",
            "mah_train_auc",
            "mah_train_pauc",
            *linear_metric_fields(),
            "pretrain_map",
            "mAP",
            "pretrain_asd_auc",
            "pretrain_asd_pauc",
        ],
        rows,
    )


def save_roc(path: Path, scores: np.ndarray, labels: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(scores) != len(labels):
        return
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true=labels, y_score=scores)
    auc, pauc = metric_pair(scores, labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title}\nAUC {auc:.4f}, pAUC {pauc:.4f}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def score_rows(model_label: str, records: Sequence[FileRecord], scores: np.ndarray) -> list[dict]:
    rows = []
    for record, score in zip(records, scores):
        rows.append(
            {
                "model": model_label,
                "device": record.device,
                "file_name": record.path.name,
                "condition": record.condition,
                "domain": record.domain,
                "section": record.section,
                "score_type": "mahalanobis_train",
                "score": float(score),
            }
        )
    return rows


def linear_score_row(model_label: str, device_name: str, probe: str, fold: str, record: FileRecord, score: float) -> dict:
    return {
        "model": model_label,
        "device": device_name,
        "probe": probe,
        "fold": fold,
        "file_name": record.path.name,
        "condition": record.condition,
        "domain": record.domain,
        "section": record.section,
        "score": float(score),
    }


def summarize_scores(
    model_label: str,
    device: str,
    records: Sequence[FileRecord],
    scores: np.ndarray,
    linear_metrics: Optional[dict[str, float]] = None,
) -> list[dict]:
    labels = np.asarray([1 if record.condition == "anomaly" else 0 for record in records], dtype=int)
    rows = []
    linear_metrics = linear_metrics or {}

    def add_row(scope: str, group: str, selected: np.ndarray) -> None:
        group_scores = scores[selected]
        group_labels = labels[selected]
        auc, pauc = metric_pair(group_scores, group_labels)
        row = {
            "model": model_label,
            "device": device,
            "scope": scope,
            "group": group,
            "score_type": "mahalanobis_train",
            "n_normal": int(np.sum(group_labels == 0)),
            "n_anomaly": int(np.sum(group_labels == 1)),
            "auc": auc,
            "pauc": pauc,
        }
        if scope == "overall" and group == "all":
            row.update({field: linear_metrics.get(field, math.nan) for field in linear_metric_fields()})
        rows.append(row)

    add_row("overall", "all", np.ones(len(records), dtype=bool))
    for section in sorted({record.section for record in records}):
        add_row("section", section, np.asarray([record.section == section for record in records]))
    for domain in sorted({record.domain for record in records}):
        if domain != "unknown":
            add_row("domain", domain, np.asarray([record.domain == domain for record in records]))
    return rows


def evaluate_one(extractor: FeatureExtractor, args: argparse.Namespace, device_name: str) -> tuple[list[dict], list[dict], list[dict]]:
    train_records, test_records = collect_device_records(args, device_name)
    if len(train_records) < 2:
        raise RuntimeError(f"{device_name}: at least two train files are required.")
    if not test_records:
        raise RuntimeError(f"{device_name}: no test files found.")

    logging.info("[%s/%s] train=%d test=%d", extractor.label, device_name, len(train_records), len(test_records))
    train_features = extract_features(extractor, train_records, args)
    test_features = extract_features(extractor, test_records, args)
    mu, cov_inv = covariance_stats(train_features, eps=args.cov_eps)
    scores = mahalanobis_scores(test_features, mu, cov_inv)
    linear_metrics: dict[str, float] = {}
    linear_rows: list[dict] = []
    if not args.skip_linear:
        linear_metrics, linear_rows = run_linear_probes(
            records=test_records,
            features=test_features,
            args=args,
            model_label=extractor.label,
            device_name=device_name,
            device=device,
        )

    if args.save_roc:
        labels = np.asarray([1 if record.condition == "anomaly" else 0 for record in test_records], dtype=int)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", extractor.label)
        save_roc(
            Path(args.save_dir) / "roc" / f"{safe_label}_{device_name}_mahalanobis.png",
            scores,
            labels,
            f"{extractor.label} {device_name}",
        )

    return (
        score_rows(extractor.label, test_records, scores),
        summarize_scores(extractor.label, device_name, test_records, scores, linear_metrics=linear_metrics),
        linear_rows,
    )


def dry_run(args: argparse.Namespace) -> None:
    for device_name in args.devices:
        train_records, test_records = collect_device_records(args, device_name)
        normal = sum(record.condition == "normal" for record in test_records)
        anomaly = sum(record.condition == "anomaly" for record in test_records)
        sections = sorted({record.section for record in test_records})
        logging.info(
            "%s: train=%d test=%d normal=%d anomaly=%d sections=%s",
            device_name,
            len(train_records),
            len(test_records),
            normal,
            anomaly,
            ",".join(sections) if sections else "none",
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_seed(args.seed)
    if args.target_device:
        args.devices = [args.target_device]

    if args.dry_run:
        dry_run(args)
        return

    device = get_device(args)
    map_values = load_pretrain_map(args.pretrain_map_csv)
    all_score_rows = []
    all_summary_rows = []
    all_linear_rows = []
    for model_kind in args.models:
        extractor = load_extractor(model_kind, args, device)
        for device_name in args.devices:
            score_data, summary_data, linear_data = evaluate_one(extractor, args, device_name)
            all_score_rows.extend(score_data)
            all_summary_rows.extend(summary_data)
            all_linear_rows.extend(linear_data)

    save_root = Path(args.save_dir)
    write_csv(
        save_root / "pretrain_scores.csv",
        ["model", "device", "file_name", "condition", "domain", "section", "score_type", "score"],
        all_score_rows,
    )
    write_csv(
        save_root / "pretrain_summary.csv",
        ["model", "device", "scope", "group", "score_type", "n_normal", "n_anomaly", "auc", "pauc", *linear_metric_fields()],
        all_summary_rows,
    )
    write_csv(
        save_root / "linear_probe_scores.csv",
        ["model", "device", "probe", "fold", "file_name", "condition", "domain", "section", "score"],
        all_linear_rows,
    )
    write_plotter_compatible_summary(save_root, all_summary_rows, map_values)
    logging.info("Wrote %s", save_root / "pretrain_summary.csv")
    logging.info("Wrote %s", save_root / "linear_probe_scores.csv")
    logging.info("Wrote %s", save_root / "results_summary.csv")


if __name__ == "__main__":
    main()

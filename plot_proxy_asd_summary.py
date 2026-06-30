#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV-driven proxy-vs-ASD plotter.

Design goal
-----------
Keep the existing scatter visualization unchanged, and replace only the
hard-coded metric arrays with a CSV loader that converts evaluator summaries to
the original data_list schema:

    {'Task_Group': <task family>, 'proxy_raw': <raw proxy>, 'asd': <AUC percent>}

Typical usage
-------------
python plot_proxy_asd_summary.py \
  --summary_csv results_summary.csv 0cb_results_summary.csv 1cb_results_summary.csv 2cb_results_summary.csv 4cb_results_summary.csv \
  --devices pump ToyConveyor \
  --asd_metric linear_loso_auc \
  --out_path outdomain_sepaug.png \
  --cache_csv plot_input_cache.csv \
  --toyconveyor_min_loss

ASD metric mapping
------------------
In-domain LP  -> linear_half_auc
Out-domain LP -> linear_loso_auc or linear_loso_auc_section_mean
MD            -> mah_train_auc
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from analysis_utils import exact_spearman


# -----------------------------------------------------------------------------
# Original visualization function: visual settings are intentionally preserved.
# -----------------------------------------------------------------------------

def plot_proxy_scatter_only(
    data_list,
    x_col,
    y_col,
    group_col,
    metric_types,
    out_path: str = "./outdomain_sepaug.png",
    x_label: str = "ASD Performance (Out-domain LP AUC %)",
):
    """Plot normalized proxy-task performance against ASD performance."""
    df = pd.DataFrame(data_list)
    if df.empty:
        raise ValueError("data_list is empty. Check CSV path, filters, and metric mapping.")

    raw_metric_col = 'proxy_raw'
    df[raw_metric_col] = pd.to_numeric(df[raw_metric_col], errors='coerce')
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df = df.dropna(subset=[raw_metric_col, x_col, group_col]).copy()
    if df.empty:
        raise ValueError("No valid rows remain after numeric conversion.")
    df[y_col] = 0.0

    for group_name, metric_type in metric_types.items():
        group_mask = (df[group_col] == group_name)
        group_data = df.loc[group_mask, raw_metric_col].values.reshape(-1, 1)
        if len(group_data) == 0:
            continue
        if len(group_data) == 1:
            df.loc[group_mask, y_col] = 0.5
            continue

        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(group_data)

        if metric_type == 'lower_is_better':
            normalized_data = 1 - normalized_data

        df.loc[group_mask, y_col] = normalized_data

    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    groups = sorted(df[group_col].unique())
    num_groups = len(groups)
    marker_map = {
        'Auto-Encoder': 'o',
        'Classification (CE)': 'X',
        'Classification (ArcFace)': 's',
        'Separation': 'D',
        'Pre-trained': 'v',
        'Contrastive learning(SimCLR)': '^',
        'Contrastive learning(SimSiam)': '*',
    }

    unknown_groups = [g for g in groups if g not in marker_map]
    if unknown_groups:
        raise ValueError(f"Unknown Task_Group(s) without marker mapping: {unknown_groups}")

    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=group_col,
        style=group_col,
        hue_order=groups,
        style_order=groups,
        s=120,
        alpha=0.85,
        ax=ax,
        markers=marker_map,
    )

    ax.set_title('Relationship between Proxy Task and ASD Performance', fontsize=16)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Normalized Proxy Task Performance (0.0 to 1.0)', fontsize=12)
    ax.set_ylim(-0.05, 1.05)

    ax.legend(
        title='Proxy Task Family',
        loc='lower center',
        bbox_to_anchor=(0.5, -0.3),
        ncol=math.ceil(num_groups / 2),
        frameon=False,
        fontsize=10,
    )

    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return df


# -----------------------------------------------------------------------------
# CSV loader layer
# -----------------------------------------------------------------------------

METRIC_TYPES: Dict[str, str] = {
    'Auto-Encoder': 'lower_is_better',
    'Classification (CE)': 'higher_is_better',
    'Classification (ArcFace)': 'higher_is_better',
    'Separation': 'higher_is_better',
    'Pre-trained': 'higher_is_better',
    'Contrastive learning(SimCLR)': 'lower_is_better',
    'Contrastive learning(SimSiam)': 'lower_is_better',
}

CHECKPOINT_SELECTION_KEYS: Tuple[str, ...] = (
    'Task_Group',
    'target_device',
    'config_id',
    'task',
    'phase',
    'trained_target',
    'backbone_name',
    'margin',
    'feature_index',
    'segment_frames_from_name',
    'frame_stack',
    'batch_size_from_name',
    'arch',
    'comp_feat',
    'lin_feat',
    'channel_size',
    'cb',
    'unsup_mode',
)


def _read_summary_csvs(paths: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df['source_csv'] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("No CSV paths were supplied.")
    return pd.concat(frames, axis=0, ignore_index=True, sort=False)


def _as_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float('nan')
        return float(value)
    except Exception:
        return float('nan')


def parse_train_loss_from_name(name: Any) -> float:
    """Parse train loss embedded in checkpoint filename.

    Supported examples:
      ToyConveyor_..._epoch200_ae0.00029516.pth
      pump_..._epoch185_loss0.292_sep0.292.pth
      model_epoch010_sep0.544.pth
    """
    text = str(name)
    patterns = [
        r'(?:^|[_\-])loss([+\-]?\d+(?:\.\d+)?(?:e[+\-]?\d+)?)',
        r'(?:^|[_\-])ae([+\-]?\d+(?:\.\d+)?(?:e[+\-]?\d+)?)',
        r'(?:^|[_\-])sep([+\-]?\d+(?:\.\d+)?(?:e[+\-]?\d+)?)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            return float(m.group(1).rstrip('.'))
        except Exception:
            pass
    return float('nan')


def add_train_loss_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'loss_from_name' in df.columns:
        loss_from_name = pd.to_numeric(df['loss_from_name'], errors='coerce')
    else:
        loss_from_name = pd.Series(np.nan, index=df.index)
    if 'model_name' in df.columns:
        name_source = df['model_name']
    elif 'model_id' in df.columns:
        name_source = df['model_id']
    else:
        name_source = pd.Series('', index=df.index)
    parsed = name_source.map(parse_train_loss_from_name)
    df['train_loss_for_selection'] = loss_from_name.fillna(parsed)
    if 'epoch_from_name' not in df.columns:
        df['epoch_from_name'] = np.nan
    return df


def infer_task_group(row: pd.Series) -> Optional[str]:
    task = str(row.get('task', '')).lower()
    phase = str(row.get('phase', '')).lower()
    model_type = str(row.get('model_type', '')).lower()
    if task in {'pretrained', 'pretrain'} or model_type in {'pretrained', 'pretrain'}:
        return 'Pre-trained'

    if task == 'ae' or phase == 'ae':
        return 'Auto-Encoder'
    if task in {'sep', 'sep_direct', 'sep_mask'} or phase in {'sep', 'sep_direct', 'sep_mask'}:
        return 'Separation'
    if task == 'ce' or phase == 'ce':
        return 'Classification (CE)'
    if task == 'arcface' or phase == 'arcface':
        return 'Classification (ArcFace)'
    if task == 'simclr' or phase == 'simclr':
        return 'Contrastive learning(SimCLR)'
    if task == 'simsiam' or phase == 'simsiam':
        return 'Contrastive learning(SimSiam)'

    name = str(row.get('model_name', row.get('model_id', row.get('model', '')))).lower()
    unsup_mode = str(row.get('unsup_mode', '')).lower()

    if model_type == 'ae' or 'auto-encoder' in name or re.search(r'[_\-]ae\d', name):
        return 'Auto-Encoder'
    if model_type == 'separation' or 'sep' in name and 'snr' in name:
        return 'Separation'
    if model_type == 'classification':
        if 'arcface' in name or 'arc' in name:
            return 'Classification (ArcFace)'
        if 'pre' in name or 'imagenet' in name:
            return 'Pre-trained'
        return 'Classification (CE)'
    if model_type == 'unsup':
        if 'simsiam' in name or unsup_mode == 'simsiam':
            return 'Contrastive learning(SimSiam)'
        return 'Contrastive learning(SimCLR)'
    if 'beat' in name or 'ced' in name or 'eat' in name:
        return 'Pre-trained'
    if 'pre-trained' in name or 'pretrained' in name:
        return 'Pre-trained'
    return None


def _mean_existing_columns(row: pd.Series, columns: Sequence[str]) -> Tuple[float, str]:
    values: List[float] = []
    used: List[str] = []
    for col in columns:
        if col not in row.index:
            continue
        value = _as_float(row[col])
        if np.isfinite(value):
            values.append(value)
            used.append(col)
    if not values:
        return float('nan'), ''
    return float(np.mean(values)), '+'.join(used)


def _first_existing_column(row: pd.Series, columns: Sequence[str]) -> Tuple[float, str]:
    for col in columns:
        if col not in row.index:
            continue
        value = _as_float(row[col])
        if np.isfinite(value):
            return value, col
    return float('nan'), ''


def _safe_name(name: Any) -> str:
    text = str(name).replace('+', 'plus')
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('_') or 'model'


def _format_key_value(value: Any) -> str:
    if pd.isna(value):
        return "na"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _safe_name(value)


def _first_value(row: pd.Series, names: Sequence[str], default: str = "na") -> Any:
    for name in names:
        if name in row.index and not pd.isna(row.get(name)):
            return row.get(name)
    return default


def _row_name_text(row: pd.Series) -> str:
    parts = [str(row.get(name, "")) for name in ("model_name", "model_id", "model", "source_csv")]
    return " ".join(parts)


def _value_or_parsed_int(row: pd.Series, column: str, patterns: Sequence[str]) -> Any:
    value = row.get(column)
    if not pd.isna(value):
        return value
    text = _row_name_text(row).lower()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return value


def _value_or_parsed_float(row: pd.Series, column: str, patterns: Sequence[str]) -> Any:
    value = row.get(column)
    if not pd.isna(value):
        return value
    text = _row_name_text(row).lower()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return value


def _arch_from_row(row: pd.Series) -> Any:
    value = _first_value(row, ["arch", "backbone_name", "model_type"])
    if value != "na":
        return value
    text = _row_name_text(row).lower()
    for pattern, arch in [
        (r"resnet\s*152|resnet152|r152", "resnet152"),
        (r"resnet\s*101|resnet101|r101", "resnet101"),
        (r"resnet\s*50|resnet50|r50", "resnet50"),
        (r"resnet\s*34|resnet34|r34", "resnet34"),
        (r"resnet\s*18|resnet18|r18", "resnet18"),
    ]:
        if re.search(pattern, text):
            return arch
    return value


def _unsup_mode_from_row(row: pd.Series) -> Any:
    value = _first_value(row, ["unsup_mode", "task"])
    if value != "na":
        return value
    text = _row_name_text(row).lower()
    if "simsiam" in text:
        return "simsiam"
    if "simclr" in text:
        return "simclr"
    return value


def infer_config_id(row: pd.Series) -> str:
    group = str(row.get('Task_Group', ''))
    if group == 'Auto-Encoder':
        comp = _value_or_parsed_int(row, 'comp_feat', [r"comp(\d+)lin\d+", r"l(\d+)h\d+"])
        lin = _value_or_parsed_int(row, 'lin_feat', [r"comp\d+lin(\d+)", r"l\d+h(\d+)"])
        return f"ae_comp{_format_key_value(comp)}_lin{_format_key_value(lin)}"
    if group == 'Separation':
        cb = _value_or_parsed_int(row, 'cb', [r"(\d+)cb"])
        channel = _value_or_parsed_int(row, 'channel_size', [r"(\d+)ch"])
        if not pd.isna(cb) or not pd.isna(channel):
            return f"sep_cb{_format_key_value(cb)}_ch{_format_key_value(channel)}"
        return f"sep_shared_{_format_key_value(_first_value(row, ['backbone_name', 'arch', 'model_type']))}"
    if group == 'Classification (CE)':
        return f"ce_{_format_key_value(_arch_from_row(row))}"
    if group == 'Classification (ArcFace)':
        arch = _format_key_value(_arch_from_row(row))
        margin = _value_or_parsed_float(row, 'margin', [r"margin[_-]?([+\-]?\d+(?:\.\d+)?)", r"_m([+\-]?\d+(?:\.\d+)?)"])
        return f"arcface_{arch}_m{_format_key_value(margin)}"
    if group in {'Contrastive learning(SimCLR)', 'Contrastive learning(SimSiam)'}:
        arch = _format_key_value(_arch_from_row(row))
        mode = _format_key_value(_unsup_mode_from_row(row))
        return f"{mode}_{arch}"
    if group == 'Pre-trained':
        return f"pretrained_{_safe_name(_first_value(row, ['model_name', 'model_id', 'model']))}"
    return _format_key_value(_first_value(row, ['model_id', 'model_name'], default='unknown'))


PRETRAIN_AS2M_MAP: Dict[str, float] = {
    'eatlarge': 49.5,
    'eatbase': 48.9,
    'eatbaseepoch30pretrain': 48.9,
    'beatsiter3': 48.0,
    'beatsiter3plus': 48.6,
    'cedbase': 50.0,
    'cedmini': 49.0,
    'cedsmall': 49.6,
    'cedtiny': 48.1,
}


def _pretrain_key(name: Any) -> str:
    text = str(name).replace('+', 'plus').lower()
    return re.sub(r'[^a-z0-9]+', '', text)


def lookup_pretrain_map(name: Any) -> float:
    key = _pretrain_key(name)
    if key in PRETRAIN_AS2M_MAP:
        return PRETRAIN_AS2M_MAP[key]
    for alias, value in sorted(PRETRAIN_AS2M_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        if alias and alias in key:
            return value
    return float('nan')


def normalize_pretrain_summary(df: pd.DataFrame) -> pd.DataFrame:
    if not {"model", "device", "auc", "pauc"}.issubset(df.columns) or "target_device" in df.columns:
        return df
    out = df.copy()
    if {"scope", "group"}.issubset(out.columns):
        out = out[
            (out["scope"].astype(str).str.lower() == "overall")
            & (out["group"].astype(str).str.lower() == "all")
        ].copy()
    out["model_id"] = out["model"].map(_safe_name)
    out["model_name"] = out["model"].astype(str)
    out["model_type"] = "pretrained"
    out["task"] = "pretrained"
    out["target_device"] = out["device"].astype(str)
    out["mah_train_auc"] = pd.to_numeric(out["auc"], errors="coerce")
    out["mah_train_pauc"] = pd.to_numeric(out["pauc"], errors="coerce")
    out["pretrain_asd_auc"] = out["mah_train_auc"]
    out["pretrain_asd_pauc"] = out["mah_train_pauc"]
    if "pretrain_map" not in out.columns:
        out["pretrain_map"] = out["model_name"].map(lookup_pretrain_map)
    if "mAP" not in out.columns:
        out["mAP"] = out["pretrain_map"]
    return out


def _format_snr_for_column(snr: float) -> str:
    return f"{float(snr):g}"


def sep_proxy_from_row(row: pd.Series, env: str, snrs: Sequence[float]) -> Tuple[float, str]:
    cols = [f"sep_si_sdr_mean_{env}_snr{_format_snr_for_column(s)}" for s in snrs]
    return _mean_existing_columns(row, cols)


def ae_proxy_from_row(row: pd.Series, ae_proxy: str) -> Tuple[float, str]:
    aliases = {
        'test_normal_l2': ['ae_l2_test_normal_mean', 'proxy_ae_l2_test_normal_mean'],
        'test_anomaly_l2': ['ae_l2_test_anomaly_mean', 'proxy_ae_l2_test_anomaly_mean'],
        'test_mean_l2': ['ae_l2_test_mean', 'proxy_ae_l2_test_mean'],
        'train_normal_l2': ['ae_l2_train_normal_mean', 'proxy_ae_l2_train_normal_mean'],
        'train_mean_l2': ['ae_l2_train_mean', 'proxy_ae_l2_train_mean'],
        'test_normal_l1': ['ae_l1_test_normal_mean', 'proxy_ae_l1_test_normal_mean'],
        'test_mean_l1': ['ae_l1_test_mean', 'proxy_ae_l1_test_mean'],
        'train_normal_l1': ['ae_l1_train_normal_mean', 'proxy_ae_l1_train_normal_mean'],
        'train_mean_l1': ['ae_l1_train_mean', 'proxy_ae_l1_train_mean'],
    }
    columns = aliases.get(ae_proxy, [ae_proxy])
    if isinstance(columns, str):
        columns = [columns]
    return _mean_existing_columns(row, columns)


def proxy_from_row(
    row: pd.Series,
    *,
    ae_proxy: str = 'test_normal_l1',
    sep_proxy_env: str = 'test_normal',
    sep_snrs: Sequence[float] = (-5.0, 0.0, 5.0),
    sep_direct_proxy: str = 'l1',
    unsup_proxy: str = 'uniformity',
    classification_proxy: str = 'global_macro_f1',
) -> Tuple[float, str]:
    group = row.get('Task_Group')
    if group == 'Auto-Encoder':
        return ae_proxy_from_row(row, ae_proxy)
    if group == 'Separation':
        value, used = sep_proxy_from_row(row, sep_proxy_env, sep_snrs)
        if np.isfinite(value):
            return value, used
        if sep_direct_proxy == 'l2':
            direct_cols = ['sep_direct_l2_snr_mean', 'sep_direct_l2_mean']
        else:
            direct_cols = ['sep_direct_l1_snr_mean', 'sep_direct_l1_mean']
        value, used = _mean_existing_columns(row, direct_cols)
        if np.isfinite(value):
            return -value, f"negative_{used}"
        return value, used
    if group in {'Classification (CE)', 'Classification (ArcFace)'}:
        priority_map = {
            'global_macro_f1': ['clf_global_macro_f1', 'clf_all_macro_f1', 'clf_proxy_macro_f1'],
            'all_macro_f1': ['clf_all_macro_f1', 'clf_global_macro_f1', 'clf_proxy_macro_f1'],
            'proxy_macro_f1': ['clf_proxy_macro_f1', 'clf_global_macro_f1', 'clf_all_macro_f1'],
            'global_micro_f1': ['clf_global_micro_f1', 'clf_all_micro_f1', 'clf_proxy_micro_f1', 'clf_target_total_micro_f1'],
            'all_micro_f1': ['clf_all_micro_f1', 'clf_global_micro_f1', 'clf_proxy_micro_f1', 'clf_target_total_micro_f1'],
            'proxy_micro_f1': ['clf_proxy_micro_f1', 'clf_global_micro_f1', 'clf_all_micro_f1', 'clf_target_total_micro_f1'],
            'target_total_micro_f1': ['clf_target_total_micro_f1', 'clf_global_micro_f1', 'clf_all_micro_f1', 'clf_proxy_micro_f1'],
        }
        value, used = _first_existing_column(
            row,
            priority_map.get(classification_proxy, priority_map['global_macro_f1']),
        )
        if not np.isfinite(value):
            value, used = _first_existing_column(
                row,
                [
                    'clf_global_macro_f1',
                    'clf_all_macro_f1',
                    'clf_proxy_macro_f1',
                    'clf_global_micro_f1',
                    'clf_proxy_micro_f1',
                    'clf_target_total_micro_f1',
                    'clf_all_micro_f1',
                    'clf_condition_normal_true_micro_f1',
                    'clf_condition_normal_micro_f1',
                ],
            )
        if np.isfinite(value) and value <= 1.5:
            value *= 100.0
        return value, used
    if group == 'Pre-trained':
        value, used = _first_existing_column(
            row,
            ['pretrain_map', 'pretrained_map', 'mAP', 'map'],
        )
        if np.isfinite(value) and value <= 1.5:
            value *= 100.0
        return value, used
    if group in {'Contrastive learning(SimCLR)', 'Contrastive learning(SimSiam)'}:
        if unsup_proxy == 'alignment':
            return _mean_existing_columns(row, ['unsup_alignment'])
        return _mean_existing_columns(row, ['unsup_uniformity'])
    return float('nan'), ''


def asd_from_row(row: pd.Series, asd_metric: str) -> Tuple[float, str]:
    if asd_metric == 'linear_loso_auc_section_mean':
        section_cols = [c for c in row.index if re.fullmatch(r'linear_loso_auc_section_\d+', str(c))]
        value, used = _mean_existing_columns(row, sorted(section_cols))
    else:
        if asd_metric not in row.index:
            return float('nan'), asd_metric
        value = _as_float(row[asd_metric])
        used = asd_metric

    if np.isfinite(value) and ('auc' in asd_metric.lower() or 'pauc' in asd_metric.lower()) and value <= 1.5:
        value *= 100.0
    return value, used


def _contains_best(row: pd.Series) -> bool:
    text = f"{row.get('model_id', '')} {row.get('model_name', '')}".lower()
    return bool(re.search(r'(^|[_\-])best([_\-.]|$)', text)) or '_best_' in text or 'best_epoch' in text


def select_one_checkpoint(group: pd.DataFrame, policy: str) -> Tuple[pd.Series, str]:
    """Select one checkpoint from a model-configuration group."""
    if group.empty:
        raise ValueError("empty group")

    if policy == 'filename_best':
        candidates = group[group.apply(_contains_best, axis=1)]
        if candidates.empty:
            raise ValueError("No filename-best checkpoint in group")
        candidates = candidates.sort_values(['train_loss_for_selection', 'epoch_from_name'], na_position='last')
        return candidates.iloc[0], 'filename_best'

    if policy == 'min_loss':
        candidates = group[np.isfinite(pd.to_numeric(group['train_loss_for_selection'], errors='coerce'))]
        if candidates.empty:
            raise ValueError("No parsable train loss in group")
        candidates = candidates.sort_values(['train_loss_for_selection', 'epoch_from_name'], na_position='last')
        return candidates.iloc[0], 'min_filename_loss'

    if policy == 'max_asd':
        candidates = group[np.isfinite(pd.to_numeric(group['asd'], errors='coerce'))]
        if candidates.empty:
            raise ValueError("No valid ASD metric in group")
        return candidates.sort_values('asd', ascending=False).iloc[0], 'max_asd_oracle'

    if policy != 'best_or_min_loss':
        raise ValueError(f"Unknown selection policy: {policy}")

    candidates = group[group.apply(_contains_best, axis=1)]
    if not candidates.empty:
        candidates = candidates.sort_values(['train_loss_for_selection', 'epoch_from_name'], na_position='last')
        return candidates.iloc[0], 'filename_best'

    candidates = group[np.isfinite(pd.to_numeric(group['train_loss_for_selection'], errors='coerce'))]
    if candidates.empty:
        if len(group) == 1:
            return group.iloc[0], 'only_row_no_loss'
        raise ValueError("No filename-best checkpoint and no parsable train loss in group")
    candidates = candidates.sort_values(['train_loss_for_selection', 'epoch_from_name'], na_position='last')
    return candidates.iloc[0], 'fallback_min_filename_loss'


def build_proxy_asd_plot_dataframe(
    summary_csv: Sequence[str],
    *,
    asd_metric: str,
    devices: Optional[Sequence[str]] = None,
    task_groups: Optional[Sequence[str]] = None,
    selection_policy: str = 'best_or_min_loss',
    toyconveyor_min_loss: bool = False,
    ae_proxy: str = 'test_normal_l1',
    sep_proxy_env: str = 'test_normal',
    sep_snrs: Sequence[float] = (-5.0, 0.0, 5.0),
    sep_direct_proxy: str = 'l1',
    unsup_proxy: str = 'uniformity',
    classification_proxy: str = 'global_macro_f1',
) -> pd.DataFrame:
    df = _read_summary_csvs(summary_csv)
    df = normalize_pretrain_summary(df)
    df = add_train_loss_column(df)
    df['Task_Group'] = df.apply(infer_task_group, axis=1)
    df = df[df['Task_Group'].notna()].copy()
    df['config_id'] = df.apply(infer_config_id, axis=1)

    if devices:
        wanted = {str(d).lower() for d in devices}
        df = df[df['target_device'].astype(str).str.lower().isin(wanted)].copy()
    if task_groups:
        wanted_groups = set(task_groups)
        df = df[df['Task_Group'].isin(wanted_groups)].copy()

    if df.empty:
        raise ValueError("No rows remain after device/task filtering.")

    proxy_rows: List[float] = []
    proxy_metric_rows: List[str] = []
    asd_rows: List[float] = []
    asd_metric_rows: List[str] = []

    for _, row in df.iterrows():
        proxy_value, proxy_metric = proxy_from_row(
            row,
            ae_proxy=ae_proxy,
            sep_proxy_env=sep_proxy_env,
            sep_snrs=sep_snrs,
            sep_direct_proxy=sep_direct_proxy,
            unsup_proxy=unsup_proxy,
            classification_proxy=classification_proxy,
        )
        asd_value, used_asd_metric = asd_from_row(row, asd_metric)
        proxy_rows.append(proxy_value)
        proxy_metric_rows.append(proxy_metric)
        asd_rows.append(asd_value)
        asd_metric_rows.append(used_asd_metric)

    df['proxy_raw'] = proxy_rows
    df['proxy_metric_name'] = proxy_metric_rows
    df['asd'] = asd_rows
    df['asd_metric_name'] = asd_metric_rows
    df = df[np.isfinite(pd.to_numeric(df['proxy_raw'], errors='coerce')) & np.isfinite(pd.to_numeric(df['asd'], errors='coerce'))].copy()
    if df.empty:
        raise ValueError("No rows have both proxy_raw and ASD metric after metric mapping.")

    group_keys = [k for k in CHECKPOINT_SELECTION_KEYS if k in df.columns]
    selected_rows: List[pd.Series] = []
    errors: List[Dict[str, Any]] = []

    for key, sub in df.groupby(group_keys, dropna=False, sort=False):
        group_policy = selection_policy
        device_value = str(sub['target_device'].iloc[0]).lower() if 'target_device' in sub else ''
        if toyconveyor_min_loss and device_value == 'toyconveyor':
            group_policy = 'min_loss'
        try:
            selected, selected_by = select_one_checkpoint(sub, group_policy)
            selected = selected.copy()
            selected['selected_by'] = selected_by
            selected_rows.append(selected)
        except Exception as exc:
            errors.append({
                'selection_group': repr(key),
                'n_rows': len(sub),
                'policy': group_policy,
                'error': repr(exc),
            })

    if not selected_rows:
        msg = "No checkpoints were selected."
        if errors:
            msg += " First selection error: " + repr(errors[0])
        raise ValueError(msg)

    out = pd.DataFrame(selected_rows).reset_index(drop=True)
    out['proxy_direction'] = out['Task_Group'].map(METRIC_TYPES)
    return out


def print_group_correlations(plot_df: pd.DataFrame) -> None:
    corr_df = plot_df.copy()
    corr_df['proxy_raw'] = pd.to_numeric(corr_df['proxy_raw'], errors='coerce')
    corr_df['proxy_quality'] = corr_df['proxy_raw']
    lower_mask = corr_df['Task_Group'].map(METRIC_TYPES).eq('lower_is_better')
    corr_df.loc[lower_mask, 'proxy_quality'] = -corr_df.loc[lower_mask, 'proxy_raw']

    for group, sub in corr_df.groupby('Task_Group'):
        if len(sub) < 3:
            continue
        rho, p = exact_spearman(sub['proxy_quality'].to_numpy(), sub['asd'].to_numpy())
        print(f"{group}: Spearman(proxy_quality, asd) rho={rho:.5f}, p={p:.5f}, n={len(sub)}")


def asd_metric_label(asd_metric: str) -> str:
    mapping = {
        'linear_loso_auc': 'ASD Performance (Out-domain LP AUC %)',
        'linear_loso_auc_section_mean': 'ASD Performance (Out-domain LP AUC %)',
        'linear_half_auc': 'ASD Performance (In-domain LP AUC %)',
        'mah_train_auc': 'ASD Performance (Mahalanobis AUC %)',
    }
    return mapping.get(asd_metric, f'ASD Performance ({asd_metric})')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--summary_csv', nargs='+', required=True, help='One or more evaluator results_summary.csv files')
    parser.add_argument('--devices', nargs='*', default=None, help='Optional target_device filter, e.g. pump ToyConveyor')
    parser.add_argument('--task_groups', nargs='*', default=None, choices=list(METRIC_TYPES.keys()), help='Optional Task_Group filter')
    parser.add_argument('--asd_metric', type=str, default='linear_loso_auc', help='ASD metric column or linear_loso_auc_section_mean')
    parser.add_argument('--selection_policy', choices=['best_or_min_loss', 'filename_best', 'min_loss', 'max_asd'], default='best_or_min_loss')
    parser.add_argument('--toyconveyor_min_loss', action='store_true', help='Force ToyConveyor groups to use minimum filename loss selection')
    parser.add_argument('--ae_proxy', type=str, default='test_normal_l1',
                        help='AE proxy alias or explicit column. Paper default: test_normal_l1 / MAE')
    parser.add_argument('--sep_proxy_env', choices=['train_data', 'test_normal', 'test_anomaly', 'test_source', 'test_target'], default='test_normal')
    parser.add_argument('--sep_snr', nargs='+', type=float, default=[-5.0, 0.0, 5.0])
    parser.add_argument('--sep_direct_proxy', choices=['l1', 'l2'], default='l1',
                        help='Shared-backbone sep_direct loss proxy. It is negated internally so higher proxy_raw remains better.')
    parser.add_argument('--unsup_proxy', choices=['uniformity', 'alignment'], default='uniformity')
    parser.add_argument('--classification_proxy',
                        choices=[
                            'global_macro_f1',
                            'all_macro_f1',
                            'proxy_macro_f1',
                            'global_micro_f1',
                            'all_micro_f1',
                            'proxy_micro_f1',
                            'target_total_micro_f1',
                        ],
                        default='global_macro_f1')
    parser.add_argument('--out_path', type=str, default='./outdomain_sepaug.png')
    parser.add_argument('--cache_csv', type=str, default='./plot_input_cache.csv')
    parser.add_argument('--print_correlations', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_df = build_proxy_asd_plot_dataframe(
        args.summary_csv,
        asd_metric=args.asd_metric,
        devices=args.devices,
        task_groups=args.task_groups,
        selection_policy=args.selection_policy,
        toyconveyor_min_loss=args.toyconveyor_min_loss,
        ae_proxy=args.ae_proxy,
        sep_proxy_env=args.sep_proxy_env,
        sep_snrs=args.sep_snr,
        sep_direct_proxy=args.sep_direct_proxy,
        unsup_proxy=args.unsup_proxy,
        classification_proxy=args.classification_proxy,
    )

    Path(args.cache_csv).parent.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(args.cache_csv, index=False)

    plot_proxy_scatter_only(
        data_list=plot_df.to_dict('records'),
        x_col='asd',
        y_col='Proxy_Performance_Normalized',
        group_col='Task_Group',
        metric_types=METRIC_TYPES,
        out_path=args.out_path,
        x_label=asd_metric_label(args.asd_metric),
    )

    print(f"Saved figure: {args.out_path}")
    print(f"Saved plot input cache: {args.cache_csv}")
    print(f"Selected rows: {len(plot_df)}")
    print(plot_df[['Task_Group', 'target_device', 'model_id', 'selected_by', 'train_loss_for_selection', 'proxy_metric_name', 'proxy_raw', 'asd_metric_name', 'asd']].to_string(index=False))
    if args.print_correlations:
        print_group_correlations(plot_df)


if __name__ == '__main__':
    main()

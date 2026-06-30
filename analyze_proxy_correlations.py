"""Aggregate proxy-vs-ASD summaries and compute paper-style correlations."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import benjamini_hochberg, exact_spearman
from plot_proxy_asd_summary import METRIC_TYPES, build_proxy_asd_plot_dataframe, infer_config_id


AGG_KEYS = [
    "Task_Group",
    "config_id",
    "proxy_metric_name",
    "asd_metric_name",
    "task",
    "phase",
    "model_type",
    "unsup_mode",
    "arch",
    "comp_feat",
    "lin_feat",
    "channel_size",
    "cb",
    "backbone_name",
    "margin",
]

EXPECTED_PAPER_CONFIG_COUNTS = {
    "Auto-Encoder": 9,
    "Classification (CE)": 5,
    "Classification (ArcFace)": 5,
    "Separation": 8,
    "Contrastive learning(SimCLR)": 5,
    "Contrastive learning(SimSiam)": 5,
    "Pre-trained": 8,
}


def add_proxy_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["proxy_raw"] = pd.to_numeric(out["proxy_raw"], errors="coerce")
    out["asd"] = pd.to_numeric(out["asd"], errors="coerce")
    out["proxy_quality"] = out["proxy_raw"]
    lower_mask = out["Task_Group"].map(METRIC_TYPES).eq("lower_is_better")
    out.loc[lower_mask, "proxy_quality"] = -out.loc[lower_mask, "proxy_raw"]
    return out


def aggregate_devices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "config_id" not in df.columns:
        df["config_id"] = df.apply(infer_config_id, axis=1)
    keys = [key for key in AGG_KEYS if key in df.columns]
    if not keys:
        raise ValueError("No aggregation keys are available.")
    numeric_cols = [
        col for col in ["proxy_quality", "proxy_raw", "asd"]
        if col in df.columns
    ]
    grouped = df.groupby(keys, dropna=False, sort=False)
    out = grouped[numeric_cols].mean().reset_index()
    out["n_devices"] = grouped["target_device"].nunique().to_numpy()
    return out


def validate_paper_config_counts(df: pd.DataFrame, context: str = "", strict: bool = False) -> bool:
    if "config_id" not in df.columns:
        df = df.copy()
        df["config_id"] = df.apply(infer_config_id, axis=1)
    if "Task_Group" not in df.columns:
        logging.warning("Cannot validate paper config counts%s: missing Task_Group column.", f" for {context}" if context else "")
        if strict:
            raise RuntimeError("Paper configuration count validation failed.")
        return False

    counts = df.groupby("Task_Group")["config_id"].nunique()
    suffix = f" for {context}" if context else ""
    ok = True
    for group, expected in EXPECTED_PAPER_CONFIG_COUNTS.items():
        actual = int(counts.get(group, 0))
        if actual != expected:
            ok = False
            logging.warning(
                "Unexpected config count%s for %s: got %d, expected %d. "
                "Check config_id parsing and summary inputs.",
                suffix,
                group,
                actual,
                expected,
            )
    if strict and not ok:
        raise RuntimeError("Paper configuration count validation failed.")
    return ok


def correlation_rows(df: pd.DataFrame, asd_metric: str, include_all_group: bool = False) -> list[dict]:
    rows = []
    groups = []
    if include_all_group:
        groups.append(("All", df))
    groups.extend(list(df.groupby("Task_Group", sort=True)))
    for task_group, sub in groups:
        rho, p_value = exact_spearman(sub["proxy_quality"].to_numpy(), sub["asd"].to_numpy())
        rows.append(
            {
                "asd_metric": asd_metric,
                "Task_Group": task_group,
                "n": int(len(sub)),
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute device-aggregated proxy-vs-ASD Spearman correlations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary_csv", nargs="+", required=True)
    parser.add_argument("--devices", nargs="*", default=None)
    parser.add_argument("--asd_metrics", nargs="+", default=["linear_half_auc", "linear_loso_auc", "mah_train_auc"])
    parser.add_argument("--out_csv", default="./proxy_correlation_summary.csv")
    parser.add_argument("--cache_prefix", default="./proxy_correlation_input")
    parser.add_argument("--no_aggregate_devices", action="store_true")
    parser.add_argument("--include_all_group", action="store_true")
    parser.add_argument("--validate_paper_counts", action="store_true",
                        help="warn when paper-family configuration counts differ from the expected Figure 4/Table 9 grid")
    parser.add_argument("--strict_paper_counts", action="store_true",
                        help="fail when --validate_paper_counts finds an unexpected paper-family configuration count")
    parser.add_argument("--ae_proxy", default="test_normal_l1")
    parser.add_argument("--classification_proxy", default="global_macro_f1")
    parser.add_argument("--sep_proxy_env", default="test_normal")
    parser.add_argument("--unsup_proxy", default="uniformity", choices=["uniformity", "alignment"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: list[dict] = []

    for asd_metric in args.asd_metrics:
        plot_df = build_proxy_asd_plot_dataframe(
            args.summary_csv,
            asd_metric=asd_metric,
            devices=args.devices,
            ae_proxy=args.ae_proxy,
            classification_proxy=args.classification_proxy,
            sep_proxy_env=args.sep_proxy_env,
            unsup_proxy=args.unsup_proxy,
        )
        plot_df = add_proxy_quality(plot_df)
        analysis_df = plot_df if args.no_aggregate_devices else aggregate_devices(plot_df)
        if args.validate_paper_counts or args.strict_paper_counts:
            validate_paper_config_counts(analysis_df, context=asd_metric, strict=args.strict_paper_counts)
        cache_path = Path(f"{args.cache_prefix}_{asd_metric}.csv")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_df.to_csv(cache_path, index=False)
        all_rows.extend(correlation_rows(analysis_df, asd_metric, include_all_group=args.include_all_group))

    out = pd.DataFrame(all_rows)
    out["q_value_bh"] = np.nan
    bh_mask = out["Task_Group"].ne("All")
    out.loc[bh_mask, "q_value_bh"] = benjamini_hochberg(out.loc[bh_mask, "p_value"].to_numpy())
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved correlation summary: {args.out_csv}")


if __name__ == "__main__":
    main()

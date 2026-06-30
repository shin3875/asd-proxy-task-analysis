"""Create the paper-style three-panel proxy-vs-ASD scatter figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analyze_proxy_correlations import add_proxy_quality, aggregate_devices, validate_paper_config_counts
from plot_proxy_asd_summary import build_proxy_asd_plot_dataframe


PANEL_METRICS = [
    ("linear_half_auc", "In-domain LP AUC (%)"),
    ("linear_loso_auc", "Out-domain LP AUC (%)"),
    ("mah_train_auc", "Mahalanobis AUC (%)"),
]


MARKER_MAP = {
    "Auto-Encoder": "o",
    "Classification (CE)": "X",
    "Classification (ArcFace)": "s",
    "Separation": "D",
    "Pre-trained": "v",
    "Contrastive learning(SimCLR)": "^",
    "Contrastive learning(SimSiam)": "*",
}


def normalize_within_family(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["proxy_quality_normalized"] = np.nan
    for task_group, sub in out.groupby("Task_Group"):
        values = pd.to_numeric(sub["proxy_quality"], errors="coerce")
        valid = values[np.isfinite(values)]
        if valid.empty:
            continue
        if len(valid) == 1 or float(valid.max()) == float(valid.min()):
            out.loc[sub.index, "proxy_quality_normalized"] = 0.5
        else:
            out.loc[sub.index, "proxy_quality_normalized"] = (values - valid.min()) / (valid.max() - valid.min())
    return out


def build_panel_dataframe(args: argparse.Namespace, asd_metric: str) -> pd.DataFrame:
    plot_df = build_proxy_asd_plot_dataframe(
        args.summary_csv,
        asd_metric=asd_metric,
        devices=args.devices,
        ae_proxy="test_normal_l1",
        classification_proxy="global_macro_f1",
        sep_proxy_env=args.sep_proxy_env,
        unsup_proxy="uniformity",
    )
    plot_df = add_proxy_quality(plot_df)
    if args.aggregate_devices:
        plot_df = aggregate_devices(plot_df)
    plot_df = normalize_within_family(plot_df)
    plot_df["panel_metric"] = asd_metric
    return plot_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paper Figure 4 style three-panel proxy-vs-ASD scatter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary_csv", nargs="+", required=True)
    parser.add_argument("--devices", nargs="*", default=None)
    parser.add_argument("--out_path", default="./paper_figure4.png")
    parser.add_argument("--cache_csv", default="./paper_figure4_cache.csv")
    parser.add_argument("--sep_proxy_env", default="test_normal")
    parser.add_argument("--aggregate_devices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate_paper_counts", action="store_true",
                        help="warn when paper-family configuration counts differ from the expected Figure 4 grid")
    parser.add_argument("--strict_paper_counts", action="store_true",
                        help="fail when --validate_paper_counts finds an unexpected paper-family configuration count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [build_panel_dataframe(args, metric) for metric, _ in PANEL_METRICS]
    full_df = pd.concat(frames, axis=0, ignore_index=True)
    if args.validate_paper_counts or args.strict_paper_counts:
        validate_paper_config_counts(full_df, context="Figure 4", strict=args.strict_paper_counts)
    Path(args.cache_csv).parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(args.cache_csv, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (metric, title) in zip(axes, PANEL_METRICS):
        sub = full_df[full_df["panel_metric"] == metric].copy()
        sns.scatterplot(
            data=sub,
            x="asd",
            y="proxy_quality_normalized",
            hue="Task_Group",
            style="Task_Group",
            markers={k: v for k, v in MARKER_MAP.items() if k in set(sub["Task_Group"])},
            s=95,
            alpha=0.85,
            ax=ax,
            legend=ax is axes[-1],
        )
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.45)
    axes[0].set_ylabel("Min-max normalized proxy quality")
    for ax in axes[1:]:
        ax.set_ylabel("")

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, title="Proxy Task Family", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(args.out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {args.out_path}")
    print(f"Saved plot input cache: {args.cache_csv}")


if __name__ == "__main__":
    main()

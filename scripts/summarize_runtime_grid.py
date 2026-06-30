"""Summarize full-grid runtime logs into a completed CSV.

The script reads ``*.time`` files produced by ``measure_full_runtime_grid.sh``
and joins them with ``docs/full_runtime_measurements_template.csv``. Raw logs,
commands, and local paths are not copied into the output table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_TEMPLATE = Path("docs/full_runtime_measurements_template.csv")
DEFAULT_OUTPUT = Path("docs/full_runtime_measurements_completed.csv")


NAME_TO_CONFIG = {
    "eval_pretrained_beats_iter3": "pretrained_BEATs-iter3",
    "eval_pretrained_beats_iter3_plus": "pretrained_BEATs-iter3plus",
    "eval_pretrained_ced_base": "pretrained_CED-base",
    "eval_pretrained_ced_mini": "pretrained_CED-mini",
    "eval_pretrained_ced_small": "pretrained_CED-small",
    "eval_pretrained_ced_tiny": "pretrained_CED-tiny",
    "eval_pretrained_eat_base": "pretrained_EAT-base",
    "eval_pretrained_eat_large": "pretrained_EAT-large",
}


def config_id_from_time_stem(stem: str) -> str | None:
    if stem in NAME_TO_CONFIG:
        return NAME_TO_CONFIG[stem]
    if stem.startswith("train_ae_"):
        return stem.removeprefix("train_ae_").replace("comp", "ae_comp", 1)
    if stem.startswith("train_ce_"):
        return "ce_" + stem.removeprefix("train_ce_")
    if stem.startswith("train_arcface_"):
        return "arcface_" + stem.removeprefix("train_arcface_")
    if stem.startswith("train_simclr_"):
        return "simclr_" + stem.removeprefix("train_simclr_")
    if stem.startswith("train_simsiam_"):
        return "simsiam_" + stem.removeprefix("train_simsiam_")
    if stem.startswith("train_sep_"):
        return "sep_" + stem.removeprefix("train_sep_")
    if stem.startswith("train_shared_"):
        return "shared_" + stem.removeprefix("train_shared_")
    return None


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in raw_line:
            key, value = raw_line.split("=", 1)
        elif ":" in raw_line:
            key, value = raw_line.split(":", 1)
        else:
            continue
        values[key.strip()] = value.strip()
    return values


def read_runtime_context(runtime_root: Path) -> dict[str, str]:
    context_path = runtime_root / "logs" / "runtime_context.txt"
    if not context_path.exists():
        return {}
    values = read_key_values(context_path)
    return {
        "gpu": values.get("cuda_device_0", ""),
        "pytorch": values.get("torch", ""),
        "cuda_visible_devices": values.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_available": values.get("cuda_available", ""),
        "cuda_device_count": values.get("cuda_device_count", ""),
    }


def read_time_files(runtime_root: Path) -> dict[str, dict[str, str]]:
    measurements: dict[str, dict[str, str]] = {}
    for path in sorted((runtime_root / "logs").glob("*.time")):
        config_id = config_id_from_time_stem(path.stem)
        if config_id is None:
            continue
        values = read_key_values(path)
        status = "completed" if values.get("exit_code") == "0" else "pending_or_failed"
        if "exit_code" in values and values["exit_code"] != "0":
            status = "failed"
        elif not values.get("exit_code"):
            status = "running_or_interrupted"
        measurements[config_id] = {
            "wall_clock_sec": values.get("elapsed_sec", ""),
            "user_sec": values.get("user_sec", ""),
            "sys_sec": values.get("sys_sec", ""),
            "max_rss_kb": values.get("max_rss_kb", ""),
            "exit_code": values.get("exit_code", ""),
            "coverage_status": status,
        }
    return measurements


def summarize(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {"completed": 0, "failed": 0, "running_or_interrupted": 0, "missing": 0}
    for row in rows:
        status = row["coverage_status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join full-grid runtime .time files with the public runtime template."
    )
    parser.add_argument("--runtime_root", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="write output even when not all expected rows are completed",
    )
    args = parser.parse_args()

    with args.template.open(newline="", encoding="utf-8") as handle:
        template_rows = list(csv.DictReader(handle))

    measurements = read_time_files(args.runtime_root)
    context = read_runtime_context(args.runtime_root)
    output_rows: list[dict[str, str]] = []

    for row in template_rows:
        merged = dict(row)
        measured = measurements.get(row["config_id"])
        if measured:
            merged.update(measured)
            if measured["coverage_status"] == "completed":
                merged["notes"] = "Measured with scripts/measure_full_runtime_grid.sh."
        else:
            merged["wall_clock_sec"] = "NA"
            merged["coverage_status"] = "missing"
        if context.get("gpu"):
            merged["gpu"] = context["gpu"]
        if context.get("pytorch"):
            merged["pytorch"] = context["pytorch"]
        merged["cuda_visible_devices"] = context.get("cuda_visible_devices", "")
        merged["cuda_available"] = context.get("cuda_available", "")
        merged["cuda_device_count"] = context.get("cuda_device_count", "")
        merged.setdefault("user_sec", "")
        merged.setdefault("sys_sec", "")
        merged.setdefault("max_rss_kb", "")
        merged.setdefault("exit_code", "")
        output_rows.append(merged)

    counts = summarize(output_rows)
    if not args.allow_partial and counts.get("completed", 0) != len(output_rows):
        raise SystemExit(
            "Runtime grid is incomplete: "
            + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        )

    fieldnames = list(template_rows[0].keys()) + [
        "user_sec",
        "sys_sec",
        "max_rss_kb",
        "exit_code",
        "cuda_visible_devices",
        "cuda_available",
        "cuda_device_count",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"wrote {args.output}")
    print(", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()

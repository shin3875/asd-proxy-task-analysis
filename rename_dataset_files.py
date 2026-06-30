#!/usr/bin/env python3
"""In-place/copy rename utility for DCASE legacy ToyConveyor/Pump filenames.

The script converts filenames such as

    normal_id_00_00000000.wav
    anomaly_id_00_00000000.npy
    fm0_normal_id_00_00000000.wav
    pp12.0_normal_id_03_00000117.npy

to the token format expected by the existing evaluation loaders:

    section_00_source_train_normal_00000000.wav
    section_00_source_test_anomaly_00000000.npy
    fm0_section_00_source_train_normal_00000000.wav
    pp12.0_section_03_source_train_normal_00000117.npy

Key properties:
  - augmentation prefix before normal_id/anomaly_id is preserved;
  - .wav and .npy are supported;
  - ToyConveyer and ToyConveyor are treated as aliases;
  - split is inferred from path components: test > aug > train;
  - aug files are written as source_train by default for loader compatibility;
  - multiple dataset roots can be processed in one command.

Examples:
  python rename_dataset_files.py \
    --root ./asd_dataset:wav \
    --root ./asd_dataset_logmel:npy \
    --root ./asd_dataset_np:npy \
    --devices ToyConveyor pump \
    --mode dry-run

  python rename_dataset_files.py \
    --root ./asd_dataset:wav \
    --root ./asd_dataset_logmel:npy \
    --root ./asd_dataset_np:npy \
    --devices ToyConveyor pump \
    --mode move

Backward-compatible single-root usage is also supported:
  python rename_dataset_files.py \
    --data_dir ./asd_dataset_np \
    --extensions npy \
    --devices ToyConveyer Pump \
    --mode move
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

SUPPORTED_EXTENSIONS = {".wav", ".npy"}

# The prefix is intentionally broad so that augmentation tags such as
# fm0_, pp12.0_, mix_abc_, etc. are preserved verbatim.
LEGACY_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<condition>normal|anomaly)_id_"
    r"(?P<section>\d{2})_(?P<index>\d+)"
    r"(?P<extra>.*?)(?P<ext>\.(?:wav|npy))$",
    re.IGNORECASE,
)

COMPATIBLE_RE = re.compile(
    r"^(?P<prefix>.*?)(?:section_\d{2})_(?:source|target)_(?:train|test|aug)_(?:normal|anomaly)_.*\.(?:wav|npy)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RootSpec:
    root: Path
    extensions: tuple[str, ...]


@dataclass(frozen=True)
class RenameItem:
    root: Path
    src: Path
    dst: Path
    condition: str
    section: str
    split: str
    domain: str
    augmentation_prefix: str
    extension: str
    status: str
    reason: str = ""


@dataclass(frozen=True)
class SkipItem:
    root: Path
    src: Path
    extension: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename legacy normal_id/anomaly_id filenames for ToyConveyor/Pump compatibility."
    )

    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help=(
            "Dataset root spec in the form PATH:EXT[,EXT...], e.g. "
            "./asd_dataset:wav or ./asd_dataset_np:npy. "
            "May be repeated. If provided, it overrides --data_dir/--extensions."
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Backward-compatible single root. Use with --extensions.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["wav", "npy"],
        help="Extensions for --data_dir mode. Use wav, npy, .wav, or .npy.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["ToyConveyor", "ToyConveyer", "pump"],
        help=(
            "Class names to process. ToyConveyer/ToyConveyor spelling is normalized automatically. "
            "Matching is case-insensitive."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "copy", "move"],
        default="dry-run",
        help="dry-run reports mappings; move renames in-place; copy writes to --output_dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output root for copy mode. Only valid for one root unless --copy_output_suffix is used.",
    )
    parser.add_argument(
        "--copy_output_suffix",
        type=str,
        default="_renamed",
        help="Suffix used for per-root output directories in copy mode when --output_dir is omitted.",
    )
    parser.add_argument(
        "--domain",
        choices=["source", "target"],
        default="source",
        help="Domain token to insert. Default: source.",
    )
    parser.add_argument(
        "--aug_split_token",
        choices=["train", "aug"],
        default="train",
        help="Split token for files under aug directories. Default: train for loader compatibility.",
    )
    parser.add_argument(
        "--unknown_split",
        choices=["train", "test", "skip", "condition"],
        default="condition",
        help=(
            "Fallback when split cannot be inferred from path. "
            "condition maps anomaly->test and normal->train. Default: condition."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing destination files. Use only after backup/manifest inspection.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV manifest path. Default: ./rename_manifest_v3.csv or under output_dir/root.",
    )
    parser.add_argument(
        "--skip_manifest",
        type=Path,
        default=None,
        help="CSV path for skipped legacy-like files. Default: ./rename_skipped_v3.csv.",
    )
    parser.add_argument(
        "--report_skipped",
        action="store_true",
        help="Print skipped legacy-like files. Useful for debugging no-change cases.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug limit per all roots. 0 means no limit.",
    )
    return parser.parse_args()


def normalize_extension(value: str) -> str:
    value = value.strip().lower()
    if not value:
        raise ValueError("Empty extension is not valid.")
    if not value.startswith("."):
        value = "." + value
    if value not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported extension: {value}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    return value


def parse_extension_list(values: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        raw: list[str] = []
        for part in values.split(","):
            raw.extend(part.split())
    else:
        raw = list(values)
    extensions = sorted({normalize_extension(v) for v in raw if v.strip()})
    if not extensions:
        raise ValueError("No valid extensions were provided.")
    return tuple(extensions)


def parse_root_spec(spec: str) -> RootSpec:
    # Use rsplit so Windows drive-like paths or unusual names are less likely to break.
    if ":" not in spec:
        raise ValueError(f"Invalid --root spec: {spec!r}. Expected PATH:EXT[,EXT...].")
    path_part, ext_part = spec.rsplit(":", 1)
    if not path_part:
        raise ValueError(f"Invalid --root spec: {spec!r}. Empty path.")
    return RootSpec(root=Path(path_part).resolve(), extensions=parse_extension_list(ext_part))


def build_root_specs(args: argparse.Namespace) -> list[RootSpec]:
    if args.root:
        specs = [parse_root_spec(spec) for spec in args.root]
    else:
        if args.data_dir is None:
            raise ValueError("Provide either repeated --root PATH:EXT specs or --data_dir.")
        specs = [RootSpec(root=args.data_dir.resolve(), extensions=parse_extension_list(args.extensions))]

    for spec in specs:
        if not spec.root.exists():
            raise FileNotFoundError(f"Root does not exist: {spec.root}")
        if not spec.root.is_dir():
            raise NotADirectoryError(f"Root is not a directory: {spec.root}")
    return specs


def normalize_device_token(value: str) -> str:
    # Normalize the common spelling mismatch ToyConveyer vs ToyConveyor.
    value = value.lower().replace("conveyer", "conveyor")
    return re.sub(r"[^a-z0-9]+", "", value)


def path_contains_device(path: Path, devices: Iterable[str]) -> bool:
    normalized_devices = {normalize_device_token(device) for device in devices}
    normalized_parts = [normalize_device_token(part) for part in path.parts]
    normalized_full = normalize_device_token(str(path))

    for device in normalized_devices:
        if not device:
            continue
        # Prefer component-level matching, but allow full-path containment for nested names.
        if device in normalized_parts or device in normalized_full:
            return True
    return False


def path_contains_split_token(part: str, token: str) -> bool:
    part = part.lower()
    token = token.lower()
    # Handles test, TEST, dev_test, source_test, train, source_train, augmented_train, etc.
    return token in re.split(r"[^a-z0-9]+", part) or token in part


def infer_split(relative_path: Path, condition: str, aug_split_token: str, unknown_split: str) -> Optional[str]:
    parts = [part.lower() for part in relative_path.parts[:-1]]  # directory components only

    # Priority is intentional. A test directory containing anomaly files must be source_test.
    if any(path_contains_split_token(part, "test") or path_contains_split_token(part, "eval") for part in parts):
        return "test"
    if any("aug" in part for part in parts):
        return aug_split_token
    if any(path_contains_split_token(part, "train") for part in parts):
        return "train"

    if unknown_split == "skip":
        return None
    if unknown_split in {"train", "test"}:
        return unknown_split
    if unknown_split == "condition":
        return "test" if condition == "anomaly" else "train"

    raise ValueError(f"Unsupported unknown_split policy: {unknown_split}")


def convert_filename(src: Path, split: str, domain: str) -> Optional[tuple[str, str, str, str, str]]:
    match = LEGACY_RE.match(src.name)
    if match is None:
        return None

    prefix = match.group("prefix") or ""
    condition = match.group("condition").lower()
    section = match.group("section")
    index = match.group("index")
    extra = match.group("extra") or ""
    ext = match.group("ext").lower()

    if extra and not extra.startswith("_"):
        extra = "_" + extra

    new_name = f"{prefix}section_{section}_{domain}_{split}_{condition}_{index}{extra}{ext}"
    return new_name, condition, section, prefix, ext


def output_root_for_copy(args: argparse.Namespace, spec: RootSpec, spec_count: int) -> Path:
    if args.output_dir is not None:
        if spec_count > 1:
            # Preserve root name under a common output parent for multi-root copy mode.
            return args.output_dir.resolve() / spec.root.name
        return args.output_dir.resolve()
    return spec.root.with_name(spec.root.name + args.copy_output_suffix).resolve()


def destination_for(args: argparse.Namespace, spec: RootSpec, relative_path: Path, new_name: str, spec_count: int) -> Path:
    if args.mode == "copy":
        return output_root_for_copy(args, spec, spec_count) / relative_path.parent / new_name
    return spec.root / relative_path.parent / new_name


def build_plan(args: argparse.Namespace, specs: list[RootSpec]) -> tuple[list[RenameItem], list[SkipItem]]:
    items: list[RenameItem] = []
    skipped: list[SkipItem] = []

    for spec in specs:
        for src in sorted(spec.root.rglob("*")):
            if not src.is_file():
                continue
            extension = src.suffix.lower()
            if extension not in spec.extensions:
                continue

            rel = src.relative_to(spec.root)

            # Avoid scanning the output tree if the user places it under a root in copy mode.
            if args.mode == "copy":
                out_root = output_root_for_copy(args, spec, len(specs))
                try:
                    src.resolve().relative_to(out_root)
                    continue
                except ValueError:
                    pass

            matched_pattern = LEGACY_RE.match(src.name) is not None
            already_compatible = COMPATIBLE_RE.match(src.name) is not None
            matched_device = path_contains_device(src, args.devices)

            if not matched_device:
                if matched_pattern or already_compatible:
                    skipped.append(SkipItem(spec.root, src, extension, "device_not_matched"))
                continue
            if already_compatible and not matched_pattern:
                skipped.append(SkipItem(spec.root, src, extension, "already_compatible"))
                continue
            if not matched_pattern:
                # Report suspicious files only, not every arbitrary file.
                lower_name = src.name.lower()
                if "_id_" in lower_name or "normal" in lower_name or "anomaly" in lower_name:
                    skipped.append(SkipItem(spec.root, src, extension, "filename_pattern_not_matched"))
                continue

            # First parse condition/section without relying on split.
            parsed = convert_filename(src, split="__split__", domain=args.domain)
            assert parsed is not None
            _, condition, section, prefix, ext = parsed
            split = infer_split(rel, condition, args.aug_split_token, args.unknown_split)
            if split is None:
                skipped.append(SkipItem(spec.root, src, extension, "split_not_inferred"))
                continue

            converted = convert_filename(src, split=split, domain=args.domain)
            assert converted is not None
            new_name, condition, section, prefix, ext = converted
            dst = destination_for(args, spec, rel, new_name, len(specs))

            status = "planned"
            reason = ""
            if src.resolve() == dst.resolve():
                status = "already_compatible"
                reason = "source_equals_destination"
            elif dst.exists() and not args.overwrite:
                status = "blocked_destination_exists"
                reason = "destination_exists"

            items.append(
                RenameItem(
                    root=spec.root,
                    src=src,
                    dst=dst,
                    condition=condition,
                    section=section,
                    split=split,
                    domain=args.domain,
                    augmentation_prefix=prefix,
                    extension=ext,
                    status=status,
                    reason=reason,
                )
            )

            if args.limit and len(items) >= args.limit:
                return items, skipped

    return items, skipped


def detect_collisions(items: list[RenameItem]) -> dict[Path, list[Path]]:
    buckets: dict[Path, list[Path]] = {}
    for item in items:
        if item.status == "blocked_destination_exists" and not item.dst.exists():
            continue
        buckets.setdefault(item.dst.resolve(), []).append(item.src)
    return {dst: srcs for dst, srcs in buckets.items() if len(srcs) > 1}


def execute(items: list[RenameItem], args: argparse.Namespace) -> list[RenameItem]:
    executed: list[RenameItem] = []
    for item in items:
        if args.mode == "dry-run":
            executed.append(item)
            continue

        if item.status == "blocked_destination_exists" and not args.overwrite:
            executed.append(item)
            continue
        if item.status == "already_compatible" and not args.overwrite:
            executed.append(item)
            continue

        item.dst.parent.mkdir(parents=True, exist_ok=True)

        if args.mode == "move":
            if args.overwrite:
                os.replace(item.src, item.dst)
            else:
                item.src.rename(item.dst)
            executed.append(replace(item, status="moved", reason=""))
        elif args.mode == "copy":
            shutil.copy2(item.src, item.dst)
            executed.append(replace(item, status="copied", reason=""))
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    return executed


def default_manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest is not None:
        return args.manifest.resolve()
    return Path("rename_manifest_v3.csv").resolve()


def default_skip_manifest_path(args: argparse.Namespace) -> Path:
    if args.skip_manifest is not None:
        return args.skip_manifest.resolve()
    return Path("rename_skipped_v3.csv").resolve()


def write_manifest(items: list[RenameItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "status",
                "reason",
                "root",
                "src",
                "dst",
                "condition",
                "section",
                "split",
                "domain",
                "augmentation_prefix",
                "extension",
            ]
        )
        for item in items:
            writer.writerow(
                [
                    item.status,
                    item.reason,
                    str(item.root),
                    str(item.src),
                    str(item.dst),
                    item.condition,
                    item.section,
                    item.split,
                    item.domain,
                    item.augmentation_prefix,
                    item.extension,
                ]
            )


def write_skip_manifest(items: list[SkipItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["reason", "root", "src", "extension"])
        for item in items:
            writer.writerow([item.reason, str(item.root), str(item.src), item.extension])


def print_summary(items: list[RenameItem], skipped: list[SkipItem], manifest: Path, skip_manifest: Path, report_skipped: bool) -> None:
    status_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    ext_counts: dict[str, int] = {}
    root_counts: dict[str, int] = {}
    prefix_count = 0

    for item in items:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1
        split_counts[item.split] = split_counts.get(item.split, 0) + 1
        ext_counts[item.extension] = ext_counts.get(item.extension, 0) + 1
        root_counts[str(item.root)] = root_counts.get(str(item.root), 0) + 1
        if item.augmentation_prefix:
            prefix_count += 1

    skip_counts: dict[str, int] = {}
    for item in skipped:
        skip_counts[item.reason] = skip_counts.get(item.reason, 0) + 1

    print("\nSummary")
    print("-------")
    print(f"candidate mappings: {len(items)}")
    for key in sorted(status_counts):
        print(f"{key}: {status_counts[key]}")
    for key in sorted(split_counts):
        print(f"split={key}: {split_counts[key]}")
    for key in sorted(ext_counts):
        print(f"ext={key}: {ext_counts[key]}")
    print(f"augmentation-prefixed items: {prefix_count}")
    for root, count in sorted(root_counts.items()):
        print(f"root items: {count}  {root}")

    print("\nSkipped legacy-like/compatible files")
    print("------------------------------------")
    if not skipped:
        print("none")
    else:
        for key in sorted(skip_counts):
            print(f"{key}: {skip_counts[key]}")

    print(f"\nmanifest: {manifest}")
    print(f"skip_manifest: {skip_manifest}")

    print("\nFirst 30 mappings")
    print("-----------------")
    for item in items[:30]:
        rel_src = item.src.relative_to(item.root)
        rel_dst = item.dst.name
        print(f"[{item.status}] {rel_src}  ->  {rel_dst}")

    if report_skipped and skipped:
        print("\nFirst 30 skipped")
        print("----------------")
        for item in skipped[:30]:
            print(f"[{item.reason}] {item.src.relative_to(item.root)}")


def main() -> None:
    args = parse_args()
    specs = build_root_specs(args)
    items, skipped = build_plan(args, specs)

    collisions = detect_collisions([item for item in items if item.status != "blocked_destination_exists"])
    if collisions:
        print("Destination collision detected. No file operations were performed.")
        for dst, srcs in list(collisions.items())[:20]:
            print(f"\nDestination: {dst}")
            for src in srcs:
                print(f"  - {src}")
        raise SystemExit(2)

    executed = execute(items, args)
    manifest = default_manifest_path(args)
    skip_manifest = default_skip_manifest_path(args)
    write_manifest(executed, manifest)
    write_skip_manifest(skipped, skip_manifest)
    print_summary(executed, skipped, manifest, skip_manifest, args.report_skipped)


if __name__ == "__main__":
    main()

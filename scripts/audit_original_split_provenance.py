#!/usr/bin/env python3
"""Audit source-image provenance leakage in the published augmented splits.

The Mendeley dataset names every augmented file as
``<source-image-id>_rot<angle>_<operation>.<extension>``. This script treats the
filename prefix before ``_rot`` as the source-image identifier and measures how
often variants of the same source occur in more than one partition.

No image files are modified.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILENAME_PATTERN = re.compile(
    r"^(?P<source_id>.+?)_rot(?P<rotation>0|90|180|270)_"
    r"(?P<operation>bright|noise|shear|zoom)$",
    re.IGNORECASE,
)


def normalized_class_name(folder_name: str) -> str:
    suffix = " images"
    return folder_name[: -len(suffix)] if folder_name.endswith(suffix) else folder_name


def collect_split(split_root: Path, split: str):
    records = []
    errors = []
    split_dir = split_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    for path in sorted(split_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = FILENAME_PATTERN.match(path.stem)
        if match is None:
            errors.append(str(path.relative_to(split_root)))
            continue
        records.append(
            {
                "split": split,
                "class_name": normalized_class_name(path.parent.name),
                "source_id": match.group("source_id"),
                "rotation": int(match.group("rotation")),
                "operation": match.group("operation").lower(),
                "relative_path": str(path.relative_to(split_root)),
            }
        )
    return records, errors


def overlap_counts(source_sets):
    train = source_sets["train"]
    valid = source_sets["valid"]
    test = source_sets["test"]
    all_three = train & valid & test
    return {
        "train_valid_inclusive": len(train & valid),
        "train_test_inclusive": len(train & test),
        "valid_test_inclusive": len(valid & test),
        "train_valid_only": len((train & valid) - test),
        "train_test_only": len((train & test) - valid),
        "valid_test_only": len((valid & test) - train),
        "all_three": len(all_three),
    }


def build_report(split_root: Path):
    records = []
    unexpected = []
    for split in SPLITS:
        split_records, split_errors = collect_split(split_root, split)
        records.extend(split_records)
        unexpected.extend(split_errors)
    if unexpected:
        sample = "\n".join(unexpected[:10])
        raise ValueError(
            f"Found {len(unexpected)} filename(s) that do not match the documented "
            f"provenance pattern. First entries:\n{sample}"
        )

    by_class = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_class[record["class_name"]][record["split"]].append(record)

    class_reports = {}
    global_sets = {split: set() for split in SPLITS}
    global_test_files_with_seen_source = 0
    for class_name in sorted(by_class):
        class_records = by_class[class_name]
        source_sets = {
            split: {record["source_id"] for record in class_records[split]}
            for split in SPLITS
        }
        for split in SPLITS:
            global_sets[split].update((class_name, source_id) for source_id in source_sets[split])

        seen_before_test = source_sets["train"] | source_sets["valid"]
        test_files_with_seen_source = sum(
            record["source_id"] in seen_before_test for record in class_records["test"]
        )
        global_test_files_with_seen_source += test_files_with_seen_source
        class_reports[class_name] = {
            "file_counts": {split: len(class_records[split]) for split in SPLITS},
            "unique_source_id_counts": {
                split: len(source_sets[split]) for split in SPLITS
            },
            "source_id_overlaps": overlap_counts(source_sets),
            "test_unique_source_ids_not_in_train_or_valid": len(
                source_sets["test"] - seen_before_test
            ),
            "test_files_whose_source_id_occurs_in_train_or_valid": test_files_with_seen_source,
        }

    global_overlaps = overlap_counts(global_sets)
    global_report = {
        "file_counts": {
            split: sum(1 for record in records if record["split"] == split)
            for split in SPLITS
        },
        "unique_class_source_id_counts": {
            split: len(global_sets[split]) for split in SPLITS
        },
        "class_source_id_overlaps": global_overlaps,
        "test_unique_class_source_ids_not_in_train_or_valid": len(
            global_sets["test"] - (global_sets["train"] | global_sets["valid"])
        ),
        "test_files_whose_class_source_id_occurs_in_train_or_valid": (
            global_test_files_with_seen_source
        ),
    }

    source_label = "/".join(split_root.resolve().parts[-2:])
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        # Store a portable dataset-relative label rather than a contributor's
        # machine-specific absolute path in the public report.
        "source_split_root": source_label,
        "source_identifier_rule": (
            "Within each class, the filename prefix before '_rot' identifies the "
            "original source image used to create an augmented variant."
        ),
        "splits": list(SPLITS),
        "status": (
            "LEAKAGE_DETECTED"
            if global_report["test_files_whose_class_source_id_occurs_in_train_or_valid"]
            else "NO_PROVENANCE_OVERLAP_DETECTED"
        ),
        "global": global_report,
        "classes": class_reports,
    }


def print_summary(report):
    print("# ORIGINAL AUGMENTED-SPLIT PROVENANCE AUDIT")
    print(f"Status: {report['status']}")
    print()
    print(
        f"{'Class':<22} {'Train':>6} {'Valid':>6} {'Test':>6} "
        f"{'T-V only':>9} {'T-Test only':>11} {'All 3':>7} {'Unseen test':>11}"
    )
    for class_name, item in report["classes"].items():
        unique = item["unique_source_id_counts"]
        overlap = item["source_id_overlaps"]
        print(
            f"{class_name:<22} {unique['train']:>6} {unique['valid']:>6} {unique['test']:>6} "
            f"{overlap['train_valid_only']:>9} {overlap['train_test_only']:>11} "
            f"{overlap['all_three']:>7} "
            f"{item['test_unique_source_ids_not_in_train_or_valid']:>11}"
        )
    print()
    global_item = report["global"]
    print(f"Published split files: {global_item['file_counts']}")
    print(
        "Test files whose class/source ID also occurs in train or validation: "
        f"{global_item['test_files_whose_class_source_id_occurs_in_train_or_valid']}"
    )
    print(
        "Test class/source IDs unseen in train and validation: "
        f"{global_item['test_unique_class_source_ids_not_in_train_or_valid']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-root",
        type=Path,
        required=True,
        help="Path to the source dataset's 'Meat Freshness/Data Splits' directory.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(args.split_root)
    print_summary(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()

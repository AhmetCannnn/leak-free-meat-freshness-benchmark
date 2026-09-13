#!/usr/bin/env python3
"""Verify files, counts, hashes, classes, and cross-split isolation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def digest(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify(root: Path, manifest_path: Path, config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    errors: list[str] = []
    split_counts = Counter()
    class_counts: dict[str, Counter] = defaultdict(Counter)
    hashes_by_split: dict[str, set[str]] = defaultdict(set)
    hashes_by_class: dict[str, set[str]] = defaultdict(set)
    manifest_paths: set[str] = set()

    for row in rows:
        split = row["split"]
        path = root / row["relative_path"]
        manifest_paths.add(row["relative_path"])
        split_counts[split] += 1
        class_counts[split][row["class_name"]] += 1
        if not path.is_file():
            errors.append(f"missing file: {row['relative_path']}")
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            errors.append(f"size mismatch: {row['relative_path']}")
        actual_hash = digest(path)
        if actual_hash != row["md5"]:
            errors.append(f"hash mismatch: {row['relative_path']}")
        if actual_hash in hashes_by_split[split]:
            errors.append(f"within-split duplicate in {split}: {actual_hash}")
        hashes_by_split[split].add(actual_hash)
        hashes_by_class[row["class_name"]].add(actual_hash)

    expected = config["expected_counts"]
    for split in ("train", "valid", "test"):
        if split_counts[split] != expected[split]:
            errors.append(f"{split} count: expected {expected[split]}, found {split_counts[split]}")
    if len(rows) != expected["total"]:
        errors.append(f"total count: expected {expected['total']}, found {len(rows)}")

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    extra_paths = sorted(actual_paths - manifest_paths)
    if extra_paths:
        errors.append(f"files not listed in manifest: {len(extra_paths)}")

    expected_classes = set(config["classes"])
    for split in ("train", "valid", "test"):
        if set(class_counts[split]) != expected_classes:
            errors.append(f"class set mismatch in {split}")
        expected_class_counts = config["expected_class_counts"][split]
        if dict(class_counts[split]) != expected_class_counts:
            errors.append(
                f"class counts mismatch in {split}: "
                f"expected {expected_class_counts}, found {dict(class_counts[split])}"
            )

    leakage = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        hash_overlap = sorted(hashes_by_split[left] & hashes_by_split[right])
        leakage[f"{left}-{right}"] = {
            "md5_overlap_count": len(hash_overlap),
        }
        if hash_overlap:
            errors.append(f"hash leakage between {left} and {right}: {len(hash_overlap)}")

    mutton_overlap = sorted(hashes_by_class["36 hr Mutton"] & hashes_by_class["48 hr Mutton"])
    if mutton_overlap:
        errors.append(
            "MD5 overlap between 36 hr Mutton and 48 hr Mutton: "
            f"{len(mutton_overlap)}"
        )

    return {
        "benchmark_version": config["benchmark_version"],
        "manifest": str(manifest_path),
        "hash_algorithm": "MD5",
        "split_counts": dict(split_counts),
        "class_counts": {key: dict(value) for key, value in class_counts.items()},
        "leakage": leakage,
        "mutton_36_48_shared_md5_count": len(mutton_overlap),
        "checks_passed": not errors,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/benchmark_manifest_md5.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/benchmark_verification.json"))
    args = parser.parse_args()
    report = verify(args.benchmark_root, args.manifest, args.config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["checks_passed"] else 1)


if __name__ == "__main__":
    main()

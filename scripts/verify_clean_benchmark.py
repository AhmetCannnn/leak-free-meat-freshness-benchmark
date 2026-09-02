#!/usr/bin/env python3
"""Verify files, counts, hashes, classes, and cross-split isolation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
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
    source_ids_by_split: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        split = row["split"]
        path = root / row["clean_relative_path"]
        split_counts[split] += 1
        class_counts[split][row["clean_class"]] += 1
        if not path.is_file():
            errors.append(f"missing file: {row['clean_relative_path']}")
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            errors.append(f"size mismatch: {row['clean_relative_path']}")
        actual_hash = digest(path)
        if actual_hash != row["sha256"]:
            errors.append(f"hash mismatch: {row['clean_relative_path']}")
        if actual_hash in hashes_by_split[split]:
            errors.append(f"within-split duplicate in {split}: {actual_hash}")
        hashes_by_split[split].add(actual_hash)
        for source_id in filter(None, row.get("source_relative_path", "").split("|")):
            source_ids_by_split[split].add(source_id)

    expected = config["expected_counts"]
    for split in ("train", "valid", "test"):
        if split_counts[split] != expected[split]:
            errors.append(f"{split} count: expected {expected[split]}, found {split_counts[split]}")
    if len(rows) != expected["total"]:
        errors.append(f"total count: expected {expected['total']}, found {len(rows)}")

    expected_classes = set(config["classes"])
    for split in ("train", "valid", "test"):
        if set(class_counts[split]) != expected_classes:
            errors.append(f"class set mismatch in {split}")

    leakage = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        hash_overlap = sorted(hashes_by_split[left] & hashes_by_split[right])
        source_overlap = sorted(source_ids_by_split[left] & source_ids_by_split[right])
        leakage[f"{left}-{right}"] = {
            "sha256_overlap_count": len(hash_overlap),
            "source_identity_overlap_count": len(source_overlap),
        }
        if hash_overlap:
            errors.append(f"hash leakage between {left} and {right}: {len(hash_overlap)}")
        if source_overlap:
            errors.append(f"source leakage between {left} and {right}: {len(source_overlap)}")

    return {
        "benchmark_version": config["benchmark_version"],
        "manifest": str(manifest_path),
        "split_counts": dict(split_counts),
        "class_counts": {key: dict(value) for key, value in class_counts.items()},
        "leakage": leakage,
        "checks_passed": not errors,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/benchmark_manifest.csv"))
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


#!/usr/bin/env python3
"""Audit exact duplicate images within and between source-dataset classes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/original_dataset_audit.json"))
    args = parser.parse_args()

    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    class_counts: dict[str, int] = defaultdict(int)
    for path in sorted(args.source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(args.source_root)
        class_name = relative.parent.as_posix()
        class_counts[class_name] += 1
        by_hash[digest(path)].append({"path": relative.as_posix(), "class": class_name})

    duplicate_groups = []
    for sha256, files in sorted(by_hash.items()):
        if len(files) > 1:
            duplicate_groups.append({
                "sha256": sha256,
                "count": len(files),
                "cross_class": len({item["class"] for item in files}) > 1,
                "files": files,
            })

    pair_overlaps: Counter[tuple[str, str]] = Counter()
    for files in by_hash.values():
        classes = sorted({item["class"] for item in files})
        for left, right in combinations(classes, 2):
            pair_overlaps[(left, right)] += 1
    report = {
        "hash_algorithm": "sha256",
        "total_files": sum(class_counts.values()),
        "unique_hashes": len(by_hash),
        "class_counts": dict(sorted(class_counts.items())),
        "duplicate_group_count": len(duplicate_groups),
        "cross_class_duplicate_group_count": sum(item["cross_class"] for item in duplicate_groups),
        "class_pair_unique_hash_overlaps": [
            {"class_a": pair[0], "class_b": pair[1], "shared_unique_hashes": count}
            for pair, count in sorted(pair_overlaps.items())
        ],
        "duplicate_groups": duplicate_groups,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audited {report['total_files']} files; found {len(duplicate_groups)} duplicate groups")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()

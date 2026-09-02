#!/usr/bin/env python3
"""Create the canonical CSV manifest for an existing clean benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_index(source_root: Path | None) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    if source_root is None:
        return index
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[digest(path)].append(path.relative_to(source_root).as_posix())
    return index


def create_manifest(benchmark_root: Path, output: Path, source_root: Path | None) -> int:
    sources = source_index(source_root)
    rows = []
    for split in SPLITS:
        split_root = benchmark_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_root}")
        for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                sha256 = digest(path)
                matches = sources.get(sha256, [])
                rows.append({
                    "split": split,
                    "clean_class": class_dir.name,
                    "clean_filename": path.name,
                    "clean_relative_path": path.relative_to(benchmark_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256,
                    "source_relative_path": "|".join(matches),
                    "source_match_count": len(matches),
                })

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("manifests/benchmark_manifest.csv"))
    args = parser.parse_args()
    count = create_manifest(args.benchmark_root, args.output, args.source_root)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()


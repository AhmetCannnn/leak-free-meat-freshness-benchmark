#!/usr/bin/env python3
"""Reconstruct the exact clean benchmark from a source tree and manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import defaultdict
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
    parser.add_argument("--manifest", type=Path, default=Path("manifests/benchmark_manifest.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {args.output}")

    index: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(args.source_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[digest(path)].append(path)

    with args.manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        matches = index.get(row["sha256"], [])
        if not matches:
            raise FileNotFoundError(f"Source image not found for SHA-256 {row['sha256']}")
        destination = args.output / row["clean_relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matches[0], destination)
    print(f"Reconstructed {len(rows)} files in {args.output}")


if __name__ == "__main__":
    main()


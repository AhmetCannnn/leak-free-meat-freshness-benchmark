# Leak-Free Meat Freshness Benchmark

Reproducibility package for an 8-class beef and mutton freshness benchmark
created after auditing structural duplication and cross-split leakage in a
publicly available source dataset.

> **Release status:** pre-release (`0.1.0`). The benchmark manifest and leakage
> verification are included. Model-result reconciliation and the archival DOI
> are still pending; no DOI or final paper result is claimed in this version.

## Main contribution

The package provides:

- an exact, versioned manifest for the 1,849-image clean benchmark;
- source-dataset duplicate auditing;
- deterministic reconstruction from source images and SHA-256 checksums;
- within-split and cross-split leakage verification;
- training and checkpoint evaluation for four compact image classifiers.

## Benchmark

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| 0 hr Beef | 140 | 30 | 31 | 201 |
| 12 hr Beef | 142 | 31 | 31 | 204 |
| 24 hr Beef | 158 | 34 | 35 | 227 |
| 36+ hr Beef | 157 | 34 | 34 | 225 |
| 0 hr Mutton | 144 | 31 | 31 | 206 |
| 12 hr Mutton | 146 | 31 | 32 | 209 |
| 24 hr Mutton | 151 | 32 | 33 | 216 |
| 36+ hr Mutton | 252 | 54 | 55 | 361 |
| **Total** | **1,290** | **277** | **282** | **1,849** |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create or refresh the canonical manifest from an existing clean benchmark:

```bash
python scripts/create_benchmark_manifest.py \
  --benchmark-root /path/to/Clean_Dataset_8class_splits \
  --source-root /path/to/source/original/images
```

Verify the benchmark:

```bash
python scripts/verify_clean_benchmark.py \
  --benchmark-root /path/to/Clean_Dataset_8class_splits
```

Reconstruct it into a new, empty directory:

```bash
python scripts/build_clean_benchmark.py \
  --source-root /path/to/source/original/images \
  --output data/clean
```

Audit exact duplicates in the source data:

```bash
python scripts/audit_original_dataset.py \
  --source-root /path/to/source/original/images
```

Train and evaluate the mobile models:

```bash
python scripts/train_mobile_models.py --dataset-root data/clean --output runs
python scripts/evaluate_checkpoints.py --dataset-root data/clean --checkpoint-root runs
```

## Reproducibility contract

The manifest, configuration, scripts, and reports in one tagged release must
be used together. A benchmark passes verification only when all configured
counts and classes match and neither SHA-256 nor source identity crosses a
split boundary.

See [the protocol](docs/benchmark_protocol.md), [data instructions](data/README.md),
and [manifest documentation](manifests/README.md).

## Data and licensing

The software is MIT licensed. That license does not relicense third-party
images. Consult `data/README.md` and the source dataset terms before
redistributing image files.

## Citation

Citation metadata is provided in `CITATION.cff`. A Zenodo DOI, `v1.0.0` tag,
and corresponding commit hash will be added after the release contents and
reported model metrics are frozen.


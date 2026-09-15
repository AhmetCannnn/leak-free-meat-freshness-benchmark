# Leak-Free Meat Freshness Benchmark

Reproducibility package for the nine-class beef and mutton freshness benchmark described in the associated manuscript. The package records the original-dataset audit, the fixed clean split, leakage verification, the four-model five-seed training protocol, and the reported aggregate results.

> **Release status:** `v1.0.0` is complete and verified. The cleaned benchmark is permanently archived at Zenodo under DOI [`10.5281/zenodo.22733892`](https://doi.org/10.5281/zenodo.22733892).

## Benchmark

The benchmark contains 1,849 original images. Offline-augmented images are excluded. Exact duplicate checks use MD5, matching the manuscript methodology. The byte-identical 36 h and 48 h Beef folders are represented once as `36+ hr Beef`; the distinct 36 h and 48 h Mutton stages remain separate.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| 0 hr Beef | 140 | 30 | 31 | 201 |
| 12 hr Beef | 142 | 31 | 31 | 204 |
| 24 hr Beef | 158 | 34 | 35 | 227 |
| 36+ hr Beef | 157 | 34 | 34 | 225 |
| 0 hr Mutton | 144 | 31 | 31 | 206 |
| 12 hr Mutton | 146 | 31 | 32 | 209 |
| 24 hr Mutton | 151 | 32 | 33 | 216 |
| 36 hr Mutton | 103 | 22 | 23 | 148 |
| 48 hr Mutton | 149 | 32 | 32 | 213 |
| **Total** | **1,290** | **277** | **282** | **1,849** |

## Repository contents

- `manifests/benchmark_manifest_md5.csv`: exact split membership and MD5 digest for every benchmark image.
- `manifests/verification_report.json`: fixed-split verification metadata used by the training notebook.
- `scripts/audit_original_dataset.py`: exact-duplicate audit for the Mendeley original images.
- `scripts/audit_original_split_provenance.py`: filename-based source-provenance audit showing that augmented variants of the same original image occur across the published training, validation, and test partitions.
- `scripts/build_clean_benchmark.py`: deterministic reconstruction from the original images and manifest.
- `scripts/verify_clean_benchmark.py`: class, count, file, MD5, and cross-split leakage checks.
- `notebooks/meat_freshness_9class_multirun.ipynb`: resumable four-model, five-seed experiment and paper-ready aggregation.
- `results/`: the verified aggregate tables, seed-level comparisons, McNemar-Holm results, and Figure 16 generated from 20 completed runs.

Image files and model checkpoints are not committed to Git. The cleaned image benchmark is archived separately under Zenodo DOI [`10.5281/zenodo.22733892`](https://doi.org/10.5281/zenodo.22733892). The source and cleaned images remain under CC BY 4.0; see `data/README.md` for attribution and the change notice.

## Install

Python 3.9 was used for the reported experiment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Reconstruct and verify the benchmark

Download version 1 of the source dataset from [Mendeley Data](https://data.mendeley.com/datasets/4tj9t3n6vj/1). Point `--source-root` to its `Meat Freshness/Original Images` directory.

```bash
python3 scripts/audit_original_dataset.py \
  --source-root "/path/to/Meat Freshness/Original Images"

python3 scripts/audit_original_split_provenance.py \
  --split-root "/path/to/Meat Freshness/Data Splits" \
  --output reports/original_split_provenance_audit.json

python3 scripts/build_clean_benchmark.py \
  --source-root "/path/to/Meat Freshness/Original Images" \
  --output Clean_Dataset_9class_splits

python3 scripts/verify_clean_benchmark.py \
  --benchmark-root Clean_Dataset_9class_splits
```

The verification command must report 1,849 files, nine expected classes, no duplicate MD5 within a split, no MD5 overlap across splits, and no MD5 overlap between the 36 h and 48 h Mutton classes.

The provenance audit uses the source dataset's documented augmented-file naming pattern: the prefix before `_rot` is the original image identifier. It reports overlap both inclusively and as mutually exclusive train-validation, train-test, validation-test, and all-three-partition groups. In the published augmented splits, every one of the 1,800 test files traces to a class/source identifier that also occurs in training or validation; no test source identifier is unseen in both partitions.

## Training and resume

The notebook defaults to a two-epoch pilot. The full command selects four models, five prespecified seeds (`42`, `123`, `2026`, `3407`, `9103`), and at most 35 epochs per run:

```bash
python3 scripts/run_notebook_streaming.py --full
```

Each completed epoch is checkpointed. Restarting the same command resumes an incomplete compatible run and skips verified completed runs. The paper-ready aggregation is blocked unless all 20 compatible runs exist.

## Reported results

The principal five-run test results are:

| Model | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
|---|---:|---:|---:|---:|
| MobileNetV3-Large | 98.51 ± 0.58 | 98.56 ± 0.53 | 98.59 ± 0.53 | 98.54 ± 0.55 |
| EfficientNet-B0 | 96.74 ± 0.98 | 96.81 ± 1.02 | 96.86 ± 0.96 | 96.76 ± 1.03 |
| ShuffleNetV2 | 93.33 ± 1.69 | 93.77 ± 1.49 | 93.69 ± 1.61 | 92.84 ± 1.84 |
| MobileNetV2 | 91.77 ± 1.53 | 91.93 ± 1.59 | 92.10 ± 1.45 | 91.77 ± 1.60 |

All displayed values are derived from the machine-readable files under `results/`. CPU-inference profiling and Apple Silicon MPS training-efficiency measurements are separate protocols in the manuscript and must not be conflated.

## Reproducibility contract

Use the manifest, configuration, scripts, reports, notebook, and results from the same tagged release. Do not combine these files with pilot outputs or files from earlier project versions. See `docs/benchmark_protocol.md` for the complete decision record.

## Citation

Citation metadata is provided in `CITATION.cff`. Cite the cleaned dataset using Zenodo DOI [`10.5281/zenodo.22733892`](https://doi.org/10.5281/zenodo.22733892) and use the matching `v1.0.0` GitHub release for code and result provenance.

# Data access

The image files are not committed to this Git repository. Download version 1 of the source dataset from:

https://data.mendeley.com/datasets/4tj9t3n6vj/1

Use the `Meat Freshness/Original Images` directory as the source root for the audit and reconstruction commands. The canonical nine-class benchmark is defined by `manifests/benchmark_manifest_md5.csv` and can be reconstructed with `scripts/build_clean_benchmark.py`.

Expected reconstructed layout:

```text
Clean_Dataset_9class_splits/
  train/<nine class folders>/
  valid/<nine class folders>/
  test/<nine class folders>/
```

The source dataset terms govern redistribution of the images. This repository distributes the benchmark definition, audit and reconstruction code, verification evidence, and aggregate experimental results; it does not relicense the source images.

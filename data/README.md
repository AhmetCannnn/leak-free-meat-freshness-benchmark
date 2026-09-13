# Data access

The image files are not committed to this Git repository. The cleaned, fixed-split benchmark is archived at Zenodo under DOI:

https://doi.org/10.5281/zenodo.22733892

The benchmark is derived from version 1 of the source dataset:

https://data.mendeley.com/datasets/4tj9t3n6vj/1

Use the `Meat Freshness/Original Images` directory as the source root for the audit and reconstruction commands. The canonical nine-class benchmark is defined by `manifests/benchmark_manifest_md5.csv` and can be reconstructed with `scripts/build_clean_benchmark.py`.

Expected reconstructed layout:

```text
Clean_Dataset_9class_splits/
  train/<nine class folders>/
  valid/<nine class folders>/
  test/<nine class folders>/
```

The source and cleaned images are governed by the source dataset's CC BY 4.0 license. The Zenodo package credits the source authors, links the original DOI `10.17632/4tj9t3n6vj.1`, and documents all cleaning, deduplication, relabeling, and split-assignment changes. The MIT License in this repository applies to the software and does not replace the CC BY 4.0 terms for the images.

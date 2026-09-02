# Benchmark protocol

## Cleaning decisions

1. Offline-augmented images are excluded.
2. Byte-identical images are detected by cryptographic digest.
3. The duplicated 48-hour beef category is not an independent class.
4. The 36-hour and 48-hour mutton categories are merged as `36+ hr Mutton`.
5. The canonical split has 1,290 train, 277 validation, and 282 test images.
6. Augmentation is dynamic and training-only.

## Leakage criteria

Verification requires no SHA-256 or source identity across splits, no duplicate
digest within a split, correct manifest metadata, and exact configured counts.
Perceptual-similarity audits are reported separately from cryptographic checks.


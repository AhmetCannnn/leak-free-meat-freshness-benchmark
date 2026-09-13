# Benchmark protocol

## Source and scope

The source is version 1 of the Mendeley Data collection linked in the repository README. Only the `Meat Freshness/Original Images` tree is used. The offline-augmented tree is excluded from benchmark membership.

## Cleaning decisions

1. Exact file-content identity is assessed with MD5, matching the manuscript audit.
2. Within-class exact duplicates are represented once.
3. The 36 h and 48 h Beef folders are byte-identical and are represented once as `36+ hr Beef`.
4. The 36 h and 48 h Mutton folders contain distinct content and remain separate classes.
5. The existing original-image split memberships are preserved; no random re-splitting is performed.
6. The fixed benchmark contains 1,290 training, 277 validation, and 282 test images.
7. Augmentation is applied dynamically to training images only.

## Leakage criteria

A benchmark passes verification only when:

- all 1,849 manifest rows resolve to files with the recorded size and MD5;
- every split contains the nine configured classes and exact expected class counts;
- there is no duplicate MD5 within any split;
- there is no shared MD5 between training, validation, and test;
- the 36 h and 48 h Mutton classes have no shared MD5.

MD5 is used here as an exact byte-identity check, not as a security mechanism or a perceptual-similarity measure.

## Model evaluation

MobileNetV3-Large, EfficientNet-B0, MobileNetV2, and ShuffleNetV2 x1.0 are evaluated with the same fixed splits and common training configuration. Each model is run with seeds 42, 123, 2026, 3407, and 9103. Overall and class-wise metrics are reported as mean ± sample standard deviation across the five runs. The notebook also generates 95% t-confidence intervals and seed-matched exact McNemar tests with Holm correction.

This is repeated-seed evaluation on a fixed split, not five-fold cross-validation.

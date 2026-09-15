# Verification reports

- `original_dataset_audit.json` records the MD5-based exact-duplicate audit of the 4,500 original source files.
- `original_split_provenance_audit.json` records source-image-ID overlap across the publisher-provided augmented training, validation, and test splits. It is generated from the filename prefix before `_rot` and confirms that every test file has a source identifier represented in training or validation.
- `benchmark_verification.json` records the verified nine-class split counts and confirms zero within-split duplication, zero cross-split MD5 overlap, and zero MD5 overlap between the 36 h and 48 h Mutton classes.

Reports in a release must be generated from the manifest and scripts contained in that same release.

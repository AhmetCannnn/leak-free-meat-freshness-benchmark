# Benchmark manifest

`benchmark_manifest_md5.csv` is the canonical machine-readable definition of the fixed nine-class benchmark. It contains one row per image with split, class name, clean filename, benchmark-relative path, byte size, and MD5 digest.

The manifest contains 1,849 data rows. It records membership and exact byte identity without redistributing the underlying source images.

`verification_report.json` is the dataset-side metadata consumed by the released training notebook. The reconstruction script copies both files into the rebuilt dataset root.

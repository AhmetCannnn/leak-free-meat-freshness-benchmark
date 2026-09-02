# Benchmark manifests

`benchmark_manifest.csv` is the canonical machine-readable definition of the
clean benchmark. Each row identifies one file and records its split, class,
filename, byte size, SHA-256 digest, and optional source provenance.

Generate it with `scripts/create_benchmark_manifest.py`. Source provenance
fields remain empty when `--source-root` is omitted.


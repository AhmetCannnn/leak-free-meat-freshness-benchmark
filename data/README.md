# Data access

The source images are not committed to Git. The source dataset is available
from Mendeley Data: https://data.mendeley.com/datasets/4tj9t3n6vj/1

The source dataset's redistribution terms must be confirmed before cleaned
images are deposited in a public archive. Until then, the versioned manifests
and reconstruction scripts are the canonical definition of the benchmark.

Expected local layout:

```text
data/source/  # downloaded source dataset
data/clean/   # reconstructed 8-class benchmark
```

The benchmark contains 1,849 raw images: 1,290 train, 277 validation, and 282
test. No offline-augmented image belongs to the clean benchmark.


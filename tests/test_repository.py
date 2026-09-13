import csv
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")


class RepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/benchmark.json").read_text(encoding="utf-8"))
        with (ROOT / "manifests/benchmark_manifest_md5.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            cls.manifest = list(csv.DictReader(stream))

    def test_required_files_exist(self):
        required = [
            "README.md",
            "LICENSE",
            "CITATION.cff",
            "requirements.txt",
            ".gitignore",
            "configs/benchmark.json",
            "manifests/benchmark_manifest_md5.csv",
            "manifests/verification_report.json",
            "reports/benchmark_verification.json",
            "results/PAPER_READY.json",
            "notebooks/meat_freshness_9class_multirun.ipynb",
        ]
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_nine_unique_classes(self):
        classes = self.config["classes"]
        self.assertEqual(len(classes), 9)
        self.assertEqual(len(set(classes)), 9)
        self.assertIn("36 hr Mutton", classes)
        self.assertIn("48 hr Mutton", classes)
        self.assertNotIn("36+ hr Mutton", classes)

    def test_manifest_schema_and_counts(self):
        self.assertEqual(len(self.manifest), self.config["expected_counts"]["total"])
        self.assertEqual(
            set(self.manifest[0]),
            {"split", "class_name", "filename", "relative_path", "size_bytes", "md5"},
        )
        split_counts = Counter(row["split"] for row in self.manifest)
        for split in SPLITS:
            self.assertEqual(split_counts[split], self.config["expected_counts"][split])

        class_counts = {
            split: Counter(
                row["class_name"] for row in self.manifest if row["split"] == split
            )
            for split in SPLITS
        }
        self.assertEqual(class_counts, {
            split: Counter(values)
            for split, values in self.config["expected_class_counts"].items()
        })

    def test_manifest_md5_values_are_unique(self):
        hashes = [row["md5"] for row in self.manifest]
        self.assertTrue(all(len(value) == 32 for value in hashes))
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_verification_report(self):
        report = json.loads(
            (ROOT / "reports/benchmark_verification.json").read_text(encoding="utf-8")
        )
        self.assertEqual(sum(report["split_counts"].values()), 1849)
        self.assertTrue(report["checks_passed"])
        self.assertEqual(report["hash_algorithm"], "MD5")
        self.assertEqual(report["mutton_36_48_shared_md5_count"], 0)
        self.assertEqual(report["errors"], [])

    def test_paper_ready_results(self):
        record = json.loads((ROOT / "results/PAPER_READY.json").read_text(encoding="utf-8"))
        self.assertEqual(record["experiment_version"], "nine-class-v1")
        self.assertEqual(record["completed_runs"], 20)
        self.assertEqual(len(record["models"]), 4)
        self.assertEqual(record["seeds"], [42, 123, 2026, 3407, 9103])


if __name__ == "__main__":
    unittest.main()

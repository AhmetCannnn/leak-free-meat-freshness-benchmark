import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_required_files_exist(self):
        required = ["README.md", "LICENSE", "CITATION.cff", "requirements.txt", ".gitignore", "configs/benchmark.json"]
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_registered_counts(self):
        config = json.loads((ROOT / "configs/benchmark.json").read_text())
        counts = config["expected_counts"]
        self.assertEqual(counts["train"] + counts["valid"] + counts["test"], counts["total"])
        self.assertEqual(counts["total"], 1849)

    def test_eight_unique_classes(self):
        classes = json.loads((ROOT / "configs/benchmark.json").read_text())["classes"]
        self.assertEqual(len(classes), 8)
        self.assertEqual(len(set(classes)), 8)


if __name__ == "__main__":
    unittest.main()


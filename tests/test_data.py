"""Kiểm tra làm sạch dữ liệu và stratified_split."""

import unittest

import pandas as pd

from src.data import clean_data, stratified_split


def synthetic_frame(n_ham=80, n_spam=20):
    raw = pd.DataFrame({
        "source_id": range(n_ham + n_spam),
        "label": ["ham"] * n_ham + ["spam"] * n_spam,
        "text": [f"tin nhan so {i}" for i in range(n_ham + n_spam)],
    })
    return clean_data(raw)[0]


class CleanDataTests(unittest.TestCase):
    def test_removes_empty_duplicate_and_conflicting_messages(self):
        raw = pd.DataFrame({
            "source_id": range(1, 9),
            "label": ["ham", "ham", "spam", "ham", "spam", "ham", "spam", "ham"],
            "text": ["  HELLO   Friend ", "hello friend", "Prize now", " ",
                     "Same message", " SAME   MESSAGE ", None, "Meet tomorrow"],
        })
        cleaned, summary = clean_data(raw)
        self.assertEqual(cleaned["text"].tolist(), ["hello friend", "prize now", "meet tomorrow"])
        self.assertEqual(summary["empty_removed"], 2)
        self.assertEqual(summary["conflicting_rows_removed"], 2)
        self.assertEqual(summary["duplicates_removed"], 1)
        self.assertEqual(summary["class_counts"], {"ham": 2, "spam": 1})

    def test_unknown_label_is_rejected(self):
        with self.assertRaises(ValueError):
            clean_data(pd.DataFrame({"source_id": [1], "label": ["other"], "text": ["hi"]}))


class StratifiedSplitTests(unittest.TestCase):
    def test_splits_are_disjoint_and_cover_all_rows(self):
        frame = synthetic_frame()
        splits = stratified_split(frame, (0.7, 0.15, 0.15), seed=42)
        ids = [set(part["source_id"]) for part in splits.values()]
        self.assertEqual(set.union(*ids), set(frame["source_id"]))
        self.assertEqual(sum(len(s) for s in ids), len(frame))
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(ids[i] & ids[j])

    def test_each_split_keeps_class_ratio(self):
        splits = stratified_split(synthetic_frame(80, 20), (0.7, 0.15, 0.15), seed=42)
        expected = {"train": (56, 14), "validation": (12, 3), "test": (12, 3)}
        for name, part in splits.items():
            counts = ((part["label"] == "ham").sum(), (part["label"] == "spam").sum())
            self.assertEqual(counts, expected[name])
            self.assertAlmostEqual((part["label"] == "spam").mean(), 0.2)

    def test_same_seed_gives_same_split(self):
        frame = synthetic_frame()
        first = stratified_split(frame, seed=42)
        second = stratified_split(frame, seed=42)
        other = stratified_split(frame, seed=7)
        self.assertEqual(first["test"]["source_id"].tolist(), second["test"]["source_id"].tolist())
        self.assertNotEqual(first["test"]["source_id"].tolist(), other["test"]["source_id"].tolist())

    def test_invalid_ratios_are_rejected(self):
        for ratios in [(0.7, 0.3), (0.5, 0.3, 0.3), (1.0, 0.0, 0.0)]:
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                stratified_split(synthetic_frame(), ratios)


if __name__ == "__main__":
    unittest.main()

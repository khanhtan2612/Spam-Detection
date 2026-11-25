"""Kiểm tra score function bằng ví dụ 6 tin nhắn tính tay (1 = spam, 0 = ham)."""

import unittest

import numpy as np

from src.metrics import classification_scores, confusion_matrix

# 2 spam + 4 ham. Mô hình bắt 1 spam, bỏ sót 1 spam, báo nhầm 1 ham.
Y_TRUE = [1, 1, 0, 0, 0, 0]
Y_PRED = [1, 0, 1, 0, 0, 0]


class HandExampleTests(unittest.TestCase):
    def test_confusion_matrix(self):
        # [[TN, FP], [FN, TP]] = [[3, 1], [1, 1]]
        np.testing.assert_array_equal(confusion_matrix(Y_TRUE, Y_PRED), [[3, 1], [1, 1]])

    def test_scores(self):
        scores = classification_scores(Y_TRUE, Y_PRED)
        self.assertAlmostEqual(scores["accuracy"], 4 / 6)         # (TP + TN) / 6
        self.assertAlmostEqual(scores["spam_precision"], 1 / 2)   # TP / (TP + FP)
        self.assertAlmostEqual(scores["spam_recall"], 1 / 2)      # TP / (TP + FN)
        self.assertAlmostEqual(scores["spam_f1"], 1 / 2)
        self.assertAlmostEqual(scores["ham_f1"], 3 / 4)           # P_ham = R_ham = 3/4
        self.assertAlmostEqual(scores["macro_f1"], 5 / 8)         # (0.5 + 0.75) / 2


class EdgeCaseTests(unittest.TestCase):
    def test_no_spam_predicted_gives_zero_precision(self):
        scores = classification_scores([1, 0, 0], [0, 0, 0])
        self.assertEqual(scores["spam_precision"], 0.0)
        self.assertEqual(scores["spam_f1"], 0.0)

    def test_perfect_prediction(self):
        scores = classification_scores([1, 0, 1, 0], [1, 0, 1, 0])
        self.assertEqual(scores, {"accuracy": 1.0, "spam_precision": 1.0, "spam_recall": 1.0,
                                  "spam_f1": 1.0, "ham_f1": 1.0, "macro_f1": 1.0})


if __name__ == "__main__":
    unittest.main()

"""Kiểm tra TF-IDF: công thức, chọn vocabulary và không học từ validation."""

import unittest

import numpy as np

from src.tfidf import TfidfVectorizer, tokenize

TRAIN = ["free prize call now", "free cash prize", "call me later", "meet me later", "free call"]
VALIDATION = ["validationonly free", "validationonly prize"]


class TokenizeTests(unittest.TestCase):
    def test_keeps_words_with_at_least_two_characters(self):
        self.assertEqual(tokenize("I'm OK, u 2 WIN!! 100%"), ["ok", "win", "100"])


class TfidfTests(unittest.TestCase):
    def test_matches_formula_by_hand(self):
        vec = TfidfVectorizer(max_features=10, min_df=1).fit(["free free prize", "free call"])
        self.assertEqual(vec.feature_names(), ["call", "free", "prize"])
        # N = 2; df: call=1, free=2, prize=1 → idf = ln(3/(1+df)) + 1
        idf = np.array([np.log(3 / 2) + 1, 1.0, np.log(3 / 2) + 1])
        np.testing.assert_allclose(vec.idf_, idf)
        row = np.array([0.0, (1 + np.log(2)) * idf[1], 1.0 * idf[2]])  # tf = 1 + ln f
        np.testing.assert_allclose(vec.transform(["free free prize"])[0], row / np.linalg.norm(row))

    def test_min_df_and_max_features(self):
        vec = TfidfVectorizer(max_features=3, min_df=2).fit(TRAIN)
        # df ≥ 2: call, free, later, me, prize. Tổng số lần: free 3, call 3, các từ khác 2.
        # Giữ 3 từ nhiều nhất; hòa thì theo thứ tự chữ cái → call, free, later.
        self.assertEqual(vec.feature_names(), ["call", "free", "later"])

    def test_rows_have_unit_norm_or_are_zero(self):
        vec = TfidfVectorizer(min_df=1).fit(TRAIN)
        X = vec.transform(TRAIN + ["hoàn toàn mới"])
        np.testing.assert_allclose(np.linalg.norm(X[:-1], axis=1), 1.0)
        np.testing.assert_array_equal(X[-1], 0.0)

    def test_validation_words_never_enter_vocabulary(self):
        vec = TfidfVectorizer(min_df=2).fit(TRAIN)
        vocabulary, idf = dict(vec.vocabulary_), vec.idf_.copy()
        # 'validationonly' xuất hiện 2 lần ở validation: nếu học nhầm thì đủ min_df=2.
        X_val = vec.transform(VALIDATION)
        self.assertNotIn("validationonly", vec.vocabulary_)
        self.assertEqual(vec.vocabulary_, vocabulary)
        np.testing.assert_array_equal(vec.idf_, idf)
        self.assertEqual(X_val.shape, (2, len(vocabulary)))


if __name__ == "__main__":
    unittest.main()

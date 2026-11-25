"""Kiểm tra pipeline trên dữ liệu tự tạo nhỏ (không phải kết quả thực nghiệm)."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.pipeline import SpamPipeline, load_pipeline, save_pipeline
from src.predict import predict_message
from src.train import choose_k

TRAIN_TEXTS = [
    "meeting project lunch tomorrow team", "meeting project report tomorrow team",
    "meeting project lunch today friend", "meeting project report today friend",
    "family dinner home tonight friend", "family dinner home tomorrow team",
    "free prize claim cash winner", "free prize claim money winner",
    "free reward claim cash offer", "free reward claim money offer",
    "win cash reward call offer", "win money prize call winner",
]
TRAIN_LABELS = ["ham"] * 6 + ["spam"] * 6
VALIDATION_TEXTS = ["validationonly meeting project lunch", "validationonly free prize claim"]


def fitted(method):
    return SpamPipeline(method, None if method == "baseline" else 2).fit(TRAIN_TEXTS, TRAIN_LABELS)


class PipelineTests(unittest.TestCase):
    def test_three_pipelines_learn_toy_data(self):
        for method in ("baseline", "svd", "pca"):
            with self.subTest(method=method):
                model = fitted(method)
                np.testing.assert_array_equal(model.predict(VALIDATION_TEXTS), [0, 1])
                p = model.predict_proba(VALIDATION_TEXTS)
                self.assertTrue(np.all((p > 0) & (p < 1)))

    def test_predicting_validation_does_not_change_learned_parameters(self):
        model = fitted("pca")
        vocabulary, mean = dict(model.vectorizer.vocabulary_), model.reducer.mean_.copy()
        model.predict(VALIDATION_TEXTS)
        self.assertNotIn("validationonly", model.vectorizer.vocabulary_)
        self.assertEqual(model.vectorizer.vocabulary_, vocabulary)
        np.testing.assert_array_equal(model.reducer.mean_, mean)
        X_train = model.vectorizer.transform(TRAIN_TEXTS)
        np.testing.assert_allclose(model.reducer.mean_, X_train.mean(axis=0))

    def test_save_and_load_give_same_probabilities(self):
        for method in ("baseline", "svd", "pca"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                model = fitted(method)
                save_pipeline(model, Path(folder) / method)
                restored = load_pipeline(Path(folder) / method)
                np.testing.assert_array_equal(model.predict_proba(VALIDATION_TEXTS),
                                              restored.predict_proba(VALIDATION_TEXTS))
                if model.reducer is not None:
                    self.assertEqual(model.reducer.score(), restored.reducer.score())

    def test_predict_message(self):
        result = predict_message(fitted("svd"), "free prize claim now")
        self.assertEqual(result["prediction"], "spam")
        with self.assertRaises(ValueError):
            predict_message(fitted("svd"), "   ")


class ChooseKTests(unittest.TestCase):
    def test_best_validation_macro_f1_and_ties_choose_smaller_k(self):
        validation = pd.DataFrame([
            {"method": "pca", "k": 100, "macro_f1": 0.90},
            {"method": "pca", "k": 50, "macro_f1": 0.90},
            {"method": "pca", "k": 10, "macro_f1": 0.80},
            {"method": "svd", "k": 10, "macro_f1": 0.99},
            {"method": "svd", "k": 50, "macro_f1": 0.89},
        ])
        self.assertEqual(choose_k(validation, "pca"), 50)
        self.assertEqual(choose_k(validation, "svd"), 10)


if __name__ == "__main__":
    unittest.main()

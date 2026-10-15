"""Kiem tra nho bang du lieu tu tao; khong phai ket qua thuc nghiem UCI."""

import io
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.data import clean_data, split_data
from src.models import ToDense, make_pipeline
from src.predict import predict_message
from src.train import choose_k


TRAIN_TEXTS = [
    "meeting project lunch tomorrow team",
    "meeting project report tomorrow team",
    "meeting project lunch today friend",
    "meeting project report today friend",
    "family dinner home tonight friend",
    "family dinner home tomorrow team",
    "free prize claim cash winner",
    "free prize claim money winner",
    "free reward claim cash offer",
    "free reward claim money offer",
    "win cash reward call offer",
    "win money prize call winner",
]
TRAIN_LABELS = ["ham"] * 6 + ["spam"] * 6
VALIDATION_TEXTS = [
    "validationonlytoken meeting project lunch",
    "validationonlytoken free prize claim",
]


class DataTests(unittest.TestCase):
    def test_cleanup_removes_empty_duplicates_and_entire_conflicting_groups(self):
        raw = pd.DataFrame({
            "source_id": list(range(1, 9)),
            "label": ["ham", "ham", "spam", "ham", "spam", "ham", "spam", "ham"],
            "text": ["  HELLO   Friend ", "hello friend", "Prize now", " ",
                     "Same message", " SAME   MESSAGE ", None, "Meet tomorrow"],
        })
        original = raw.copy(deep=True)
        cleaned, audit = clean_data(raw)
        self.assertEqual(cleaned["text"].tolist(), ["hello friend", "prize now", "meet tomorrow"])
        self.assertEqual(audit["empty_removed"], 2)
        self.assertEqual(audit["conflicting_groups"], 1)
        self.assertEqual(audit["conflicting_rows_removed"], 2)
        self.assertEqual(audit["duplicates_removed"], 1)
        self.assertEqual(audit["clean_rows"], 3)
        self.assertEqual(audit["class_counts"], {"ham": 2, "spam": 1})
        self.assertEqual(cleaned["text_sha256"].nunique(), len(cleaned))
        pd.testing.assert_frame_equal(raw, original)

    def test_invalid_label_is_rejected(self):
        raw = pd.DataFrame({"source_id": [1], "label": ["unknown"], "text": ["hello"]})
        with self.assertRaises(ValueError):
            clean_data(raw)

    def test_stratified_splits_are_disjoint_complete_and_reproducible(self):
        raw = pd.DataFrame({
            "source_id": range(100),
            "label": ["ham"] * 80 + ["spam"] * 20,
            "text": [f"distinct synthetic message {i}" for i in range(100)],
        })
        frame, _ = clean_data(raw)
        splits = split_data(frame, seed=42)
        repeated = split_data(frame, seed=42)
        expected_sizes = {"train": 70, "validation": 15, "test": 15}
        id_sets, hash_sets = [], []
        for name, part in splits.items():
            self.assertEqual(len(part), expected_sizes[name])
            self.assertEqual(part["label"].value_counts(normalize=True).to_dict(), {"ham": .8, "spam": .2})
            self.assertEqual(part["source_id"].tolist(), repeated[name]["source_id"].tolist())
            id_sets.append(set(part["source_id"]))
            hash_sets.append(set(part["text_sha256"]))
        self.assertEqual(set.union(*id_sets), set(frame["source_id"]))
        for i in range(len(id_sets)):
            for j in range(i + 1, len(id_sets)):
                self.assertFalse(id_sets[i] & id_sets[j])
                self.assertFalse(hash_sets[i] & hash_sets[j])


class PipelineTests(unittest.TestCase):
    def test_validation_token_never_enters_fitted_vocabulary_or_idf(self):
        for method in ("baseline", "pca", "svd"):
            with self.subTest(method=method):
                model = make_pipeline(method, None if method == "baseline" else 2)
                model.fit(TRAIN_TEXTS, TRAIN_LABELS)
                tfidf = model.named_steps["tfidf"]
                vocabulary = tfidf.vocabulary_.copy()
                idf = tfidf.idf_.copy()
                self.assertNotIn("validationonlytoken", vocabulary)
                # Sentinel occurs twice in validation: min_df=2 would admit it if refit leaked.
                model.predict(VALIDATION_TEXTS)
                model.predict_proba(VALIDATION_TEXTS)
                self.assertEqual(tfidf.vocabulary_, vocabulary)
                np.testing.assert_array_equal(tfidf.idf_, idf)
                self.assertNotIn("validationonlytoken", tfidf.vocabulary_)

    def test_pca_uses_training_mean_and_transform_preserves_fitted_state(self):
        model = make_pipeline("pca", 2)
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        tfidf = model.named_steps["tfidf"]
        reducer = model.named_steps["reducer"]
        X_train = tfidf.transform(TRAIN_TEXTS).toarray()
        np.testing.assert_allclose(reducer.mean_, X_train.mean(axis=0), atol=1e-12)
        before = {name: getattr(reducer, name).copy() for name in (
            "mean_", "components_", "singular_values_", "explained_variance_",
        )}
        X_validation = tfidf.transform(VALIDATION_TEXTS).toarray()
        expected = (X_validation - before["mean_"]) @ before["components_"].T
        actual = model[:-1].transform(VALIDATION_TEXTS)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(reducer, name), value)

    def test_baseline_representation_and_svd_input_remain_sparse(self):
        baseline = make_pipeline("baseline")
        baseline.fit(TRAIN_TEXTS, TRAIN_LABELS)
        self.assertTrue(sparse.issparse(baseline[:-1].transform(TRAIN_TEXTS)))
        self.assertNotIn("dense", baseline.named_steps)
        original_fit_transform = TruncatedSVD.fit_transform
        received_sparse = []

        def checked_fit_transform(reducer, X, y=None):
            received_sparse.append(sparse.issparse(X))
            return original_fit_transform(reducer, X, y)

        with patch.object(TruncatedSVD, "fit_transform", new=checked_fit_transform):
            svd_model = make_pipeline("svd", 2)
            svd_model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        self.assertEqual(received_sparse, [True])
        self.assertNotIn("dense", svd_model.named_steps)
        self.assertEqual(svd_model[:-1].transform(TRAIN_TEXTS).shape, (len(TRAIN_TEXTS), 2))

    def test_dense_budget_blocks_conversion_before_allocating(self):
        X = sparse.csr_matrix(np.arange(12, dtype=np.float64).reshape(3, 4))
        with patch.object(type(X), "toarray", side_effect=AssertionError("must not allocate")):
            with self.assertRaises(MemoryError):
                ToDense(max_bytes=95).transform(X)
        actual = ToDense(max_bytes=96).transform(X)
        self.assertIsInstance(actual, np.ndarray)
        self.assertEqual(actual.nbytes, 96)
        np.testing.assert_array_equal(actual, X.toarray())

    def test_pipeline_save_load_preserves_predictions_and_probabilities(self):
        for method in ("baseline", "pca", "svd"):
            with self.subTest(method=method):
                model = make_pipeline(method, None if method == "baseline" else 2)
                model.fit(TRAIN_TEXTS, TRAIN_LABELS)
                # Round-trip joblib in memory: no filesystem-specific temp ACLs.
                payload = io.BytesIO()
                joblib.dump(model, payload)
                self.assertGreater(payload.tell(), 0)
                payload.seek(0)
                restored = joblib.load(payload)
                np.testing.assert_array_equal(model.predict(VALIDATION_TEXTS), restored.predict(VALIDATION_TEXTS))
                np.testing.assert_allclose(model.predict_proba(VALIDATION_TEXTS), restored.predict_proba(VALIDATION_TEXTS), rtol=0, atol=0)
                result = predict_message(restored, VALIDATION_TEXTS[0])
                spam_index = list(restored.classes_).index("spam")
                self.assertEqual(result["prediction"], restored.predict(VALIDATION_TEXTS[:1])[0])
                self.assertAlmostEqual(result["spam_probability"], restored.predict_proba(VALIDATION_TEXTS[:1])[0, spam_index])

    def test_predict_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            predict_message(None, " \t\n")


class SelectionTests(unittest.TestCase):
    def test_select_k_only_needs_validation_and_breaks_ties_by_smaller_k(self):
        validation = pd.DataFrame([
            {"method": "pca", "k": 100, "macro_f1": .90},
            {"method": "svd", "k": 10, "macro_f1": .99},
            {"method": "pca", "k": 50, "macro_f1": .90},
            {"method": "pca", "k": 10, "macro_f1": .80},
            {"method": "svd", "k": 50, "macro_f1": .89},
        ])
        original = validation.copy(deep=True)
        self.assertEqual(choose_k(validation, "pca"), 50)
        self.assertEqual(choose_k(validation, "svd"), 10)
        pd.testing.assert_frame_equal(validation, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)

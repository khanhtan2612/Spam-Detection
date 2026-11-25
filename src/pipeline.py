"""Ghép ba bước: văn bản → TF-IDF → SVD/PCA (nếu có) → Logistic Regression.

Ba pipeline chỉ khác nhau ở bước giữa: baseline (không giảm chiều), svd, pca.
"""

import json
from pathlib import Path

import numpy as np

from .decomposition import PCA, TruncatedSVD
from .logistic import LogisticRegressionGD
from .tfidf import TfidfVectorizer

SEED = 42
MAX_FEATURES = 1000
MIN_DF = 2
K_VALUES = (10, 50, 100)
METHODS = ("baseline", "svd", "pca")


def to_binary(labels):
    """'spam' → 1, 'ham' → 0."""
    return (np.asarray(labels) == "spam").astype(int)


class SpamPipeline:
    """fit() học TF-IDF, reducer và classifier trên train; predict() chỉ dùng lại tham số đã học."""

    def __init__(self, method="baseline", k=None):
        if method not in METHODS:
            raise ValueError(f"method phải là một trong {METHODS}")
        self.method, self.k = method, k
        self.vectorizer = TfidfVectorizer(MAX_FEATURES, MIN_DF)
        self.reducer = {"baseline": None, "svd": TruncatedSVD, "pca": PCA}[method]
        self.reducer = self.reducer(k, seed=SEED) if self.reducer else None
        self.classifier = LogisticRegressionGD()

    def features(self, texts):
        """Văn bản → X (TF-IDF) → Z = X·V_k nếu có giảm chiều."""
        X = self.vectorizer.transform(texts)
        return X if self.reducer is None else self.reducer.transform(X)

    def fit(self, texts, labels):
        """Học TF-IDF → V_k (và μ nếu là PCA) → w, b, tất cả chỉ trên dữ liệu được đưa vào."""
        X = self.vectorizer.fit_transform(texts)
        if self.reducer is not None:
            X = self.reducer.fit_transform(X)
        self.classifier.fit(X, to_binary(labels))
        return self

    def predict_proba(self, texts):
        """P(spam) của từng tin nhắn."""
        return self.classifier.predict_proba(self.features(texts))

    def predict(self, texts):
        """1 = spam, 0 = ham (ngưỡng 0.5)."""
        return self.classifier.predict(self.features(texts))

    def score(self, texts, labels):
        """Score function của classifier trên tập văn bản có nhãn."""
        return self.classifier.score(self.features(texts), to_binary(labels))


def save_pipeline(model, path):
    """Lưu mảng số vào path.npz và vocabulary/cấu hình vào path.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"idf": model.vectorizer.idf_, "w": model.classifier.w, "b": model.classifier.b}
    if model.reducer is not None:
        arrays["components"] = model.reducer.components_
        arrays["singular_values"] = model.reducer.singular_values_
        arrays["total_energy"] = model.reducer.total_energy_
    if model.method == "pca":
        arrays["mean"] = model.reducer.mean_
        arrays["explained_variance"] = model.reducer.explained_variance_
    np.savez(path.with_suffix(".npz"), **arrays)
    info = {"method": model.method, "k": model.k, "vocabulary": model.vectorizer.feature_names()}
    path.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")


def load_pipeline(path):
    """Đọc lại pipeline đã lưu bằng save_pipeline; không fit lại."""
    path = Path(path)
    info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    arrays = np.load(path.with_suffix(".npz"))
    model = SpamPipeline(info["method"], info["k"])
    model.vectorizer.vocabulary_ = {word: j for j, word in enumerate(info["vocabulary"])}
    model.vectorizer.idf_ = arrays["idf"]
    if model.reducer is not None:
        model.reducer.components_ = arrays["components"]
        model.reducer.singular_values_ = arrays["singular_values"]
        model.reducer.total_energy_ = float(arrays["total_energy"])
    if model.method == "pca":
        model.reducer.mean_ = arrays["mean"]
        model.reducer.explained_variance_ = arrays["explained_variance"]
    model.classifier.w, model.classifier.b = arrays["w"], float(arrays["b"])
    return model

"""Nơi định nghĩa mô hình: văn bản → TF-IDF → giảm chiều (nếu có) → nhãn.

Đọc make_pipeline() để thấy khác biệt duy nhất giữa ba phương án.
Module này chỉ tạo cấu hình; model.fit(...) trong train.py mới học từ dữ liệu.
"""

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import normalize_text

DEFAULT_SEED = 42
MAX_FEATURES = 1000
K_VALUES = (10, 50, 100)


class ToDense(TransformerMixin, BaseEstimator):
    """Đổi sparse sang dense cho PCA, chặn ma trận đầu vào quá lớn.

    fit() không học tham số. transform() chỉ đổi cách lưu, không đổi giá trị.
    Giới hạn 128 MiB không bao gồm bộ nhớ trung gian của thuật toán PCA.
    """
    def __init__(self, max_bytes=128 * 1024**2):
        self.max_bytes = max_bytes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        estimated_bytes = X.shape[0] * X.shape[1] * X.dtype.itemsize
        if estimated_bytes > self.max_bytes:
            raise MemoryError(f"Dense cần {estimated_bytes:,} byte; vượt giới hạn {self.max_bytes:,}.")
        return X.toarray() if sparse.issparse(X) else np.asarray(X)


def make_pipeline(method="baseline", k=None, seed=DEFAULT_SEED):
    """Tạo Pipeline chưa fit: method là baseline/pca/svd; k là số chiều giữ lại.

    Pipeline tự nối các bước: fit trên train, sau đó dùng cùng các tham số
    đã học để transform và predict trên validation/test hoặc SMS mới.
    """
    # Bước 1: mỗi SMS thành vector có tối đa MAX_FEATURES tọa độ.
    steps = [("tfidf", TfidfVectorizer(
        max_features=MAX_FEATURES, min_df=2, ngram_range=(1, 1), norm="l2",
        sublinear_tf=True, dtype=np.float64, preprocessor=normalize_text,
    ))]
    # Bước 2: baseline giữ nguyên TF-IDF; PCA/SVD tạo vector k chiều.
    if method == "pca":
        steps += [("dense", ToDense()), ("reducer", PCA(
            n_components=k, svd_solver="randomized", whiten=False, random_state=seed
        ))]
    elif method == "svd":
        steps += [("reducer", TruncatedSVD(n_components=k, algorithm="randomized", random_state=seed))]
    elif method != "baseline":
        raise ValueError(f"Phương pháp không hợp lệ: {method}")
    # Bước 3: cùng một classifier để so sánh tác động của giảm chiều.
    steps += [("classifier", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"))]
    return Pipeline(steps)

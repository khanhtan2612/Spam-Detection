"""TF-IDF tự cài: văn bản → ma trận X (N tin nhắn × D từ)."""

from collections import Counter
import re

import numpy as np

from .data import normalize_text

TOKEN_PATTERN = re.compile(r"\b\w\w+\b")


def tokenize(text):
    """Tách từ: chuỗi chữ/số dài từ 2 ký tự, regex \\b\\w\\w+\\b."""
    return TOKEN_PATTERN.findall(normalize_text(text))


class TfidfVectorizer:
    """tf = 1 + log f,  idf = log((1+N)/(1+df)) + 1,  rồi chuẩn hóa L2 từng hàng.

    fit() chỉ học vocabulary và idf từ train; transform() dùng lại đúng hai thứ đó.
    """

    def __init__(self, max_features=1000, min_df=2):
        self.max_features = max_features
        self.min_df = min_df

    def fit(self, texts):
        """Giữ từ có df ≥ min_df, lấy max_features từ có tổng số lần xuất hiện lớn nhất."""
        documents = [tokenize(text) for text in texts]
        df = Counter(word for doc in documents for word in set(doc))
        total = Counter(word for doc in documents for word in doc)
        candidates = sorted(word for word in df if df[word] >= self.min_df)
        # Sắp theo số lần xuất hiện giảm dần; bằng nhau thì theo thứ tự chữ cái.
        kept = sorted(candidates, key=lambda word: -total[word])[:self.max_features]
        self.vocabulary_ = {word: j for j, word in enumerate(sorted(kept))}
        n = len(documents)
        doc_freq = np.array([df[word] for word in self.vocabulary_], dtype=float)
        self.idf_ = np.log((1 + n) / (1 + doc_freq)) + 1
        return self

    def transform(self, texts):
        """Trả ma trận dense X có shape (số tin nhắn, D); từ ngoài vocabulary bị bỏ qua."""
        X = np.zeros((len(texts), len(self.vocabulary_)))
        for i, text in enumerate(texts):
            counts = Counter(w for w in tokenize(text) if w in self.vocabulary_)
            for word, f in counts.items():
                X[i, self.vocabulary_[word]] = 1 + np.log(f)
        X *= self.idf_
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms == 0, 1, norms)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    def feature_names(self):
        """Danh sách từ theo đúng thứ tự cột của X."""
        return list(self.vocabulary_)

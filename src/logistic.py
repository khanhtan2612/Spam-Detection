"""Logistic Regression tự cài, đi đúng mạch: mô hình → loss → gradient → tối ưu → score."""

import numpy as np

from .metrics import classification_scores


def sigmoid(z):
    """σ(z) = 1 / (1 + e^(−z)), đưa điểm z về xác suất trong (0, 1)."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip để e^(−z) không tràn số.


class LogisticRegressionGD:
    """Logistic Regression học bằng gradient descent, phạt L2 với hệ số lam (λ).

    η = 2: L ≤ ¼·λ_max([X 1]ᵀ[X 1])/N + λ/N ≤ ¼·2 + λ/N ≈ 0,5, vì λ_max ≤ trace ≤ 2N
    (hàng TF-IDF có chuẩn ≤ 1; Z của SVD/PCA có ‖Z‖²_F ≤ N). Khi η < 2/L, J giảm sau mỗi bước.
    tol = 1e-10: dừng khi J gần như không đổi; epochs = 20000 là giới hạn trên.
    """

    def __init__(self, lam=1.0, lr=2.0, epochs=20000, tol=1e-10):
        self.lam = lam
        self.lr = lr
        self.epochs = epochs
        self.tol = tol

    # 1. Mô hình
    def predict_proba(self, X):
        """p = σ(z) với z = Xw + b: xác suất mỗi tin nhắn là spam."""
        return sigmoid(X @ self.w + self.b)

    # 2. Loss function
    def loss(self, X, y):
        """J(w, b) = −(1/N) Σ [y·log p + (1−y)·log(1−p)] + (λ/2N)·‖w‖².

        p được clip vào [1e-12, 1 − 1e-12] để tránh log(0).
        """
        n = len(y)
        p = np.clip(self.predict_proba(X), 1e-12, 1 - 1e-12)
        cross_entropy = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return cross_entropy + self.lam / (2 * n) * np.sum(self.w ** 2)

    # 3. Gradient
    def gradient(self, X, y):
        """∂J/∂w = (1/N)·Xᵀ(p − y) + (λ/N)·w,   ∂J/∂b = (1/N)·Σ(p − y)."""
        n = len(y)
        error = self.predict_proba(X) - y
        grad_w = X.T @ error / n + self.lam / n * self.w
        grad_b = np.mean(error)
        return grad_w, grad_b

    # 4. Tối ưu
    def fit(self, X, y):
        """Gradient descent: w ← w − η·∂J/∂w, b ← b − η·∂J/∂b.

        Bắt đầu từ w = 0, b = 0. Dừng khi đủ epochs hoặc |J_mới − J_cũ| < tol.
        loss_history_ lưu J sau mỗi epoch (phần tử đầu là J tại w = 0).
        """
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.loss_history_ = [self.loss(X, y)]
        for _ in range(self.epochs):
            grad_w, grad_b = self.gradient(X, y)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            self.loss_history_.append(self.loss(X, y))
            if abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol:
                break
        self.n_epochs_ = len(self.loss_history_) - 1
        return self

    # 5. Dự đoán
    def predict(self, X, threshold=0.5):
        """ŷ = 1 (spam) nếu p ≥ 0.5, ngược lại ŷ = 0 (ham)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # 6. Score function
    def score(self, X, y):
        """accuracy = (TP+TN)/N, P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2PR/(P+R) cho lớp spam;
        macro-F1 = (F1_spam + F1_ham)/2.
        """
        return classification_scores(y, self.predict(X))


def gradient_check(X, y, lam=1.0, eps=1e-6, seed=0):
    """So gradient giải tích với sai phân trung tâm (J(θ+ε) − J(θ−ε)) / 2ε.

    Thử tại một điểm (w, b) ngẫu nhiên; trả sai số tuyệt đối lớn nhất.
    """
    rng = np.random.default_rng(seed)
    model = LogisticRegressionGD(lam=lam)
    model.w, model.b = rng.normal(size=X.shape[1]), float(rng.normal())
    grad_w, grad_b = model.gradient(X, y)
    numeric = []
    for j in range(X.shape[1] + 1):  # j = X.shape[1] ứng với b.
        values = []
        for step in (eps, -eps):
            shifted = LogisticRegressionGD(lam=lam)
            shifted.w, shifted.b = model.w.copy(), model.b
            if j < X.shape[1]:
                shifted.w[j] += step
            else:
                shifted.b += step
            values.append(shifted.loss(X, y))
        numeric.append((values[0] - values[1]) / (2 * eps))
    analytic = np.append(grad_w, grad_b)
    return float(np.max(np.abs(analytic - np.array(numeric))))

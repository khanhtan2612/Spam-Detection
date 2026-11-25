"""SVD và PCA tự cài bằng orthogonal iteration (nhân ma trận + QR).

Cùng mạch với Logistic Regression:
    loss  = sai số tái tạo ‖X − X_k‖²_F = Σ_{j>k} σ_j²
    score = tỷ lệ năng lượng (SVD) hoặc phương sai (PCA) giữ lại.
"""

import warnings

import numpy as np


def top_eigenvectors(A, k, tol=1e-8, max_iter=5000, seed=42):
    """k vector riêng ứng với k trị riêng lớn nhất của A = XᵀX: lặp Q ← QR(A·Q), λ_j = q_jᵀ A q_j.

    Dừng khi mọi λ_j đổi ít hơn tol·λ_1 giữa hai vòng. Trả (λ giảm dần, Q, số vòng lặp).
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((A.shape[0], k)))
    eigenvalues = np.zeros(k)
    for iteration in range(1, max_iter + 1):
        Q, R = np.linalg.qr(A @ Q)
        Q *= np.where(np.diag(R) < 0, -1, 1)  # Giữ dấu cột ổn định giữa các vòng.
        new_eigenvalues = np.einsum("ij,ij->j", Q, A @ Q)  # λ_j = q_jᵀ A q_j
        change = np.max(np.abs(new_eigenvalues - eigenvalues))
        eigenvalues = new_eigenvalues
        if change < tol * eigenvalues.max():
            break
    else:
        warnings.warn(f"Orthogonal iteration chưa hội tụ sau {max_iter} vòng.")
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], Q[:, order], iteration


class TruncatedSVD:
    """X ≈ X_k = U_k Σ_k V_kᵀ. V_k là k vector riêng lớn nhất của XᵀX, σ_j = √λ_j, Z = X·V_k."""

    def __init__(self, k, tol=1e-8, seed=42):
        self.k = k
        self.tol = tol
        self.seed = seed

    def _prepare(self, X):
        return X  # PCA ghi đè hàm này để trừ mean.

    def fit(self, X):
        """Học V_k (components_, mỗi hàng là một v_j) và σ (singular_values_) từ X train."""
        A = self._prepare(X)
        eigenvalues, V, self.n_iter_ = top_eigenvectors(A.T @ A, self.k, self.tol, seed=self.seed)
        self.singular_values_ = np.sqrt(np.maximum(eigenvalues, 0))
        self.components_ = V.T
        self.total_energy_ = float(np.sum(A ** 2))  # ‖X‖²_F = Σ_j σ_j²
        return self

    def transform(self, X):
        """Z = X·V_k: mỗi tin nhắn còn k tọa độ."""
        return self._prepare(X) @ self.components_.T

    def fit_transform(self, X):
        """fit trên X rồi trả Z = X·V_k."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        """X_k = Z·V_kᵀ: tái tạo gần đúng X từ k tọa độ."""
        return Z @ self.components_

    def loss(self):
        """Loss = ‖X − X_k‖²_F = Σ_{j>k} σ_j² = ‖X‖²_F − Σ_{j≤k} σ_j²."""
        return self.total_energy_ - float(np.sum(self.singular_values_ ** 2))

    def score(self):
        """Score = Σ_{j≤k} σ_j² / ‖X‖²_F: tỷ lệ năng lượng giữ lại."""
        return float(np.sum(self.singular_values_ ** 2)) / self.total_energy_


class PCA(TruncatedSVD):
    """PCA = SVD của X_c = X − μ, với μ là mean của train.

    Phương sai của thành phần j là σ_j² / (N − 1); score = tỷ lệ phương sai giữ lại.
    """

    def fit(self, X):
        """Học μ từ train, rồi làm SVD trên X_c = X − μ."""
        self.mean_ = X.mean(axis=0)
        super().fit(X)
        self.explained_variance_ = self.singular_values_ ** 2 / (len(X) - 1)
        return self

    def _prepare(self, X):
        return X - self.mean_  # Validation/test cũng trừ μ của train.

    def inverse_transform(self, Z):
        """X_k = Z·V_kᵀ + μ."""
        return Z @ self.components_ + self.mean_

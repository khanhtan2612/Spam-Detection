"""Đối chiếu SVD/PCA tự viết với np.linalg.svd / np.linalg.eigh (chỉ dùng trong test)."""

import unittest

import numpy as np

from src.decomposition import PCA, TruncatedSVD, top_eigenvectors

TOL = 1e-14  # Nhỏ hơn mặc định (1e-8) để so sát với NumPy.
# Trị riêng hội tụ nhanh hơn vector riêng (sai số vector ~ căn bậc hai sai số trị riêng),
# nên σ được so tới 1e-8 còn projector của không gian con so tới 1e-6.
SIGMA_RTOL, PROJECTOR_ATOL = 1e-8, 1e-6


def sample_matrix(n=40, d=8, seed=0):
    """Ma trận 40×8 có các giá trị kỳ dị tách biệt và mean khác 0."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) * (5 * 0.6 ** np.arange(d)) + rng.uniform(0.5, 1.5, d)


def projector(rows):
    """P = VᵀV: phép chiếu lên không gian con, không phụ thuộc dấu của từng vector."""
    return rows.T @ rows


class TopEigenvectorTests(unittest.TestCase):
    def test_matches_numpy_eigh(self):
        X = sample_matrix()
        A = X.T @ X
        values, vectors, _ = top_eigenvectors(A, 3, tol=TOL)
        expected_values, expected_vectors = np.linalg.eigh(A)
        np.testing.assert_allclose(values, expected_values[::-1][:3], rtol=SIGMA_RTOL)
        top = expected_vectors[:, ::-1][:, :3]
        np.testing.assert_allclose(vectors @ vectors.T, top @ top.T, atol=PROJECTOR_ATOL)
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=1e-12)

    def test_warns_when_not_converged_and_values_match_vectors(self):
        X = sample_matrix()
        A = X.T @ X
        with self.assertWarns(UserWarning):
            values, vectors, n_iter = top_eigenvectors(A, 3, max_iter=2)
        self.assertEqual(n_iter, 2)
        # λ_j trả về phải là q_jᵀ A q_j của chính Q trả về.
        np.testing.assert_allclose(values, np.einsum("ij,ij->j", vectors, A @ vectors))


class TruncatedSVDTests(unittest.TestCase):
    def setUp(self):
        self.X = sample_matrix()
        _, self.sigma, self.Vt = np.linalg.svd(self.X, full_matrices=False)

    def test_singular_values_and_subspace_match_numpy(self):
        for k in (1, 3, 5):
            with self.subTest(k=k):
                svd = TruncatedSVD(k, tol=TOL).fit(self.X)
                np.testing.assert_allclose(svd.singular_values_, self.sigma[:k], rtol=SIGMA_RTOL)
                np.testing.assert_allclose(projector(svd.components_), projector(self.Vt[:k]),
                                           atol=PROJECTOR_ATOL)

    def test_loss_is_tail_energy_and_score_is_kept_energy(self):
        svd = TruncatedSVD(3, tol=TOL).fit(self.X)
        X_k = svd.inverse_transform(svd.transform(self.X))
        tail = np.sum(self.sigma[3:] ** 2)
        self.assertAlmostEqual(svd.loss(), tail, places=8)
        self.assertAlmostEqual(np.sum((self.X - X_k) ** 2), tail, places=8)
        self.assertAlmostEqual(svd.score(), 1 - tail / np.sum(self.sigma ** 2), places=10)

    def test_loss_decreases_with_k_and_vanishes_at_full_rank(self):
        losses = [TruncatedSVD(k, tol=TOL).fit(self.X).loss() for k in range(1, 9)]
        self.assertTrue(all(a > b for a, b in zip(losses, losses[1:])))
        self.assertLess(abs(losses[-1]), 1e-8)


class PCATests(unittest.TestCase):
    def setUp(self):
        self.X = sample_matrix()

    def test_matches_eigh_of_covariance(self):
        pca = PCA(3, tol=TOL).fit(self.X)
        Xc = self.X - self.X.mean(axis=0)
        values, vectors = np.linalg.eigh(Xc.T @ Xc / (len(self.X) - 1))
        np.testing.assert_allclose(pca.explained_variance_, values[::-1][:3], rtol=SIGMA_RTOL)
        top = vectors[:, ::-1][:, :3]
        np.testing.assert_allclose(projector(pca.components_), top @ top.T, atol=PROJECTOR_ATOL)

    def test_pca_equals_svd_of_centered_data(self):
        pca = PCA(3, tol=TOL).fit(self.X)
        Xc = self.X - self.X.mean(axis=0)
        _, sigma, Vt = np.linalg.svd(Xc, full_matrices=False)
        np.testing.assert_allclose(pca.singular_values_, sigma[:3], rtol=SIGMA_RTOL)
        self.assertAlmostEqual(pca.score(), np.sum(sigma[:3] ** 2) / np.sum(sigma ** 2), places=10)

    def test_new_data_uses_training_mean(self):
        pca = PCA(3, tol=TOL).fit(self.X)
        mean = pca.mean_.copy()
        new = self.X[:2] + 100.0
        Z = pca.transform(new)
        np.testing.assert_allclose(Z, (new - self.X.mean(axis=0)) @ pca.components_.T)
        np.testing.assert_array_equal(pca.mean_, mean)
        full = PCA(8, tol=TOL).fit(self.X)
        np.testing.assert_allclose(full.inverse_transform(full.transform(new)), new, atol=1e-8)


if __name__ == "__main__":
    unittest.main()

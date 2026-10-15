"""Kiểm tra kiến thức toán học và việc dùng mean train của PCA."""

import json
import unittest

import numpy as np

from src.math_demo import pca_eigh, pca_inverse, pca_transform, run_demo


class TestPCAMath(unittest.TestCase):
    def setUp(self):
        self.train = np.array([
            [1.0, 2.0, 0.0],
            [3.0, 1.0, 4.0],
            [2.0, 5.0, 1.0],
            [7.0, 2.0, 3.0],
            [4.0, 4.0, 6.0],
        ])

    def test_full_pca_reconstructs_original_data(self):
        fitted = pca_eigh(self.train, k=3)
        Z = pca_transform(self.train, fitted)
        reconstructed = pca_inverse(Z, fitted)
        np.testing.assert_allclose(reconstructed, self.train, atol=1e-8, rtol=1e-6)
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(
            fitted["components"] @ fitted["components"].T, np.eye(3), atol=1e-8
        )

    def test_validation_uses_train_mean_without_refitting(self):
        fitted = pca_eigh(self.train, k=3)
        before = {key: value.copy() for key, value in fitted.items()}
        # Một mẫu validation có mean riêng bằng chính nó. Nếu tự center lại,
        # vector sau chiếu sẽ thành zero và không tái tạo đúng vị trí ban đầu.
        validation = self.train[:1] + np.array([100.0, -50.0, 20.0])
        transformed = pca_transform(validation, fitted)
        self.assertGreater(float(np.linalg.norm(transformed)), 10.0)
        np.testing.assert_allclose(pca_inverse(transformed, fitted), validation, atol=1e-8)
        np.testing.assert_allclose(fitted["mean"], self.train.mean(axis=0), atol=1e-8)
        for key in before:
            np.testing.assert_array_equal(fitted[key], before[key])

    def test_centered_svd_gives_same_pca_subspace(self):
        fitted = pca_eigh(self.train, k=2)
        centered = self.train - self.train.mean(axis=0)
        _, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
        np.testing.assert_allclose(
            fitted["eigenvalues"], singular_values ** 2 / (len(self.train) - 1),
            rtol=1e-6, atol=1e-8,
        )
        # So projector để tránh kết luận sai khi eigenvector chỉ khác dấu.
        np.testing.assert_allclose(
            fitted["components"].T @ fitted["components"], Vt[:2].T @ Vt[:2],
            rtol=1e-6, atol=1e-8,
        )

    def test_reconstruction_error_does_not_increase_with_k(self):
        errors = []
        for k in (1, 2, 3):
            fitted = pca_eigh(self.train, k)
            restored = pca_inverse(pca_transform(self.train, fitted), fitted)
            errors.append(np.linalg.norm(self.train - restored))
        self.assertTrue(all(later <= earlier + 1e-8
                            for earlier, later in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 1e-8)

    def test_invalid_k_is_rejected(self):
        for invalid_k in (0, -1, 4, 1.5, "2", None):
            with self.subTest(k=invalid_k):
                with self.assertRaises(ValueError):
                    pca_eigh(self.train, invalid_k)

    def test_invalid_input_matrix_is_rejected(self):
        bad_matrices = [
            np.array([1.0, 2.0]),
            np.empty((0, 3)),
            np.empty((3, 0)),
            np.ones((1, 3)),
            np.array([[1.0, np.nan], [2.0, 3.0]]),
            np.array([[1.0, np.inf], [2.0, 3.0]]),
        ]
        for matrix in bad_matrices:
            with self.subTest(shape=matrix.shape):
                with self.assertRaises(ValueError):
                    pca_eigh(matrix, 1)

    def test_projection_dimension_mismatch_is_rejected(self):
        fitted = pca_eigh(self.train, k=2)
        with self.assertRaises(ValueError):
            pca_transform(np.ones((2, 4)), fitted)
        with self.assertRaises(ValueError):
            pca_inverse(np.ones((2, 3)), fitted)

    def test_demo_produces_serializable_verified_results(self):
        result = run_demo()
        json.dumps(result, allow_nan=False)
        self.assertTrue(result["checks_passed"])
        self.assertEqual(result["matrix_shape"], [6, 4])
        self.assertLess(result["full_reconstruction_error"], 1e-8)
        self.assertLess(result["orthogonality_error"], 1e-8)
        self.assertLess(result["pca_eigen_vs_svd_eigenvalue_error"], 1e-8)
        self.assertLess(result["projector_error"], 1e-8)
        self.assertLess(
            result["sklearn_pca_full_comparison"]["reconstruction_difference"], 1e-8
        )


if __name__ == "__main__":
    unittest.main()

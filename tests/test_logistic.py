"""Kiểm tra Logistic Regression: gradient check, loss giảm, dự đoán theo ngưỡng 0.5."""

import unittest
import warnings

import numpy as np

from src.logistic import LogisticRegressionGD, gradient_check, sigmoid


def toy_data(n=60, d=5, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_w = np.array([2.0, -1.0, 0.5, 0.0, 1.0])[:d]
    y = (X @ true_w + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


class SigmoidTests(unittest.TestCase):
    def test_values_and_no_overflow(self):
        self.assertEqual(sigmoid(0.0), 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            p = sigmoid(np.array([-1000.0, 1000.0]))
        np.testing.assert_allclose(p, [0.0, 1.0], atol=1e-200)


class GradientTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_differences(self):
        X, y = toy_data(n=20, d=4)
        for lam in (0.0, 1.0, 5.0):
            with self.subTest(lam=lam):
                self.assertLess(gradient_check(X, y, lam=lam), 1e-6)

    def test_loss_at_zero_weights_is_log_two(self):
        X, y = toy_data()
        model = LogisticRegressionGD()
        model.w, model.b = np.zeros(X.shape[1]), 0.0
        self.assertAlmostEqual(model.loss(X, y), np.log(2))


class FitTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = toy_data()
        self.model = LogisticRegressionGD(lam=1.0, lr=1.0, epochs=20000, tol=1e-12).fit(self.X, self.y)

    def test_loss_never_increases_and_converges(self):
        history = np.array(self.model.loss_history_)
        self.assertTrue(np.all(np.diff(history) <= 0))
        self.assertLess(self.model.n_epochs_, 20000)
        grad_w, grad_b = self.model.gradient(self.X, self.y)
        self.assertLess(max(np.abs(grad_w).max(), abs(grad_b)), 1e-5)  # ∇J ≈ 0 tại cực tiểu

    def test_predict_uses_threshold_half(self):
        p = self.model.predict_proba(self.X)
        np.testing.assert_array_equal(self.model.predict(self.X), (p >= 0.5).astype(int))
        self.assertGreater(self.model.score(self.X, self.y)["accuracy"], 0.9)

    def test_larger_lambda_gives_smaller_weights(self):
        strong = LogisticRegressionGD(lam=50.0, lr=1.0, epochs=20000, tol=1e-12).fit(self.X, self.y)
        self.assertLess(np.linalg.norm(strong.w), np.linalg.norm(self.model.w))


if __name__ == "__main__":
    unittest.main()

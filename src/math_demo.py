"""Minh họa PCA và SVD trên ma trận nhỏ, độc lập với dữ liệu SMS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    """Đổi input thành ma trận float và phát hiện dữ liệu không hợp lệ."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError(f"{name} phải là ma trận hai chiều không rỗng.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} chứa NaN hoặc giá trị vô hạn.")
    return matrix


def pca_eigh(X: np.ndarray, k: int) -> dict:
    """PCA qua covariance; mỗi hàng components là một hướng chính."""
    X = _matrix(X, "X")
    if X.shape[0] < 2:
        raise ValueError("PCA cần ít nhất hai mẫu để tính covariance.")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= X.shape[1]:
        raise ValueError("k phải là số nguyên từ 1 đến số đặc trưng.")

    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (len(X) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    # Covariance là nửa xác định dương; giá trị âm rất nhỏ có thể do làm tròn.
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order[:k]].T
    return {"mean": mean, "components": components, "eigenvalues": eigenvalues}


def pca_transform(X: np.ndarray, fit: dict) -> np.ndarray:
    """Dùng mean đã học: validation/test không được tự tính lại mean."""
    X = _matrix(X, "X")
    mean = np.asarray(fit["mean"], dtype=np.float64)
    components = _matrix(fit["components"], "components")
    if X.shape[1] != len(mean) or components.shape[1] != len(mean):
        raise ValueError("Số đặc trưng không khớp PCA đã fit.")
    return (X - mean) @ components.T


def pca_inverse(Z: np.ndarray, fit: dict) -> np.ndarray:
    """Tái tạo gần đúng dữ liệu gốc, gồm bước cộng lại mean."""
    Z = _matrix(Z, "Z")
    components = _matrix(fit["components"], "components")
    if Z.shape[1] != components.shape[0]:
        raise ValueError("Số chiều Z không khớp PCA đã fit.")
    return Z @ components + np.asarray(fit["mean"], dtype=np.float64)


def run_demo() -> dict:
    """Chạy các kiểm chứng; trả dict chỉ chứa kiểu có thể lưu thành JSON."""
    from sklearn.decomposition import PCA

    seed = 42
    rng = np.random.default_rng(seed)
    # Ma trận 6x4 có mean khác zero và các eigenvalue phân biệt.
    mixing = np.array(
        [[3.0, 0.5, 0.0, 0.0], [0.0, 2.0, 0.2, 0.0],
         [0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.0, 0.4]]
    )
    X = rng.normal(size=(6, 4)) @ mixing + np.array([3.0, -2.0, 5.0, 0.5])
    U, singular_values, Vt = np.linalg.svd(X, full_matrices=False)
    reconstructed = (U * singular_values) @ Vt
    full_error = float(np.linalg.norm(X - reconstructed, ord="fro"))
    left_error = float(np.linalg.norm(U.T @ U - np.eye(4), ord="fro"))
    right_error = float(np.linalg.norm(Vt @ Vt.T - np.eye(4), ord="fro"))

    k = 2
    fitted = pca_eigh(X, k)
    centered = X - fitted["mean"]
    _, centered_singular_values, centered_Vt = np.linalg.svd(centered, full_matrices=False)
    svd_eigenvalues = centered_singular_values ** 2 / (len(X) - 1)
    eigenvalue_error = float(np.max(np.abs(fitted["eigenvalues"] - svd_eigenvalues)))

    # Dấu vector có thể khác nhau. So projector giúp so đúng không gian con.
    eig_projector = fitted["components"].T @ fitted["components"]
    svd_projector = centered_Vt[:k].T @ centered_Vt[:k]
    projector_error = float(np.linalg.norm(eig_projector - svd_projector, ord="fro"))

    errors_by_k = []
    for current_k in range(1, X.shape[1] + 1):
        X_k = (U[:, :current_k] * singular_values[:current_k]) @ Vt[:current_k]
        svd_error = float(np.linalg.norm(X - X_k, ord="fro"))
        tail_energy = float(np.sum(singular_values[current_k:] ** 2))
        pca_fit = pca_eigh(X, current_k)
        pca_X_k = pca_inverse(pca_transform(X, pca_fit), pca_fit)
        pca_error = float(np.linalg.norm(X - pca_X_k, ord="fro"))
        errors_by_k.append({
            "k": current_k,
            "svd_reconstruction_error": svd_error,
            "pca_reconstruction_error": pca_error,
            "svd_squared_error": svd_error ** 2,
            "svd_tail_energy": tail_energy,
        })
        np.testing.assert_allclose(svd_error ** 2, tail_energy, rtol=1e-6, atol=1e-8)

    sklearn_pca = PCA(n_components=k, svd_solver="full", whiten=False).fit(X)
    sklearn_projector = sklearn_pca.components_.T @ sklearn_pca.components_
    sklearn_reconstructed = sklearn_pca.inverse_transform(sklearn_pca.transform(X))
    numpy_reconstructed = pca_inverse(pca_transform(X, fitted), fitted)
    sklearn_comparison = {
        "solver": "full",
        "k": k,
        "eigenvalue_error": float(np.max(np.abs(
            fitted["eigenvalues"][:k] - sklearn_pca.explained_variance_
        ))),
        "projector_error": float(np.linalg.norm(eig_projector - sklearn_projector, ord="fro")),
        "reconstruction_difference": float(np.linalg.norm(
            numpy_reconstructed - sklearn_reconstructed, ord="fro"
        )),
    }

    # Các assertion khiến CLI báo lỗi nếu quan hệ toán học bị triển khai sai.
    np.testing.assert_allclose(reconstructed, X, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(U.T @ U, np.eye(4), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(Vt @ Vt.T, np.eye(4), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(fitted["eigenvalues"], svd_eigenvalues, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(eig_projector, svd_projector, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(eig_projector, sklearn_projector, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(numpy_reconstructed, sklearn_reconstructed, rtol=1e-6, atol=1e-8)
    assert np.all(np.diff(fitted["eigenvalues"]) < -1e-8), "Demo cần eigenvalues phân biệt."
    assert all(errors_by_k[i + 1]["svd_reconstruction_error"] <=
               errors_by_k[i]["svd_reconstruction_error"] + 1e-8
               for i in range(len(errors_by_k) - 1))
    assert all(errors_by_k[i + 1]["pca_reconstruction_error"] <=
               errors_by_k[i]["pca_reconstruction_error"] + 1e-8
               for i in range(len(errors_by_k) - 1))

    return {
        "seed": seed,
        "matrix_shape": list(X.shape),
        "matrix": X.tolist(),
        "svd_shapes": {"U": list(U.shape), "singular_values": list(singular_values.shape),
                       "Vt": list(Vt.shape)},
        "singular_values": singular_values.tolist(),
        "centered_singular_values": centered_singular_values.tolist(),
        "pca_eigenvalues": fitted["eigenvalues"].tolist(),
        "full_reconstruction_error": full_error,
        "orthogonality_error": max(left_error, right_error),
        "left_orthogonality_error": left_error,
        "right_orthogonality_error": right_error,
        "pca_eigen_vs_svd_eigenvalue_error": eigenvalue_error,
        "projector_error": projector_error,
        "comparison_k": k,
        "errors_by_k": errors_by_k,
        "sklearn_pca_full_comparison": sklearn_comparison,
        "checks_passed": True,
        "notes": [
            "Sai số tái tạo được đo bằng chuẩn Frobenius.",
            "PCA trừ mean trước khi chiếu và cộng mean khi tái tạo.",
            "PCA qua covariance và SVD của X_c cho cùng không gian con.",
            "So projector để không nhầm các vector chỉ khác dấu.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minh họa toán học PCA và SVD bằng NumPy.")
    parser.add_argument("--output", type=Path, default=Path("outputs/results/math_demo.json"))
    args = parser.parse_args()
    result = run_demo()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Ma trận minh họa: 6 mẫu x 4 đặc trưng; seed=42")
    print(f"Sai số tái tạo SVD đầy đủ: {result['full_reconstruction_error']:.3e}")
    print(f"Sai số trực giao: {result['orthogonality_error']:.3e}")
    print("Sai số eigenvalue PCA so với sigma^2/(N-1): "
          f"{result['pca_eigen_vs_svd_eigenvalue_error']:.3e}")
    print(f"Sai số projector PCA so với SVD (k=2): {result['projector_error']:.3e}")
    print(" k | Sai số SVD | Sai số PCA")
    for row in result["errors_by_k"]:
        print(f"{row['k']:2d} | {row['svd_reconstruction_error']:.6f} | "
              f"{row['pca_reconstruction_error']:.6f}")
    print("Kiểm chứng với sklearn PCA(svd_solver='full'): ĐẠT")
    print(f"Tất cả kiểm tra toán học: ĐẠT. Đã lưu {args.output}")


if __name__ == "__main__":
    main()

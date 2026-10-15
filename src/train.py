"""Chạy thí nghiệm: train → chọn k trên validation → đánh giá test một lần.

Chạy từ thư mục gốc: python -m src.train. Các hàm bên dưới chia luồng
thành từng bước để có thể đọc và đối chiếu với báo cáo của đồ án.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import sys
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from threadpoolctl import threadpool_limits

from .data import DEFAULT_DATA, load_data, split_data
from .models import DEFAULT_SEED, K_VALUES, MAX_FEATURES, make_pipeline
from .predict import predict_message


def save_json(path, values):
    """Ghi dict/list values thành JSON UTF-8 tại path."""
    Path(path).write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def array_bytes(X):
    """Trả dung lượng mảng dense hoặc các mảng thành phần của sparse CSR."""
    if sparse.issparse(X):
        return X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    return X.nbytes


def score_predictions(y_true, y_pred):
    """Nhận nhãn thật/dự đoán; trả accuracy, macro-F1 và các chỉ số lớp spam."""
    spam_true = np.asarray(y_true) == "spam"
    spam_pred = np.asarray(y_pred) == "spam"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "spam_precision": float(precision_score(spam_true, spam_pred, zero_division=0)),
        "spam_recall": float(recall_score(spam_true, spam_pred, zero_division=0)),
        "spam_f1": float(f1_score(spam_true, spam_pred, zero_division=0)),
    }


def feature_fingerprint(model):
    """Băm vocabulary có thứ tự và IDF của pipeline đã fit để đối chiếu các nhánh."""
    vectorizer = model.named_steps["tfidf"]
    features = json.dumps(vectorizer.get_feature_names_out().tolist()).encode()
    return hashlib.sha256(features + vectorizer.idf_.tobytes()).hexdigest()


def fit_evaluate(method, k, train, evaluation, split_name):
    """Fit một pipeline trên train rồi dự đoán evaluation.

    Hai bảng cần cột text/label. method là baseline/pca/svd; baseline dùng
    k=None. Trả về (pipeline đã fit, một dòng metrics, các nhãn dự đoán).
    Hàm này không chọn k và không fit trên bảng evaluation.
    """
    model = make_pipeline(method, k)
    start = perf_counter()
    with warnings.catch_warnings():
        # Dừng nếu chưa hội tụ để không âm thầm dùng kết quả chưa đạt yêu cầu.
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train["text"].tolist(), train["label"].tolist())
    fit_seconds = perf_counter() - start

    texts = evaluation["text"].tolist()
    predictions = model.predict(texts)  # Warm-up trước khi đo inference.
    inference_times = []
    for _ in range(5):
        start = perf_counter()
        model.predict(texts)
        inference_times.append(perf_counter() - start)

    # X có N hàng tin nhắn và D cột TF-IDF; chỉ transform lại dữ liệu train.
    X = model.named_steps["tfidf"].transform(train["text"])
    # model[:-1] gồm mọi bước trừ classifier: văn bản → biểu diễn đặc trưng.
    # Z có N×D phần tử ở baseline, N×k ở PCA/SVD (Z của baseline vẫn sparse).
    Z = model[:-1].transform(train["text"].tolist())
    D = X.shape[1]

    # Mỗi row trở thành một dòng CSV. Các chỉ số lưu dung lượng thực của X/Z;
    # pca_dense_input_bytes chỉ ước lượng riêng ma trận đầu vào, không phải peak RAM.
    row = {
        "method": method, "split": split_name, "seed": DEFAULT_SEED,
        "n_train": len(train), "n_evaluation": len(evaluation), "D": D,
        "k": k, "output_dimensions": Z.shape[1],
        "dimension_reduction": 1 - Z.shape[1] / D,
        "fit_seconds": fit_seconds, "inference_batch_seconds": float(np.median(inference_times)),
        "tfidf_csr_bytes": array_bytes(X), "representation_bytes": array_bytes(Z),
        "pca_dense_input_bytes": X.shape[0] * D * 8 if method == "pca" else 0,
        "feature_fingerprint": feature_fingerprint(model),
        "pca_variance_retention": None, "svd_energy_retention": None,
        **score_predictions(evaluation["label"], predictions),
    }
    # PCA đo variance của dữ liệu đã center; SVD đo energy không center.
    # Các đại lượng này được tính trên train, không dùng nhãn validation/test.
    if method == "pca":
        row["pca_variance_retention"] = float(model.named_steps["reducer"].explained_variance_ratio_.sum())
    if method == "svd":
        row["svd_energy_retention"] = float(np.sum(Z**2) / X.multiply(X).sum())
    return model, row, predictions


def choose_k(validation, method):
    """Chọn k có validation macro-F1 cao nhất; bằng điểm thì lấy k nhỏ hơn."""
    candidates = validation.loc[validation["method"].eq(method)]
    return int(candidates.sort_values(["macro_f1", "k"], ascending=[False, True]).iloc[0]["k"])


def prepare_output_and_logger(output_root):
    """Tạo thư mục kết quả; trả đường dẫn results và logger ghi cả terminal/file."""
    for name in ("results", "figures", "models"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    results = output_root / "results"
    logger = logging.getLogger("spam_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(results / "training.log", mode="w", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return results, logger


def prepare_data_and_audit(data_path, results, logger):
    """Đọc/làm sạch/chia SMS, lưu thống kê và danh sách split; trả dict các tập."""
    frame, summary = load_data(data_path)
    splits = split_data(frame, seed=DEFAULT_SEED)
    summary["splits"] = {
        name: {"rows": len(part), "class_counts": {k: int(v) for k, v in part["label"].value_counts().items()}}
        for name, part in splits.items()
    }
    summary["raw_file_sha256"] = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    save_json(results / "data_summary.json", summary)
    assignments = pd.concat([
        part[["source_id", "text_sha256", "label"]].assign(split=name)
        for name, part in splits.items()
    ]).sort_values("source_id")
    assert not assignments["text_sha256"].duplicated().any()
    assignments.to_csv(results / "split_assignments.csv", index=False)
    logger.info("Dữ liệu: %s", json.dumps(summary, ensure_ascii=False))
    return splits


def run_validation_sweep(splits, results, logger):
    """Thử baseline và grid k trên train/validation; trả bảng metrics và grid hợp lệ."""
    baseline, row, _ = fit_evaluate("baseline", None, splits["train"], splits["validation"], "validation")
    validation_rows = [row]
    dimension = row["D"]
    valid_k = [k for k in K_VALUES if 1 <= k < min(len(splits["train"]) - 1, dimension)]
    if not valid_k:
        raise ValueError("Dữ liệu quá nhỏ cho grid k={10,50,100}.")
    del baseline
    for method in ("pca", "svd"):
        for k in valid_k:
            _, row, _ = fit_evaluate(method, k, splits["train"], splits["validation"], "validation")
            validation_rows.append(row)
            logger.info("Validation %s k=%d: macro-F1=%.6f; fit=%.4fs", method, k, row["macro_f1"], row["fit_seconds"])
    validation = pd.DataFrame(validation_rows)
    # Cùng fingerprint chứng tỏ ba nhánh dùng cùng vocabulary và IDF đã fit.
    assert validation["feature_fingerprint"].nunique() == 1
    validation.to_csv(results / "validation_metrics.csv", index=False)
    return validation, valid_k


def refit_and_evaluate_test(splits, selected, output_root, logger):
    """Refit với k đã khóa trên train+validation; lưu model, test metrics và demo.

    selected chỉ chứa k đã chọn trước bước này. Hàm không chọn lại k từ test.
    """
    results = output_root / "results"
    development = pd.concat([splits["train"], splits["validation"]])
    test_rows, prediction_rows, matrices, demos = [], [], {}, []
    example_messages = [
        "Can we meet at the library tomorrow?",
        "Congratulations! You have won a free prize. Call now to claim your cash reward!",
    ]
    for method in ("baseline", "pca", "svd"):
        model, row, predictions = fit_evaluate(method, selected.get(method), development, splits["test"], "test")
        model_path = output_root / "models" / f"{method}.joblib"
        joblib.dump(model, model_path)
        restored = joblib.load(model_path)
        # Xác suất từng lớp phải giữ nguyên sau lưu/load; đây không phải retrain.
        np.testing.assert_allclose(model.predict_proba(example_messages), restored.predict_proba(example_messages))
        row["model_file_bytes"] = model_path.stat().st_size
        test_rows.append(row)
        matrices[method] = confusion_matrix(splits["test"]["label"], predictions, labels=["ham", "spam"]).tolist()
        for source_id, label, prediction in zip(splits["test"]["source_id"], splits["test"]["label"], predictions):
            prediction_rows.append({"method": method, "source_id": int(source_id), "label": label, "prediction": prediction})
        for message in example_messages:
            demos.append({"method": method, **predict_message(restored, message)})
        logger.info("TEST %s: accuracy=%.6f macro-F1=%.6f spam-recall=%.6f", method, row["accuracy"], row["macro_f1"], row["spam_recall"])
    test = pd.DataFrame(test_rows)
    # Sau refit, các nhánh vẫn phải có cùng TF-IDF trên tập development.
    assert test["feature_fingerprint"].nunique() == 1
    test.to_csv(results / "test_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(results / "test_predictions.csv", index=False)
    save_json(results / "confusion_matrices.json", {"labels": ["ham", "spam"], "matrices": matrices})
    save_json(results / "demo_predictions.json", demos)


def save_run_config(results, valid_k):
    """Lưu cấu hình, phiên bản và định nghĩa thời gian đo để tái lập thí nghiệm."""
    save_json(results / "run_config.json", {
        "seed": DEFAULT_SEED, "split_proportions": [0.70, 0.15, 0.15], "max_features": MAX_FEATURES,
        "min_df": 2, "sublinear_tf": True, "norm": "l2", "ngram_range": [1, 1],
        "classifier": {"name": "LogisticRegression", "C": 1.0, "solver": "lbfgs", "max_iter": 2000},
        "reducers": {"pca": "randomized, whiten=False", "svd": "randomized"},
        "valid_k": valid_k, "threshold": 0.5, "blas_threads": 1,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl")},
        "timing_note": "fit includes TF-IDF + densify/reducer (if any) + classifier; inference median of 5 warm batches",
        "information_metrics_fit_on": "training partition of each run",
    })


def run(data_path, output_root):
    """Điều phối sáu bước từ file dữ liệu SMS đến kết quả, model và biểu đồ."""
    # 1. Chuẩn bị thư mục đầu ra và nhật ký.
    output_root = Path(output_root)
    results, logger = prepare_output_and_logger(output_root)

    # 2. Đọc/làm sạch dữ liệu, chia tập và lưu thống kê kiểm tra.
    splits = prepare_data_and_audit(data_path, results, logger)

    # 3. Chạy baseline và thử các k; chỉ xem kết quả validation.
    validation, valid_k = run_validation_sweep(splits, results, logger)

    # 4. Khóa k TRƯỚC khi xem kết quả test; không sửa lựa chọn dựa trên test.
    selected = {method: choose_k(validation, method) for method in ("pca", "svd")}
    save_json(results / "selected_k.json", {
        "criterion": "Maximum validation macro-F1; ties choose smaller k",
        "selected_k": selected, "candidate_k": valid_k,
    })
    logger.info("Đã khóa k theo validation: %s", selected)

    # 5. Refit trên train+validation; đánh giá test, lưu model và chạy demo.
    refit_and_evaluate_test(splits, selected, output_root, logger)

    # 6. Lưu cấu hình môi trường và tạo biểu đồ từ các kết quả đã ghi.
    save_run_config(results, valid_k)
    from .plots import make_plots
    make_plots(output_root)
    logger.info("Hoàn tất: %s (results/, figures/, models/). Test không được dùng để chọn k.", output_root)


def main():
    """Đọc tham số terminal và chạy thí nghiệm với một BLAS thread."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    # Một thread đủ cho dữ liệu nhỏ, tránh overhead và giảm tải CPU.
    with threadpool_limits(limits=1):
        run(args.data, args.output_root)


if __name__ == "__main__":
    main()

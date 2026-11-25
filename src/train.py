"""Chạy toàn bộ thí nghiệm: chọn k trên validation → refit trên train+validation → test MỘT lần.

Chạy từ thư mục gốc: python -m src.train
"""

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .data import DEFAULT_DATA, load_data, stratified_split
from .logistic import LogisticRegressionGD
from .metrics import confusion_matrix
from .pipeline import K_VALUES, MAX_FEATURES, METHODS, MIN_DF, SEED, SpamPipeline, save_pipeline, to_binary
from .plots import make_plots
from .predict import predict_message

DEMO_MESSAGES = [
    "Can we meet at the library tomorrow?",
    "Congratulations! You have won a free prize. Call now to claim your cash reward!",
]


def save_json(path, values):
    """Ghi values thành file JSON UTF-8."""
    Path(path).write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_data(data_path, results):
    """Đọc, làm sạch, chia 70/15/15 theo lớp; lưu thống kê vào data_summary.json."""
    frame, summary = load_data(data_path)
    splits = stratified_split(frame, (0.7, 0.15, 0.15), seed=SEED)
    summary["splits"] = {
        name: {"rows": len(part), "ham": int((part["label"] == "ham").sum()),
               "spam": int((part["label"] == "spam").sum())}
        for name, part in splits.items()
    }
    save_json(results / "data_summary.json", summary)
    return splits


def fit_and_score(method, k, train, evaluation):
    """Fit pipeline trên train rồi gọi score trên evaluation. Trả (model, một dòng kết quả)."""
    model = SpamPipeline(method, k).fit(train["text"].tolist(), train["label"])
    row = {"method": method, "k": k,
           **model.score(evaluation["text"].tolist(), evaluation["label"])}
    reducer = model.reducer
    row["reconstruction_loss"] = reducer.loss() if reducer else None   # ‖X − X_k‖²_F
    row["retained_ratio"] = reducer.score() if reducer else None       # năng lượng / phương sai giữ lại
    row["decomposition_iterations"] = reducer.n_iter_ if reducer else None
    row["gd_epochs"] = model.classifier.n_epochs_
    row["final_train_loss"] = model.classifier.loss_history_[-1]
    return model, row


def run_validation_sweep(splits):
    """Fit trên train, score trên validation cho baseline và mọi k của SVD/PCA."""
    rows, spectrum = [], {}
    for method, k in [("baseline", None)] + [(m, k) for m in ("svd", "pca") for k in K_VALUES]:
        model, row = fit_and_score(method, k, splits["train"], splits["validation"])
        rows.append(row)
        print(f"Validation {method:8s} k={k}: macro-F1={row['macro_f1']:.4f}, "
              f"GD {row['gd_epochs']} epoch")
        if k == max(K_VALUES):  # Lưu σ của k lớn nhất để vẽ score theo mọi k ≤ 100.
            spectrum[method] = {"singular_values": model.reducer.singular_values_.tolist(),
                                "total_energy": model.reducer.total_energy_}
    return as_table(rows), spectrum


def as_table(rows):
    """Ghép các dòng kết quả thành DataFrame; cột số nguyên giữ kiểu nguyên (baseline để trống)."""
    table = pd.DataFrame(rows)
    for column in ("k", "decomposition_iterations"):
        table[column] = table[column].astype("Int64")
    return table


def choose_k(validation, method):
    """Chọn k có macro-F1 validation cao nhất; bằng nhau thì lấy k nhỏ hơn."""
    rows = validation.loc[validation["method"] == method]
    return int(rows.sort_values(["macro_f1", "k"], ascending=[False, True]).iloc[0]["k"])


def refit_and_test(splits, selected, output_root):
    """Refit ba pipeline trên train+validation với k đã khóa, đánh giá test đúng một lần."""
    development = pd.concat([splits["train"], splits["validation"]])
    test_texts, y_test = splits["test"]["text"].tolist(), to_binary(splits["test"]["label"])
    rows, matrices, loss_history, demos = [], {}, {}, []
    for method in METHODS:
        model, row = fit_and_score(method, selected.get(method), development, splits["test"])
        rows.append(row)
        matrices[method] = confusion_matrix(y_test, model.predict(test_texts)).tolist()
        loss_history[method] = [round(float(j), 10) for j in model.classifier.loss_history_]
        save_pipeline(model, output_root / "models" / method)
        demos += [{"method": method, **predict_message(model, text)} for text in DEMO_MESSAGES]
        print(f"TEST {method:8s}: accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}")
    results = output_root / "results"
    test = as_table(rows)
    test.to_csv(results / "test_metrics.csv", index=False)
    save_json(results / "confusion_matrices.json", {"labels": ["ham", "spam"], "matrices": matrices})
    save_json(results / "loss_history.json", loss_history)
    save_json(results / "demo_predictions.json", demos)
    return test


def save_run_config(results):
    """Ghi lại các tham số đã dùng để chạy lại cho ra cùng kết quả."""
    classifier = LogisticRegressionGD()
    save_json(results / "run_config.json", {
        "seed": SEED, "split": [0.7, 0.15, 0.15],
        "tfidf": {"max_features": MAX_FEATURES, "min_df": MIN_DF, "token_pattern": r"\b\w\w+\b",
                  "tf": "1 + log f", "idf": "log((1+N)/(1+df)) + 1", "norm": "l2"},
        "decomposition": {"k_values": list(K_VALUES), "method": "orthogonal iteration",
                          "tol": 1e-8, "max_iter": 5000},
        "logistic_regression": {"lambda": classifier.lam, "learning_rate": classifier.lr,
                                "max_epochs": classifier.epochs, "tol": classifier.tol,
                                "threshold": 0.5},
        "selection": "validation macro-F1 cao nhất; bằng nhau thì lấy k nhỏ hơn",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    })


def run(data_path=DEFAULT_DATA, output_root=Path("outputs")):
    """Chạy cả thí nghiệm và ghi kết quả, model, biểu đồ vào output_root."""
    output_root = Path(output_root)
    results = output_root / "results"
    for folder in ("results", "figures", "models"):
        (output_root / folder).mkdir(parents=True, exist_ok=True)
    # 1. Dữ liệu: làm sạch và chia tập.
    splits = prepare_data(data_path, results)
    # 2. Thử k trên validation (test chưa được dùng).
    validation, spectrum = run_validation_sweep(splits)
    validation.to_csv(results / "validation_metrics.csv", index=False)
    save_json(results / "spectrum.json", spectrum)
    # 3. Khóa k trước khi nhìn test.
    selected = {method: choose_k(validation, method) for method in ("svd", "pca")}
    save_json(results / "selected_k.json", selected)
    print("k đã chọn trên validation:", selected)
    # 4. Refit trên train+validation, đánh giá test một lần, lưu model.
    refit_and_test(splits, selected, output_root)
    save_run_config(results)
    make_plots(output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(args.data, args.output_root)

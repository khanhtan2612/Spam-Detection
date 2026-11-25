"""Vẽ biểu đồ từ các file kết quả trong outputs/results/ để số trên hình khớp số đã lưu."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"baseline": "#444444", "svd": "#2f6db3", "pca": "#d9822b"}
NAMES = {"baseline": "Baseline (TF-IDF)", "svd": "SVD", "pca": "PCA"}


def _save(fig, path):
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_class_distribution(class_counts, path):
    """Cột số tin ham/spam sau làm sạch."""
    fig, ax = plt.subplots(figsize=(5, 3.2), layout="constrained")
    labels = ["ham", "spam"]
    bars = ax.bar(labels, [class_counts[c] for c in labels], color=["#2f6db3", "#d9822b"])
    ax.bar_label(bars)
    ax.set(title="Số tin nhắn mỗi lớp sau làm sạch", ylabel="Số tin nhắn")
    _save(fig, path)


def plot_retention(spectrum, path):
    """Score theo k: Σ_{j≤k} σ_j² / ‖X‖²_F (SVD: năng lượng, PCA: phương sai)."""
    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    for method in ("svd", "pca"):
        sigma = np.asarray(spectrum[method]["singular_values"])
        score = np.cumsum(sigma ** 2) / spectrum[method]["total_energy"]
        k = np.arange(1, len(sigma) + 1)
        ax.plot(k, score, color=COLORS[method], label=NAMES[method])
        ax.scatter([10, 50, 100], score[[9, 49, 99]], color=COLORS[method], zorder=3)
    ax.set(xlabel="Số chiều k", ylabel="Score = tỷ lệ giữ lại", ylim=(0, 0.5),
           title="Score của SVD/PCA trên train (loss = phần còn lại)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_learning_rates(histories, path):
    """J theo epoch với vài giá trị η để chọn learning rate."""
    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    for lr, history in histories.items():
        ax.plot(history, label=f"η = {lr}")
    ax.set(xlabel="Epoch", ylabel="Loss J(w, b)", yscale="log", title="Chọn learning rate η")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_loss_curves(loss_history, path):
    """Loss J theo epoch khi huấn luyện ba pipeline cuối cùng."""
    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    for method, history in loss_history.items():
        ax.plot(history, color=COLORS[method], linestyle="--" if method == "pca" else "-",
                label=f"{NAMES[method]} ({len(history) - 1} epoch)")
    ax.set(xlabel="Epoch", ylabel="Loss J(w, b)", yscale="log",
           title="Gradient descent: loss giảm dần rồi hội tụ")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_validation_f1(validation, path):
    """Macro-F1 validation theo k của SVD/PCA, kèm đường baseline."""
    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    for method in ("svd", "pca"):
        rows = validation.loc[validation["method"] == method].sort_values("k")
        ax.plot(rows["k"], rows["macro_f1"], marker="o", color=COLORS[method], label=NAMES[method])
    baseline = validation.loc[validation["method"] == "baseline", "macro_f1"].iloc[0]
    ax.axhline(baseline, color=COLORS["baseline"], linestyle="--", label=NAMES["baseline"])
    ax.set(xlabel="Số chiều k", ylabel="Macro-F1", title="Chọn k trên validation")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_confusion_matrices(matrices, path):
    """Vẽ bằng plt.imshow, mỗi ô ghi số tin nhắn; hàng = nhãn thật, cột = nhãn dự đoán."""
    fig = plt.figure(figsize=(3.4 * len(matrices), 3.2), layout="constrained")
    for index, (method, matrix) in enumerate(matrices.items(), start=1):
        matrix = np.asarray(matrix)
        plt.subplot(1, len(matrices), index)
        plt.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                plt.text(j, i, matrix[i, j], ha="center", va="center", color=color, fontsize=13)
        plt.xticks([0, 1], ["ham", "spam"])
        plt.yticks([0, 1], ["ham", "spam"])
        plt.xlabel("Dự đoán")
        plt.ylabel("Thực tế")
        plt.title(NAMES[method])
    _save(fig, path)


def make_plots(root=Path("outputs")):
    """Vẽ lại mọi biểu đồ của thí nghiệm chính từ outputs/results/."""
    root = Path(root)
    results, figures = root / "results", root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((results / name).read_text(encoding="utf-8"))
    plot_class_distribution(read("data_summary.json")["class_counts"], figures / "class_distribution.png")
    plot_retention(read("spectrum.json"), figures / "retention.png")
    plot_validation_f1(pd.read_csv(results / "validation_metrics.csv"), figures / "validation_f1.png")
    plot_loss_curves(read("loss_history.json"), figures / "loss_curve.png")
    plot_confusion_matrices(read("confusion_matrices.json")["matrices"], figures / "test_confusion.png")


if __name__ == "__main__":
    make_plots()

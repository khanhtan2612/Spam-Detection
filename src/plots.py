"""Vẽ trực tiếp từ CSV/JSON để số trong hình khớp kết quả."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def make_plots(root=Path("outputs")):
    root = Path(root)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(root / "results/validation_metrics.csv")
    baseline = validation.loc[validation["method"].eq("baseline")].iloc[0]
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120})

    fig, ax = plt.subplots(figsize=(6.8, 4.1), layout="constrained")
    for method in ("pca", "svd"):
        rows = validation.loc[validation["method"].eq(method)].sort_values("k")
        ax.plot(rows["k"], rows["macro_f1"], marker="o", label=method.upper())
    ax.axhline(baseline["macro_f1"], color="black", linestyle="--", label="Baseline")
    ax.set(xlabel="Số chiều k", ylabel="Macro-F1", title="Chất lượng phân loại trên validation")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(figures / "validation_f1.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), layout="constrained")
    for ax, method, metric, title in zip(
        axes, ("pca", "svd"), ("pca_variance_retention", "svd_energy_retention"),
        ("PCA: variance sau centering", "SVD: energy không centering")
    ):
        rows = validation.loc[validation["method"].eq(method)].sort_values("k")
        ax.plot(rows["k"], rows[metric], marker="o")
        ax.set(xlabel="Số chiều k", ylabel="Tỷ lệ giữ lại", title=title, ylim=(0, 1))
        ax.grid(alpha=0.2)
    fig.savefig(figures / "retention.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.1), layout="constrained")
    for method in ("pca", "svd"):
        rows = validation.loc[validation["method"].eq(method)].sort_values("k")
        ax.plot(rows["k"], rows["fit_seconds"], marker="o", label=method.upper())
    ax.axhline(baseline["fit_seconds"], color="black", linestyle="--", label="Baseline")
    ax.set(xlabel="Số chiều k", ylabel="Thời gian fit (giây)", title="Fit toàn pipeline, một CPU thread")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(figures / "fit_time.png")
    plt.close(fig)

    # Dung lượng representation; đây không phải peak RAM hoặc dung lượng model.
    fig, ax = plt.subplots(figsize=(6.8, 4.1), layout="constrained")
    for method in ("pca", "svd"):
        rows = validation.loc[validation["method"].eq(method)].sort_values("k")
        ax.plot(rows["k"], rows["representation_bytes"] / 1024**2, marker="o", label=method.upper())
    ax.axhline(baseline["tfidf_csr_bytes"] / 1024**2, color="black", linestyle="--", label="TF-IDF CSR")
    ax.set(xlabel="Số chiều k", ylabel="Dung lượng (MiB)", title="Dung lượng biểu diễn train thực tế")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(figures / "representation_size.png")
    plt.close(fig)

    matrices = json.loads((root / "results/confusion_matrices.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), layout="constrained")
    for ax, method in zip(axes, ("baseline", "pca", "svd")):
        display = ConfusionMatrixDisplay(np.asarray(matrices["matrices"][method]),
                                         display_labels=matrices["labels"])
        display.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set(title=method.upper(), xlabel="Dự đoán", ylabel="Thực tế")
    fig.savefig(figures / "test_confusion.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    make_plots(parser.parse_args().output_root)

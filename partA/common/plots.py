import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#这个文件叫做plots是因为它包含了一些用于绘制图表和可视化结果的函数。具体来说，这些函数可以生成学习曲线、比较不同模型的性能、绘制特征重要性图以及展示模型架构等。这些可视化工具对于分析模型的训练过程、评估模型性能以及理解模型行为非常有帮助。
#因此，将这些函数集中在一个名为plots的文件中，可以方便地进行图表绘制和结果可视化。
def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def learning_curve(history, path, metric="rmse", title="", final_test=None):
    fig, axis = plt.subplots(figsize=(7, 4.5))
    epochs = [row["epoch"] for row in history]
    axis.plot(epochs, [row[f"train_{metric}"] for row in history], label="Train")
    if f"validation_{metric}" in history[0]:
        axis.plot(epochs, [row[f"validation_{metric}"] for row in history], label="Validation")
    if final_test is not None:
        axis.scatter([epochs[-1]], [final_test[metric]], color="tab:red", marker="o", label="Test (one final evaluation)", zorder=4)
    axis.set(xlabel="Epoch", ylabel="RMSE (SGD)" if metric == "rmse" else "R squared", title=title)
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def comparison_plot(rows, path, metric="rmse"):
    fig, axis = plt.subplots(figsize=(7, 4.5))
    names = [row["model"] for row in rows]
    means = [row[f"test_{metric}_mean"] for row in rows]
    deviations = [row[f"test_{metric}_sample_std"] for row in rows]
    axis.bar(names, means, yerr=deviations, capsize=5)
    axis.set(ylabel="Test RMSE (SGD)" if metric == "rmse" else "Test R squared")
    axis.tick_params(axis="x", rotation=12)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def diagnostic_plot(results, path, metric):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for name, result in results.items():
        history = result["history"]
        epochs = [row["epoch"] for row in history]
        for axis, split in zip(axes, ("train", "validation")):
            axis.plot(epochs, [row[f"{split}_{metric}"] for row in history], label=name)
            axis.set(xlabel="Epoch", title=f"Diagnostic {split}")
            axis.grid(alpha=0.2)
    axes[0].set_ylabel("RMSE (SGD)" if metric == "rmse" else "R squared")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def importance_plot(names, values, path, title):
    order = np.argsort(-np.asarray(values))
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.barh([names[index] for index in order[::-1]], [values[index] for index in order[::-1]])
    axis.set(xlabel="Mean absolute FeatureAblation attribution (SGD)", title=title)
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def architecture_plot(preprocessing, width, embedding_dim, parameters, path):
    fig, axis = plt.subplots(figsize=(11, 3))
    labels = [
        f"4 categorical inputs\n4 embedding tables\nwidth {embedding_dim} each",
        f"Concatenation\n{4 * embedding_dim + 6} features\n6 continuous inputs",
        f"Linear\n{4 * embedding_dim + 6} → {width}",
        f"LayerNorm\n{width}",
        f"ReLU\n{width}",
        f"Linear\n{width} → 1",
    ]
    locations = np.linspace(0.08, 0.92, len(labels))
    for index, (location, label) in enumerate(zip(locations, labels)):
        axis.text(location, 0.6, label, ha="center", va="center", fontsize=9, bbox={"boxstyle": "round,pad=0.5", "facecolor": "#edf4fa", "edgecolor": "#32658a"}, transform=axis.transAxes)
        if index:
            axis.annotate("", xy=(location - 0.065, 0.6), xytext=(locations[index - 1] + 0.065, 0.6), xycoords="axes fraction", arrowprops={"arrowstyle": "->"})
    axis.text(0.5, 0.12, f"Trainable parameters: {parameters:,}; output transformed back to SGD", ha="center", transform=axis.transAxes)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

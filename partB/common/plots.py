from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Render without a display.
import matplotlib.pyplot as plt

from .runtime import load_json


def search_curves(results, output_dir, variable, smoke=False):
    # 分别画验证损失和准确率。
    output_dir = Path(output_dir)
    prefix = f"{'smoke_' if smoke else ''}{output_dir.name}"
    for metric, ylabel in (("loss", "Validation BCE loss"), ("accuracy", "Validation accuracy")):
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for result in results:
            history = load_json(Path(result["checkpoint"]).parent / "history.json")  # Use each trial's saved history.
            value = result[variable]
            label = f"lr={value:g}" if variable == "learning_rate" else f"batch={value}"
            axis.plot([row["epoch"] for row in history], [row[f"val_{metric}"] for row in history], marker="o", markersize=3, label=label)
        axis.set(xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()  # 避免图例和标签被裁切。
        figure.savefig(output_dir / f"{prefix}_validation_{metric}.png", dpi=180)
        plt.close(figure)

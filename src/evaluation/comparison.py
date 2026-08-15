from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_model_comparison(metrics: pd.DataFrame, output_path: str | Path) -> None:
    score_columns = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    plot_data = metrics.melt(id_vars=["model"], value_vars=score_columns, var_name="metric", value_name="score")
    ax = plot_data.pivot(index="metric", columns="model", values="score").plot(kind="bar", figsize=(10, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()

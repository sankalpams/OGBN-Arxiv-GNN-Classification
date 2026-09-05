import sys
from pathlib import Path

PROJECT_ROOT = (
    Path().resolve().parent
    if Path().resolve().name == "notebooks"
    else Path().resolve()
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay
from src.config import RAW_DATA_DIR, MODELS_DIR
from src.data import load_ogbn_arxiv
from src.models import GCN, GAT
from src.evaluation import evaluate_model
from src.evaluation.comparison import plot_model_comparison

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

dataset, data, split_idx = load_ogbn_arxiv(RAW_DATA_DIR)

data = data.to(device)

split_idx = {
    key: value.to(device)
    for key, value in split_idx.items()
}

print("Dataset loaded successfully.")
print(f"Number of features: {data.num_features}")
print(f"Number of classes: {dataset.num_classes}")

models = {
    "GCN": GCN(
        data.num_features,
        256,
        dataset.num_classes
    ),
    "GAT": GAT(
        data.num_features,
        64,
        dataset.num_classes,
        heads=4
    )
}

out = PROJECT_ROOT / "results" / "evaluation"
out.mkdir(parents=True, exist_ok=True)

rows = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Evaluating {name}")
    print("=" * 60)

    ckpt = MODELS_DIR / f"best_{name.lower()}.pt"

    if not ckpt.exists():
        print(f"[SKIP] Checkpoint not found: {ckpt}")
        print("Please run the model training notebooks first.")
        continue

    model.load_state_dict(
        torch.load(
            ckpt,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    print(f"\n{name} - Validation Set")

    val_metrics, val_y_true, val_y_pred = evaluate_model(
        model,
        data,
        split_idx["valid"],
        "validation"
    )

    val_row = {
        "model": name,
        "split": "Validation",
        "accuracy": val_metrics["accuracy"],
        "precision": val_metrics.get("precision_weighted", val_metrics.get("precision")),
        "recall": val_metrics.get("recall_weighted", val_metrics.get("recall")),
        "f1": val_metrics.get("f1_weighted", val_metrics.get("f1")),
        "precision_weighted": val_metrics.get("precision_weighted", val_metrics.get("precision")),
        "recall_weighted": val_metrics.get("recall_weighted", val_metrics.get("recall")),
        "f1_weighted": val_metrics.get("f1_weighted", val_metrics.get("f1")),
    }
    rows.append(val_row)

    print("Validation Metrics:")
    for metric, value in val_metrics.items():
        if metric == "split":
            continue
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    ConfusionMatrixDisplay.from_predictions(
        val_y_true,
        val_y_pred,
        xticks_rotation="vertical",
        include_values=False
    )

    plt.title(f"{name} - Validation Confusion Matrix")

    plt.savefig(
        out / f"confusion_matrix_{name.lower()}_validation.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\n{name} - Test Set")

    test_metrics, test_y_true, test_y_pred = evaluate_model(
        model,
        data,
        split_idx["test"],
        "test"
    )

    test_row = {
        "model": name,
        "split": "Test",
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics.get("precision_weighted", test_metrics.get("precision")),
        "recall": test_metrics.get("recall_weighted", test_metrics.get("recall")),
        "f1": test_metrics.get("f1_weighted", test_metrics.get("f1")),
        "precision_weighted": test_metrics.get("precision_weighted", test_metrics.get("precision")),
        "recall_weighted": test_metrics.get("recall_weighted", test_metrics.get("recall")),
        "f1_weighted": test_metrics.get("f1_weighted", test_metrics.get("f1")),
    }
    rows.append(test_row)

    print("Test Metrics:")
    for metric, value in test_metrics.items():
        if metric == "split":
            continue
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    ConfusionMatrixDisplay.from_predictions(
        test_y_true,
        test_y_pred,
        xticks_rotation="vertical",
        include_values=False
    )

    plt.title(f"{name} - Test Confusion Matrix")

    plt.savefig(
        out / f"confusion_matrix_{name.lower()}_test.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

if rows:

    metrics_df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("ALL MODEL EVALUATION RESULTS")
    print("=" * 60)

    display_df = metrics_df[["model", "split", "accuracy", "precision", "recall", "f1"]]
    print(display_df.to_string(index=False))

    metrics_df[["model", "split", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]].to_csv(
        out / "metrics.csv",
        index=False
    )

    comparison_df = metrics_df.pivot(
        index="model",
        columns="split",
        values=[
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )

    print("\nValidation vs Test Performance:")
    print(comparison_df.to_string())

    comparison_df.to_csv(
        out / "model_comparison.csv"
    )

    # Plot comparison for test split (and overall)
    test_metrics_df = metrics_df[metrics_df["split"] == "Test"]
    plot_model_comparison(
        test_metrics_df,
        out / "model_comparison.png"
    )

    average_scores = (
        metrics_df
        .groupby("model")[
            [
                "accuracy",
                "precision",
                "recall",
                "f1"
            ]
        ]
        .mean()
    )

    print("\n" + "=" * 60)
    print("BEST MODEL ANALYSIS")
    print("=" * 60)

    print(average_scores.to_string())

    best_model = average_scores["f1"].idxmax()

    print(
        f"\nOverall best model based on average F1 Score: "
        f"{best_model}"
    )

    print("\n" + "=" * 60)
    print("STRENGTHS AND WEAKNESSES")
    print("=" * 60)

    for model_name in average_scores.index:

        scores = average_scores.loc[model_name]

        strongest_metric = scores.idxmax()
        weakest_metric = scores.idxmin()

        print(f"\n{model_name}")
        print("-" * 40)

        print(
            f"Strength: Highest average performance in "
            f"{strongest_metric.upper()} "
            f"({scores[strongest_metric]:.4f})"
        )

        print(
            f"Weakness: Lowest average performance in "
            f"{weakest_metric.upper()} "
            f"({scores[weakest_metric]:.4f})"
        )

    print("\n" + "=" * 60)
    print("VALIDATION vs TEST GENERALIZATION")
    print("=" * 60)

    for model_name in metrics_df["model"].unique():

        model_results = metrics_df[
            metrics_df["model"] == model_name
        ]

        validation = model_results[
            model_results["split"] == "Validation"
        ].iloc[0]

        test = model_results[
            model_results["split"] == "Test"
        ].iloc[0]

        print(f"\n{model_name}")
        print("-" * 40)

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]:

            difference = (
                test[metric] -
                validation[metric]
            )

            print(
                f"{metric.capitalize()}: "
                f"Validation = {validation[metric]:.4f}, "
                f"Test = {test[metric]:.4f}, "
                f"Difference = {difference:+.4f}"
            )

else:

    print(
        "\nNo model checkpoints found. "
        "Run the model training notebooks first."
    )

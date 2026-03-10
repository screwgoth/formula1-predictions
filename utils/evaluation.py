"""
Evaluation helpers: metrics computation and visualisation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_regression(y_true, y_pred, model_name: str = "Model") -> dict:
    """Return a dict of MAE, RMSE, R² for a regression model."""
    return {
        "Model": model_name,
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "R²": round(r2_score(y_true, y_pred), 3),
    }


def evaluate_classification(
    y_true, y_pred, y_prob=None, model_name: str = "Model"
) -> dict:
    """Return a dict of Accuracy, Precision, Recall, F1, and optionally ROC-AUC."""
    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "F1 Score": round(f1_score(y_true, y_pred, zero_division=0), 3),
    }
    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = round(roc_auc_score(y_true, y_prob), 3)
        except ValueError:
            metrics["ROC-AUC"] = np.nan
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results_list: list[dict],
    metric_key: str,
    title: str = "Model Comparison",
) -> None:
    """Grouped bar chart comparing models on a single metric."""
    df = pd.DataFrame(results_list)

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = sns.color_palette("viridis", len(df))
    bars = ax.bar(df["Model"], df[metric_key], color=colours, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, df[metric_key]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel(metric_key, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    model, feature_names: list[str], top_n: int = 15, title: str = "Feature Importance"
) -> None:
    """Horizontal bar chart of tree-based feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = sns.color_palette("viridis", len(indices))
    ax.barh(range(len(indices)), importances[indices], color=colours, edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> None:
    """Annotated heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_prediction_vs_actual(
    y_true, y_pred, title: str = "Predicted vs Actual Finishing Position"
) -> None:
    """Scatter plot with identity line."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, color="#4C72B0", edgecolors="white", s=60)

    lims = [0.5, 20.5]
    ax.plot(lims, lims, "--", color="#C44E52", linewidth=2, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Position", fontsize=12)
    ax.set_ylabel("Predicted Position", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

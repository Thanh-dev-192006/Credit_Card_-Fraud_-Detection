"""
Model evaluation: metrics, threshold tuning, cost analysis, and plots.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def evaluate_all(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config_path: str = "configs/model_config.yaml",
) -> pd.DataFrame:
    """
    Evaluate all models on the held-out test set, log to MLflow, return summary DataFrame.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rows = []
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        threshold, metrics = tune_threshold(y_prob, y_test, cfg)
        metrics["model"] = name
        metrics["threshold"] = threshold
        rows.append(metrics)

        _plot_pr_curve(y_test, y_prob, name, cfg)
        _plot_roc_curve(y_test, y_prob, name, cfg)
        _plot_confusion_matrix(y_test, (y_prob >= threshold).astype(int), name, cfg)

    summary = pd.DataFrame(rows).set_index("model")
    summary.to_csv(Path(cfg["experiment"]["artifact_location"]) / "reports" / "test_metrics.csv")
    logger.info("\n%s", summary.to_string())
    return summary


def tune_threshold(
    y_prob: np.ndarray,
    y_true: pd.Series,
    cfg: dict,
) -> tuple[float, dict]:
    """
    Sweep thresholds and pick the one that maximises the configured metric.
    Also computes business cost at the chosen threshold.
    """
    metric_name = cfg["threshold"]["optimize_metric"]
    cost_fp = cfg["threshold"]["cost_fp"]
    cost_fn = cfg["threshold"]["cost_fn"]

    thresholds = np.linspace(0.01, 0.99, 200)
    best_thresh, best_score = 0.5, -np.inf

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric_name == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric_name == "cost":
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            score = -(fp * cost_fp + fn * cost_fn)   # negate: higher = cheaper
        else:
            raise ValueError(f"Unknown threshold metric: {metric_name}")

        if score > best_score:
            best_score, best_thresh = score, t

    y_pred_best = (y_prob >= best_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "pr_auc":          average_precision_score(y_true, y_prob),
        "roc_auc":         roc_auc_score(y_true, y_prob),
        "f1_fraud":        f1_score(y_true, y_pred_best, zero_division=0),
        "business_cost":   fp * cost_fp + fn * cost_fn,
        "true_positives":  int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives":  int(tn),
    }
    return best_thresh, metrics


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_pr_curve(y_true, y_prob, name: str, cfg: dict) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"PR-AUC={auc:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {name}")
    ax.legend()
    _save_plot(fig, cfg, f"pr_curve_{name}.png")


def _plot_roc_curve(y_true, y_prob, name: str, cfg: dict) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC-AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curve — {name}")
    ax.legend()
    _save_plot(fig, cfg, f"roc_curve_{name}.png")


def _plot_confusion_matrix(y_true, y_pred, name: str, cfg: dict) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(ax=ax)
    ax.set_title(f"Confusion Matrix — {name}")
    _save_plot(fig, cfg, f"confusion_matrix_{name}.png")


def _save_plot(fig: plt.Figure, cfg: dict, filename: str) -> None:
    plots_dir = Path(cfg["experiment"]["artifact_location"]) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    mlflow.log_artifact(str(path))
    plt.close(fig)

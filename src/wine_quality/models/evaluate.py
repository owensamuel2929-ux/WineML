"""Metric computation for both framing tasks.

The classifier is the headline model, so its metrics are chosen with the class
imbalance in mind. ROC-AUC alone flatters a model on a 13.6% positive class;
**average precision (PR-AUC)** is reported as the primary metric because it
reflects performance on the minority "good wine" class that actually matters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float | list[list[int]]]:
    """Compute classification metrics for the "good wine" task.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Hard binary predictions.
        y_proba: Predicted probability of the positive class.

    Returns:
        Metric name to value, including the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        # Primary metric: insensitive to the large number of true negatives.
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # Probability calibration quality — relevant if the score is shown as a
        # confidence in the dashboard.
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for the sensory-score task.

    Args:
        y_true: Ground-truth quality scores.
        y_pred: Predicted quality scores (continuous).

    Returns:
        Metric name to value, including error statistics in score units.
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "max_error": float(np.abs(residuals).max()),
        "mean_residual": float(residuals.mean()),
        "std_residual": float(residuals.std()),
    }


def metrics_to_display(metrics: dict[str, Any]) -> dict[str, str]:
    """Format a metrics mapping for readable logging or table output.

    Args:
        metrics: Raw metric mapping.

    Returns:
        Mapping of metric name to a formatted string.
    """
    formatted: dict[str, str] = {}
    for name, value in metrics.items():
        if isinstance(value, float):
            formatted[name] = f"{value:.4f}"
        elif isinstance(value, int):
            formatted[name] = str(value)
        elif isinstance(value, list):
            formatted[name] = "matrix"
        else:
            formatted[name] = str(value)
    return formatted
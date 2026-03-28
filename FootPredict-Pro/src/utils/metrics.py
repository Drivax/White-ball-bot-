"""
FootPredict-Pro — Evaluation metrics.

Implements football-specific prediction metrics:
  - Ranked Probability Score (RPS)
  - Brier Score
  - Log-Loss
  - Calibration error
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def ranked_probability_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = 3,
) -> float:
    """
    Compute the mean Ranked Probability Score (RPS) for multi-class predictions.

    RPS is the standard evaluation metric for football match outcome prediction.
    Lower is better. Industry benchmark for a good model: RPS < 0.20.

    Args:
        y_true: Array of true class labels (0=Home Win, 1=Draw, 2=Away Win).
        y_prob: Array of shape (n_samples, n_classes) with predicted probabilities.
        n_classes: Number of outcome classes (default 3 for 1X2).

    Returns:
        Mean RPS across all predictions.

    References:
        Constantinou & Fenton (2012) — Solving the Problem of Inadequate Scoring
        Rules for Assessing Probabilistic Football-Match Predictions.
    """
    n_samples = len(y_true)
    rps_sum = 0.0

    for i in range(n_samples):
        # Build cumulative predicted probabilities
        cum_pred = np.cumsum(y_prob[i])
        # Build cumulative true (one-hot) probabilities
        one_hot = np.zeros(n_classes)
        one_hot[int(y_true[i])] = 1.0
        cum_true = np.cumsum(one_hot)

        # RPS for this match: sum of squared differences of cumulative probs
        rps_sum += np.sum((cum_pred[:-1] - cum_true[:-1]) ** 2) / (n_classes - 1)

    return rps_sum / n_samples


def brier_score_multiclass(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = 3,
) -> float:
    """
    Compute the mean multi-class Brier Score.

    Industry benchmark: Brier < 0.22.

    Args:
        y_true: True class labels.
        y_prob: Predicted probability matrix (n_samples, n_classes).
        n_classes: Number of classes.

    Returns:
        Mean Brier score.
    """
    n_samples = len(y_true)
    brier_sum = 0.0
    for i in range(n_samples):
        one_hot = np.zeros(n_classes)
        one_hot[int(y_true[i])] = 1.0
        brier_sum += np.sum((y_prob[i] - one_hot) ** 2)
    return brier_sum / n_samples


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = 3,
) -> dict:
    """
    Compute all evaluation metrics for outcome prediction.

    Args:
        y_true: True class labels (0=Home Win, 1=Draw, 2=Away Win).
        y_prob: Predicted probabilities (n_samples, n_classes).
        n_classes: Number of classes.

    Returns:
        Dictionary with keys: rps, brier, log_loss, accuracy, calibration_error.
    """
    y_pred = np.argmax(y_prob, axis=1)
    accuracy = float(np.mean(y_pred == y_true))

    rps = ranked_probability_score(y_true, y_prob, n_classes)
    brier = brier_score_multiclass(y_true, y_prob, n_classes)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll = log_loss(y_true, y_prob, labels=list(range(n_classes)))

    cal_err = calibration_error(y_true, y_prob, n_classes)

    return {
        "rps": round(rps, 4),
        "brier": round(brier, 4),
        "log_loss": round(ll, 4),
        "accuracy": round(accuracy, 4),
        "calibration_error": round(cal_err, 4),
    }


def calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = 3,
    n_bins: int = 10,
) -> float:
    """
    Compute the Expected Calibration Error (ECE) averaged across classes.

    Args:
        y_true: True class labels.
        y_prob: Predicted probabilities (n_samples, n_classes).
        n_classes: Number of classes.
        n_bins: Number of probability bins.

    Returns:
        Average ECE across all classes.
    """
    n_samples = len(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_total = 0.0

    for c in range(n_classes):
        probs_c = y_prob[:, c]
        labels_c = (y_true == c).astype(float)
        ece_c = 0.0
        for b in range(n_bins):
            mask = (probs_c >= bins[b]) & (probs_c < bins[b + 1])
            if mask.sum() == 0:
                continue
            bin_conf = probs_c[mask].mean()
            bin_acc = labels_c[mask].mean()
            ece_c += (mask.sum() / n_samples) * abs(bin_conf - bin_acc)
        ece_total += ece_c

    return ece_total / n_classes


def print_metrics_table(metrics: dict, title: str = "Prediction Metrics") -> None:
    """Pretty-print a metrics dictionary."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        table.add_column("Benchmark", justify="right")

        benchmarks = {
            "rps": "< 0.200",
            "brier": "< 0.220",
            "log_loss": "< 1.000",
            "accuracy": "> 0.530",
            "calibration_error": "< 0.030",
        }

        colors = {
            "rps": lambda v: "green" if v < 0.200 else "red",
            "brier": lambda v: "green" if v < 0.220 else "red",
            "log_loss": lambda v: "green" if v < 1.000 else "red",
            "accuracy": lambda v: "green" if v > 0.530 else "red",
            "calibration_error": lambda v: "green" if v < 0.030 else "red",
        }

        for key, value in metrics.items():
            color = colors.get(key, lambda v: "white")(value)
            table.add_row(
                key.replace("_", " ").title(),
                f"[{color}]{value:.4f}[/{color}]",
                benchmarks.get(key, "—"),
            )

        console.print(table)
    except ImportError:
        print(f"\n=== {title} ===")
        for key, value in metrics.items():
            print(f"  {key:25s}: {value:.4f}")

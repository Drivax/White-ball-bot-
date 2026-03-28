"""
FootPredict-Pro — Probability calibration utilities.

Wraps sklearn's calibration methods (Platt scaling, Isotonic Regression)
with multi-class support for football match outcome predictions.
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class MulticlassCalibrator:
    """
    Calibrate multi-class (1X2) probability predictions.

    Supports both Platt scaling (logistic) and Isotonic Regression.
    Fits one calibrator per class using a one-vs-rest scheme on the
    raw predicted probabilities from the base ensemble.

    Usage:
        cal = MulticlassCalibrator(method="isotonic")
        cal.fit(raw_probs_train, y_train)
        calibrated = cal.transform(raw_probs_test)
    """

    def __init__(
        self,
        method: Literal["platt", "isotonic"] = "isotonic",
        n_classes: int = 3,
    ) -> None:
        """
        Args:
            method: Calibration method ("platt" = logistic, "isotonic").
            n_classes: Number of outcome classes (3 for Home/Draw/Away).
        """
        self.method = method
        self.n_classes = n_classes
        self._calibrators: list = []
        self._is_fitted: bool = False

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "MulticlassCalibrator":
        """
        Fit calibrators on held-out probability predictions.

        Args:
            probs: Raw predicted probabilities (n_samples, n_classes).
            y: True class labels (n_samples,).

        Returns:
            self (fitted calibrator).
        """
        self._calibrators = []
        for c in range(self.n_classes):
            y_c = (y == c).astype(float)
            p_c = probs[:, c]

            if self.method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(p_c, y_c)
            else:  # platt
                cal = LogisticRegression(C=1.0, max_iter=1000)
                cal.fit(p_c.reshape(-1, 1), y_c)

            self._calibrators.append(cal)

        self._is_fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration and re-normalize so probabilities sum to 1.

        Args:
            probs: Raw predicted probabilities (n_samples, n_classes).

        Returns:
            Calibrated probabilities (n_samples, n_classes).
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before transform().")

        calibrated = np.zeros_like(probs, dtype=float)
        for c, cal in enumerate(self._calibrators):
            p_c = probs[:, c]
            if self.method == "isotonic":
                calibrated[:, c] = cal.predict(p_c)
            else:
                calibrated[:, c] = cal.predict_proba(p_c.reshape(-1, 1))[:, 1]

        # Ensure non-negative and normalize to sum to 1
        calibrated = np.clip(calibrated, 0.0, 1.0)
        row_sums = calibrated.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        calibrated = calibrated / row_sums

        return calibrated

    def fit_transform(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform same data (for diagnostics only)."""
        return self.fit(probs, y).transform(probs)

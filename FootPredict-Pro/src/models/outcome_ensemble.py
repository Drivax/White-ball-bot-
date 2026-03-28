"""
FootPredict-Pro — Stacked ensemble outcome classifier.

Combines XGBoost + LightGBM + CatBoost + Logistic Regression via
soft voting with calibrated probabilities (Isotonic or Platt scaling).

Architecture:
  Level 0: XGBoost, LightGBM, CatBoost, Logistic Regression (base models)
  Level 1: Weighted soft vote → calibration layer
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.utils.calibration import MulticlassCalibrator
from src.utils.helpers import load_model, save_model


class OutcomeEnsemble:
    """
    Stacked ensemble for predicting 1X2 match outcome.

    Base models: XGBoost, LightGBM, CatBoost, Logistic Regression.
    Meta-level: Weighted soft vote + isotonic calibration.

    Usage:
        ensemble = OutcomeEnsemble(config)
        ensemble.fit(X_train, y_train)
        probs = ensemble.predict_proba(X_test)  # shape (n, 3)
    """

    OUTCOME_LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}

    def __init__(
        self,
        xgb_params: Optional[dict] = None,
        lgb_params: Optional[dict] = None,
        cb_params: Optional[dict] = None,
        lr_params: Optional[dict] = None,
        weights: Optional[Dict[str, float]] = None,
        calibration: str = "isotonic",
    ) -> None:
        """
        Args:
            xgb_params: XGBoost hyperparameters.
            lgb_params: LightGBM hyperparameters.
            cb_params: CatBoost hyperparameters.
            lr_params: Logistic Regression hyperparameters.
            weights: Model weights for soft voting.
            calibration: "isotonic" or "platt".
        """
        self.xgb_params = xgb_params or {}
        self.lgb_params = lgb_params or {}
        self.cb_params = cb_params or {}
        self.lr_params = lr_params or {}
        self.weights = weights or {
            "xgboost": 0.30,
            "lightgbm": 0.30,
            "catboost": 0.25,
            "logistic": 0.15,
        }
        self.calibration = calibration

        self._models: Dict[str, object] = {}
        self._scaler: Optional[StandardScaler] = None
        self._calibrator: Optional[MulticlassCalibrator] = None
        self._is_fitted: bool = False
        self._feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple] = None,
        calibration_frac: float = 0.2,
    ) -> "OutcomeEnsemble":
        """
        Train all base models and the calibration layer.

        Args:
            X: Feature matrix.
            y: Labels (0=Home Win, 1=Draw, 2=Away Win).
            eval_set: Optional (X_val, y_val) for early stopping.
            calibration_frac: Fraction of training data held out for calibration.

        Returns:
            self.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values.astype(float)
        else:
            X_arr = X.astype(float)

        n_cal = max(50, int(len(y) * calibration_frac))
        X_train, y_train = X_arr[:-n_cal], y[:-n_cal]
        X_cal, y_cal = X_arr[-n_cal:], y[-n_cal:]

        logger.info(
            f"Training ensemble on {len(y_train)} samples "
            f"(calibration: {len(y_cal)})"
        )

        # Standardize (needed for Logistic Regression)
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_cal_scaled = self._scaler.transform(X_cal)

        # --- XGBoost ---
        xgb_model = self._train_xgboost(X_train, y_train, eval_set)
        self._models["xgboost"] = xgb_model

        # --- LightGBM ---
        lgb_model = self._train_lightgbm(X_train, y_train, eval_set)
        self._models["lightgbm"] = lgb_model

        # --- CatBoost ---
        cb_model = self._train_catboost(X_train, y_train, eval_set)
        self._models["catboost"] = cb_model

        # --- Logistic Regression ---
        lr_model = self._train_logistic(X_train_scaled, y_train)
        self._models["logistic"] = lr_model

        # --- Calibration ---
        raw_cal_probs = self._weighted_vote(
            X_cal, X_cal_scaled, apply_calibration=False
        )
        self._calibrator = MulticlassCalibrator(method=self.calibration)
        self._calibrator.fit(raw_cal_probs, y_cal)

        self._is_fitted = True
        logger.info("Ensemble training complete.")
        return self

    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, eval_set) -> object:
        """Train XGBoost classifier."""
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not available. Skipping.")
            return None

        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
            **self.xgb_params,
        }

        model = xgb.XGBClassifier(**params)
        fit_params: dict = {}
        if eval_set:
            fit_params["eval_set"] = [eval_set]
            fit_params["verbose"] = False
            fit_params["early_stopping_rounds"] = 50

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y, **fit_params)

        logger.debug(f"XGBoost trained: {model.n_estimators} estimators")
        return model

    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, eval_set) -> object:
        """Train LightGBM classifier."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM not available. Skipping.")
            return None

        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            **self.lgb_params,
        }

        model = lgb.LGBMClassifier(**params)
        fit_params: dict = {}
        if eval_set:
            fit_params["eval_set"] = [eval_set]
            fit_params["callbacks"] = [lgb.early_stopping(50, verbose=False)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y, **fit_params)

        logger.debug("LightGBM trained")
        return model

    def _train_catboost(self, X: np.ndarray, y: np.ndarray, eval_set) -> object:
        """Train CatBoost classifier."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            logger.warning("CatBoost not available. Skipping.")
            return None

        params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "loss_function": "MultiClass",
            "random_seed": 42,
            "verbose": 0,
            **self.cb_params,
        }

        model = CatBoostClassifier(**params)
        fit_params: dict = {}
        if eval_set:
            fit_params["eval_set"] = eval_set
            fit_params["early_stopping_rounds"] = 50

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y, **fit_params)

        logger.debug("CatBoost trained")
        return model

    def _train_logistic(self, X_scaled: np.ndarray, y: np.ndarray) -> object:
        """Train Logistic Regression (needs scaled features)."""
        params = {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
            "multi_class": "multinomial",
            "solver": "lbfgs",
            **self.lr_params,
        }
        model = LogisticRegression(**params)
        model.fit(X_scaled, y)
        logger.debug("Logistic Regression trained")
        return model

    def _weighted_vote(
        self,
        X: np.ndarray,
        X_scaled: np.ndarray,
        apply_calibration: bool = True,
    ) -> np.ndarray:
        """
        Compute weighted soft vote of base model probabilities.

        Args:
            X: Raw features.
            X_scaled: Scaled features (for Logistic Regression).
            apply_calibration: Whether to apply calibration.

        Returns:
            Probability matrix (n_samples, 3).
        """
        available_models = {k: v for k, v in self._models.items() if v is not None}
        total_weight = sum(self.weights[k] for k in available_models)

        combined = np.zeros((len(X), 3), dtype=float)

        for name, model in available_models.items():
            w = self.weights[name] / total_weight
            if name == "logistic":
                probs = model.predict_proba(X_scaled)
            else:
                probs = model.predict_proba(X)

            # Ensure shape (n, 3) even if model returns (n, 2) in edge cases
            if probs.shape[1] == 3:
                combined += w * probs
            else:
                logger.warning(f"Unexpected proba shape from {name}: {probs.shape}")

        # Re-normalize
        row_sums = combined.sum(axis=1, keepdims=True)
        combined = combined / np.where(row_sums == 0, 1.0, row_sums)

        if apply_calibration and self._calibrator:
            combined = self._calibrator.transform(combined)

        return combined

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict calibrated outcome probabilities.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Probability matrix (n_samples, 3): [P(H), P(D), P(A)].
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(float)
        else:
            X_arr = X.astype(float)

        X_scaled = self._scaler.transform(X_arr)
        return self._weighted_vote(X_arr, X_scaled, apply_calibration=True)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict most likely outcome class.

        Returns:
            Array of class labels (0=H, 1=D, 2=A).
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def feature_importance(self) -> Dict[str, float]:
        """
        Return normalized XGBoost feature importances.

        Returns:
            Dict of feature_name -> importance score.
        """
        if not self._is_fitted or "xgboost" not in self._models:
            return {}

        xgb_model = self._models["xgboost"]
        if xgb_model is None:
            return {}

        importances = xgb_model.feature_importances_
        if self._feature_names:
            return dict(zip(self._feature_names, importances.tolist()))
        return {f"f{i}": float(v) for i, v in enumerate(importances)}

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        from src.utils.helpers import save_model
        save_model(self, path)
        logger.info(f"OutcomeEnsemble saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "OutcomeEnsemble":
        """Load model from disk."""
        from src.utils.helpers import load_model
        model = load_model(path)
        logger.info(f"OutcomeEnsemble loaded from {path}")
        return model

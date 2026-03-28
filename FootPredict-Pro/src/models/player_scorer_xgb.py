"""
FootPredict-Pro — Player goal-scorer XGBoost model.

Trains an XGBoost regressor on player-level features to predict
λ_player (expected goals for a player in a single match).

Then converts λ to P(≥1 goal) via the Poisson survival function:
    P(goals ≥ 1) = 1 - P(goals = 0) = 1 - e^(-λ)

This is the correct mathematical transform from expected goals to
goal-scoring probability.

Training data format: one row per player per match, with features
including shot context, player form, position, and opponent strength.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import poisson


class PlayerScorerModel:
    """
    XGBoost regressor that predicts player-level expected goals (λ).

    Converts predicted λ → P(≥1 goal) via Poisson survival function.

    Usage:
        model = PlayerScorerModel()
        model.fit(player_features_df, goals_series)
        p_goal = model.predict_goal_probability(player_feature_dict)
    """

    def __init__(self, xgb_params: Optional[dict] = None) -> None:
        """
        Args:
            xgb_params: XGBoost hyperparameters dict.
        """
        self.xgb_params = xgb_params or {}
        self._model = None
        self._feature_names: List[str] = []
        self._is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_set: Optional[Tuple] = None,
    ) -> "PlayerScorerModel":
        """
        Train the XGBoost regressor on player-match data.

        Args:
            X: Feature DataFrame (one row per player per match).
            y: Goals scored (0, 1, 2, ...).
            eval_set: Optional (X_val, y_val) for early stopping.

        Returns:
            self.
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required. Run: pip install xgboost")

        self._feature_names = list(X.columns)
        X_arr = X.values.astype(float)

        params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "reg:tweedie",  # Good for count/xG regression
            "tweedie_variance_power": 1.5,
            "random_state": 42,
            "n_jobs": -1,
            **self.xgb_params,
        }

        self._model = xgb.XGBRegressor(**params)

        fit_params: dict = {}
        if eval_set:
            X_v, y_v = eval_set
            fit_params["eval_set"] = [(X_v.values.astype(float), y_v)]
            fit_params["verbose"] = False
            fit_params["early_stopping_rounds"] = 30

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X_arr, y.astype(float), **fit_params)

        self._is_fitted = True
        logger.info(
            f"PlayerScorerModel trained on {len(y)} samples, "
            f"{len(self._feature_names)} features"
        )
        return self

    def predict_lambda(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict expected goals (λ) for each player-match.

        Args:
            X: Feature matrix.

        Returns:
            Array of λ values (non-negative).
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before prediction.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(float)
        else:
            X_arr = X.astype(float)

        lambdas = self._model.predict(X_arr)
        return np.clip(lambdas, 0.0, None)

    def predict_goal_probability(
        self, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """
        Predict P(goals ≥ 1) for each player-match.

        Uses Poisson survival function: P(X≥1) = 1 - e^(-λ)

        Args:
            X: Feature matrix.

        Returns:
            Array of P(≥1 goal) values in [0, 1].
        """
        lambdas = self.predict_lambda(X)
        return 1.0 - np.exp(-lambdas)

    def predict_score_distribution(
        self, X: pd.DataFrame | np.ndarray, max_goals: int = 5
    ) -> np.ndarray:
        """
        Predict full P(goals = k) distribution for k = 0..max_goals.

        Args:
            X: Feature matrix.
            max_goals: Maximum goals to compute probability for.

        Returns:
            Array of shape (n_samples, max_goals+1).
        """
        lambdas = self.predict_lambda(X)
        n = len(lambdas)
        dist = np.zeros((n, max_goals + 1))
        for k in range(max_goals + 1):
            dist[:, k] = poisson.pmf(k, lambdas)
        # Normalize to sum to 1 across shown range
        dist = dist / dist.sum(axis=1, keepdims=True)
        return dist

    def feature_importance(self) -> Dict[str, float]:
        """Return feature importances from trained XGBoost model."""
        if not self._is_fitted or self._model is None:
            return {}
        importances = self._model.feature_importances_
        return dict(zip(self._feature_names, importances.tolist()))

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        from src.utils.helpers import save_model
        save_model(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "PlayerScorerModel":
        """Load model from disk."""
        from src.utils.helpers import load_model
        return load_model(path)


# ---------------------------------------------------------------------------
# Training data builder
# ---------------------------------------------------------------------------

def build_player_training_data(
    player_stats_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build feature matrix and labels for training the player goal model.

    Args:
        player_stats_df: DataFrame with per-player-per-match rows.
            Required columns: goals, player_xg_avg, player_goals_avg,
            player_shots_avg, player_shots_on_target_avg,
            player_pos_weight, player_is_home, opp_defensive_pressure,
            team_xg_avg.

    Returns:
        Tuple of (X_features, y_goals).
    """
    FEATURE_COLS = [
        "player_xg_avg",
        "player_goals_avg",
        "player_assists_avg",
        "player_shots_avg",
        "player_shots_on_target_avg",
        "player_form_score",
        "player_goal_involvement_rate",
        "player_pos_weight",
        "player_is_home",
        "team_xg_avg",
        "opp_xg_conceded_avg",
        "opp_defensive_pressure",
    ]

    available = [c for c in FEATURE_COLS if c in player_stats_df.columns]
    X = player_stats_df[available].fillna(0.0)
    y = player_stats_df["goals"].fillna(0).values.astype(float)

    return X, y


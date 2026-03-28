"""
FootPredict-Pro — End-to-end feature engineering pipeline.

Orchestrates team + player feature builders into a single pipeline
that takes raw match data and outputs a feature matrix ready for
model training or inference.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.feature_engineering.team_features import TeamFeatureBuilder
from src.feature_engineering.player_features import PlayerFeatureBuilder


class FeaturePipeline:
    """
    End-to-end feature engineering pipeline.

    Combines team-level and player-level feature builders.

    Usage:
        pipeline = FeaturePipeline()
        X, y = pipeline.fit_transform(matches_df)
        # For inference:
        features = pipeline.transform_single(home_team, away_team, ...)
    """

    # Columns expected in raw match DataFrame
    REQUIRED_COLUMNS = ["date", "home_team", "away_team", "home_goals", "away_goals"]

    def __init__(
        self,
        form_window: int = 10,
        form_short_window: int = 5,
        decay_factor: float = 0.85,
        h2h_matches: int = 5,
        min_matches: int = 5,
    ) -> None:
        self.team_builder = TeamFeatureBuilder(
            form_window=form_window,
            form_short_window=form_short_window,
            decay_factor=decay_factor,
            h2h_matches=h2h_matches,
            min_matches=min_matches,
        )
        self.player_builder = PlayerFeatureBuilder(
            form_window=5,
            decay_factor=decay_factor,
        )
        self._feature_columns: List[str] = []
        self._is_fitted: bool = False

    def fit_transform(
        self,
        matches: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Build features for all historical matches (training mode).

        Args:
            matches: Raw match DataFrame with required columns.

        Returns:
            Tuple of (feature_DataFrame, label_array).
            Labels: 0=Home Win, 1=Draw, 2=Away Win.
        """
        self._validate_input(matches)
        matches = matches.copy().sort_values("date").reset_index(drop=True)

        # Ensure result_label column
        if "result_label" not in matches.columns:
            if "result" in matches.columns:
                result_map = {"H": 0, "D": 1, "A": 2}
                matches["result_label"] = matches["result"].map(result_map)
            else:
                matches = _compute_result(matches)

        # Build team features
        feature_df = self.team_builder.build(matches)

        # Extract feature columns (all newly added columns)
        meta_cols = set(matches.columns) | {"match_idx"}
        self._feature_columns = [
            c for c in feature_df.columns
            if c not in meta_cols and c != "result_label"
        ]

        # Drop rows with no valid labels
        feature_df = feature_df.dropna(subset=["result_label"])

        X = feature_df[self._feature_columns].fillna(0.0)
        y = feature_df["result_label"].values.astype(int)

        self._is_fitted = True
        logger.info(
            f"Feature pipeline: {X.shape[0]} samples × {X.shape[1]} features"
        )
        return X, y

    def transform_single(
        self,
        home_team: str,
        away_team: str,
        home_history: List[dict],
        away_history: List[dict],
        past_matches: pd.DataFrame,
        home_lineup: Optional[List[str]] = None,
        away_lineup: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Build features for a single upcoming match (inference mode).

        Args:
            home_team: Home team name.
            away_team: Away team name.
            home_history: List of home team's past match dicts.
            away_history: List of away team's past match dicts.
            past_matches: DataFrame of all past matches (for H2H).
            home_lineup: Optional list of home starting XI player names.
            away_lineup: Optional list of away starting XI player names.

        Returns:
            Single-row feature DataFrame.
        """
        features = self.team_builder._compute_match_features(
            home_team, away_team, home_history, away_history, past_matches
        )
        features_df = pd.DataFrame([features])

        # Fill any missing columns
        for col in self._feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0.0

        return features_df[self._feature_columns].fillna(0.0)

    @property
    def feature_names(self) -> List[str]:
        """Return list of feature column names."""
        return self._feature_columns.copy()

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Raise ValueError if required columns are missing."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input DataFrame missing required columns: {missing}"
            )


def _compute_result(matches: pd.DataFrame) -> pd.DataFrame:
    """Add result and result_label columns from goals."""
    matches = matches.copy()
    matches["result"] = matches.apply(
        lambda r: "H" if r["home_goals"] > r["away_goals"]
        else ("D" if r["home_goals"] == r["away_goals"] else "A"),
        axis=1,
    )
    result_map = {"H": 0, "D": 1, "A": 2}
    matches["result_label"] = matches["result"].map(result_map)
    return matches

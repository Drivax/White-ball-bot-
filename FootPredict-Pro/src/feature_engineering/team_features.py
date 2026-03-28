"""
FootPredict-Pro — Team-level feature engineering.

Computes all team-level features for match outcome prediction:
  - Recent form (exponentially weighted, last 5/10 games)
  - Home/away performance splits
  - Goals scored/conceded rolling averages
  - xG differentials (when available)
  - Head-to-head statistics
  - League table position + points
  - Strength of schedule (opponent quality)
  - Attacking and defensive ratings

All features are computed using only past data (no future leakage).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.helpers import exponential_decay_weights, rolling_weighted_average


# ---------------------------------------------------------------------------
# Core feature builder
# ---------------------------------------------------------------------------

class TeamFeatureBuilder:
    """
    Builds team-level features for all matches in a DataFrame.

    Uses a strict temporal ordering — features for match i are computed
    from matches 0..i-1 only (no future leakage).

    Usage:
        builder = TeamFeatureBuilder(form_window=10, decay_factor=0.85)
        feature_df = builder.build(matches_df)
    """

    def __init__(
        self,
        form_window: int = 10,
        form_short_window: int = 5,
        decay_factor: float = 0.85,
        h2h_matches: int = 5,
        min_matches: int = 5,
        home_advantage: float = 0.1,
    ) -> None:
        """
        Args:
            form_window: Long-form rolling window size.
            form_short_window: Short-form window (last 5 games).
            decay_factor: Exponential decay for time-weighting.
            h2h_matches: Number of H2H matches to consider.
            min_matches: Minimum matches before computing ratings.
            home_advantage: Additive home advantage for attack rating.
        """
        self.form_window = form_window
        self.form_short_window = form_short_window
        self.decay_factor = decay_factor
        self.h2h_matches = h2h_matches
        self.min_matches = min_matches
        self.home_advantage = home_advantage

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Build the full feature matrix for all matches.

        Args:
            matches: DataFrame with columns: date, home_team, away_team,
                     home_goals, away_goals, result (H/D/A).
                     Optional: home_xg, away_xg (xG per match).

        Returns:
            DataFrame with one row per match and all features.
        """
        matches = matches.copy().sort_values("date").reset_index(drop=True)
        logger.info(f"Building features for {len(matches)} matches...")

        # Compute team histories incrementally
        feature_rows = []
        team_history: Dict[str, List[dict]] = {}

        for idx, row in matches.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            home_hist = team_history.get(home, [])
            away_hist = team_history.get(away, [])

            features = self._compute_match_features(
                home, away, home_hist, away_hist, matches.iloc[:idx]
            )
            features["match_idx"] = idx
            feature_rows.append(features)

            # Update history AFTER computing features (no leakage)
            match_date = row["date"]
            home_goals = row.get("home_goals", 0) or 0
            away_goals = row.get("away_goals", 0) or 0
            result = row.get("result", "")
            home_xg = row.get("home_xg", home_goals)
            away_xg = row.get("away_xg", away_goals)

            home_entry = {
                "date": match_date,
                "opponent": away,
                "is_home": True,
                "goals_scored": home_goals,
                "goals_conceded": away_goals,
                "xg_scored": home_xg,
                "xg_conceded": away_xg,
                "points": 3 if result == "H" else (1 if result == "D" else 0),
                "result": result,
            }
            away_entry = {
                "date": match_date,
                "opponent": home,
                "is_home": False,
                "goals_scored": away_goals,
                "goals_conceded": home_goals,
                "xg_scored": away_xg,
                "xg_conceded": home_xg,
                "points": 3 if result == "A" else (1 if result == "D" else 0),
                "result": result,
            }

            team_history.setdefault(home, []).append(home_entry)
            team_history.setdefault(away, []).append(away_entry)

        feature_df = pd.DataFrame(feature_rows)

        # Merge with original matches
        result_df = matches.copy()
        for col in feature_df.columns:
            if col != "match_idx":
                result_df[col] = feature_df[col].values

        logger.info(f"Feature matrix shape: {result_df.shape}")
        return result_df

    def _compute_match_features(
        self,
        home_team: str,
        away_team: str,
        home_hist: List[dict],
        away_hist: List[dict],
        past_matches: pd.DataFrame,
    ) -> dict:
        """Compute all features for a single match."""
        feats: dict = {}

        # Form features
        feats.update(
            self._form_features(home_hist, prefix="home", window=self.form_window)
        )
        feats.update(
            self._form_features(away_hist, prefix="away", window=self.form_window)
        )
        feats.update(
            self._form_features(
                home_hist, prefix="home_short", window=self.form_short_window
            )
        )
        feats.update(
            self._form_features(
                away_hist, prefix="away_short", window=self.form_short_window
            )
        )

        # Home/away specific form
        home_home_hist = [h for h in home_hist if h["is_home"]]
        away_away_hist = [h for h in away_hist if not h["is_home"]]
        feats.update(
            self._form_features(home_home_hist, prefix="home_ha", window=self.form_window)
        )
        feats.update(
            self._form_features(away_away_hist, prefix="away_ha", window=self.form_window)
        )

        # H2H features
        feats.update(
            self._h2h_features(home_team, away_team, past_matches)
        )

        # Differential features
        for stat in ["goals_scored", "goals_conceded", "xg_scored", "xg_conceded", "points"]:
            h_key = f"home_{stat}_avg"
            a_key = f"away_{stat}_avg"
            if h_key in feats and a_key in feats:
                feats[f"diff_{stat}"] = feats[h_key] - feats[a_key]

        return feats

    def _form_features(
        self, history: List[dict], prefix: str, window: int
    ) -> dict:
        """
        Compute form features from team match history.

        Args:
            history: List of past match dicts (chronological).
            prefix: Column prefix (e.g., "home", "away_short").
            window: Number of recent matches to consider.

        Returns:
            Dict of feature_name -> value.
        """
        feats = {}
        recent = history[-window:] if len(history) >= window else history

        n = len(recent)
        feats[f"{prefix}_n_matches"] = n

        if n == 0:
            # No history: use neutral values
            for stat in ["goals_scored", "goals_conceded", "xg_scored",
                         "xg_conceded", "points", "win_rate", "draw_rate",
                         "loss_rate", "clean_sheet_rate", "form_score"]:
                feats[f"{prefix}_{stat}_avg"] = _neutral_value(stat)
            return feats

        weights = exponential_decay_weights(n, self.decay_factor)

        for stat in ["goals_scored", "goals_conceded", "xg_scored", "xg_conceded"]:
            vals = np.array([m.get(stat, 0) or 0 for m in recent], dtype=float)
            feats[f"{prefix}_{stat}_avg"] = float(np.dot(weights, vals))

        # Points per game
        pts = np.array([m.get("points", 0) or 0 for m in recent], dtype=float)
        feats[f"{prefix}_points_avg"] = float(np.dot(weights, pts))

        # Win/draw/loss rates
        results = [m.get("result", "") for m in recent]
        is_home_list = [m.get("is_home", True) for m in recent]

        wins = np.array([
            1.0 if (r == "H" and ih) or (r == "A" and not ih) else 0.0
            for r, ih in zip(results, is_home_list)
        ])
        draws = np.array([1.0 if r == "D" else 0.0 for r in results])
        losses = 1.0 - wins - draws

        feats[f"{prefix}_win_rate_avg"] = float(np.dot(weights, wins))
        feats[f"{prefix}_draw_rate_avg"] = float(np.dot(weights, draws))
        feats[f"{prefix}_loss_rate_avg"] = float(np.dot(weights, losses))

        # Clean sheets
        cs = np.array([
            1.0 if m.get("goals_conceded", 1) == 0 else 0.0
            for m in recent
        ])
        feats[f"{prefix}_clean_sheet_rate_avg"] = float(np.dot(weights, cs))

        # Composite form score (pts weighted + goal diff)
        gd = np.array([
            m.get("goals_scored", 0) - m.get("goals_conceded", 0)
            for m in recent
        ], dtype=float)
        form_raw = pts + 0.3 * gd
        feats[f"{prefix}_form_score_avg"] = float(np.dot(weights, form_raw))

        return feats

    def _h2h_features(
        self,
        home_team: str,
        away_team: str,
        past_matches: pd.DataFrame,
    ) -> dict:
        """Compute head-to-head features between two teams."""
        feats: dict = {}
        prefix = "h2h"

        if past_matches.empty:
            for key in ["home_wins", "draws", "away_wins", "home_goals_avg",
                        "away_goals_avg", "n_matches"]:
                feats[f"{prefix}_{key}"] = _neutral_h2h_value(key)
            return feats

        # Find past meetings (both home/away orientations)
        h2h = past_matches[
            (
                (past_matches["home_team"] == home_team) &
                (past_matches["away_team"] == away_team)
            ) | (
                (past_matches["home_team"] == away_team) &
                (past_matches["away_team"] == home_team)
            )
        ].tail(self.h2h_matches)

        n = len(h2h)
        feats[f"{prefix}_n_matches"] = n

        if n == 0:
            for key in ["home_wins", "draws", "away_wins", "home_goals_avg",
                        "away_goals_avg"]:
                feats[f"{prefix}_{key}"] = _neutral_h2h_value(key)
            return feats

        # From the perspective of home_team
        home_wins = 0
        draws = 0
        away_wins = 0
        home_goals_total = 0.0
        away_goals_total = 0.0

        for _, m in h2h.iterrows():
            if m["home_team"] == home_team:
                hg = m.get("home_goals", 0) or 0
                ag = m.get("away_goals", 0) or 0
                home_goals_total += hg
                away_goals_total += ag
                if hg > ag:
                    home_wins += 1
                elif hg == ag:
                    draws += 1
                else:
                    away_wins += 1
            else:
                # Reversed fixture
                hg = m.get("away_goals", 0) or 0
                ag = m.get("home_goals", 0) or 0
                home_goals_total += hg
                away_goals_total += ag
                if hg > ag:
                    home_wins += 1
                elif hg == ag:
                    draws += 1
                else:
                    away_wins += 1

        feats[f"{prefix}_home_wins"] = home_wins / n
        feats[f"{prefix}_draws"] = draws / n
        feats[f"{prefix}_away_wins"] = away_wins / n
        feats[f"{prefix}_home_goals_avg"] = home_goals_total / n
        feats[f"{prefix}_away_goals_avg"] = away_goals_total / n

        return feats


def _neutral_value(stat: str) -> float:
    """Return a neutral/prior value for a stat when no history exists."""
    defaults = {
        "goals_scored": 1.35,
        "goals_conceded": 1.35,
        "xg_scored": 1.35,
        "xg_conceded": 1.35,
        "points": 1.0,
        "win_rate": 0.33,
        "draw_rate": 0.26,
        "loss_rate": 0.41,
        "clean_sheet_rate": 0.25,
        "form_score": 1.0,
    }
    return defaults.get(stat, 0.0)


def _neutral_h2h_value(key: str) -> float:
    """Return neutral H2H value when no history exists."""
    defaults = {
        "home_wins": 0.45,
        "draws": 0.26,
        "away_wins": 0.29,
        "home_goals_avg": 1.35,
        "away_goals_avg": 1.10,
        "n_matches": 0,
    }
    return defaults.get(key, 0.0)

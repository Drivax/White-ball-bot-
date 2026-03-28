"""
FootPredict-Pro — Player-level feature engineering.

Computes per-player features for the goal-scorer prediction model:
  - Rolling xG (expected goals) per player
  - Goal involvement rate (goals + assists per 90 min)
  - Position-adjusted contribution scores
  - Opponent defensive strength vs position
  - Starting lineup xG aggregates
  - Player form (last 5 games rolling weighted average)

When player-level data is unavailable, falls back to team-average estimates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.helpers import exponential_decay_weights


# ---------------------------------------------------------------------------
# Position configuration
# ---------------------------------------------------------------------------

POSITION_CATEGORIES = {
    # Strikers / centre forwards
    "CF": "striker", "ST": "striker", "SS": "striker",
    # Wingers / attacking mids
    "LW": "forward", "RW": "forward", "AM": "forward", "CAM": "forward",
    # Central midfielders
    "CM": "midfielder", "DM": "midfielder", "CDM": "midfielder",
    "LM": "midfielder", "RM": "midfielder",
    # Defenders
    "CB": "defender", "LB": "defender", "RB": "defender",
    "LWB": "defender", "RWB": "defender",
    # Goalkeeper
    "GK": "goalkeeper",
}

POSITION_WEIGHTS = {
    "striker": 1.0,
    "forward": 0.85,
    "midfielder": 0.55,
    "defender": 0.20,
    "goalkeeper": 0.02,
    "unknown": 0.40,
}

# Average xG per position category per 90 minutes (league averages)
POSITION_XG_PRIORS = {
    "striker": 0.40,
    "forward": 0.22,
    "midfielder": 0.09,
    "defender": 0.04,
    "goalkeeper": 0.002,
    "unknown": 0.10,
}


# ---------------------------------------------------------------------------
# Player Feature Builder
# ---------------------------------------------------------------------------

class PlayerFeatureBuilder:
    """
    Builds player-level features for the goal-scorer prediction model.

    Maintains a per-player rolling history and computes features for
    a given match using only past data (no leakage).
    """

    def __init__(
        self,
        form_window: int = 5,
        decay_factor: float = 0.85,
        min_minutes: int = 45,
    ) -> None:
        """
        Args:
            form_window: Rolling window for player form.
            decay_factor: Exponential decay per match.
            min_minutes: Minimum minutes to include in form calculation.
        """
        self.form_window = form_window
        self.decay_factor = decay_factor
        self.min_minutes = min_minutes
        self._player_history: Dict[str, List[dict]] = {}

    def reset(self) -> None:
        """Clear accumulated player history."""
        self._player_history = {}

    def get_player_features(
        self,
        player_name: str,
        position: Optional[str],
        team: str,
        opponent_team: str,
        is_home: bool,
        team_xg_avg: float = 1.35,
        opponent_xg_conceded_avg: Optional[float] = None,
    ) -> dict:
        """
        Compute feature vector for a single player in an upcoming match.

        Args:
            player_name: Player's name (used as history key).
            position: Position code (e.g., "ST", "CM"). None = unknown.
            team: Player's team name.
            opponent_team: Opponent team name.
            is_home: Whether the player's team is at home.
            team_xg_avg: Team's rolling average xG per match (for context).
            opponent_xg_conceded_avg: Opponent's average xG conceded per match.

        Returns:
            Feature dictionary for this player.
        """
        pos_cat = _map_position(position)
        history = self._player_history.get(player_name, [])
        recent = history[-self.form_window:]

        feats: dict = {
            "player_position_cat": pos_cat,
            "player_pos_weight": POSITION_WEIGHTS.get(pos_cat, 0.40),
            "player_is_home": float(is_home),
            "player_n_matches": len(recent),
        }

        if len(recent) == 0:
            # No history: use positional priors
            prior_xg = POSITION_XG_PRIORS.get(pos_cat, 0.10)
            feats["player_xg_avg"] = prior_xg
            feats["player_goals_avg"] = prior_xg * 0.85
            feats["player_assists_avg"] = prior_xg * 0.50
            feats["player_shots_avg"] = prior_xg / 0.10  # ~ 10% shot conversion
            feats["player_shots_on_target_avg"] = prior_xg / 0.35
            feats["player_form_score"] = prior_xg
            feats["player_goal_involvement_rate"] = prior_xg + prior_xg * 0.50
        else:
            n = len(recent)
            weights = exponential_decay_weights(n, self.decay_factor)

            xg_vals = np.array([m.get("xg", 0) or 0 for m in recent])
            goals_vals = np.array([m.get("goals", 0) or 0 for m in recent])
            assists_vals = np.array([m.get("assists", 0) or 0 for m in recent])
            shots_vals = np.array([m.get("shots", 0) or 0 for m in recent])
            sot_vals = np.array([m.get("shots_on_target", 0) or 0 for m in recent])

            feats["player_xg_avg"] = float(np.dot(weights, xg_vals))
            feats["player_goals_avg"] = float(np.dot(weights, goals_vals))
            feats["player_assists_avg"] = float(np.dot(weights, assists_vals))
            feats["player_shots_avg"] = float(np.dot(weights, shots_vals))
            feats["player_shots_on_target_avg"] = float(np.dot(weights, sot_vals))
            feats["player_form_score"] = float(
                np.dot(weights, goals_vals + 0.5 * assists_vals + 0.1 * xg_vals)
            )
            feats["player_goal_involvement_rate"] = (
                feats["player_goals_avg"] + feats["player_assists_avg"]
            )

        # Team and opponent context
        feats["team_xg_avg"] = team_xg_avg
        feats["opp_xg_conceded_avg"] = (
            opponent_xg_conceded_avg if opponent_xg_conceded_avg is not None
            else 1.35
        )
        feats["opp_defensive_pressure"] = (
            feats["opp_xg_conceded_avg"] / 1.35  # normalized vs league average
        )

        # Adjusted xG estimate for this match
        feats["player_adj_xg_estimate"] = (
            feats["player_xg_avg"]
            * feats["opp_defensive_pressure"]
            * (1.05 if is_home else 0.95)
        )

        return feats

    def update_player_history(
        self,
        player_name: str,
        match_stats: dict,
    ) -> None:
        """
        Add a completed match to a player's history.

        Args:
            player_name: Player's name.
            match_stats: Dict with keys: goals, assists, shots,
                         shots_on_target, xg, minutes, date.
        """
        self._player_history.setdefault(player_name, []).append(match_stats)

    def get_lineup_features(
        self,
        lineup: List[str],
        positions: Optional[Dict[str, str]],
        team: str,
        opponent_team: str,
        is_home: bool,
        team_xg_avg: float = 1.35,
        opp_xg_conceded_avg: Optional[float] = None,
    ) -> Tuple[dict, dict]:
        """
        Compute aggregate lineup features and individual player features.

        Args:
            lineup: List of player names (starting XI).
            positions: Dict mapping player_name -> position code.
            team: Team name.
            opponent_team: Opponent team name.
            is_home: Whether this team is at home.
            team_xg_avg: Team's rolling xG per match.
            opp_xg_conceded_avg: Opponent's xG conceded per match.

        Returns:
            Tuple of (lineup_summary_features, player_features_dict).
        """
        positions = positions or {}
        player_features_dict: dict = {}

        for player in lineup:
            pos = positions.get(player)
            pf = self.get_player_features(
                player_name=player,
                position=pos,
                team=team,
                opponent_team=opponent_team,
                is_home=is_home,
                team_xg_avg=team_xg_avg,
                opponent_xg_conceded_avg=opp_xg_conceded_avg,
            )
            player_features_dict[player] = pf

        # Lineup aggregate features
        all_xg = [pf["player_adj_xg_estimate"] for pf in player_features_dict.values()]
        lineup_feats = {
            "lineup_total_xg": sum(all_xg),
            "lineup_avg_xg": np.mean(all_xg) if all_xg else 0.0,
            "lineup_top3_xg": sum(sorted(all_xg, reverse=True)[:3]),
            "lineup_n_strikers": sum(
                1 for pf in player_features_dict.values()
                if pf["player_position_cat"] == "striker"
            ),
            "lineup_n_forwards": sum(
                1 for pf in player_features_dict.values()
                if pf["player_position_cat"] in ("striker", "forward")
            ),
        }

        return lineup_feats, player_features_dict


def _map_position(position_code: Optional[str]) -> str:
    """Map raw position code to category."""
    if not position_code:
        return "unknown"
    upper = position_code.upper().strip()
    return POSITION_CATEGORIES.get(upper, "unknown")


def find_top_scorer(
    player_features: Dict[str, dict],
    n: int = 1,
) -> List[Tuple[str, float]]:
    """
    Identify the top N most likely scorers and their P(≥1 goal).

    Uses adjusted xG estimate to compute Poisson P(≥1 goal) = 1 - e^(-λ).

    Args:
        player_features: Dict of player_name -> feature dict.
        n: Number of top scorers to return.

    Returns:
        List of (player_name, p_goal) tuples sorted by probability descending.
    """
    scorer_probs = []
    for player, feats in player_features.items():
        lambda_est = max(feats.get("player_adj_xg_estimate", 0.0), 0.0)
        p_goal = 1.0 - np.exp(-lambda_est)
        scorer_probs.append((player, round(p_goal, 4)))

    scorer_probs.sort(key=lambda x: x[1], reverse=True)
    return scorer_probs[:n]

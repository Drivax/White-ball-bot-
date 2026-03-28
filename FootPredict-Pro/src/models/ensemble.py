"""
FootPredict-Pro — Master ensemble combiner.

Orchestrates the full prediction pipeline:
  1. Dixon-Coles Poisson → scoreline distribution + expected goals
  2. OutcomeEnsemble (XGB+LGB+CB+LR) → calibrated 1X2 probabilities
  3. PlayerScorerModel → P(≥1 goal) per player

Blends the two outcome models (Poisson + ML ensemble) using
configurable weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.models.poisson_dixon_coles import DixonColesModel
from src.models.outcome_ensemble import OutcomeEnsemble
from src.models.player_scorer_xgb import PlayerScorerModel


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class PlayerPrediction:
    """Goal-scoring prediction for a single player."""
    name: str
    team: str
    position: Optional[str]
    lambda_xg: float             # Expected goals (λ)
    p_goal: float                # P(≥1 goal)


@dataclass
class MatchPrediction:
    """Complete prediction for a single match."""
    home_team: str
    away_team: str

    # Outcome probabilities (blended)
    p_home_win: float
    p_draw: float
    p_away_win: float

    # Source breakdown
    poisson_p_home: float
    poisson_p_draw: float
    poisson_p_away: float
    ml_p_home: float
    ml_p_draw: float
    ml_p_away: float

    # Expected goals
    home_xg: float
    away_xg: float

    # Top scorelines
    top_scorelines: List[Tuple[int, int, float]] = field(default_factory=list)

    # Player predictions
    home_top_scorer: Optional[PlayerPrediction] = None
    away_top_scorer: Optional[PlayerPrediction] = None
    all_home_players: List[PlayerPrediction] = field(default_factory=list)
    all_away_players: List[PlayerPrediction] = field(default_factory=list)

    # Metadata
    inference_time_ms: float = 0.0
    confidence: str = "medium"  # high / medium / low

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "outcome": {
                "home_win": round(self.p_home_win, 4),
                "draw": round(self.p_draw, 4),
                "away_win": round(self.p_away_win, 4),
            },
            "models": {
                "poisson": {
                    "home_win": round(self.poisson_p_home, 4),
                    "draw": round(self.poisson_p_draw, 4),
                    "away_win": round(self.poisson_p_away, 4),
                },
                "ml_ensemble": {
                    "home_win": round(self.ml_p_home, 4),
                    "draw": round(self.ml_p_draw, 4),
                    "away_win": round(self.ml_p_away, 4),
                },
            },
            "expected_goals": {
                "home": round(self.home_xg, 3),
                "away": round(self.away_xg, 3),
            },
            "top_scorelines": [
                {"home": h, "away": a, "probability": round(p, 4)}
                for h, a, p in self.top_scorelines
            ],
            "home_top_scorer": _player_to_dict(self.home_top_scorer),
            "away_top_scorer": _player_to_dict(self.away_top_scorer),
            "inference_time_ms": round(self.inference_time_ms, 1),
            "confidence": self.confidence,
        }

    def __str__(self) -> str:
        lines = [
            f"\n{'─' * 55}",
            f"  {self.home_team}  vs  {self.away_team}",
            f"{'─' * 55}",
            f"  Outcome:  Home {self.p_home_win:.1%}  |  Draw {self.p_draw:.1%}  |  Away {self.p_away_win:.1%}",
            f"  xG:       {self.home_team} {self.home_xg:.2f}  |  {self.away_team} {self.away_xg:.2f}",
        ]
        if self.top_scorelines:
            h, a, p = self.top_scorelines[0]
            lines.append(f"  Top score: {h}-{a} ({p:.1%})")

        if self.home_top_scorer:
            lines.append(
                f"  🏠 {self.home_top_scorer.name}: P(goal) = {self.home_top_scorer.p_goal:.1%}"
            )
        if self.away_top_scorer:
            lines.append(
                f"  ✈️  {self.away_top_scorer.name}: P(goal) = {self.away_top_scorer.p_goal:.1%}"
            )
        lines.append(f"{'─' * 55}")
        return "\n".join(lines)


def _player_to_dict(p: Optional[PlayerPrediction]) -> Optional[dict]:
    if p is None:
        return None
    return {
        "name": p.name,
        "team": p.team,
        "position": p.position,
        "lambda_xg": round(p.lambda_xg, 4),
        "p_goal": round(p.p_goal, 4),
    }


# ---------------------------------------------------------------------------
# Master ensemble
# ---------------------------------------------------------------------------

class MasterEnsemble:
    """
    Combines all models into a single prediction interface.

    Usage:
        ensemble = MasterEnsemble.load("models/")
        prediction = ensemble.predict("Man City", "Arsenal")
        print(prediction)
    """

    def __init__(
        self,
        outcome_model: Optional[OutcomeEnsemble] = None,
        dixon_coles_model: Optional[DixonColesModel] = None,
        player_model: Optional[PlayerScorerModel] = None,
        poisson_weight: float = 0.60,
        ml_weight: float = 0.40,
    ) -> None:
        """
        Args:
            outcome_model: Trained OutcomeEnsemble.
            dixon_coles_model: Trained DixonColesModel.
            player_model: Trained PlayerScorerModel.
            poisson_weight: Weight for Dixon-Coles in blended outcome.
            ml_weight: Weight for ML ensemble in blended outcome.
        """
        self.outcome_model = outcome_model
        self.dixon_coles = dixon_coles_model
        self.player_model = player_model
        self.poisson_weight = poisson_weight
        self.ml_weight = ml_weight

    def predict(
        self,
        home_team: str,
        away_team: str,
        features: Optional[np.ndarray] = None,
        home_lineup: Optional[List[str]] = None,
        away_lineup: Optional[List[str]] = None,
        home_player_features: Optional[Dict[str, dict]] = None,
        away_player_features: Optional[Dict[str, dict]] = None,
    ) -> MatchPrediction:
        """
        Generate a full match prediction.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            features: Optional pre-computed feature vector (for ML ensemble).
            home_lineup: Home starting XI player names.
            away_lineup: Away starting XI player names.
            home_player_features: Pre-computed player feature dicts for home.
            away_player_features: Pre-computed player feature dicts for away.

        Returns:
            MatchPrediction dataclass with full prediction.
        """
        start_time = time.perf_counter()

        # --- Poisson (Dixon-Coles) prediction ---
        if self.dixon_coles and self.dixon_coles._is_fitted:
            p_home_dc, p_draw_dc, p_away_dc = self.dixon_coles.outcome_probabilities(
                home_team, away_team
            )
            home_xg, away_xg = self.dixon_coles.expected_goals_for(home_team, away_team)
            top_scorelines = self.dixon_coles.most_likely_score(home_team, away_team, top_n=5)
        else:
            logger.warning("Dixon-Coles not available. Using flat priors.")
            p_home_dc, p_draw_dc, p_away_dc = 0.45, 0.27, 0.28
            home_xg, away_xg = 1.35, 1.10
            top_scorelines = [(1, 1, 0.10), (2, 1, 0.08), (1, 0, 0.08)]

        # --- ML ensemble prediction ---
        if self.outcome_model and self.outcome_model._is_fitted and features is not None:
            probs = self.outcome_model.predict_proba(features)[0]
            p_home_ml, p_draw_ml, p_away_ml = float(probs[0]), float(probs[1]), float(probs[2])
        else:
            # Fall back to Poisson probabilities
            p_home_ml, p_draw_ml, p_away_ml = p_home_dc, p_draw_dc, p_away_dc

        # --- Blend outcomes ---
        w_p = self.poisson_weight
        w_m = self.ml_weight
        total_w = w_p + w_m

        p_home = (w_p * p_home_dc + w_m * p_home_ml) / total_w
        p_draw = (w_p * p_draw_dc + w_m * p_draw_ml) / total_w
        p_away = (w_p * p_away_dc + w_m * p_away_ml) / total_w

        # Normalize
        total = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

        # --- Player goal predictions ---
        home_players, away_players = [], []

        if home_player_features:
            home_players = self._score_players(
                home_player_features, home_team, "home"
            )
        elif home_lineup:
            home_players = self._default_player_preds(home_lineup, home_team, home_xg)

        if away_player_features:
            away_players = self._score_players(
                away_player_features, away_team, "away"
            )
        elif away_lineup:
            away_players = self._default_player_preds(away_lineup, away_team, away_xg)

        home_top = home_players[0] if home_players else None
        away_top = away_players[0] if away_players else None

        # --- Confidence ---
        max_prob = max(p_home, p_draw, p_away)
        confidence = "high" if max_prob > 0.55 else ("medium" if max_prob > 0.45 else "low")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            p_home_win=p_home,
            p_draw=p_draw,
            p_away_win=p_away,
            poisson_p_home=p_home_dc,
            poisson_p_draw=p_draw_dc,
            poisson_p_away=p_away_dc,
            ml_p_home=p_home_ml,
            ml_p_draw=p_draw_ml,
            ml_p_away=p_away_ml,
            home_xg=home_xg,
            away_xg=away_xg,
            top_scorelines=top_scorelines,
            home_top_scorer=home_top,
            away_top_scorer=away_top,
            all_home_players=home_players,
            all_away_players=away_players,
            inference_time_ms=elapsed_ms,
            confidence=confidence,
        )

    def _score_players(
        self,
        player_features: Dict[str, dict],
        team: str,
        side: str,
    ) -> List[PlayerPrediction]:
        """Convert player feature dicts to PlayerPrediction list, sorted by P(goal)."""
        predictions = []
        for player_name, feats in player_features.items():
            lam = max(feats.get("player_adj_xg_estimate", 0.0), 0.0)
            p_goal = 1.0 - np.exp(-lam)
            predictions.append(PlayerPrediction(
                name=player_name,
                team=team,
                position=feats.get("player_position_cat"),
                lambda_xg=round(lam, 4),
                p_goal=round(p_goal, 4),
            ))

        predictions.sort(key=lambda x: x.p_goal, reverse=True)
        return predictions

    def _default_player_preds(
        self,
        lineup: List[str],
        team: str,
        team_xg: float,
    ) -> List[PlayerPrediction]:
        """
        Generate default player predictions when no player-specific data.

        Distributes team xG equally (adjusted by position prior).
        """
        from src.feature_engineering.player_features import POSITION_XG_PRIORS

        # Simple equal distribution
        n = max(len(lineup), 1)
        per_player_xg = team_xg / n

        predictions = []
        for player in lineup:
            lam = per_player_xg
            p_goal = 1.0 - np.exp(-lam)
            predictions.append(PlayerPrediction(
                name=player,
                team=team,
                position=None,
                lambda_xg=round(lam, 4),
                p_goal=round(p_goal, 4),
            ))

        predictions.sort(key=lambda x: x.p_goal, reverse=True)
        return predictions

    def save(self, model_dir: str | Path) -> None:
        """Save all component models to directory."""
        from src.utils.helpers import save_model, ensure_dir
        model_dir = ensure_dir(model_dir)

        if self.dixon_coles:
            save_model(self.dixon_coles, model_dir / "dixon_coles.joblib")
        if self.outcome_model:
            save_model(self.outcome_model, model_dir / "outcome_ensemble.joblib")
        if self.player_model:
            save_model(self.player_model, model_dir / "player_scorer.joblib")

        logger.info(f"Models saved to {model_dir}")

    @classmethod
    def load(cls, model_dir: str | Path) -> "MasterEnsemble":
        """Load all component models from directory."""
        from src.utils.helpers import load_model
        model_dir = Path(model_dir)

        dc = None
        outcome = None
        player = None

        dc_path = model_dir / "dixon_coles.joblib"
        if dc_path.exists():
            dc = load_model(dc_path)
            logger.info("Loaded DixonColesModel")

        oe_path = model_dir / "outcome_ensemble.joblib"
        if oe_path.exists():
            outcome = load_model(oe_path)
            logger.info("Loaded OutcomeEnsemble")

        ps_path = model_dir / "player_scorer.joblib"
        if ps_path.exists():
            player = load_model(ps_path)
            logger.info("Loaded PlayerScorerModel")

        return cls(
            outcome_model=outcome,
            dixon_coles_model=dc,
            player_model=player,
        )

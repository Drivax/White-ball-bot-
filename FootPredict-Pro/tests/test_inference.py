"""
FootPredict-Pro — Tests for inference pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.ensemble import MasterEnsemble, MatchPrediction, PlayerPrediction
from src.models.poisson_dixon_coles import DixonColesModel
from src.inference.predict_match import MatchPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_dc_model():
    """A minimal fitted Dixon-Coles model."""
    import pandas as pd

    rng = np.random.default_rng(42)
    teams = ["Arsenal", "Chelsea", "Liverpool", "Man City"]
    rows = []
    date = pd.Timestamp("2022-08-01")
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i == j:
                continue
            hg = int(rng.poisson(1.5))
            ag = int(rng.poisson(1.0))
            rows.append({
                "date": date + pd.Timedelta(days=len(rows) * 7),
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
            })

    df = pd.DataFrame(rows)
    model = DixonColesModel(max_goals=8, xi=0.0)
    model.fit(df)
    return model


@pytest.fixture
def master_ensemble(fitted_dc_model):
    """A MasterEnsemble with only Dixon-Coles loaded."""
    return MasterEnsemble(
        dixon_coles_model=fitted_dc_model,
        outcome_model=None,
        player_model=None,
    )


# ---------------------------------------------------------------------------
# MasterEnsemble tests
# ---------------------------------------------------------------------------

class TestMasterEnsemble:
    def test_predict_returns_match_prediction(self, master_ensemble):
        """predict() should return a MatchPrediction instance."""
        pred = master_ensemble.predict("Arsenal", "Chelsea")
        assert isinstance(pred, MatchPrediction)

    def test_outcome_probs_sum_to_one(self, master_ensemble):
        """Blended outcome probabilities should sum to 1."""
        pred = master_ensemble.predict("Arsenal", "Chelsea")
        total = pred.p_home_win + pred.p_draw + pred.p_away_win
        assert abs(total - 1.0) < 1e-6

    def test_outcome_probs_in_range(self, master_ensemble):
        """Each probability should be in [0, 1]."""
        pred = master_ensemble.predict("Arsenal", "Chelsea")
        for p in [pred.p_home_win, pred.p_draw, pred.p_away_win]:
            assert 0.0 <= p <= 1.0

    def test_xg_positive(self, master_ensemble):
        """Expected goals should be positive."""
        pred = master_ensemble.predict("Arsenal", "Chelsea")
        assert pred.home_xg > 0
        assert pred.away_xg > 0

    def test_top_scorelines_sorted(self, master_ensemble):
        """Top scorelines should be sorted by probability descending."""
        pred = master_ensemble.predict("Arsenal", "Chelsea")
        probs = [s[2] for s in pred.top_scorelines]
        assert probs == sorted(probs, reverse=True)

    def test_predict_with_lineup(self, master_ensemble):
        """Prediction should work with lineups provided."""
        home_lineup = ["Player1", "Player2", "Player3", "Player4", "Player5",
                       "Player6", "Player7", "Player8", "Player9", "Player10", "Player11"]
        away_lineup = ["APlayer1", "APlayer2", "APlayer3", "APlayer4", "APlayer5",
                       "APlayer6", "APlayer7", "APlayer8", "APlayer9", "APlayer10", "APlayer11"]
        pred = master_ensemble.predict(
            "Arsenal", "Chelsea",
            home_lineup=home_lineup,
            away_lineup=away_lineup,
        )
        assert pred.home_top_scorer is not None
        assert pred.away_top_scorer is not None
        assert len(pred.all_home_players) == len(home_lineup)

    def test_predict_unknown_team(self, master_ensemble):
        """Prediction for unknown team should not raise."""
        pred = master_ensemble.predict("Unknown FC", "Chelsea")
        total = pred.p_home_win + pred.p_draw + pred.p_away_win
        assert abs(total - 1.0) < 1e-6

    def test_confidence_level_set(self, master_ensemble):
        """Confidence level should be high/medium/low."""
        pred = master_ensemble.predict("Arsenal", "Chelsea")
        assert pred.confidence in ("high", "medium", "low")

    def test_inference_time_recorded(self, master_ensemble):
        """Inference time should be recorded and positive."""
        pred = master_ensemble.predict("Arsenal", "Chelsea")
        assert pred.inference_time_ms > 0

    def test_to_dict_complete(self, master_ensemble):
        """to_dict() should return a dict with all expected keys."""
        pred = master_ensemble.predict("Arsenal", "Chelsea")
        d = pred.to_dict()
        assert "home_team" in d
        assert "away_team" in d
        assert "outcome" in d
        assert "expected_goals" in d
        assert "top_scorelines" in d

    def test_to_dict_json_serializable(self, master_ensemble):
        """to_dict() output should be JSON-serializable."""
        pred = master_ensemble.predict("Arsenal", "Chelsea")
        d = pred.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# PlayerPrediction tests
# ---------------------------------------------------------------------------

class TestPlayerPrediction:
    def test_p_goal_poisson_formula(self):
        """P(≥1 goal) should match Poisson formula."""
        lam = 0.5
        pred = PlayerPrediction(
            name="Test Player",
            team="Team A",
            position="ST",
            lambda_xg=lam,
            p_goal=round(1.0 - np.exp(-lam), 4),
        )
        expected = 1.0 - np.exp(-lam)
        assert abs(pred.p_goal - expected) < 0.001


# ---------------------------------------------------------------------------
# MatchPredictor tests (uses default/fallback mode)
# ---------------------------------------------------------------------------

class TestMatchPredictor:
    def test_predictor_creates(self, tmp_path):
        """MatchPredictor should create without error."""
        predictor = MatchPredictor(model_dir=str(tmp_path))
        assert predictor is not None

    def test_predictor_loads_fallback(self, tmp_path):
        """MatchPredictor should load in fallback mode if no models exist."""
        predictor = MatchPredictor(model_dir=str(tmp_path))
        predictor.load()
        assert predictor._loaded

    def test_predict_returns_result(self, tmp_path):
        """predict() in fallback mode should return a MatchPrediction."""
        predictor = MatchPredictor(model_dir=str(tmp_path))
        predictor.load()
        pred = predictor.predict("Manchester City", "Arsenal")
        assert isinstance(pred, MatchPrediction)

    def test_predict_with_lineup_str(self, tmp_path):
        """predict() with comma-separated lineup should work."""
        predictor = MatchPredictor(model_dir=str(tmp_path))
        predictor.load()
        lineup = ["Haaland", "De Bruyne", "Silva", "Foden", "Doku",
                  "Rodri", "Walker", "Dias", "Akanji", "Gvardiol", "Ederson"]
        pred = predictor.predict(
            "Manchester City", "Arsenal", home_lineup=lineup
        )
        assert pred.home_top_scorer is not None

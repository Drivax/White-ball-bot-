"""
FootPredict-Pro — Tests for feature engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering.team_features import TeamFeatureBuilder, _neutral_value
from src.feature_engineering.player_features import (
    PlayerFeatureBuilder,
    POSITION_WEIGHTS,
    find_top_scorer,
    _map_position,
)
from src.feature_engineering.pipeline import FeaturePipeline
from src.utils.helpers import exponential_decay_weights, rolling_weighted_average


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """Create a small synthetic match DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 60

    teams = ["Team A", "Team B", "Team C", "Team D"]
    rows = []
    date = pd.Timestamp("2022-08-01")
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        hg = int(rng.poisson(1.4))
        ag = int(rng.poisson(1.1))
        rows.append({
            "date": date + pd.Timedelta(days=i * 7),
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "result": "H" if hg > ag else ("D" if hg == ag else "A"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_exponential_decay_weights_sum_to_one(self):
        w = exponential_decay_weights(10, 0.85)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_exponential_decay_weights_most_recent_highest(self):
        w = exponential_decay_weights(10, 0.85)
        # Last element (most recent) should be highest
        assert w[-1] == w.max()

    def test_exponential_decay_weights_length(self):
        for n in [1, 5, 10, 20]:
            w = exponential_decay_weights(n, 0.85)
            assert len(w) == n

    def test_rolling_weighted_average_returns_series(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_weighted_average(s, window=3, decay=0.9)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)

    def test_rolling_weighted_average_increasing_series(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_weighted_average(s, window=3, decay=0.9)
        # For increasing series, weighted avg should be increasing
        assert result.iloc[-1] > result.iloc[0]


# ---------------------------------------------------------------------------
# Team features tests
# ---------------------------------------------------------------------------

class TestTeamFeatureBuilder:
    def test_build_returns_dataframe(self, sample_matches):
        builder = TeamFeatureBuilder()
        result = builder.build(sample_matches)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_matches)

    def test_build_adds_form_features(self, sample_matches):
        builder = TeamFeatureBuilder()
        result = builder.build(sample_matches)
        # Check for expected feature columns
        feature_cols = [c for c in result.columns if c.startswith("home_")]
        assert len(feature_cols) > 0

    def test_build_no_leakage(self, sample_matches):
        """Verify that features for match i don't use match i's result."""
        builder = TeamFeatureBuilder(form_window=5)
        result = builder.build(sample_matches)
        # First match should have 0 prior matches
        assert result.iloc[0]["home_n_matches"] == 0
        assert result.iloc[0]["away_n_matches"] == 0

    def test_h2h_features_present(self, sample_matches):
        builder = TeamFeatureBuilder()
        result = builder.build(sample_matches)
        h2h_cols = [c for c in result.columns if c.startswith("h2h_")]
        assert len(h2h_cols) > 0

    def test_neutral_value(self):
        assert _neutral_value("goals_scored") > 0
        assert _neutral_value("win_rate") == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# Player features tests
# ---------------------------------------------------------------------------

class TestPlayerFeatureBuilder:
    def test_get_player_features_no_history(self):
        builder = PlayerFeatureBuilder()
        feats = builder.get_player_features(
            player_name="Test Player",
            position="ST",
            team="Team A",
            opponent_team="Team B",
            is_home=True,
        )
        assert "player_xg_avg" in feats
        assert feats["player_xg_avg"] > 0  # Should use position prior
        assert feats["player_pos_weight"] == POSITION_WEIGHTS["striker"]

    def test_get_player_features_with_history(self):
        builder = PlayerFeatureBuilder()
        # Add some match history
        for i in range(5):
            builder.update_player_history("Test Player", {
                "goals": 1 if i % 2 == 0 else 0,
                "assists": 0,
                "shots": 3,
                "shots_on_target": 2,
                "xg": 0.35,
                "minutes": 90,
                "date": pd.Timestamp(f"2023-0{i+1}-01"),
            })

        feats = builder.get_player_features(
            player_name="Test Player",
            position="ST",
            team="Team A",
            opponent_team="Team B",
            is_home=True,
        )
        assert feats["player_xg_avg"] == pytest.approx(0.35)
        assert feats["player_n_matches"] == 5

    def test_map_position(self):
        assert _map_position("ST") == "striker"
        assert _map_position("CF") == "striker"
        assert _map_position("GK") == "goalkeeper"
        assert _map_position("CM") == "midfielder"
        assert _map_position("CB") == "defender"
        assert _map_position(None) == "unknown"
        assert _map_position("XYZ") == "unknown"

    def test_find_top_scorer(self):
        player_feats = {
            "Player A": {"player_adj_xg_estimate": 0.5},
            "Player B": {"player_adj_xg_estimate": 0.3},
            "Player C": {"player_adj_xg_estimate": 0.1},
        }
        top = find_top_scorer(player_feats, n=1)
        assert len(top) == 1
        assert top[0][0] == "Player A"
        assert top[0][1] == pytest.approx(1.0 - np.exp(-0.5), abs=0.001)

    def test_get_lineup_features(self):
        builder = PlayerFeatureBuilder()
        lineup = ["Player A", "Player B", "Player C"]
        positions = {"Player A": "ST", "Player B": "CM", "Player C": "CB"}
        lineup_feats, player_feats = builder.get_lineup_features(
            lineup=lineup,
            positions=positions,
            team="Team A",
            opponent_team="Team B",
            is_home=True,
        )
        assert "lineup_total_xg" in lineup_feats
        assert "lineup_top3_xg" in lineup_feats
        assert len(player_feats) == len(lineup)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestFeaturePipeline:
    def test_fit_transform_shapes(self, sample_matches):
        pipeline = FeaturePipeline()
        X, y = pipeline.fit_transform(sample_matches)
        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(y)
        assert len(X) > 0
        assert X.shape[1] > 10  # Should have many features

    def test_fit_transform_labels_valid(self, sample_matches):
        pipeline = FeaturePipeline()
        X, y = pipeline.fit_transform(sample_matches)
        assert set(y).issubset({0, 1, 2})

    def test_no_nan_in_features(self, sample_matches):
        pipeline = FeaturePipeline()
        X, y = pipeline.fit_transform(sample_matches)
        assert not X.isna().any().any(), "Feature matrix contains NaN values"

    def test_feature_names_accessible(self, sample_matches):
        pipeline = FeaturePipeline()
        X, y = pipeline.fit_transform(sample_matches)
        assert len(pipeline.feature_names) == X.shape[1]

    def test_validates_required_columns(self):
        pipeline = FeaturePipeline()
        bad_df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing required columns"):
            pipeline.fit_transform(bad_df)

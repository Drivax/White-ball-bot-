"""
FootPredict-Pro — Tests for models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.poisson_dixon_coles import DixonColesModel, _tau
from src.utils.metrics import (
    ranked_probability_score,
    brier_score_multiclass,
    compute_all_metrics,
    calibration_error,
)
from src.utils.calibration import MulticlassCalibrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_match_df() -> pd.DataFrame:
    """Create a small match DataFrame for testing Dixon-Coles."""
    rng = np.random.default_rng(42)
    teams = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Spurs"]
    rows = []
    date = pd.Timestamp("2022-08-01")

    # Generate round-robin schedule (each team plays each other twice)
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i == j:
                continue
            hg = int(rng.poisson(1.5))
            ag = int(rng.poisson(1.1))
            rows.append({
                "date": date + pd.Timedelta(days=len(rows) * 7),
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
            })

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_probs():
    """Generate synthetic prediction probabilities for metric testing."""
    rng = np.random.default_rng(0)
    n = 100
    # Simulate realistic 1X2 probabilities
    raw = rng.dirichlet(alpha=[3.0, 2.0, 2.0], size=n)
    # True labels: sample from the distribution
    labels = np.array([rng.choice(3, p=raw[i]) for i in range(n)])
    return raw, labels


# ---------------------------------------------------------------------------
# Dixon-Coles tests
# ---------------------------------------------------------------------------

class TestDixonColesModel:
    def test_tau_low_scores(self):
        """Test the tau correction function for low scores."""
        lam, mu, rho = 1.5, 1.1, -0.1
        # 0-0 and 1-1 should differ from 1.0
        tau_00 = _tau(0, 0, lam, mu, rho)
        tau_11 = _tau(1, 1, lam, mu, rho)
        tau_20 = _tau(2, 0, lam, mu, rho)

        assert tau_00 != 1.0
        assert tau_11 != 1.0
        assert tau_20 == 1.0  # High scores → correction = 1

    def test_fit_runs_on_small_data(self, small_match_df):
        """Model should fit without errors on small dataset."""
        model = DixonColesModel(xi=0.0018, max_goals=5)
        model.fit(small_match_df)
        assert model._is_fitted

    def test_attack_defense_params_set(self, small_match_df):
        """Fitted model should have attack/defense params for all teams."""
        model = DixonColesModel(max_goals=5)
        model.fit(small_match_df)
        teams = set(small_match_df["home_team"]) | set(small_match_df["away_team"])
        for team in teams:
            assert team in model.attack
            assert team in model.defense

    def test_outcome_probs_sum_to_one(self, small_match_df):
        """Outcome probabilities should sum to 1."""
        model = DixonColesModel(max_goals=5)
        model.fit(small_match_df)
        ph, pd_, pa = model.outcome_probabilities("Arsenal", "Chelsea")
        assert abs(ph + pd_ + pa - 1.0) < 1e-6

    def test_outcome_probs_in_range(self, small_match_df):
        """Each probability should be in [0, 1]."""
        model = DixonColesModel(max_goals=5)
        model.fit(small_match_df)
        ph, pd_, pa = model.outcome_probabilities("Arsenal", "Chelsea")
        for p in [ph, pd_, pa]:
            assert 0.0 <= p <= 1.0

    def test_score_matrix_sums_to_one(self, small_match_df):
        """Score matrix should approximately sum to 1."""
        model = DixonColesModel(max_goals=8)
        model.fit(small_match_df)
        matrix = model.score_matrix("Arsenal", "Chelsea")
        assert abs(matrix.sum() - 1.0) < 0.01

    def test_home_advantage_positive(self, small_match_df):
        """Home advantage should be non-negative."""
        model = DixonColesModel()
        model.fit(small_match_df)
        assert model.home_advantage >= 0

    def test_most_likely_score_returns_list(self, small_match_df):
        """most_likely_score should return sorted list of tuples."""
        model = DixonColesModel(max_goals=5)
        model.fit(small_match_df)
        scores = model.most_likely_score("Arsenal", "Chelsea", top_n=5)
        assert len(scores) == 5
        # Should be sorted by probability descending
        probs = [s[2] for s in scores]
        assert probs == sorted(probs, reverse=True)

    def test_unknown_team_uses_average(self, small_match_df):
        """Prediction for unknown team should not raise and use averages."""
        model = DixonColesModel(max_goals=5)
        model.fit(small_match_df)
        # Unknown team should use average parameters (not crash)
        ph, pd_, pa = model.outcome_probabilities("Unknown FC", "Arsenal")
        assert abs(ph + pd_ + pa - 1.0) < 1e-6

    def test_raises_before_fit(self):
        """score_matrix should raise RuntimeError if not fitted."""
        model = DixonColesModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.score_matrix("Arsenal", "Chelsea")

    def test_expected_goals_reasonable(self, small_match_df):
        """Expected goals should be in reasonable range."""
        model = DixonColesModel(max_goals=8)
        model.fit(small_match_df)
        home_xg, away_xg = model.expected_goals_for("Arsenal", "Chelsea")
        assert 0.3 <= home_xg <= 5.0
        assert 0.3 <= away_xg <= 5.0


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_rps_perfect_prediction(self):
        """Perfect prediction should give RPS close to 0."""
        y_true = np.array([0, 1, 2, 0])
        y_prob = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        rps = ranked_probability_score(y_true, y_prob)
        assert rps == pytest.approx(0.0, abs=1e-10)

    def test_rps_worst_prediction(self):
        """Completely wrong prediction should give high RPS."""
        y_true = np.array([0, 0])
        y_prob = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        rps = ranked_probability_score(y_true, y_prob)
        assert rps > 0.5

    def test_rps_range(self, synthetic_probs):
        """RPS should always be in [0, 1]."""
        probs, labels = synthetic_probs
        rps = ranked_probability_score(labels, probs)
        assert 0.0 <= rps <= 1.0

    def test_brier_perfect(self):
        """Brier score for perfect prediction should be 0."""
        y_true = np.array([0, 1, 2])
        y_prob = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        brier = brier_score_multiclass(y_true, y_prob)
        assert brier == pytest.approx(0.0, abs=1e-10)

    def test_compute_all_metrics_keys(self, synthetic_probs):
        """compute_all_metrics should return expected keys."""
        probs, labels = synthetic_probs
        metrics = compute_all_metrics(labels, probs)
        expected_keys = {"rps", "brier", "log_loss", "accuracy", "calibration_error"}
        assert expected_keys.issubset(set(metrics.keys()))

    def test_compute_all_metrics_ranges(self, synthetic_probs):
        """All metrics should be in reasonable ranges."""
        probs, labels = synthetic_probs
        metrics = compute_all_metrics(labels, probs)
        assert 0.0 <= metrics["rps"] <= 1.0
        assert 0.0 <= metrics["brier"] <= 1.0
        assert metrics["log_loss"] > 0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["calibration_error"] <= 1.0

    def test_calibration_error_perfect(self):
        """Perfectly calibrated probabilities should have low ECE."""
        # For 100 samples, true labels match the probabilities
        rng = np.random.default_rng(123)
        n = 200
        probs = rng.dirichlet([3, 2, 2], size=n)
        labels = np.array([rng.choice(3, p=probs[i]) for i in range(n)])
        ece = calibration_error(labels, probs)
        # ECE should be relatively small for well-calibrated probs
        assert ece < 0.20  # Reasonable threshold for stochastic test


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_fit_transform_shape(self):
        """Calibrated output should have same shape as input."""
        rng = np.random.default_rng(42)
        probs = rng.dirichlet([3, 2, 2], size=100)
        labels = rng.integers(0, 3, size=100)
        cal = MulticlassCalibrator(method="isotonic")
        cal.fit(probs, labels)
        calibrated = cal.transform(probs)
        assert calibrated.shape == probs.shape

    def test_calibrated_probs_sum_to_one(self):
        """Calibrated probabilities should sum to 1 per row."""
        rng = np.random.default_rng(42)
        probs = rng.dirichlet([3, 2, 2], size=100)
        labels = rng.integers(0, 3, size=100)
        cal = MulticlassCalibrator(method="isotonic")
        cal.fit(probs, labels)
        calibrated = cal.transform(probs)
        row_sums = calibrated.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_calibrated_probs_in_range(self):
        """Calibrated probabilities should be in [0, 1]."""
        rng = np.random.default_rng(42)
        probs = rng.dirichlet([3, 2, 2], size=100)
        labels = rng.integers(0, 3, size=100)
        cal = MulticlassCalibrator(method="platt")
        cal.fit(probs, labels)
        calibrated = cal.transform(probs)
        assert (calibrated >= 0.0).all()
        assert (calibrated <= 1.0).all()

    def test_raises_if_not_fitted(self):
        """transform() before fit() should raise RuntimeError."""
        cal = MulticlassCalibrator()
        probs = np.array([[0.5, 0.3, 0.2]])
        with pytest.raises(RuntimeError, match="must be fitted"):
            cal.transform(probs)

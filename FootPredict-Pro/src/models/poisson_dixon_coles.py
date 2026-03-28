"""
FootPredict-Pro — Dixon-Coles adjusted bivariate Poisson model.

Implements the Dixon-Coles (1997) model for predicting football scorelines.
Key improvements over basic Poisson:
  1. Low-score correlation adjustment (ρ parameter) corrects the
     independence assumption for 0-0, 1-0, 0-1, 1-1 scorelines.
  2. Time-weighted estimation using exponential decay (xi parameter)
     so recent matches have more influence on parameter estimates.

References:
  Dixon, M.J. & Coles, S.G. (1997).
  "Modelling Association Football Scores and Inefficiencies in the
  Football Betting Market." Applied Statistics, 46(2), 265-280.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from loguru import logger


# ---------------------------------------------------------------------------
# Dixon-Coles correction function
# ---------------------------------------------------------------------------

def _tau(home_goals: int, away_goals: int, lam: float, mu: float, rho: float) -> float:
    """
    Dixon-Coles low-score correction term τ(x, y, λ, μ, ρ).

    Adjusts the joint probability for low-scoring outcomes (0-0, 1-0, 0-1, 1-1)
    to correct for the independence assumption violation.

    Args:
        home_goals: Number of home goals.
        away_goals: Number of away goals.
        lam: Expected home goals.
        mu: Expected away goals.
        rho: Correlation parameter (negative → positive correlation, typical -0.1).

    Returns:
        Correction multiplier.
    """
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam * mu * rho
    elif home_goals == 1 and away_goals == 0:
        return 1.0 + mu * rho
    elif home_goals == 0 and away_goals == 1:
        return 1.0 + lam * rho
    elif home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    else:
        return 1.0


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class DixonColesModel:
    """
    Dixon-Coles bivariate Poisson model for football score prediction.

    Estimates:
      - attack[team]: Attack strength parameter.
      - defense[team]: Defense strength parameter.
      - home_advantage: Additive home advantage.
      - rho: Low-score correlation correction.

    Usage:
        model = DixonColesModel(xi=0.0018, max_goals=10)
        model.fit(matches_df)
        score_probs = model.score_matrix("Manchester City", "Arsenal")
        outcome_probs = model.outcome_probabilities("Manchester City", "Arsenal")
    """

    def __init__(
        self,
        xi: float = 0.0018,
        max_goals: int = 10,
        min_matches: int = 5,
    ) -> None:
        """
        Args:
            xi: Time decay parameter per day (higher = faster decay).
                0.0018 ≈ half-life of 385 days (≈ 1 season).
            max_goals: Maximum goals per team in score matrix.
            min_matches: Minimum team matches to include in estimation.
        """
        self.xi = xi
        self.max_goals = max_goals
        self.min_matches = min_matches

        # Fitted parameters
        self.attack: Dict[str, float] = {}
        self.defense: Dict[str, float] = {}
        self.home_advantage: float = 0.0
        self.rho: float = 0.0
        self.teams: List[str] = []
        self._is_fitted: bool = False
        self._reference_date: Optional[pd.Timestamp] = None

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        """
        Fit the model on historical match data.

        Args:
            matches: DataFrame with columns: date, home_team, away_team,
                     home_goals, away_goals.

        Returns:
            self (fitted model).
        """
        matches = matches.copy().dropna(
            subset=["date", "home_team", "away_team", "home_goals", "away_goals"]
        )
        matches["date"] = pd.to_datetime(matches["date"])
        matches = matches.sort_values("date").reset_index(drop=True)

        # Filter to teams with enough matches
        team_counts = (
            pd.concat([
                matches["home_team"].rename("team"),
                matches["away_team"].rename("team"),
            ])
            .value_counts()
        )
        valid_teams = team_counts[team_counts >= self.min_matches].index.tolist()
        matches = matches[
            matches["home_team"].isin(valid_teams) &
            matches["away_team"].isin(valid_teams)
        ]

        if len(matches) < 20:
            logger.warning(
                f"Only {len(matches)} valid matches after filtering. "
                "Model quality will be poor."
            )

        self.teams = sorted(matches["home_team"].unique().tolist() +
                            matches["away_team"].unique().tolist())
        self.teams = sorted(set(self.teams))
        n_teams = len(self.teams)

        logger.info(
            f"Fitting Dixon-Coles on {len(matches)} matches, "
            f"{n_teams} teams..."
        )

        # Compute time weights
        self._reference_date = matches["date"].max()
        days_ago = (self._reference_date - matches["date"]).dt.days.values
        weights = np.exp(-self.xi * days_ago)

        # Build index maps
        team_idx = {t: i for i, t in enumerate(self.teams)}

        # Initial parameter values
        # [attack_0..n-1, defense_0..n-1, home_adv, rho]
        # Constraint: sum of attack params = 0 (fix one team as reference)
        n_params = 2 * n_teams + 2
        x0 = np.zeros(n_params)
        x0[n_teams + n_teams] = 0.3   # home advantage
        x0[n_params - 1] = -0.1       # rho (slight positive correlation)

        # Bounds: defense must be positive-ish, rho in [-0.99, 0]
        bounds = (
            [(None, None)] * n_teams +           # attack
            [(None, None)] * n_teams +           # defense
            [(0, None)] +                         # home advantage >= 0
            [(-0.99, 0.99)]                       # rho
        )

        def _neg_log_likelihood(params: np.ndarray) -> float:
            attack = params[:n_teams]
            defense = params[n_teams: 2 * n_teams]
            home_adv = params[2 * n_teams]
            rho = params[2 * n_teams + 1]

            ll = 0.0
            for i, row in matches.iterrows():
                hi = team_idx[row["home_team"]]
                ai = team_idx[row["away_team"]]
                hg = int(row["home_goals"])
                ag = int(row["away_goals"])
                w = weights[i]

                lam = np.exp(attack[hi] - defense[ai] + home_adv)
                mu = np.exp(attack[ai] - defense[hi])

                # Clamp to prevent exp overflow
                lam = np.clip(lam, 1e-6, 20.0)
                mu = np.clip(mu, 1e-6, 20.0)

                tau_val = _tau(hg, ag, lam, mu, rho)
                if tau_val <= 0:
                    return 1e12

                ll += w * (
                    np.log(tau_val)
                    + poisson.logpmf(hg, lam)
                    + poisson.logpmf(ag, mu)
                )

            return -ll

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                _neg_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-9},
            )

        if not result.success:
            logger.warning(f"Optimization did not fully converge: {result.message}")

        params = result.x
        for i, team in enumerate(self.teams):
            self.attack[team] = float(params[i])
            self.defense[team] = float(params[n_teams + i])
        self.home_advantage = float(params[2 * n_teams])
        self.rho = float(params[2 * n_teams + 1])

        logger.info(
            f"Fitted. Home advantage={self.home_advantage:.3f}, "
            f"rho={self.rho:.4f}"
        )
        self._is_fitted = True
        return self

    def _expected_goals(
        self,
        home_team: str,
        away_team: str,
    ) -> Tuple[float, float]:
        """
        Compute expected goals λ (home) and μ (away).

        Returns team averages for unknown teams.
        """
        avg_attack = np.mean(list(self.attack.values())) if self.attack else 0.0
        avg_defense = np.mean(list(self.defense.values())) if self.defense else 0.0

        h_att = self.attack.get(home_team, avg_attack)
        h_def = self.defense.get(home_team, avg_defense)
        a_att = self.attack.get(away_team, avg_attack)
        a_def = self.defense.get(away_team, avg_defense)

        lam = np.exp(h_att - a_def + self.home_advantage)
        mu = np.exp(a_att - h_def)
        return float(np.clip(lam, 0.1, 10.0)), float(np.clip(mu, 0.1, 10.0))

    def score_matrix(
        self,
        home_team: str,
        away_team: str,
    ) -> np.ndarray:
        """
        Compute the (max_goals+1) × (max_goals+1) score probability matrix.

        Element [i, j] = P(home goals = i, away goals = j).

        Args:
            home_team: Home team name.
            away_team: Away team name.

        Returns:
            2D numpy array of score probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        lam, mu = self._expected_goals(home_team, away_team)
        n = self.max_goals + 1
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                tau_val = _tau(i, j, lam, mu, self.rho)
                matrix[i, j] = (
                    tau_val
                    * poisson.pmf(i, lam)
                    * poisson.pmf(j, mu)
                )

        # Normalize (should sum very close to 1)
        matrix = matrix / matrix.sum()
        return matrix

    def outcome_probabilities(
        self,
        home_team: str,
        away_team: str,
    ) -> Tuple[float, float, float]:
        """
        Compute P(Home Win), P(Draw), P(Away Win).

        Args:
            home_team: Home team name.
            away_team: Away team name.

        Returns:
            Tuple (p_home, p_draw, p_away).
        """
        matrix = self.score_matrix(home_team, away_team)
        p_home = float(np.tril(matrix, k=-1).sum())
        p_draw = float(np.trace(matrix))
        p_away = float(np.triu(matrix, k=1).sum())
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    def most_likely_score(
        self,
        home_team: str,
        away_team: str,
        top_n: int = 5,
    ) -> List[Tuple[int, int, float]]:
        """
        Return the top N most likely exact scorelines.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            top_n: Number of scorelines to return.

        Returns:
            List of (home_goals, away_goals, probability) tuples.
        """
        matrix = self.score_matrix(home_team, away_team)
        n = self.max_goals + 1
        scores = []
        for i in range(n):
            for j in range(n):
                scores.append((i, j, float(matrix[i, j])))
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_n]

    def expected_goals_for(
        self,
        home_team: str,
        away_team: str,
    ) -> Tuple[float, float]:
        """
        Return expected goals for each team.

        Args:
            home_team: Home team name.
            away_team: Away team name.

        Returns:
            Tuple (home_xg, away_xg).
        """
        return self._expected_goals(home_team, away_team)

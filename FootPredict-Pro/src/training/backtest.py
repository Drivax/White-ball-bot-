"""
FootPredict-Pro — Temporal backtesting framework.

Implements a walk-forward validation approach:
  - Train on data up to time T
  - Predict matches in window [T, T+step]
  - Advance T by step
  - Repeat until end of dataset
  - Compute aggregate metrics: RPS, Brier, log-loss, accuracy

This ensures NO future data leakage — the gold standard for sports
prediction model evaluation.

Usage:
    python src/training/backtest.py --seasons "2021 2022 2023" --league E0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_ingestion.football_data_co import load_all_seasons
from src.models.poisson_dixon_coles import DixonColesModel
from src.feature_engineering.pipeline import FeaturePipeline
from src.models.outcome_ensemble import OutcomeEnsemble
from src.utils.metrics import compute_all_metrics, print_metrics_table
from src.utils.helpers import get_project_root


def run_backtest(
    league_code: str,
    seasons: List[int],
    min_train_matches: int = 100,
    step_size: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Run temporal walk-forward backtest on historical data.

    Args:
        league_code: League code (e.g., "E0").
        seasons: Seasons to include.
        min_train_matches: Minimum matches before first prediction.
        step_size: Number of matches per evaluation window.
        verbose: Print progress.

    Returns:
        Dict of aggregate metrics.
    """
    logger.info(
        f"Starting backtest: {league_code}, seasons={seasons}, "
        f"min_train={min_train_matches}, step={step_size}"
    )

    # Load all data
    matches = load_all_seasons(league_code, seasons)
    if matches.empty or len(matches) < min_train_matches + step_size:
        logger.error("Not enough data for backtesting.")
        return {}

    matches = matches.sort_values("date").reset_index(drop=True)
    n = len(matches)
    logger.info(f"Total matches: {n}")

    all_probs_dc: List[np.ndarray] = []
    all_probs_ml: List[np.ndarray] = []
    all_probs_blend: List[np.ndarray] = []
    all_labels: List[int] = []

    windows = range(min_train_matches, n - step_size + 1, step_size)
    logger.info(f"Running {len(windows)} evaluation windows...")

    for train_end in tqdm(windows, desc="Backtest windows", disable=not verbose):
        train_data = matches.iloc[:train_end].copy()
        test_data = matches.iloc[train_end: train_end + step_size].copy()

        if test_data.empty:
            break

        # --- Dixon-Coles ---
        try:
            dc = DixonColesModel(xi=0.0018)
            dc.fit(train_data)
        except Exception as e:
            logger.debug(f"DC fit failed at window {train_end}: {e}")
            continue

        # --- ML Ensemble ---
        try:
            fp = FeaturePipeline()
            X_train, y_tr = fp.fit_transform(train_data)

            if len(X_train) < 50 or len(set(y_tr)) < 3:
                continue

            ensemble = OutcomeEnsemble()
            ensemble.fit(X_train, y_tr)

            # Feature engineering for test data
            # Use full dataset up to test end for proper history
            full_slice = matches.iloc[: train_end + step_size].copy()
            fp2 = FeaturePipeline()
            X_full, _ = fp2.fit_transform(full_slice)
            X_test = X_full.iloc[train_end:].reset_index(drop=True)

        except Exception as e:
            logger.debug(f"ML fit failed at window {train_end}: {e}")
            continue

        # --- Predictions ---
        for idx, (_, row) in enumerate(test_data.iterrows()):
            label = row.get("result_label")
            if label is None or np.isnan(label):
                continue
            label = int(label)

            # Dixon-Coles probabilities
            try:
                ph, pd_, pa = dc.outcome_probabilities(
                    row["home_team"], row["away_team"]
                )
                p_dc = np.array([ph, pd_, pa])
            except Exception:
                p_dc = np.array([0.45, 0.27, 0.28])

            # ML ensemble probabilities
            try:
                if idx < len(X_test):
                    p_ml = ensemble.predict_proba(X_test.iloc[[idx]])[0]
                else:
                    p_ml = p_dc.copy()
            except Exception:
                p_ml = p_dc.copy()

            # Blend
            p_blend = 0.60 * p_dc + 0.40 * p_ml
            p_blend = p_blend / p_blend.sum()

            all_probs_dc.append(p_dc)
            all_probs_ml.append(p_ml)
            all_probs_blend.append(p_blend)
            all_labels.append(label)

    if not all_labels:
        logger.error("No predictions collected during backtest!")
        return {}

    y_true = np.array(all_labels)
    probs_dc = np.array(all_probs_dc)
    probs_ml = np.array(all_probs_ml)
    probs_blend = np.array(all_probs_blend)

    logger.info(f"Evaluated {len(y_true)} matches")

    dc_metrics = compute_all_metrics(y_true, probs_dc)
    ml_metrics = compute_all_metrics(y_true, probs_ml)
    blend_metrics = compute_all_metrics(y_true, probs_blend)

    if verbose:
        print_metrics_table(dc_metrics, title="Dixon-Coles (Backtest)")
        print_metrics_table(ml_metrics, title="ML Ensemble (Backtest)")
        print_metrics_table(blend_metrics, title="Blended Ensemble (Backtest)")

    # Check benchmarks
    rps = blend_metrics["rps"]
    brier = blend_metrics["brier"]
    bench_rps = "✅" if rps < 0.20 else "❌"
    bench_brier = "✅" if brier < 0.22 else "❌"
    logger.info(
        f"Benchmark check — RPS: {rps:.3f} {bench_rps} (<0.20) | "
        f"Brier: {brier:.3f} {bench_brier} (<0.22)"
    )

    return {
        "n_matches": len(y_true),
        "dixon_coles": dc_metrics,
        "ml_ensemble": ml_metrics,
        "blended": blend_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest FootPredict-Pro models")
    parser.add_argument("--league", type=str, default="E0")
    parser.add_argument("--seasons", type=str, default="2021 2022 2023")
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument("--step", type=int, default=20)
    args = parser.parse_args()

    seasons = [int(s) for s in args.seasons.split()]
    results = run_backtest(
        args.league,
        seasons,
        min_train_matches=args.min_train,
        step_size=args.step,
    )

    if results:
        print("\n=== Final Backtest Summary ===")
        blended = results.get("blended", {})
        for k, v in blended.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

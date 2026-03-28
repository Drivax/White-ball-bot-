"""
FootPredict-Pro — Main training pipeline.

Orchestrates the full training workflow:
  1. Load match data (from CSVs or API-Football cache)
  2. Build feature matrix via FeaturePipeline
  3. Train DixonColesModel on raw match data
  4. Train OutcomeEnsemble on feature matrix
  5. Save all models with versioning
  6. Report evaluation metrics

Usage:
    python src/training/train.py --league E0 --seasons "2021 2022 2023"
    python src/training/train.py --all-leagues
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_ingestion.football_data_co import load_all_seasons
from src.feature_engineering.pipeline import FeaturePipeline

from src.models.outcome_ensemble import OutcomeEnsemble
from src.models.ensemble import MasterEnsemble
from src.utils.helpers import ensure_dir, get_project_root
from src.utils.metrics import compute_all_metrics, print_metrics_table


def train(
    league_code: str,
    seasons: List[int],
    output_dir: Optional[str] = None,
    test_size: float = 0.15,
    verbose: bool = True,
) -> MasterEnsemble:
    """
    Full training pipeline for a given league.

    Args:
        league_code: football-data.co.uk league code (e.g., "E0").
        seasons: List of season start years (e.g., [2021, 2022, 2023]).
        output_dir: Directory to save trained models.
        test_size: Fraction of data to hold out for evaluation.
        verbose: Print detailed training output.

    Returns:
        Trained MasterEnsemble.
    """
    root = get_project_root()
    if output_dir is None:
        output_dir = str(root / "models")

    logger.info(
        f"Starting training: league={league_code}, seasons={seasons}"
    )

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading match data...")
    matches = load_all_seasons(league_code, seasons)

    if matches.empty or len(matches) < 50:
        logger.error(
            f"Insufficient data: {len(matches)} matches. "
            "Need at least 50. Check data source."
        )
        sys.exit(1)

    logger.info(f"Loaded {len(matches)} matches")

    # ------------------------------------------------------------------
    # 2. Temporal train/test split (no shuffling — respect time order)
    # ------------------------------------------------------------------
    matches = matches.sort_values("date").reset_index(drop=True)
    n_test = max(30, int(len(matches) * test_size))
    train_matches = matches.iloc[:-n_test].copy()
    test_matches = matches.iloc[-n_test:].copy()

    logger.info(
        f"Train: {len(train_matches)} | Test (hold-out): {len(test_matches)}"
    )

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    logger.info("Building feature matrix...")
    pipeline = FeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_matches)

    # Build test features using history from training matches
    # (we rebuild using the full pipeline on combined, then slice)
    all_matches = pd.concat([train_matches, test_matches], ignore_index=True)
    X_all, y_all = FeaturePipeline().fit_transform(all_matches)
    X_test = X_all.iloc[-n_test:].reset_index(drop=True)
    y_test = y_all[-n_test:]

    logger.info(
        f"Feature matrix: train {X_train.shape}, test {X_test.shape}"
    )

    # ------------------------------------------------------------------
    # 4. Train Dixon-Coles model
    # ------------------------------------------------------------------
    logger.info("Training Dixon-Coles Poisson model...")
    from src.models.poisson_dixon_coles import DixonColesModel
    dc_model = DixonColesModel(xi=0.0018, max_goals=10)
    dc_model.fit(train_matches)

    # Evaluate Dixon-Coles on test set
    dc_probs = []
    for _, row in test_matches.iterrows():
        ph, pd_, pa = dc_model.outcome_probabilities(
            row["home_team"], row["away_team"]
        )
        dc_probs.append([ph, pd_, pa])
    dc_probs_arr = np.array(dc_probs)

    dc_metrics = compute_all_metrics(y_test, dc_probs_arr)
    if verbose:
        print_metrics_table(dc_metrics, title="Dixon-Coles Metrics (Test Set)")

    # ------------------------------------------------------------------
    # 5. Train outcome ensemble
    # ------------------------------------------------------------------
    logger.info("Training outcome ensemble (XGB + LGB + CB + LR)...")

    # Small validation set for early stopping
    val_size = max(20, int(len(X_train) * 0.1))
    X_tr = X_train.iloc[:-val_size]
    y_tr = y_train[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_val = y_train[-val_size:]

    ensemble = OutcomeEnsemble()
    ensemble.fit(
        X_tr, y_tr,
        eval_set=(X_val.values, y_val),
    )

    # Evaluate ensemble on test set
    ml_probs = ensemble.predict_proba(X_test)
    ml_metrics = compute_all_metrics(y_test, ml_probs)
    if verbose:
        print_metrics_table(ml_metrics, title="ML Ensemble Metrics (Test Set)")

    # ------------------------------------------------------------------
    # 6. Blended evaluation
    # ------------------------------------------------------------------
    blended = 0.60 * dc_probs_arr + 0.40 * ml_probs
    blended = blended / blended.sum(axis=1, keepdims=True)
    blended_metrics = compute_all_metrics(y_test, blended)
    if verbose:
        print_metrics_table(blended_metrics, title="Blended Ensemble Metrics (Test Set)")

    # ------------------------------------------------------------------
    # 7. Save models
    # ------------------------------------------------------------------
    master = MasterEnsemble(
        outcome_model=ensemble,
        dixon_coles_model=dc_model,
        player_model=None,
    )

    out = ensure_dir(output_dir)
    master.save(out)
    logger.info(f"All models saved to {out}")

    # Save feature pipeline
    import joblib
    joblib.dump(pipeline, out / "feature_pipeline.joblib")
    logger.info("Feature pipeline saved.")

    return master


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train FootPredict-Pro models"
    )
    parser.add_argument(
        "--league", type=str, default="E0",
        help="League code (e.g. E0, SP1, D1)"
    )
    parser.add_argument(
        "--seasons", type=str, default="2021 2022 2023",
        help="Space-separated season start years"
    )
    parser.add_argument(
        "--all-leagues", action="store_true",
        help="Train on all leagues in config"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for models"
    )
    args = parser.parse_args()

    if args.all_leagues:
        try:
            from src.utils.config_loader import load_config
            cfg = load_config()
            leagues = [(lg.football_data_code or "E0") for lg in cfg.leagues]
        except Exception:
            leagues = ["E0", "SP1", "D1", "I1", "F1"]

        seasons = [2021, 2022, 2023]
        for league in leagues:
            if league:
                try:
                    train(league, seasons, args.output)
                except Exception as e:
                    logger.warning(f"Failed for {league}: {e}")
    else:
        seasons = [int(s) for s in args.seasons.split()]
        train(args.league, seasons, args.output)


if __name__ == "__main__":
    main()

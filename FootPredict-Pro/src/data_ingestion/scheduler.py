"""
FootPredict-Pro — Automated data refresh scheduler.

Schedules periodic data fetches to keep the prediction system up-to-date
with the latest match results, lineups, and player statistics.

Features:
  - Daily fixture result updates
  - Pre-match lineup fetch (runs closer to kickoff)
  - Weekly model retraining trigger
  - Health monitoring with alerts

Usage:
    python src/data_ingestion/scheduler.py
    python src/data_ingestion/scheduler.py --once  # Run immediately and exit
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def update_results(league_ids: Optional[List[int]] = None) -> None:
    """
    Fetch latest finished match results from API-Football.

    Args:
        league_ids: League IDs to update. Defaults to all configured leagues.
    """
    try:
        from src.utils.config_loader import load_config
        cfg = load_config()

        if not cfg.api.api_football.key or cfg.api.api_football.key == "YOUR_API_FOOTBALL_KEY":
            logger.warning("API key not configured. Skipping API fetch.")
            return

        from src.data_ingestion.api_football import APIFootballClient, fixtures_to_dataframe
        from src.utils.helpers import ensure_dir, get_project_root

        client = APIFootballClient(api_key=cfg.api.api_football.key)
        root = get_project_root()
        out_dir = ensure_dir(root / "data" / "raw")

        target_leagues = league_ids or [lg.id for lg in cfg.leagues]
        current_season = datetime.now().year

        for league_id in target_leagues:
            try:
                fixtures = client.get_fixtures(league_id, current_season)
                df = fixtures_to_dataframe(fixtures)
                if not df.empty:
                    out_path = out_dir / f"fixtures_league{league_id}_latest.csv"
                    df.to_csv(out_path, index=False)
                    logger.info(
                        f"Updated {len(df)} fixtures for league {league_id}"
                    )
            except Exception as e:
                logger.error(f"Failed to update league {league_id}: {e}")

    except Exception as e:
        logger.error(f"update_results failed: {e}")


def retrain_models() -> None:
    """Trigger model retraining pipeline."""
    try:
        logger.info("Triggering weekly model retraining...")
        from src.training.train import train
        train(league_code="E0", seasons=[2022, 2023, 2024])
        logger.info("Retraining complete.")
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")


def run_scheduler(league_ids: Optional[List[int]] = None) -> None:
    """
    Run the continuous data refresh scheduler.

    Schedule:
        - Every 6 hours: Update match results
        - Every Sunday 03:00: Retrain models

    Args:
        league_ids: League IDs to schedule updates for.
    """
    try:
        import schedule
    except ImportError:
        logger.error("schedule package required. Run: pip install schedule")
        sys.exit(1)

    logger.info("Starting FootPredict-Pro data scheduler...")

    # Schedule tasks
    schedule.every(6).hours.do(update_results, league_ids=league_ids)
    schedule.every().sunday.at("03:00").do(retrain_models)

    # Run once immediately on startup
    update_results(league_ids)

    logger.info("Scheduler running. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


def main() -> None:
    parser = argparse.ArgumentParser(description="FootPredict-Pro data scheduler")
    parser.add_argument(
        "--once", action="store_true",
        help="Run data update once and exit (no loop)"
    )
    parser.add_argument(
        "--leagues", type=str, default=None,
        help="Comma-separated league IDs (e.g. '39,140,78')"
    )
    args = parser.parse_args()

    league_ids = (
        [int(x) for x in args.leagues.split(",")]
        if args.leagues
        else None
    )

    if args.once:
        logger.info("Running single data update...")
        update_results(league_ids)
        logger.info("Done.")
    else:
        run_scheduler(league_ids)


if __name__ == "__main__":
    main()

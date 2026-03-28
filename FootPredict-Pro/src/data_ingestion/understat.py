"""
FootPredict-Pro — Understat xG data scraper.

Understat (understat.com) provides shot-level xG data for major European
leagues. This module scrapes match-level and player-level xG statistics
to supplement the basic football-data.co.uk match results.

Note: Web scraping may be fragile if Understat changes its structure.
Consider their API if one becomes available.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger

from src.utils.helpers import ensure_dir, get_project_root


UNDERSTAT_LEAGUES = {
    "EPL": "EPL",
    "La_liga": "La_liga",
    "Bundesliga": "Bundesliga",
    "Serie_A": "Serie_A",
    "Ligue_1": "Ligue_1",
    "RFPL": "RFPL",  # Russian Premier League
}


class UnderstatScraper:
    """
    Scrapes xG data from Understat.com.

    Provides match-level and player-level xG statistics that are not
    available in the football-data.co.uk CSV fallback.
    """

    BASE_URL = "https://understat.com"

    def __init__(self, delay: float = 1.5) -> None:
        """
        Args:
            delay: Seconds to wait between requests (be polite!).
        """
        self.delay = delay
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (compatible; FootPredict-Pro/1.0; "
                "+https://github.com/footpredict-pro)"
            )
        })

    def get_league_matches(
        self,
        league: str,
        season: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all match xG data for a league-season from Understat.

        Args:
            league: Understat league name (e.g., "EPL", "La_liga").
            season: Season start year (e.g., 2023 for 2023-24).

        Returns:
            List of match dicts with xG statistics.
        """
        if league not in UNDERSTAT_LEAGUES:
            logger.warning(
                f"League '{league}' not in known Understat leagues. "
                f"Known: {list(UNDERSTAT_LEAGUES.keys())}"
            )

        url = f"{self.BASE_URL}/league/{league}/{season}"
        logger.info(f"Scraping Understat: {url}")

        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return []

        # Understat embeds JSON data in JavaScript vars in the HTML
        matches_data = self._extract_json_var(resp.text, "datesData")
        if not matches_data:
            logger.warning(f"No match data found for {league}/{season}")
            return []

        time.sleep(self.delay)
        return self._parse_matches(matches_data)

    def _extract_json_var(self, html: str, var_name: str) -> Optional[Any]:
        """
        Extract a JSON variable from Understat's embedded JavaScript.

        Understat stores data as:
            var datesData = JSON.parse('...');

        Args:
            html: Page HTML content.
            var_name: JavaScript variable name to extract.

        Returns:
            Parsed Python object or None if not found.
        """
        pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)
        if not match:
            return None

        try:
            # Understat double-escapes JSON, need to unescape
            raw = match.group(1)
            raw = raw.encode().decode("unicode_escape")
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to parse {var_name}: {e}")
            return None

    def _parse_matches(self, data: List[dict]) -> List[Dict[str, Any]]:
        """
        Parse raw Understat match data into standardized format.

        Args:
            data: Raw match list from Understat JSON.

        Returns:
            List of standardized match dicts.
        """
        matches = []
        for item in data:
            try:
                match = {
                    "fixture_id": item.get("id"),
                    "date": item.get("datetime"),
                    "home_team": item.get("h", {}).get("title"),
                    "away_team": item.get("a", {}).get("title"),
                    "home_goals": int(item.get("goals", {}).get("h", 0)),
                    "away_goals": int(item.get("goals", {}).get("a", 0)),
                    "home_xg": float(item.get("xG", {}).get("h", 0)),
                    "away_xg": float(item.get("xG", {}).get("a", 0)),
                    "is_result": item.get("isResult", False),
                    "forecast_home": float(item.get("forecast", {}).get("w", 0)),
                    "forecast_draw": float(item.get("forecast", {}).get("d", 0)),
                    "forecast_away": float(item.get("forecast", {}).get("l", 0)),
                }
                if match["home_team"] and match["away_team"]:
                    matches.append(match)
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Skipping malformed match: {e}")

        logger.info(f"Parsed {len(matches)} matches from Understat")
        return matches


def merge_xg_data(
    matches_df: "pd.DataFrame",
    understat_matches: List[Dict[str, Any]],
) -> "pd.DataFrame":
    """
    Merge Understat xG data into an existing match DataFrame.

    Matches on home_team + away_team + approximate date.

    Args:
        matches_df: Existing match DataFrame (from football-data.co.uk).
        understat_matches: xG data from Understat.

    Returns:
        matches_df with home_xg and away_xg columns added.
    """
    import pandas as pd

    if not understat_matches:
        logger.warning("No Understat data to merge.")
        return matches_df

    xg_df = pd.DataFrame(understat_matches)
    xg_df["date"] = pd.to_datetime(xg_df["date"], errors="coerce").dt.date

    matches_df = matches_df.copy()
    matches_df["_date_key"] = pd.to_datetime(matches_df["date"]).dt.date

    # Simple merge on team names + date
    xg_df["_merge_key"] = (
        xg_df["home_team"].str.lower().str.strip() + "_" +
        xg_df["away_team"].str.lower().str.strip() + "_" +
        xg_df["date"].astype(str)
    )
    matches_df["_merge_key"] = (
        matches_df["home_team"].str.lower().str.strip() + "_" +
        matches_df["away_team"].str.lower().str.strip() + "_" +
        matches_df["_date_key"].astype(str)
    )

    xg_lookup = xg_df.set_index("_merge_key")[["home_xg", "away_xg"]].to_dict(orient="index")

    def _get_xg(row, col):
        return xg_lookup.get(row["_merge_key"], {}).get(col, None)

    matches_df["home_xg"] = matches_df.apply(lambda r: _get_xg(r, "home_xg"), axis=1)
    matches_df["away_xg"] = matches_df.apply(lambda r: _get_xg(r, "away_xg"), axis=1)

    n_merged = matches_df["home_xg"].notna().sum()
    logger.info(f"Merged xG data for {n_merged}/{len(matches_df)} matches")

    matches_df = matches_df.drop(columns=["_date_key", "_merge_key"])
    return matches_df


def main() -> None:
    """CLI: scrape Understat xG data."""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Understat xG data")
    parser.add_argument("--league", type=str, default="EPL")
    parser.add_argument("--seasons", type=str, default="2022 2023")
    args = parser.parse_args()

    seasons = [int(s) for s in args.seasons.split()]
    scraper = UnderstatScraper()

    root = get_project_root()
    out_dir = ensure_dir(root / "data" / "raw" / "understat")

    for season in seasons:
        matches = scraper.get_league_matches(args.league, season)
        if matches:
            import pandas as pd
            df = pd.DataFrame(matches)
            out_path = out_dir / f"{args.league}_{season}_xg.csv"
            df.to_csv(out_path, index=False)
            logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

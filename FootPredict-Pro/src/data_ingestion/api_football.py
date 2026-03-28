"""
FootPredict-Pro — Data ingestion from API-Football (api-football.com).

Fetches:
  - Historical match results with basic stats
  - Fixture lineups (starting XIs + substitutes)
  - Player statistics (goals, assists, shots, xG when available)
  - Injury/suspension reports

Requires free API key from https://www.api-football.com
Free tier: 100 requests/day (sufficient for daily updates).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from src.utils.helpers import ensure_dir, get_project_root


class APIFootballClient:
    """
    Client for the API-Football v3 REST API.

    Handles authentication, rate limiting, retries, and response caching.
    """

    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(
        self,
        api_key: str,
        cache_dir: Optional[str] = None,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: int = 5,
    ) -> None:
        """
        Args:
            api_key: API-Football API key.
            cache_dir: Directory to cache JSON responses.
            timeout: Request timeout in seconds.
            retry_attempts: Number of retry attempts on failure.
            retry_delay: Seconds between retries.
        """
        self.api_key = api_key
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._session = requests.Session()
        self._session.headers.update({
            "x-apisports-key": api_key,
            "Accept": "application/json",
        })

        root = get_project_root()
        self.cache_dir = Path(cache_dir) if cache_dir else root / "data" / "raw" / "api_football"
        ensure_dir(self.cache_dir)

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a GET request with retries and optional caching.

        Args:
            endpoint: API endpoint path (e.g., "/fixtures").
            params: Query parameters.

        Returns:
            Parsed JSON response as dict.

        Raises:
            requests.HTTPError: On non-200 response after retries.
        """
        # Build cache key from endpoint + sorted params
        cache_key = endpoint.strip("/").replace("/", "_")
        param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
        cache_file = self.cache_dir / f"{cache_key}_{param_str}.json"

        if cache_file.exists():
            logger.debug(f"Cache hit: {cache_file.name}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        url = f"{self.BASE_URL}{endpoint}"
        last_error: Optional[Exception] = None

        for attempt in range(self.retry_attempts):
            try:
                logger.debug(f"GET {url} params={params} (attempt {attempt+1})")
                resp = self._session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

                # API-Football wraps errors in response body
                if data.get("errors"):
                    logger.warning(f"API errors: {data['errors']}")

                # Cache the successful response
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)

                return data

            except requests.RequestException as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt+1}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"All {self.retry_attempts} attempts failed for {url}. "
            f"Last error: {last_error}"
        )

    def get_fixtures(
        self,
        league_id: int,
        season: int,
        status: str = "FT",
    ) -> List[Dict[str, Any]]:
        """
        Fetch all finished fixtures for a league/season.

        Args:
            league_id: API-Football league ID (e.g., 39 = Premier League).
            season: Season year (e.g., 2023 for 2023/24).
            status: Fixture status filter ("FT" = full time).

        Returns:
            List of fixture dictionaries.
        """
        data = self._get(
            "/fixtures",
            {"league": league_id, "season": season, "status": status},
        )
        fixtures = data.get("response", [])
        logger.info(
            f"Fetched {len(fixtures)} fixtures for league={league_id} season={season}"
        )
        return fixtures

    def get_fixture_lineups(self, fixture_id: int) -> List[Dict[str, Any]]:
        """
        Fetch starting lineups for a fixture.

        Args:
            fixture_id: The fixture ID from get_fixtures().

        Returns:
            List of team lineup dicts (home + away).
        """
        data = self._get("/fixtures/lineups", {"fixture": fixture_id})
        return data.get("response", [])

    def get_fixture_statistics(self, fixture_id: int) -> List[Dict[str, Any]]:
        """
        Fetch per-team match statistics for a fixture.

        Args:
            fixture_id: The fixture ID.

        Returns:
            List of team statistics dicts.
        """
        data = self._get("/fixtures/statistics", {"fixture": fixture_id})
        return data.get("response", [])

    def get_fixture_player_stats(self, fixture_id: int) -> List[Dict[str, Any]]:
        """
        Fetch per-player statistics for a fixture.

        Args:
            fixture_id: The fixture ID.

        Returns:
            List of team player statistics dicts.
        """
        data = self._get("/fixtures/players", {"fixture": fixture_id})
        return data.get("response", [])

    def get_team_statistics(
        self,
        team_id: int,
        league_id: int,
        season: int,
    ) -> Dict[str, Any]:
        """
        Fetch aggregate team statistics for a season.

        Args:
            team_id: Team ID.
            league_id: League ID.
            season: Season year.

        Returns:
            Team statistics dict.
        """
        data = self._get(
            "/teams/statistics",
            {"team": team_id, "league": league_id, "season": season},
        )
        return data.get("response", {})

    def get_injuries(
        self,
        league_id: int,
        season: int,
        fixture_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch injury reports.

        Args:
            league_id: League ID.
            season: Season year.
            fixture_id: Optional specific fixture ID.

        Returns:
            List of injury records.
        """
        params: Dict[str, Any] = {"league": league_id, "season": season}
        if fixture_id:
            params["fixture"] = fixture_id
        data = self._get("/injuries", params)
        return data.get("response", [])


# ---------------------------------------------------------------------------
# Data transformation helpers
# ---------------------------------------------------------------------------

def fixtures_to_dataframe(fixtures: List[Dict[str, Any]]) -> "pd.DataFrame":
    """
    Convert raw API-Football fixture list to a clean pandas DataFrame.

    Args:
        fixtures: List of fixture dicts from get_fixtures().

    Returns:
        DataFrame with standardized columns.
    """
    import pandas as pd

    rows = []
    for fix in fixtures:
        try:
            row = {
                "fixture_id": fix["fixture"]["id"],
                "date": fix["fixture"]["date"],
                "league_id": fix["league"]["id"],
                "league_name": fix["league"]["name"],
                "season": fix["league"]["season"],
                "home_team_id": fix["teams"]["home"]["id"],
                "home_team": fix["teams"]["home"]["name"],
                "away_team_id": fix["teams"]["away"]["id"],
                "away_team": fix["teams"]["away"]["name"],
                "home_goals": fix["goals"]["home"],
                "away_goals": fix["goals"]["away"],
                "home_ht": fix["score"]["halftime"]["home"],
                "away_ht": fix["score"]["halftime"]["away"],
                "status": fix["fixture"]["status"]["short"],
            }
            rows.append(row)
        except (KeyError, TypeError) as e:
            logger.warning(f"Skipping fixture due to missing data: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["result"] = df.apply(_compute_result, axis=1)
    return df


def _compute_result(row: "pd.Series") -> Optional[str]:
    """Compute H/D/A result from goals."""
    h, a = row.get("home_goals"), row.get("away_goals")
    if h is None or a is None:
        return None
    if h > a:
        return "H"
    elif h == a:
        return "D"
    else:
        return "A"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI: fetch fixtures for a given league and season."""
    import argparse
    import sys

    from loguru import logger

    parser = argparse.ArgumentParser(
        description="Fetch API-Football data for FootPredict-Pro"
    )
    parser.add_argument("--league", type=int, required=True, help="League ID (e.g. 39)")
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g. 2023)")
    parser.add_argument("--api-key", type=str, default=None, help="Override API key")
    args = parser.parse_args()

    # Load config for API key
    try:
        from src.utils.config_loader import load_config
        cfg = load_config()
        api_key = args.api_key or cfg.api.api_football.key
    except Exception:
        api_key = args.api_key or ""

    if not api_key or api_key == "YOUR_API_FOOTBALL_KEY":
        logger.error(
            "API key not set. Configure api.api_football.key in config.yaml "
            "or pass --api-key."
        )
        sys.exit(1)

    client = APIFootballClient(api_key=api_key)
    fixtures = client.get_fixtures(league_id=args.league, season=args.season)
    df = fixtures_to_dataframe(fixtures)

    root = get_project_root()
    out_dir = ensure_dir(root / "data" / "raw")
    out_path = out_dir / f"fixtures_league{args.league}_season{args.season}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} fixtures to {out_path}")


if __name__ == "__main__":
    main()

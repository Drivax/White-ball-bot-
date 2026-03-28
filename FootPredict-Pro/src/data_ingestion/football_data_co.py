"""
FootPredict-Pro — football-data.co.uk CSV data ingestion.

Downloads historical match result CSVs from football-data.co.uk —
a free, no-registration fallback data source covering major European
leagues from the 1990s to the current season.

CSV columns include: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR,
B365H, B365D, B365A (Bet365 odds), plus many more bookmaker odds.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

from src.utils.helpers import ensure_dir, get_project_root


# ---------------------------------------------------------------------------
# League codes mapping (football-data.co.uk format)
# ---------------------------------------------------------------------------

LEAGUE_CODES: Dict[str, str] = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "E2": "England League 1",
    "E3": "England League 2",
    "EC": "England Conference",
    "SC0": "Scotland Premiership",
    "D1": "Germany Bundesliga 1",
    "D2": "Germany Bundesliga 2",
    "I1": "Italy Serie A",
    "I2": "Italy Serie B",
    "SP1": "Spain La Liga",
    "SP2": "Spain Segunda Division",
    "F1": "France Ligue 1",
    "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie",
    "B1": "Belgium First Division A",
    "P1": "Portugal Primeira Liga",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
}

# Season codes used in URLs (e.g. "2324" for 2023-24)
SEASON_SUFFIXES = {
    2018: "1819",
    2019: "1920",
    2020: "2021",
    2021: "2122",
    2022: "2223",
    2023: "2324",
    2024: "2425",
}

BASE_URL = "https://www.football-data.co.uk/mmz4281"


def _build_url(league_code: str, season: int) -> str:
    """Build the CSV download URL for a given league and season."""
    suffix = SEASON_SUFFIXES.get(season)
    if suffix is None:
        raise ValueError(
            f"Season {season} not in known suffixes. "
            f"Available: {list(SEASON_SUFFIXES.keys())}"
        )
    return f"{BASE_URL}/{suffix}/{league_code}.csv"


def fetch_csv(
    league_code: str,
    season: int,
    save_raw: bool = True,
) -> pd.DataFrame:
    """
    Download and parse a football-data.co.uk CSV file.

    Args:
        league_code: League code (e.g., "E0" for Premier League).
        season: Season start year (e.g., 2023 for 2023-24).
        save_raw: If True, save raw CSV to data/raw/.

    Returns:
        Cleaned DataFrame with standardized columns.

    Raises:
        requests.HTTPError: If download fails.
        ValueError: If league_code or season is not recognized.
    """
    if league_code not in LEAGUE_CODES:
        logger.warning(
            f"League code '{league_code}' not in known list. "
            f"Attempting anyway. Known: {list(LEAGUE_CODES.keys())}"
        )

    url = _build_url(league_code, season)
    logger.info(f"Fetching: {url}")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        content = resp.content
    except requests.HTTPError as e:
        logger.error(f"Failed to download {url}: {e}")
        raise

    # Save raw CSV
    if save_raw:
        root = get_project_root()
        out_dir = ensure_dir(root / "data" / "raw" / "football_data_co")
        out_path = out_dir / f"{league_code}_{season}.csv"
        with open(out_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved raw CSV: {out_path}")

    # Parse and standardize
    df = _parse_csv(content)
    df["league_code"] = league_code
    df["season"] = season
    return df


def _parse_csv(content: bytes) -> pd.DataFrame:
    """
    Parse football-data.co.uk raw CSV bytes into standardized DataFrame.

    Standardizes column names and adds computed columns.

    Args:
        content: Raw CSV bytes.

    Returns:
        Standardized DataFrame.
    """
    # Try different encodings
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not decode CSV with any known encoding.")

    # Drop completely empty rows (common at end of football-data CSVs)
    df = df.dropna(how="all")

    # Rename columns to standard names
    rename_map = {
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "result",
        "HTHG": "home_ht",
        "HTAG": "away_ht",
        "HTR": "ht_result",
        "Referee": "referee",
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_shots_on_target",
        "AST": "away_shots_on_target",
        "HC": "home_corners",
        "AC": "away_corners",
        "HF": "home_fouls",
        "AF": "away_fouls",
        "HY": "home_yellows",
        "AY": "away_yellows",
        "HR": "home_reds",
        "AR": "away_reds",
        # Bet365 odds
        "B365H": "odds_home",
        "B365D": "odds_draw",
        "B365A": "odds_away",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"])

    # Ensure numeric goal columns
    for col in ["home_goals", "away_goals"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows with valid results
    if "result" in df.columns:
        df = df[df["result"].isin(["H", "D", "A"])]
    elif "home_goals" in df.columns and "away_goals" in df.columns:
        df = df.dropna(subset=["home_goals", "away_goals"])
        df["result"] = df.apply(
            lambda r: "H" if r["home_goals"] > r["away_goals"]
            else ("D" if r["home_goals"] == r["away_goals"] else "A"),
            axis=1,
        )

    # Encode numeric result label (0=H, 1=D, 2=A)
    result_map = {"H": 0, "D": 1, "A": 2}
    df["result_label"] = df["result"].map(result_map)

    logger.info(f"Parsed {len(df)} matches from CSV")
    return df


def load_all_seasons(
    league_code: str,
    seasons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Download and concatenate multiple seasons for a league.

    Args:
        league_code: League code (e.g., "E0").
        seasons: List of season start years. Defaults to [2021, 2022, 2023].

    Returns:
        Combined DataFrame sorted by date.
    """
    if seasons is None:
        seasons = [2021, 2022, 2023]

    dfs = []
    for season in seasons:
        try:
            df = fetch_csv(league_code, season)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch {league_code} season {season}: {e}")

    if not dfs:
        logger.error("No data fetched for any season!")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    logger.info(
        f"Loaded {len(combined)} total matches for {league_code} "
        f"seasons {seasons}"
    )
    return combined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI: download CSVs from football-data.co.uk."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download football-data.co.uk CSV data"
    )
    parser.add_argument(
        "--league", type=str, default="E0",
        help=f"League code. Options: {', '.join(LEAGUE_CODES.keys())}"
    )
    parser.add_argument(
        "--seasons", type=str, default="2021 2022 2023",
        help="Space-separated season start years (e.g. '2021 2022 2023')"
    )
    args = parser.parse_args()

    season_list = [int(s) for s in args.seasons.split()]
    df = load_all_seasons(args.league, season_list)

    root = get_project_root()
    out_dir = ensure_dir(root / "data" / "processed")
    out_path = out_dir / f"{args.league}_combined.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved combined dataset: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()

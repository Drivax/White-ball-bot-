"""
FootPredict-Pro — Live upcoming-fixtures scraper.

Fetches real upcoming football fixtures from multiple free, no-key-required
public sources so the bot never has to rely on stale hardcoded match lists.

Priority order (each source tried in sequence; first success wins):
  1. SofaScore unofficial public API  (no key, comprehensive)
  2. BBC Sport fixtures HTML scrape    (no key, major leagues)
  3. TheSportsDB free API              (key "3", limited but reliable)

Returns a list of fixture dicts in the same format used by STATIC_FIXTURES in
predict_upcoming.py::

    {
        "date": "2026-04-07",          # ISO-8601 date string
        "competition": "Premier League",
        "home": "Arsenal",
        "away": "Chelsea",
        "neutral": False,
    }

Only fixtures from the following tracked competitions are returned so the
downstream prediction model (which has team-strength priors only for major
clubs) produces meaningful output:

    Premier League, La Liga, Bundesliga, Serie A, Ligue 1,
    UEFA Champions League, UEFA Europa League,
    FIFA World Cup Qualifiers (UEFA / CONMEBOL / CONCACAF / AFC / CAF),
    UEFA Nations League, Copa America, Euro Qualifiers.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


# ---------------------------------------------------------------------------
# Tracked competition keywords
# ---------------------------------------------------------------------------

#: Lower-case substrings.  A fixture whose competition name contains ANY of
#: these will be included.  Broaden or narrow as needed.
TRACKED_COMPS: tuple[str, ...] = (
    "premier league",
    "la liga",
    "primera division",
    "bundesliga",
    "serie a",
    "ligue 1",
    "champions league",
    "europa league",
    "world cup",
    "wc qualifier",
    "wc 2026",
    "nations league",
    "copa america",
    "euro 2028",
    "euro qualifier",
    "euro 2024",     # keep for backward compatibility with older source data
    "conmebol",
    "concacaf",
    "qualif",        # catches "qualification" in any language
)


def _is_tracked(competition_name: str) -> bool:
    """Return True if the competition should be included."""
    n = competition_name.lower()
    return any(kw in n for kw in TRACKED_COMPS)


# ---------------------------------------------------------------------------
# Source 1: SofaScore unofficial public API
# ---------------------------------------------------------------------------
# Docs/community: https://github.com/nickel-org/sofascore-api (unofficial)
# Endpoint: GET https://api.sofascore.com/api/v1/sport/football/scheduled-events/{date}
# Returns all scheduled football events for a single day – no API key needed.
# Rate-limit: ~60 req/min from a single IP.

_SOFASCORE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
}


def _fetch_sofascore_day(day: str, timeout: int = 15) -> List[Dict[str, Any]]:
    """
    Fetch scheduled football fixtures for *day* (ISO-8601) from SofaScore.

    Args:
        day: ISO-8601 date string, e.g. ``"2026-04-07"``.
        timeout: Request timeout in seconds.

    Returns:
        List of fixture dicts (may be empty).
    """
    url = f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{day}"
    resp = requests.get(url, headers=_SOFASCORE_HEADERS, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    fixtures: List[Dict[str, Any]] = []
    for event in data.get("events", []):
        try:
            comp = (
                event.get("tournament", {}).get("name", "")
                or event.get("tournament", {}).get("uniqueTournament", {}).get("name", "")
            )
            home = event["homeTeam"]["name"]
            away = event["awayTeam"]["name"]
            status = event.get("status", {}).get("type", "")
            # Skip events that have already finished or are in progress
            if status in ("finished", "inprogress", "postponed", "canceled"):
                continue
            if not _is_tracked(comp):
                continue
            fixtures.append({
                "date": day,
                "competition": comp,
                "home": home,
                "away": away,
                "neutral": False,
            })
        except (KeyError, TypeError):
            continue

    logger.debug(f"SofaScore [{day}]: {len(fixtures)} tracked fixtures")
    return fixtures


def fetch_sofascore(
    date_from: str,
    date_to: str,
    timeout: int = 15,
    delay: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Fetch upcoming fixtures from SofaScore for a date range.

    Args:
        date_from: ISO-8601 start date.
        date_to:   ISO-8601 end date (inclusive).
        timeout:   Per-request timeout in seconds.
        delay:     Seconds to wait between day requests (be polite).

    Returns:
        Combined list of fixture dicts across all days in the range.
    """
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)
    all_fixtures: List[Dict[str, Any]] = []
    current = d0
    while current <= d1:
        try:
            day_fixtures = _fetch_sofascore_day(current.isoformat(), timeout)
            all_fixtures.extend(day_fixtures)
        except Exception as exc:
            logger.warning(f"SofaScore fetch failed for {current}: {exc}")
        current += timedelta(days=1)
        if current <= d1:
            time.sleep(delay)

    logger.info(f"SofaScore: fetched {len(all_fixtures)} fixtures ({date_from} → {date_to})")
    return all_fixtures


# ---------------------------------------------------------------------------
# Source 2: BBC Sport fixtures HTML scrape
# ---------------------------------------------------------------------------
# BBC Sport renders fixture data in a structured JSON blob embedded in the page
# at https://www.bbc.com/sport/football/fixtures.  We parse the __INITIAL_DATA__
# JSON object that BBC injects into all sport pages.

def fetch_bbc_sport(
    date_from: str,
    date_to: str,
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    """
    Scrape upcoming football fixtures from BBC Sport.

    BBC Sport embeds a ``__INITIAL_DATA__`` JSON blob in every fixtures page.
    We retrieve the page, extract that blob, and parse the relevant fields.

    Args:
        date_from: ISO-8601 start date (used for date-range filtering only;
                   BBC Sport always shows the current upcoming week).
        date_to:   ISO-8601 end date.
        timeout:   Request timeout in seconds.

    Returns:
        List of fixture dicts.
    """
    import json
    import re

    url = "https://www.bbc.com/sport/football/fixtures"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-GB,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    # Extract the __INITIAL_DATA__ JSON blob
    match = re.search(
        r'window\.__INITIAL_DATA__\s*=\s*({.*?});\s*</script>',
        resp.text,
        re.DOTALL,
    )
    if not match:
        raise ValueError("Could not find __INITIAL_DATA__ in BBC Sport page")

    data = json.loads(match.group(1))

    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)
    fixtures: List[Dict[str, Any]] = []

    # Traverse the data tree — the exact path varies with BBC's page structure
    # but competitions are typically under data.body.items[].body.items[]
    for section in _bbc_iter_sections(data):
        comp_name = section.get("title", "")
        if not _is_tracked(comp_name):
            continue
        for match_item in section.get("events", []):
            try:
                match_date_str = match_item.get("startTime", "")[:10]
                match_date = date.fromisoformat(match_date_str)
                if not (d0 <= match_date <= d1):
                    continue
                home = match_item["homeTeam"]["name"]
                away = match_item["awayTeam"]["name"]
                status = match_item.get("status", {}).get("type", "")
                if status in ("finished", "postponed", "canceled"):
                    continue
                fixtures.append({
                    "date": match_date_str,
                    "competition": comp_name,
                    "home": home,
                    "away": away,
                    "neutral": False,
                })
            except (KeyError, TypeError, ValueError):
                continue

    logger.info(f"BBC Sport: fetched {len(fixtures)} fixtures ({date_from} → {date_to})")
    return fixtures


def _bbc_iter_sections(data: Any) -> List[Dict[str, Any]]:
    """
    Recursively walk BBC's JSON blob and yield fixture-section dicts.

    BBC's page structure changes occasionally; this breadth-first walk
    is more robust than hard-coding a fixed key path.
    """
    sections: List[Dict[str, Any]] = []
    if not isinstance(data, dict):
        return sections
    # The key that holds match events tends to be named "fixtureSection" or
    # has an "events" list alongside a "title" string.
    if "events" in data and "title" in data:
        sections.append(data)
    for v in data.values():
        if isinstance(v, dict):
            sections.extend(_bbc_iter_sections(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    sections.extend(_bbc_iter_sections(item))
    return sections


# ---------------------------------------------------------------------------
# Source 3: TheSportsDB free API (key = "3")
# ---------------------------------------------------------------------------
# Free tier: https://www.thesportsdb.com/api.php
# Endpoint: /api/v1/json/3/eventsday.php?d=YYYY-MM-DD&s=Soccer

_SPORTSDB_URL = "https://www.thesportsdb.com/api/v1/json/3/eventsday.php"

_SPORTSDB_LEAGUE_MAP: Dict[str, str] = {
    "English Premier League": "Premier League",
    "Spanish La Liga": "La Liga",
    "German Bundesliga": "Bundesliga",
    "Italian Serie A": "Serie A",
    "French Ligue 1": "Ligue 1",
    "UEFA Champions League": "UEFA Champions League",
    "UEFA Europa League": "UEFA Europa League",
    "FIFA World Cup": "WC 2026 Qualifiers",
    "UEFA Nations League": "UEFA Nations League",
}


def _fetch_sportsdb_day(day: str, timeout: int = 15) -> List[Dict[str, Any]]:
    """
    Fetch football events for *day* from TheSportsDB free API.

    Args:
        day: ISO-8601 date string.
        timeout: Request timeout in seconds.

    Returns:
        List of fixture dicts.
    """
    resp = requests.get(
        _SPORTSDB_URL,
        params={"d": day, "s": "Soccer"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    fixtures: List[Dict[str, Any]] = []
    for event in (data.get("events") or []):
        try:
            league_raw = event.get("strLeague", "")
            if not _is_tracked(league_raw):
                continue
            comp = _SPORTSDB_LEAGUE_MAP.get(league_raw, league_raw)
            home = event["strHomeTeam"]
            away = event["strAwayTeam"]
            fixtures.append({
                "date": day,
                "competition": comp,
                "home": home,
                "away": away,
                "neutral": False,
            })
        except (KeyError, TypeError):
            continue

    logger.debug(f"TheSportsDB [{day}]: {len(fixtures)} tracked fixtures")
    return fixtures


def fetch_sportsdb(
    date_from: str,
    date_to: str,
    timeout: int = 15,
    delay: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Fetch upcoming fixtures from TheSportsDB for a date range.

    Args:
        date_from: ISO-8601 start date.
        date_to:   ISO-8601 end date (inclusive).
        timeout:   Per-request timeout in seconds.
        delay:     Seconds to wait between day requests.

    Returns:
        Combined list of fixture dicts.
    """
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)
    all_fixtures: List[Dict[str, Any]] = []
    current = d0
    while current <= d1:
        try:
            day_fixtures = _fetch_sportsdb_day(current.isoformat(), timeout)
            all_fixtures.extend(day_fixtures)
        except Exception as exc:
            logger.warning(f"TheSportsDB fetch failed for {current}: {exc}")
        current += timedelta(days=1)
        if current <= d1:
            time.sleep(delay)

    logger.info(
        f"TheSportsDB: fetched {len(all_fixtures)} fixtures ({date_from} → {date_to})"
    )
    return all_fixtures


# ---------------------------------------------------------------------------
# Public entry point — tries sources in order, returns first non-empty result
# ---------------------------------------------------------------------------

def fetch_live_fixtures(
    date_from: str,
    date_to: Optional[str] = None,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """
    Fetch upcoming tracked football fixtures from the best available live source.

    Tries sources in this order and returns the first successful non-empty result:
      1. SofaScore unofficial API
      2. TheSportsDB free API
      3. BBC Sport HTML scrape

    Args:
        date_from: ISO-8601 start date (default: today).
        date_to:   ISO-8601 end date (default: ``date_from + 7 days``).
        timeout:   Per-request timeout in seconds.

    Returns:
        List of fixture dicts, sorted by date then competition.
        Returns an empty list if all sources fail.
    """
    if date_to is None:
        d0 = date.fromisoformat(date_from)
        date_to = (d0 + timedelta(days=7)).isoformat()

    sources = [
        ("SofaScore", lambda: fetch_sofascore(date_from, date_to, timeout=timeout)),
        ("TheSportsDB", lambda: fetch_sportsdb(date_from, date_to, timeout=timeout)),
        ("BBC Sport", lambda: fetch_bbc_sport(date_from, date_to, timeout=timeout)),
    ]

    for name, fetcher in sources:
        try:
            logger.info(f"Trying live fixture source: {name}")
            fixtures = fetcher()
            if fixtures:
                fixtures.sort(key=lambda f: (f["date"], f["competition"]))
                logger.info(
                    f"✓ {name} returned {len(fixtures)} fixtures "
                    f"({date_from} → {date_to})"
                )
                return fixtures
            logger.warning(f"{name} returned 0 fixtures — trying next source.")
        except Exception as exc:
            logger.warning(f"{name} failed: {exc} — trying next source.")

    logger.error(
        "All live fixture sources failed. "
        "The caller should fall back to a cached or static fixture list."
    )
    return []

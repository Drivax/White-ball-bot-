"""
FootPredict-Pro — League-specific web scrapers.

Provides a ``FootballScraper`` class that gathers upcoming fixtures from
multiple free public sources.  Each competition is tried in the order below
and the first non-empty result wins:

  1. **ESPN public API** (free, no key required, JSON, most reliable)
  2. **Direct league website HTML scrape** via ``requests`` + BeautifulSoup
     (static or semi-static pages: ligue1.com, laliga.com, bundesliga.com,
      legaseriea.it, worldfootball.net)
  3. **Selenium-based scrape** (JavaScript-rendered pages that fail plain
     requests: premierleague.com, laliga.com, ligue1.com, uefa.com)

Selenium is only imported and invoked when the lighter methods fail, so the
class degrades gracefully in headless CI environments where no browser is
available.

Returns a list of fixture dicts in the canonical FootPredict-Pro format::

    {
        "date":        "2026-04-07",     # ISO-8601
        "competition": "Premier League",
        "home":        "Arsenal",
        "away":        "Chelsea",
        "neutral":     False,
    }

Typical usage (called from ``fixtures_scraper.py``)::

    from src.data_ingestion.league_scrapers import FootballScraper

    scraper = FootballScraper()
    fixtures = scraper.scrape_all(date_from="2026-03-29", date_to="2026-04-05")
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ESPN public API – no key required.
# League slug → FootPredict canonical competition name.
_ESPN_LEAGUES: Dict[str, str] = {
    "eng.1":          "Premier League",
    "esp.1":          "La Liga",
    "ger.1":          "Bundesliga",
    "ita.1":          "Serie A",
    "fra.1":          "Ligue 1",
    "uefa.champions": "UEFA Champions League",
    "uefa.europa":    "UEFA Europa League",
}

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# worldfootball.net slug → canonical name (static HTML fallback)
_WFB_LEAGUES: Dict[str, str] = {
    "eng-premier-league-2025-2026":        "Premier League",
    "esp-primera-division-2025-2026":      "La Liga",
    "bundesliga-2025-2026":                "Bundesliga",
    "ita-serie-a-2025-2026":               "Serie A",
    "fra-ligue-1-2025-2026":              "Ligue 1",
    "champions-league-2025-2026":          "UEFA Champions League",
}

# Official league fixture URLs (may need Selenium for JS rendering)
_OFFICIAL_URLS: Dict[str, Dict[str, Any]] = {
    "Ligue 1": {
        "url":      "https://ligue1.com/fr/calendar",
        "selectors": {
            "match":  "div.match-item",
            "date":   "span.match-date",
            "home":   "span.team-1",
            "away":   "span.team-2",
            "time":   "span.match-time",
        },
    },
    "Premier League": {
        "url":      "https://www.premierleague.com/fixtures",
        "selectors": {
            "match":  "li.match-fixture",
            "date":   "time.league-phase__date",
            "home":   "span.match-fixture__short-name:nth-of-type(1)",
            "away":   "span.match-fixture__short-name:nth-of-type(2)",
            "time":   "span.match-fixture__time",
        },
    },
    "La Liga": {
        "url":      "https://www.laliga.com/en-GB/fixtures",
        "selectors": {
            "match":  "div.match-card",
            "date":   "span.match-date",
            "home":   "span.team-local",
            "away":   "span.team-visitor",
            "time":   "span.match-time",
        },
    },
    "UEFA Champions League": {
        "url":      "https://www.uefa.com/uefachampionsleague/fixtures-results/",
        "selectors": {
            "match":  "div.fixture",
            "date":   "span.fixture__date",
            "home":   "span.fixture__participant--1 .fixture__team-name",
            "away":   "span.fixture__participant--2 .fixture__team-name",
            "time":   "span.fixture__time",
        },
    },
}


# ---------------------------------------------------------------------------
# ESPN public API source
# ---------------------------------------------------------------------------

def _parse_espn_events(
    data: Dict[str, Any],
    competition: str,
    date_str: str,
    d0: date,
    d1: date,
) -> List[Dict[str, Any]]:
    """Parse ESPN scoreboard JSON into fixture dicts."""
    fixtures: List[Dict[str, Any]] = []
    for event in data.get("events", []):
        try:
            competition_list = event.get("competitions", [{}])
            comp_data = competition_list[0] if competition_list else {}

            # Skip already-finished matches
            status = comp_data.get("status", {}).get("type", {})
            if status.get("completed", False):
                continue

            competitors = comp_data.get("competitors", [])
            if len(competitors) < 2:
                continue

            # ESPN orders home first (homeAway == "home")
            home_team = away_team = None
            for c in competitors:
                name = c.get("team", {}).get("displayName", "")
                if c.get("homeAway") == "home":
                    home_team = name
                else:
                    away_team = name

            if not home_team or not away_team:
                # fall back to order
                home_team = competitors[0]["team"]["displayName"]
                away_team = competitors[1]["team"]["displayName"]

            # Date from event field
            event_date_raw = event.get("date", date_str)[:10]
            try:
                event_date = date.fromisoformat(event_date_raw)
            except ValueError:
                event_date = date.fromisoformat(date_str)

            if not (d0 <= event_date <= d1):
                continue

            fixtures.append({
                "date":        event_date.isoformat(),
                "competition": competition,
                "home":        home_team,
                "away":        away_team,
                "neutral":     False,
            })
        except (KeyError, IndexError, TypeError):
            continue
    return fixtures


def _fetch_espn_league_day(
    league_slug: str,
    competition: str,
    day: str,
    d0: date,
    d1: date,
    session: requests.Session,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """Fetch a single day from ESPN's free scoreboard API."""
    date_param = day.replace("-", "")
    url = f"{_ESPN_BASE}/{league_slug}/scoreboard"
    resp = session.get(url, params={"dates": date_param}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return _parse_espn_events(data, competition, day, d0, d1)


def _fetch_espn_league_range(
    league_slug: str,
    competition: str,
    date_from: str,
    date_to: str,
    d0: date,
    d1: date,
    session: requests.Session,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """
    Fetch fixtures for a date range from ESPN's free scoreboard API.

    ESPN accepts a ``dates`` parameter in the format ``YYYYMMDD-YYYYMMDD``
    which returns all events across the full range in one request — far more
    reliable than querying day-by-day.

    Args:
        league_slug:  ESPN league slug (e.g. ``"eng.1"``).
        competition:  Canonical competition name.
        date_from:    ISO-8601 start date string.
        date_to:      ISO-8601 end date string.
        d0:           Start date object (for range filtering).
        d1:           End date object (for range filtering).
        session:      ``requests.Session`` to reuse.
        timeout:      Request timeout in seconds.

    Returns:
        List of fixture dicts within [d0, d1].
    """
    date_range = f"{date_from.replace('-', '')}-{date_to.replace('-', '')}"
    url = f"{_ESPN_BASE}/{league_slug}/scoreboard"
    resp = session.get(url, params={"dates": date_range}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return _parse_espn_events(data, competition, date_from, d0, d1)


def fetch_espn(
    date_from: str,
    date_to: str,
    timeout: int = 15,
    delay: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Fetch upcoming football fixtures from ESPN's free public API.

    Makes one request per league using a date-range parameter
    (``dates=YYYYMMDD-YYYYMMDD``) rather than querying each day separately.
    The range format is reliably supported by ESPN and returns all scheduled
    events across the full window in a single call.

    Args:
        date_from: ISO-8601 start date.
        date_to:   ISO-8601 end date (inclusive).
        timeout:   Per-request timeout in seconds.
        delay:     Seconds to wait between league requests (courtesy rate-limit).

    Returns:
        Combined, deduplicated list of fixture dicts.
    """
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)

    session = requests.Session()
    session.headers.update({
        "User-Agent": _DEFAULT_HEADERS["User-Agent"],
        "Accept": "application/json",
    })

    all_fixtures: List[Dict[str, Any]] = []
    seen: set = set()

    slugs = list(_ESPN_LEAGUES.items())
    for idx, (slug, competition) in enumerate(slugs):
        try:
            league_fixtures = _fetch_espn_league_range(
                slug, competition, date_from, date_to, d0, d1, session, timeout
            )
            for fix in league_fixtures:
                key = (fix["date"], fix["competition"], fix["home"], fix["away"])
                if key not in seen:
                    seen.add(key)
                    all_fixtures.append(fix)
        except Exception as exc:
            logger.debug(f"ESPN [{slug}]: {exc}")
        if delay > 0 and idx < len(slugs) - 1:
            time.sleep(delay)

    logger.info(
        f"ESPN API: fetched {len(all_fixtures)} fixtures ({date_from} → {date_to})"
    )
    return all_fixtures


# ---------------------------------------------------------------------------
# BeautifulSoup scraper — worldfootball.net (static HTML)
# ---------------------------------------------------------------------------

def _parse_wfb_date(raw: str, year: int) -> Optional[str]:
    """
    Parse a worldfootball.net date cell like ``"04/07/26"`` or ``"07.04.2026"``
    into ISO-8601.  Returns ``None`` if parsing fails.
    """
    from datetime import datetime as _dt

    raw = raw.strip()
    # Try DD.MM.YYYY or DD/MM/YYYY
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return _dt.strptime(raw, fmt).date().isoformat()
        except (ValueError, AttributeError):
            pass
    # Try partial "DD.MM." → assume current year
    m = re.match(r"(\d{1,2})[./](\d{1,2})[./]?", raw)
    if m:
        try:
            return date(year, int(m.group(2)), int(m.group(1))).isoformat()
        except ValueError:
            pass
    return None


def _scrape_worldfootball_league(
    slug: str,
    competition: str,
    d0: date,
    d1: date,
    session: requests.Session,
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    """
    Scrape fixtures for one league from worldfootball.net.

    worldfootball.net renders tables in plain HTML, making it a reliable
    BeautifulSoup target.
    """
    url = f"https://www.worldfootball.net/schedule/{slug}/"
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "lxml")
    fixtures: List[Dict[str, Any]] = []

    # Fixture rows are in a standard <table class="standard_tabelle">
    table = soup.find("table", class_="standard_tabelle")
    if not table:
        # Try the first main content table
        table = soup.find("table", {"class": re.compile(r"tabelle|schedule")})
    if not table:
        logger.debug(f"worldfootball.net [{slug}]: no fixture table found")
        return fixtures

    # worldfootball.net schedule table columns (0-indexed):
    #   0: date   1: time   2: home team   3: score/dash   4: away team
    # When fewer columns are present the score column is absent.
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        try:
            date_raw  = cells[0].get_text(strip=True)
            home_cell = cells[2].get_text(strip=True)
            away_cell = cells[4].get_text(strip=True) if len(cells) > 4 else cells[3].get_text(strip=True)
            result_cell = cells[3].get_text(strip=True) if len(cells) > 4 else ""

            # Skip rows where a score (e.g. "2:1") is already recorded — match played
            if re.search(r"\d+:\d+", result_cell):
                continue
            # Skip empty team names
            if not home_cell or not away_cell or home_cell == away_cell:
                continue

            date_str = _parse_wfb_date(date_raw, d0.year)
            if not date_str:
                continue

            match_date = date.fromisoformat(date_str)
            if not (d0 <= match_date <= d1):
                continue

            fixtures.append({
                "date":        date_str,
                "competition": competition,
                "home":        home_cell,
                "away":        away_cell,
                "neutral":     False,
            })
        except (IndexError, ValueError):
            continue

    logger.debug(
        f"worldfootball.net [{slug}]: {len(fixtures)} fixtures in range"
    )
    return fixtures


def fetch_worldfootball(
    date_from: str,
    date_to: str,
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    """
    Scrape upcoming fixtures from worldfootball.net for all tracked leagues.

    worldfootball.net serves plain HTML, so this works without a browser.

    Args:
        date_from: ISO-8601 start date.
        date_to:   ISO-8601 end date (inclusive).
        timeout:   Per-request timeout in seconds.

    Returns:
        List of fixture dicts.
    """
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)

    session = requests.Session()
    session.headers.update(_DEFAULT_HEADERS)

    all_fixtures: List[Dict[str, Any]] = []
    for slug, competition in _WFB_LEAGUES.items():
        try:
            league_fixtures = _scrape_worldfootball_league(
                slug, competition, d0, d1, session, timeout
            )
            all_fixtures.extend(league_fixtures)
        except Exception as exc:
            logger.warning(f"worldfootball.net [{slug}]: {exc}")

    logger.info(
        f"worldfootball.net: fetched {len(all_fixtures)} fixtures "
        f"({date_from} → {date_to})"
    )
    return all_fixtures


# ---------------------------------------------------------------------------
# BeautifulSoup scraper — official league sites (requests first)
# ---------------------------------------------------------------------------

def _scrape_official_bs4(
    competition: str,
    cfg: Dict[str, Any],
    d0: date,
    d1: date,
    session: requests.Session,
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    """
    Attempt to scrape an official league website with requests + BeautifulSoup.

    Returns an empty list if the page appears to require JavaScript (no match
    elements found), signalling the caller to try Selenium instead.
    """
    url = cfg["url"]
    sel = cfg["selectors"]

    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "lxml")
    match_elements = soup.select(sel["match"])

    fixtures: List[Dict[str, Any]] = []
    for match_el in match_elements:
        try:
            date_el = match_el.select_one(sel["date"])
            home_el = match_el.select_one(sel["home"])
            away_el = match_el.select_one(sel["away"])

            if not all([date_el, home_el, away_el]):
                continue

            home = home_el.get_text(strip=True)
            away = away_el.get_text(strip=True)
            if not home or not away:
                continue

            date_raw = date_el.get_text(strip=True)
            # Handle ISO dates embedded in datetime attributes
            dt_attr = date_el.get("datetime") or date_el.get("data-date") or date_raw
            date_str = _parse_wfb_date(str(dt_attr), d0.year)
            if not date_str:
                continue

            match_date = date.fromisoformat(date_str)
            if not (d0 <= match_date <= d1):
                continue

            fixtures.append({
                "date":        date_str,
                "competition": competition,
                "home":        home,
                "away":        away,
                "neutral":     False,
            })
        except Exception:
            continue

    return fixtures


# ---------------------------------------------------------------------------
# Selenium-based scraper (JavaScript-rendered sites)
# ---------------------------------------------------------------------------

def _scrape_official_selenium(
    competition: str,
    cfg: Dict[str, Any],
    d0: date,
    d1: date,
    wait_seconds: int = 8,
) -> List[Dict[str, Any]]:
    """
    Scrape an official league website using a headless Selenium Chrome driver.

    Used as a fallback when the plain requests approach returns 0 fixtures
    (indicating JavaScript rendering is required).

    Args:
        competition:  Canonical competition name.
        cfg:          Dict with ``url`` and ``selectors`` keys.
        d0:           Start date filter.
        d1:           End date filter.
        wait_seconds: Time (seconds) to allow JS to load before parsing.

    Returns:
        List of fixture dicts (may be empty if scraping fails or driver
        unavailable).
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError:
        logger.warning(
            "selenium not installed — skipping Selenium scrape for "
            f"{competition}. Install with: pip install selenium"
        )
        return []

    # Try to auto-manage ChromeDriver via webdriver-manager
    chrome_service: Optional[Any] = None
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        chrome_service = Service(ChromeDriverManager().install())
    except Exception:
        chrome_service = None  # Use PATH-installed chromedriver

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(f"user-agent={_DEFAULT_HEADERS['User-Agent']}")

    driver = None
    fixtures: List[Dict[str, Any]] = []
    url = cfg["url"]
    sel = cfg["selectors"]

    try:
        if chrome_service:
            driver = webdriver.Chrome(service=chrome_service, options=options)
        else:
            driver = webdriver.Chrome(options=options)

        driver.get(url)

        # Wait for match elements to appear
        try:
            WebDriverWait(driver, wait_seconds).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, sel["match"]))
            )
        except Exception:
            logger.debug(f"Selenium [{competition}]: timed out waiting for {sel['match']}")

        # Parse rendered HTML with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "lxml")
        match_elements = soup.select(sel["match"])

        for match_el in match_elements:
            try:
                date_el = match_el.select_one(sel["date"])
                home_el = match_el.select_one(sel["home"])
                away_el = match_el.select_one(sel["away"])

                if not all([date_el, home_el, away_el]):
                    continue

                home = home_el.get_text(strip=True)
                away = away_el.get_text(strip=True)
                if not home or not away:
                    continue

                date_raw = date_el.get_text(strip=True)
                dt_attr = (
                    date_el.get("datetime")
                    or date_el.get("data-date")
                    or date_raw
                )
                date_str = _parse_wfb_date(str(dt_attr), d0.year)
                if not date_str:
                    continue

                match_date = date.fromisoformat(date_str)
                if not (d0 <= match_date <= d1):
                    continue

                fixtures.append({
                    "date":        date_str,
                    "competition": competition,
                    "home":        home,
                    "away":        away,
                    "neutral":     False,
                })
            except Exception:
                continue

        logger.info(
            f"Selenium [{competition}]: found {len(fixtures)} fixtures"
        )
    except Exception as exc:
        logger.warning(f"Selenium scrape failed for {competition}: {exc}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    return fixtures


# ---------------------------------------------------------------------------
# Main public class — FootballScraper
# ---------------------------------------------------------------------------

class FootballScraper:
    """
    Multi-source football fixture scraper.

    Scrapes upcoming match data from a cascade of sources:

      1. ESPN public API  (free JSON, no key needed)
      2. WorldFootball.net HTML  (static, very reliable)
      3. Official league websites via requests + BeautifulSoup
      4. Official league websites via Selenium (JS fallback)

    Each competition's results are merged and deduplicated.

    Example::

        scraper = FootballScraper()
        fixtures = scraper.scrape_all("2026-03-29", "2026-04-05")
        for fix in fixtures:
            print(fix["date"], fix["competition"], fix["home"], "vs", fix["away"])
    """

    def __init__(self, timeout: int = 20, selenium_wait: int = 8):
        self.timeout = timeout
        self.selenium_wait = selenium_wait
        self._session = requests.Session()
        self._session.headers.update(_DEFAULT_HEADERS)

    # ── Per-league convenience wrappers ──────────────────────────────────────

    def scrape_ligue1(
        self, date_from: str, date_to: str
    ) -> List[Dict[str, Any]]:
        """Scrape upcoming Ligue 1 matches."""
        return self._scrape_competition("Ligue 1", date_from, date_to)

    def scrape_premier_league(
        self, date_from: str, date_to: str
    ) -> List[Dict[str, Any]]:
        """Scrape upcoming Premier League matches."""
        return self._scrape_competition("Premier League", date_from, date_to)

    def scrape_la_liga(
        self, date_from: str, date_to: str
    ) -> List[Dict[str, Any]]:
        """Scrape upcoming La Liga matches."""
        return self._scrape_competition("La Liga", date_from, date_to)

    def scrape_bundesliga(
        self, date_from: str, date_to: str
    ) -> List[Dict[str, Any]]:
        """Scrape upcoming Bundesliga matches."""
        return self._scrape_competition("Bundesliga", date_from, date_to)

    def scrape_serie_a(
        self, date_from: str, date_to: str
    ) -> List[Dict[str, Any]]:
        """Scrape upcoming Serie A matches."""
        return self._scrape_competition("Serie A", date_from, date_to)

    def scrape_champions_league(
        self, date_from: str, date_to: str
    ) -> List[Dict[str, Any]]:
        """Scrape upcoming UEFA Champions League matches."""
        return self._scrape_competition("UEFA Champions League", date_from, date_to)

    # ── Internal per-competition scraper ─────────────────────────────────────

    def _scrape_competition(
        self, competition: str, date_from: str, date_to: str
    ) -> List[Dict[str, Any]]:
        """
        Collect fixtures for *competition* trying ESPN → worldfootball →
        official BS4 → official Selenium in order.
        """
        d0 = date.fromisoformat(date_from)
        d1 = date.fromisoformat(date_to)

        # 1. ESPN API — single range request for this competition
        try:
            espn_slug = {v: k for k, v in _ESPN_LEAGUES.items()}.get(competition)
            if espn_slug:
                espn_fixtures = _fetch_espn_league_range(
                    espn_slug, competition, date_from, date_to, d0, d1,
                    self._session, self.timeout,
                )
                if espn_fixtures:
                    logger.info(f"ESPN [{competition}]: {len(espn_fixtures)} fixtures")
                    return espn_fixtures
        except Exception as exc:
            logger.debug(f"ESPN [{competition}]: {exc}")

        # 2. worldfootball.net (BeautifulSoup)
        wfb_slug = {v: k for k, v in _WFB_LEAGUES.items()}.get(competition)
        if wfb_slug:
            try:
                wfb_fixtures = _scrape_worldfootball_league(
                    wfb_slug, competition, d0, d1, self._session, self.timeout
                )
                if wfb_fixtures:
                    logger.info(
                        f"worldfootball.net [{competition}]: {len(wfb_fixtures)} fixtures"
                    )
                    return wfb_fixtures
            except Exception as exc:
                logger.debug(f"worldfootball.net [{competition}]: {exc}")

        # 3. Official website — requests + BeautifulSoup
        official_cfg = _OFFICIAL_URLS.get(competition)
        if official_cfg:
            try:
                bs4_fixtures = _scrape_official_bs4(
                    competition, official_cfg, d0, d1,
                    self._session, self.timeout,
                )
                if bs4_fixtures:
                    logger.info(
                        f"Official BS4 [{competition}]: {len(bs4_fixtures)} fixtures"
                    )
                    return bs4_fixtures
                logger.debug(
                    f"Official BS4 [{competition}]: 0 fixtures (likely JS-rendered). "
                    "Trying Selenium."
                )
            except Exception as exc:
                logger.debug(f"Official BS4 [{competition}]: {exc}")

            # 4. Official website — Selenium (JS fallback)
            try:
                sel_fixtures = _scrape_official_selenium(
                    competition, official_cfg, d0, d1, self.selenium_wait
                )
                if sel_fixtures:
                    logger.info(
                        f"Selenium [{competition}]: {len(sel_fixtures)} fixtures"
                    )
                    return sel_fixtures
            except Exception as exc:
                logger.debug(f"Selenium [{competition}]: {exc}")

        logger.warning(f"FootballScraper: no fixtures found for {competition}.")
        return []

    # ── Full multi-competition scrape ────────────────────────────────────────

    def scrape_all(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scrape all tracked competitions for a date range.

        Prefers the ESPN API (single bulk call per league per day) over the
        per-competition methods, falling back to worldfootball.net and Selenium
        if ESPN returns nothing.

        Args:
            date_from: ISO-8601 start date (default: today).
            date_to:   ISO-8601 end date (default: today + 7 days).

        Returns:
            Combined, deduplicated, date-sorted list of fixture dicts.
        """
        today = date.today()
        date_from = date_from or today.isoformat()
        date_to   = date_to   or (today + timedelta(days=7)).isoformat()

        logger.info(
            f"FootballScraper.scrape_all: {date_from} → {date_to}"
        )

        # ── 1. ESPN bulk fetch (covers all tracked leagues at once) ───────────
        all_fixtures: List[Dict[str, Any]] = []
        try:
            espn_fixtures = fetch_espn(date_from, date_to, timeout=self.timeout)
            if espn_fixtures:
                logger.info(
                    f"FootballScraper: ESPN returned {len(espn_fixtures)} fixtures total"
                )
                all_fixtures.extend(espn_fixtures)
        except Exception as exc:
            logger.warning(f"FootballScraper: ESPN bulk fetch failed: {exc}")

        # ── 2. worldfootball.net (BeautifulSoup) — supplement/fallback ────────
        competitions_covered = {f["competition"] for f in all_fixtures}
        missing = set(_WFB_LEAGUES.values()) - competitions_covered
        if missing:
            try:
                wfb_fixtures = fetch_worldfootball(
                    date_from, date_to, timeout=self.timeout
                )
                # Only add fixtures for competitions ESPN missed
                for fix in wfb_fixtures:
                    if fix["competition"] in missing:
                        all_fixtures.append(fix)
                logger.info(
                    f"FootballScraper: worldfootball.net added "
                    f"{sum(1 for f in wfb_fixtures if f['competition'] in missing)} fixtures"
                )
            except Exception as exc:
                logger.warning(f"FootballScraper: worldfootball.net failed: {exc}")

        # ── 3. Official-site Selenium fallback for still-missing competitions ──
        competitions_covered = {f["competition"] for f in all_fixtures}
        for competition, cfg in _OFFICIAL_URLS.items():
            if competition in competitions_covered:
                continue
            d0 = date.fromisoformat(date_from)
            d1 = date.fromisoformat(date_to)

            # Try requests/BS4 first
            try:
                bs4_fixtures = _scrape_official_bs4(
                    competition, cfg, d0, d1, self._session, self.timeout
                )
                if bs4_fixtures:
                    all_fixtures.extend(bs4_fixtures)
                    logger.info(
                        f"FootballScraper: Official BS4 added "
                        f"{len(bs4_fixtures)} {competition} fixtures"
                    )
                    continue
            except Exception as exc:
                logger.debug(f"Official BS4 [{competition}]: {exc}")

            # Fall back to Selenium
            try:
                sel_fixtures = _scrape_official_selenium(
                    competition, cfg, d0, d1, self.selenium_wait
                )
                if sel_fixtures:
                    all_fixtures.extend(sel_fixtures)
                    logger.info(
                        f"FootballScraper: Selenium added "
                        f"{len(sel_fixtures)} {competition} fixtures"
                    )
            except Exception as exc:
                logger.debug(f"Selenium [{competition}]: {exc}")

        # Deduplicate and sort
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for fix in all_fixtures:
            key = (fix["date"], fix["competition"], fix["home"], fix["away"])
            if key not in seen:
                seen.add(key)
                unique.append(fix)

        unique.sort(key=lambda f: (f["date"], f["competition"]))
        logger.info(
            f"FootballScraper.scrape_all: {len(unique)} unique fixtures total"
        )
        return unique

    # ── Display / save helpers ───────────────────────────────────────────────

    def display_fixtures(self, fixtures: List[Dict[str, Any]]) -> None:
        """Print fixtures to stdout in a human-readable format."""
        if not fixtures:
            print("No fixtures found.")
            return

        current_competition = ""
        for fix in fixtures:
            if fix["competition"] != current_competition:
                current_competition = fix["competition"]
                print(f"\n{'=' * 60}")
                print(f"  {current_competition}")
                print(f"{'=' * 60}")
            print(f"  {fix['date']}  {fix['home']} vs {fix['away']}")

    def save_to_json(
        self,
        fixtures: List[Dict[str, Any]],
        filename: str = "upcoming_fixtures.json",
    ) -> None:
        """Save fixtures to a JSON file."""
        try:
            with open(filename, "w", encoding="utf-8") as fh:
                json.dump(fixtures, fh, ensure_ascii=False, indent=2)
            logger.info(f"Fixtures saved to {filename}")
            print(f"✓ Fixtures saved to {filename}")
        except OSError as exc:
            logger.error(f"Failed to save fixtures: {exc}")


# ---------------------------------------------------------------------------
# CLI entry-point (python -m src.data_ingestion.league_scrapers)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    today = date.today()
    parser = argparse.ArgumentParser(
        description="Scrape upcoming football fixtures and optionally save to JSON."
    )
    parser.add_argument(
        "--date-from", default=today.isoformat(),
        help=f"Start date ISO-8601 (default: {today})"
    )
    parser.add_argument(
        "--date-to", default=(today + timedelta(days=7)).isoformat(),
        help="End date ISO-8601"
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Save fixtures to this JSON file path"
    )
    args = parser.parse_args()

    scraper = FootballScraper()
    fixtures = scraper.scrape_all(args.date_from, args.date_to)
    scraper.display_fixtures(fixtures)

    if args.output_json:
        scraper.save_to_json(fixtures, args.output_json)
    else:
        scraper.save_to_json(fixtures)

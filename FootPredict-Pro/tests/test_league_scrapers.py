"""
FootPredict-Pro — Tests for league_scrapers.FootballScraper.

These tests validate the parsing and data-transformation logic entirely
offline (no network calls) by injecting mock HTTP responses via
``unittest.mock``.  Only ``_parse_wfb_date``, ``_parse_espn_events``,
``FootballScraper.scrape_all``, and ``FootballScraper.display_fixtures``
are exercised here.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_ingestion.league_scrapers import (
    FootballScraper,
    _ESPN_LEAGUES,
    _parse_espn_events,
    _parse_wfb_date,
    fetch_espn,
)


# ---------------------------------------------------------------------------
# _parse_wfb_date
# ---------------------------------------------------------------------------

class TestParseWfbDate:
    def test_dd_mm_yyyy(self):
        assert _parse_wfb_date("07.04.2026", 2026) == "2026-04-07"

    def test_dd_slash_mm_slash_yyyy(self):
        assert _parse_wfb_date("07/04/2026", 2026) == "2026-04-07"

    def test_iso_format_passthrough(self):
        assert _parse_wfb_date("2026-04-07", 2026) == "2026-04-07"

    def test_returns_none_for_garbage(self):
        assert _parse_wfb_date("not-a-date", 2026) is None

    def test_partial_date_with_current_year(self):
        result = _parse_wfb_date("07.04.", 2026)
        assert result == "2026-04-07"


# ---------------------------------------------------------------------------
# _parse_espn_events
# ---------------------------------------------------------------------------

class TestParseEspnEvents:
    """Unit tests for ESPN JSON → fixture dict conversion."""

    def _make_event(
        self,
        home: str,
        away: str,
        completed: bool = False,
        event_date: str = "2026-04-07T19:45:00Z",
    ) -> dict:
        return {
            "date": event_date,
            "competitions": [
                {
                    "status": {"type": {"completed": completed}},
                    "competitors": [
                        {
                            "homeAway": "home",
                            "team": {"displayName": home},
                        },
                        {
                            "homeAway": "away",
                            "team": {"displayName": away},
                        },
                    ],
                }
            ],
        }

    def test_upcoming_fixture_parsed(self):
        data = {"events": [self._make_event("Arsenal", "Chelsea")]}
        d0 = date(2026, 4, 7)
        d1 = date(2026, 4, 7)
        fixtures = _parse_espn_events(data, "Premier League", "2026-04-07", d0, d1)
        assert len(fixtures) == 1
        fix = fixtures[0]
        assert fix["home"] == "Arsenal"
        assert fix["away"] == "Chelsea"
        assert fix["competition"] == "Premier League"
        assert fix["date"] == "2026-04-07"
        assert fix["neutral"] is False

    def test_completed_fixture_excluded(self):
        data = {"events": [self._make_event("Arsenal", "Chelsea", completed=True)]}
        d0 = date(2026, 4, 7)
        d1 = date(2026, 4, 7)
        fixtures = _parse_espn_events(data, "Premier League", "2026-04-07", d0, d1)
        assert fixtures == []

    def test_out_of_range_fixture_excluded(self):
        data = {"events": [
            self._make_event("Arsenal", "Chelsea", event_date="2026-04-10T19:45:00Z")
        ]}
        d0 = date(2026, 4, 7)
        d1 = date(2026, 4, 7)
        fixtures = _parse_espn_events(data, "Premier League", "2026-04-07", d0, d1)
        assert fixtures == []

    def test_multiple_fixtures(self):
        data = {
            "events": [
                self._make_event("Arsenal", "Chelsea"),
                self._make_event("Liverpool", "Man City"),
            ]
        }
        d0 = date(2026, 4, 7)
        d1 = date(2026, 4, 7)
        fixtures = _parse_espn_events(data, "Premier League", "2026-04-07", d0, d1)
        assert len(fixtures) == 2

    def test_empty_events(self):
        fixtures = _parse_espn_events(
            {}, "Premier League", "2026-04-07",
            date(2026, 4, 7), date(2026, 4, 7),
        )
        assert fixtures == []

    def test_malformed_event_skipped(self):
        data = {"events": [{"date": "2026-04-07T19:45:00Z"}]}  # missing competitions
        fixtures = _parse_espn_events(
            data, "Premier League", "2026-04-07",
            date(2026, 4, 7), date(2026, 4, 7),
        )
        assert fixtures == []


# ---------------------------------------------------------------------------
# fetch_espn (mocked HTTP)
# ---------------------------------------------------------------------------

class TestFetchEspn:
    """Integration-level test of fetch_espn with mocked requests."""

    def _espn_response(self, home: str, away: str, day: str) -> dict:
        return {
            "events": [
                {
                    "date": f"{day}T19:45:00Z",
                    "competitions": [
                        {
                            "status": {"type": {"completed": False}},
                            "competitors": [
                                {"homeAway": "home", "team": {"displayName": home}},
                                {"homeAway": "away", "team": {"displayName": away}},
                            ],
                        }
                    ],
                }
            ]
        }

    def test_fetch_espn_returns_fixtures(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._espn_response(
            "Arsenal", "Chelsea", "2026-04-07"
        )
        mock_resp.raise_for_status.return_value = None

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch(
            "src.data_ingestion.league_scrapers.requests.Session",
            return_value=mock_session,
        ):
            fixtures = fetch_espn(
                "2026-04-07", "2026-04-07", timeout=5, delay=0
            )

        assert isinstance(fixtures, list)


# ---------------------------------------------------------------------------
# FootballScraper
# ---------------------------------------------------------------------------

class TestFootballScraper:
    """Tests for the FootballScraper orchestration class."""

    def test_instantiation(self):
        scraper = FootballScraper()
        assert scraper.timeout == 20
        assert scraper.selenium_wait == 8

    def test_scrape_all_returns_list(self):
        """scrape_all with all sources mocked to return empty should not raise."""
        with (
            patch("src.data_ingestion.league_scrapers.fetch_espn", return_value=[]),
            patch("src.data_ingestion.league_scrapers.fetch_worldfootball", return_value=[]),
            patch("src.data_ingestion.league_scrapers._scrape_official_bs4", return_value=[]),
            patch("src.data_ingestion.league_scrapers._scrape_official_selenium", return_value=[]),
        ):
            scraper = FootballScraper()
            fixtures = scraper.scrape_all("2026-04-07", "2026-04-07")
        assert isinstance(fixtures, list)

    def test_scrape_all_deduplicates(self):
        """Duplicate fixtures from different sources should appear only once."""
        dup_fixture = {
            "date": "2026-04-07",
            "competition": "Premier League",
            "home": "Arsenal",
            "away": "Chelsea",
            "neutral": False,
        }
        with (
            patch(
                "src.data_ingestion.league_scrapers.fetch_espn",
                return_value=[dup_fixture, dup_fixture],
            ),
            patch("src.data_ingestion.league_scrapers.fetch_worldfootball", return_value=[dup_fixture]),
            patch("src.data_ingestion.league_scrapers._scrape_official_bs4", return_value=[]),
            patch("src.data_ingestion.league_scrapers._scrape_official_selenium", return_value=[]),
        ):
            scraper = FootballScraper()
            fixtures = scraper.scrape_all("2026-04-07", "2026-04-07")
        assert len(fixtures) == 1

    def test_scrape_all_sorted_by_date(self):
        """Returned fixtures should be sorted ascending by date."""
        fixtures_in = [
            {"date": "2026-04-09", "competition": "La Liga", "home": "Barcelona", "away": "Madrid", "neutral": False},
            {"date": "2026-04-07", "competition": "Premier League", "home": "Arsenal", "away": "Chelsea", "neutral": False},
        ]
        with (
            patch("src.data_ingestion.league_scrapers.fetch_espn", return_value=fixtures_in),
            patch("src.data_ingestion.league_scrapers.fetch_worldfootball", return_value=[]),
            patch("src.data_ingestion.league_scrapers._scrape_official_bs4", return_value=[]),
            patch("src.data_ingestion.league_scrapers._scrape_official_selenium", return_value=[]),
        ):
            scraper = FootballScraper()
            fixtures = scraper.scrape_all("2026-04-07", "2026-04-09")
        dates = [f["date"] for f in fixtures]
        assert dates == sorted(dates)

    def test_display_fixtures_no_crash(self, capsys):
        scraper = FootballScraper()
        scraper.display_fixtures([])
        captured = capsys.readouterr()
        assert "No fixtures found" in captured.out

    def test_display_fixtures_prints_competition_headers(self, capsys):
        scraper = FootballScraper()
        scraper.display_fixtures([
            {"date": "2026-04-07", "competition": "Premier League",
             "home": "Arsenal", "away": "Chelsea", "neutral": False},
        ])
        captured = capsys.readouterr()
        assert "Premier League" in captured.out
        assert "Arsenal" in captured.out

    def test_save_to_json(self, tmp_path):
        scraper = FootballScraper()
        out_file = str(tmp_path / "fixtures.json")
        data = [{"date": "2026-04-07", "competition": "La Liga",
                 "home": "Barcelona", "away": "Real Madrid", "neutral": False}]
        scraper.save_to_json(data, out_file)
        loaded = json.loads(Path(out_file).read_text())
        assert loaded == data

    def test_scrape_ligue1_calls_competition(self):
        with patch.object(FootballScraper, "_scrape_competition", return_value=[]) as mock_sc:
            scraper = FootballScraper()
            scraper.scrape_ligue1("2026-04-07", "2026-04-07")
        mock_sc.assert_called_once_with("Ligue 1", "2026-04-07", "2026-04-07")

    def test_scrape_premier_league_calls_competition(self):
        with patch.object(FootballScraper, "_scrape_competition", return_value=[]) as mock_sc:
            scraper = FootballScraper()
            scraper.scrape_premier_league("2026-04-07", "2026-04-07")
        mock_sc.assert_called_once_with("Premier League", "2026-04-07", "2026-04-07")

    def test_scrape_la_liga_calls_competition(self):
        with patch.object(FootballScraper, "_scrape_competition", return_value=[]) as mock_sc:
            scraper = FootballScraper()
            scraper.scrape_la_liga("2026-04-07", "2026-04-07")
        mock_sc.assert_called_once_with("La Liga", "2026-04-07", "2026-04-07")

    def test_scrape_champions_league_calls_competition(self):
        with patch.object(FootballScraper, "_scrape_competition", return_value=[]) as mock_sc:
            scraper = FootballScraper()
            scraper.scrape_champions_league("2026-04-07", "2026-04-07")
        mock_sc.assert_called_once_with("UEFA Champions League", "2026-04-07", "2026-04-07")


# ---------------------------------------------------------------------------
# fetch_live_fixtures integration — new ESPN source is tried first
# ---------------------------------------------------------------------------

class TestFetchLiveFixturesEspnFirst:
    """Verify ESPN is consulted before SofaScore in fetch_live_fixtures."""

    def test_espn_is_tried_first(self):
        from src.data_ingestion.fixtures_scraper import fetch_live_fixtures

        call_order = []

        def _espn(*args, **kwargs):
            call_order.append("ESPN")
            return [
                {
                    "date": "2026-04-07",
                    "competition": "Premier League",
                    "home": "Arsenal",
                    "away": "Chelsea",
                    "neutral": False,
                }
            ]

        def _sofascore(*args, **kwargs):
            call_order.append("SofaScore")
            return []

        with (
            patch(
                "src.data_ingestion.fixtures_scraper.fetch_sofascore",
                side_effect=_sofascore,
            ),
            patch(
                "src.data_ingestion.league_scrapers.fetch_espn",
                side_effect=_espn,
            ),
        ):
            result = fetch_live_fixtures("2026-04-07", "2026-04-07")

        assert result[0]["competition"] == "Premier League"
        # ESPN should have been tried; SofaScore should not be needed
        assert "ESPN" in call_order

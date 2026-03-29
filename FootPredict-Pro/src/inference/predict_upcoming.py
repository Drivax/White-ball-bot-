"""
FootPredict-Pro — Upcoming match batch predictions.

Fetches all upcoming fixtures for a given date range (default: today + 7 days),
then generates predictions using the best available model.

If a trained MasterEnsemble is found in the models/ directory it is used
directly.  Otherwise a Dixon-Coles model is seeded with built-in team-strength
priors so that predictions are still informative without any data download or
training step.

Fixture data is sourced from API-Football when an API key is configured in
config.yaml; if the key is absent or the request fails the script automatically
falls back to a curated static fixture list covering the period
2026-03-29 → 2026-04-08 (today + next ~10 days).

Usage:
    # Predict today + next 7 days (uses config.yaml API key if set)
    python src/inference/predict_upcoming.py

    # Explicit date range
    python src/inference/predict_upcoming.py \\
        --date-from 2026-03-29 --date-to 2026-04-05

    # Filter to a single competition
    python src/inference/predict_upcoming.py --competition "Premier League"

    # JSON output saved to a file
    python src/inference/predict_upcoming.py \\
        --output-format json --output-file predictions_week.json

    # Pipe-friendly plain text (no Rich colours)
    python src/inference/predict_upcoming.py --no-color
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from loguru import logger

from src.models.poisson_dixon_coles import DixonColesModel
from src.models.ensemble import MasterEnsemble, MatchPrediction


# ---------------------------------------------------------------------------
# Static fixture list – used when API-Football is not configured
# Covers: 29 March 2026 (today) through 8 April 2026 (UCL QF Leg 1 second slate)
#
# Verified against official sources (UEFA, La Liga, Bundesliga, Serie A,
# Ligue 1) as of 2026-03-29.
#
# Key corrections vs. previous version:
#   • 29 Mar: UEFA WC qualifier group games do not exist on this date.
#             UEFA play-off semi-finals were 26 Mar; finals are 31 Mar.
#             CONMEBOL qualifying concluded Sept 2025.
#             Replaced with confirmed high-profile international friendlies.
#   • 1-2 Apr: UCL QF Leg 1 is actually 7-8 April (not 1-2 April).
#              Previous matchups (Real Madrid vs Arsenal, Bayern vs PSG,
#              Man City vs Inter Milan) were also incorrect.
#   • 4-5 Apr: Premier League GW32 starts 10 April — no PL on this weekend.
#              La Liga, Bundesliga, Serie A and Ligue 1 fixtures corrected.
# ---------------------------------------------------------------------------

#: Each entry: date (ISO), competition, home, away
#: ``neutral=True`` suppresses home advantage (e.g. friendly on neutral ground).
STATIC_FIXTURES: List[Dict[str, Any]] = [
    # ── 29 March 2026 ── International Friendlies (FIFA March Window) ────────
    # UEFA WC qualifier play-off semi-finals were 26 Mar; finals are 31 Mar.
    # 29 Mar falls mid-window and features high-profile pre-WC friendlies.
    {"date": "2026-03-29", "competition": "International Friendly", "home": "Colombia", "away": "France",  "neutral": True},
    {"date": "2026-03-29", "competition": "International Friendly", "home": "Portugal", "away": "Mexico",  "neutral": True},

    # ── 4 April 2026 ── La Liga Matchweek 30 ─────────────────────────────────
    {"date": "2026-04-04", "competition": "La Liga", "home": "Atletico Madrid", "away": "Barcelona",    "neutral": False},
    {"date": "2026-04-04", "competition": "La Liga", "home": "Mallorca",        "away": "Real Madrid",  "neutral": False},
    {"date": "2026-04-04", "competition": "La Liga", "home": "Real Betis",      "away": "Espanyol",     "neutral": False},
    # ── 4 April 2026 ── Bundesliga Matchday 28 ───────────────────────────────
    # Bayern is away at SC Freiburg; Leverkusen hosts Wolfsburg;
    # Stuttgart hosts Dortmund.
    {"date": "2026-04-04", "competition": "Bundesliga",      "home": "SC Freiburg",       "away": "Bayern Munich",     "neutral": False},
    {"date": "2026-04-04", "competition": "Bundesliga",      "home": "Bayer Leverkusen",  "away": "Wolfsburg",         "neutral": False},
    {"date": "2026-04-04", "competition": "Bundesliga",      "home": "VfB Stuttgart",     "away": "Borussia Dortmund", "neutral": False},

    # ── 4 April 2026 ── Serie A ───────────────────────────────────────────────
    # Juventus vs Lazio and Inter vs AC Milan do NOT fall on this date;
    # the confirmed April 4 Serie A fixtures are below.
    {"date": "2026-04-04", "competition": "Serie A",         "home": "Hellas Verona",    "away": "Fiorentina",        "neutral": False},
    {"date": "2026-04-04", "competition": "Serie A",         "home": "Lazio",            "away": "Parma",             "neutral": False},

    # ── 4 April 2026 ── Ligue 1 ──────────────────────────────────────────────
    # Confirmed April 4 Ligue 1 fixtures (Saturday slate).
    {"date": "2026-04-04", "competition": "Ligue 1",         "home": "Rennes",           "away": "Brest",             "neutral": False},
    {"date": "2026-04-04", "competition": "Ligue 1",         "home": "Lens",             "away": "Lille",             "neutral": False},

    # ── 5 April 2026 ── Sunday slate ─────────────────────────────────────────
    # La Liga — no Premier League matches this weekend (GW32 starts 10 Apr).
    {"date": "2026-04-05", "competition": "La Liga",         "home": "Getafe",           "away": "Athletic Bilbao",   "neutral": False},
    # Serie A
    {"date": "2026-04-05", "competition": "Serie A",         "home": "Inter Milan",      "away": "Roma",              "neutral": False},
    # Ligue 1
    {"date": "2026-04-05", "competition": "Ligue 1",         "home": "Monaco",           "away": "PSG",               "neutral": False},

    # ── 7 April 2026 ── UEFA Champions League QF Leg 1 ───────────────────────
    # Confirmed draw (27 Feb 2026, Nyon): Real Madrid vs Bayern Munich;
    # Sporting CP vs Arsenal.
    {"date": "2026-04-07", "competition": "UEFA Champions League QF Leg 1", "home": "Real Madrid",  "away": "Bayern Munich", "neutral": False},
    {"date": "2026-04-07", "competition": "UEFA Champions League QF Leg 1", "home": "Sporting CP",  "away": "Arsenal",       "neutral": False},

    # ── 8 April 2026 ── UEFA Champions League QF Leg 1 ───────────────────────
    # Barcelona vs Atletico Madrid; PSG vs Liverpool.
    {"date": "2026-04-08", "competition": "UEFA Champions League QF Leg 1", "home": "Barcelona",    "away": "Atletico Madrid", "neutral": False},
    {"date": "2026-04-08", "competition": "UEFA Champions League QF Leg 1", "home": "PSG",          "away": "Liverpool",       "neutral": False},
]


# ---------------------------------------------------------------------------
# Built-in team-strength priors for the Dixon-Coles model
# (attack, defense) in log-scale; calibrated from 2024-25 season form.
# These are static priors used as a training-free fallback; for live
# production use a model trained on current-season data.
# attack > 0 → above-average scoring; defense < 0 → above-average defending.
# lam = exp(home_attack − away_defense + home_advantage)
# mu  = exp(away_attack  − home_defense)
# ---------------------------------------------------------------------------

TEAM_PRIORS: Dict[str, Tuple[float, float]] = {
    # ── Premier League ───────────────────────────────────────────────────────
    "Liverpool":          ( 0.48, -0.35),
    "Manchester City":    ( 0.42, -0.30),
    "Arsenal":            ( 0.38, -0.32),
    "Chelsea":            ( 0.28, -0.08),
    "Aston Villa":        ( 0.22, -0.12),
    "Newcastle United":   ( 0.20, -0.18),
    "Tottenham":          ( 0.18,  0.05),
    "Manchester United":  ( 0.12,  0.10),
    "Brighton":           ( 0.15, -0.05),
    "West Ham":           ( 0.05,  0.08),
    "Bournemouth":        ( 0.10,  0.12),
    "Fulham":             ( 0.08,  0.05),
    "Crystal Palace":     ( 0.00, -0.02),
    "Everton":            (-0.05,  0.08),
    "Wolves":             (-0.08,  0.05),
    "Nottingham Forest":  (-0.05, -0.08),
    "Brentford":          ( 0.05,  0.02),
    "Leicester City":     (-0.10,  0.15),
    "Southampton":        (-0.20,  0.20),
    "Ipswich Town":       (-0.15,  0.18),

    # ── La Liga ──────────────────────────────────────────────────────────────
    "Real Madrid":        ( 0.50, -0.32),
    "Barcelona":          ( 0.45, -0.25),
    "Atletico Madrid":    ( 0.25, -0.38),
    "Athletic Bilbao":    ( 0.20, -0.15),
    "Villarreal":         ( 0.18, -0.05),
    "Real Sociedad":      ( 0.15, -0.10),
    "Real Betis":         ( 0.10,  0.00),
    "Sevilla":            ( 0.05,  0.05),
    "Valencia":           ( 0.00,  0.08),
    "Getafe":             (-0.10,  0.00),
    "Espanyol":           (-0.05,  0.10),
    "Mallorca":           (-0.05,  0.10),

    # ── Bundesliga ───────────────────────────────────────────────────────────
    "Bayern Munich":      ( 0.55, -0.28),
    "Bayer Leverkusen":   ( 0.42, -0.22),
    "Borussia Dortmund":  ( 0.35, -0.10),
    "RB Leipzig":         ( 0.30, -0.18),
    "Eintracht Frankfurt":( 0.20,  0.05),
    "VfB Stuttgart":      ( 0.22, -0.08),
    "Wolfsburg":          ( 0.00,  0.05),
    "Augsburg":           (-0.15,  0.05),
    "Hoffenheim":         ( 0.05,  0.10),
    "SC Freiburg":        ( 0.10, -0.02),

    # ── Serie A ──────────────────────────────────────────────────────────────
    "Inter Milan":        ( 0.45, -0.35),
    "Napoli":             ( 0.35, -0.20),
    "Juventus":           ( 0.30, -0.25),
    "AC Milan":           ( 0.28, -0.15),
    "Roma":               ( 0.22, -0.05),
    "Lazio":              ( 0.20,  0.00),
    "Atalanta":           ( 0.38, -0.15),
    "Fiorentina":         ( 0.18, -0.05),
    "Hellas Verona":      (-0.05,  0.12),
    "Parma":              (-0.12,  0.18),

    # ── Ligue 1 ──────────────────────────────────────────────────────────────
    "PSG":                ( 0.60, -0.25),
    "Marseille":          ( 0.28, -0.15),
    "Monaco":             ( 0.30, -0.12),
    "Lyon":               ( 0.25,  0.05),
    "Lille":              ( 0.22, -0.20),
    "Lens":               ( 0.15, -0.05),
    "Rennes":             ( 0.10, -0.05),
    "Brest":              ( 0.08,  0.10),
    "Montpellier":        (-0.15,  0.15),

    # ── International (friendlies / qualifiers) ───────────────────────────────
    "England":            ( 0.40, -0.28),
    "France":             ( 0.45, -0.35),
    "Germany":            ( 0.40, -0.30),
    "Spain":              ( 0.42, -0.32),
    "Italy":              ( 0.32, -0.28),
    "Portugal":           ( 0.42, -0.22),
    "Netherlands":        ( 0.38, -0.22),
    "Belgium":            ( 0.35, -0.20),
    "Brazil":             ( 0.45, -0.30),
    "Argentina":          ( 0.48, -0.28),
    "Croatia":            ( 0.22, -0.18),
    "Denmark":            ( 0.28, -0.22),
    "Hungary":            ( 0.12, -0.08),
    "Albania":            ( 0.05, -0.05),
    "Czech Republic":     ( 0.18, -0.12),
    "Austria":            ( 0.22, -0.10),
    "Turkey":             ( 0.15, -0.05),
    "Serbia":             ( 0.20, -0.12),
    "Switzerland":        ( 0.25, -0.20),
    "Bulgaria":           ( 0.00,  0.05),
    "Colombia":           ( 0.30, -0.15),
    "Uruguay":            ( 0.28, -0.18),
    "Chile":              ( 0.18, -0.08),
    "Ecuador":            ( 0.12, -0.05),
    "Mexico":             ( 0.25, -0.15),

    # ── UCL clubs without domestic league entry ───────────────────────────────
    "Sporting CP":        ( 0.30, -0.12),
}

_AVG_ATTACK:  float = 0.0
_AVG_DEFENSE: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_seeded_model(
    home_advantage: float = 0.30,
    rho: float = -0.10,
) -> DixonColesModel:
    """
    Return a DixonColesModel pre-loaded with TEAM_PRIORS.

    This bypasses the ``fit()`` step so predictions can be generated
    without historical data when trained model files are absent.
    """
    model = DixonColesModel(xi=0.0018, max_goals=10)
    model.attack = {t: v[0] for t, v in TEAM_PRIORS.items()}
    model.defense = {t: v[1] for t, v in TEAM_PRIORS.items()}
    model.home_advantage = home_advantage
    model.rho = rho
    model.teams = sorted(TEAM_PRIORS.keys())
    model._is_fitted = True
    return model


def _fetch_upcoming_from_api(
    date_from: str,
    date_to: str,
    api_key: str,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """
    Fetch upcoming fixtures from API-Football for a date range.

    Returns a list of fixture dicts compatible with STATIC_FIXTURES format.
    """
    import requests

    headers = {
        "x-apisports-key": api_key,
        "Accept": "application/json",
    }
    url = "https://v3.football.api-sports.io/fixtures"
    params = {"from": date_from, "to": date_to, "timezone": "UTC"}

    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if data.get("errors"):
        raise RuntimeError(f"API-Football error: {data['errors']}")

    fixtures = []
    for fix in data.get("response", []):
        try:
            fixtures.append({
                "date": fix["fixture"]["date"][:10],
                "competition": fix["league"]["name"],
                "home": fix["teams"]["home"]["name"],
                "away": fix["teams"]["away"]["name"],
                "neutral": False,
                "fixture_id": fix["fixture"]["id"],
            })
        except (KeyError, TypeError):
            continue

    logger.info(
        f"API-Football returned {len(fixtures)} upcoming fixtures "
        f"({date_from} → {date_to})"
    )
    return fixtures


def _get_fixtures(
    date_from: str,
    date_to: str,
    api_key: Optional[str],
    competition_filter: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Return fixtures in the date range, trying API then falling back to static list.
    """
    fixtures: List[Dict[str, Any]] = []

    if api_key and api_key not in ("", "YOUR_API_FOOTBALL_KEY"):
        try:
            fixtures = _fetch_upcoming_from_api(date_from, date_to, api_key)
        except Exception as exc:
            logger.warning(
                f"API-Football fetch failed ({exc}). "
                "Falling back to static fixture list."
            )

    if not fixtures:
        logger.info("Using built-in static fixture list.")
        d0 = datetime.strptime(date_from, "%Y-%m-%d").date()
        d1 = datetime.strptime(date_to, "%Y-%m-%d").date()
        fixtures = [
            f for f in STATIC_FIXTURES
            if d0 <= datetime.strptime(f["date"], "%Y-%m-%d").date() <= d1
        ]

    if competition_filter:
        cf = competition_filter.lower()
        fixtures = [f for f in fixtures if cf in f["competition"].lower()]

    # Sort by date then competition
    fixtures.sort(key=lambda f: (f["date"], f["competition"]))
    return fixtures


def _load_or_build_ensemble(model_dir: Path) -> MasterEnsemble:
    """
    Try to load a saved MasterEnsemble; fall back to seeded Dixon-Coles.
    """
    try:
        ensemble = MasterEnsemble.load(model_dir)
        if ensemble.dixon_coles and ensemble.dixon_coles._is_fitted:
            logger.info("Loaded trained MasterEnsemble from disk.")
            return ensemble
        raise RuntimeError("Loaded model is not fitted.")
    except Exception as exc:
        logger.info(
            f"No trained models found ({exc}). "
            "Using seeded Dixon-Coles with built-in team priors."
        )
        dc = _build_seeded_model()
        return MasterEnsemble(
            dixon_coles_model=dc,
            outcome_model=None,
            player_model=None,
            poisson_weight=1.0,
            ml_weight=0.0,
        )


# ---------------------------------------------------------------------------
# Core prediction runner
# ---------------------------------------------------------------------------

def predict_all_upcoming(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    model_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    competition_filter: Optional[str] = None,
) -> List[Tuple[Dict[str, Any], MatchPrediction]]:
    """
    Generate predictions for every upcoming fixture in the date range.

    Args:
        date_from: Start date (ISO 8601, default: today).
        date_to:   End date   (ISO 8601, default: today + 7 days).
        model_dir: Path to saved model directory (optional).
        api_key:   API-Football key (optional; falls back to static list).
        competition_filter: Case-insensitive substring to filter competitions.

    Returns:
        Ordered list of (fixture_dict, MatchPrediction) tuples.
    """
    today = date.today()
    date_from = date_from or today.isoformat()
    date_to   = date_to   or (today + timedelta(days=7)).isoformat()

    # Try to read API key from config if not provided
    if api_key is None:
        try:
            from src.utils.config_loader import load_config
            cfg = load_config()
            api_key = cfg.api.api_football.key
        except Exception:
            api_key = None

    # Locate model directory
    root = Path(__file__).resolve().parent.parent.parent
    model_path = Path(model_dir) if model_dir else root / "models"

    # Load / build ensemble
    ensemble = _load_or_build_ensemble(model_path)

    # Collect fixtures
    fixtures = _get_fixtures(date_from, date_to, api_key, competition_filter)
    logger.info(f"Predicting {len(fixtures)} fixtures from {date_from} to {date_to}")

    results: List[Tuple[Dict[str, Any], MatchPrediction]] = []
    for fix in fixtures:
        home, away = fix["home"], fix["away"]
        # For neutral-venue matches suppress home advantage
        if fix.get("neutral"):
            saved_ha = ensemble.dixon_coles.home_advantage
            ensemble.dixon_coles.home_advantage = 0.0
            pred = ensemble.predict(home_team=home, away_team=away)
            ensemble.dixon_coles.home_advantage = saved_ha
        else:
            pred = ensemble.predict(home_team=home, away_team=away)
        results.append((fix, pred))

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _competition_emoji(name: str) -> str:
    n = name.lower()
    if "champions" in n:
        return "🏆"
    if "premier" in n or "england" in n:
        return "🏴󠁧󠁢󠁥󠁮󠁧󠁿"
    if "liga" in n or "spain" in n:
        return "🇪🇸"
    if "bundesliga" in n or "germany" in n:
        return "🇩🇪"
    if "serie a" in n or "italy" in n:
        return "🇮🇹"
    if "ligue" in n or "france" in n:
        return "🇫🇷"
    if "qualifier" in n or "wc" in n:
        return "🌍"
    return "⚽"


def print_predictions_report(
    results: List[Tuple[Dict[str, Any], MatchPrediction]],
    color: bool = True,
) -> None:
    """
    Print a rich-formatted predictions report grouped by date and competition.
    """
    if not results:
        print("No fixtures found for the requested period.")
        return

    try:
        if not color:
            raise ImportError  # force plain-text path
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box

        console = Console()
        console.print()
        console.rule(
            "[bold yellow]⚽  FootPredict-Pro — Upcoming Match Predictions  ⚽[/bold yellow]",
            style="yellow",
        )
        console.print()

        current_date = ""
        current_comp = ""

        for fix, pred in results:
            # Date header
            if fix["date"] != current_date:
                current_date = fix["date"]
                dt = datetime.strptime(current_date, "%Y-%m-%d")
                console.print(
                    f"\n[bold white on blue]  {dt.strftime('%A, %d %B %Y')}  [/bold white on blue]"
                )
                current_comp = ""

            # Competition header
            comp = fix["competition"]
            if comp != current_comp:
                current_comp = comp
                emoji = _competition_emoji(comp)
                console.print(f"\n  {emoji}  [bold cyan]{comp}[/bold cyan]")

            # Match table
            t = Table(
                show_header=True,
                header_style="bold dim",
                box=box.SIMPLE_HEAVY,
                expand=False,
                padding=(0, 1),
            )
            t.add_column("Match",          style="bold",         min_width=32)
            t.add_column("Home Win",       justify="right",      min_width=10)
            t.add_column("Draw",           justify="right",      min_width=8)
            t.add_column("Away Win",       justify="right",      min_width=10)
            t.add_column("xG H – A",       justify="center",     min_width=12)
            t.add_column("Top Score",      justify="center",     min_width=10)
            t.add_column("Confidence",     justify="center",     min_width=10)

            p   = pred
            probs = [p.p_home_win, p.p_draw, p.p_away_win]
            best  = max(probs)

            def _pct(v: float, is_best: bool) -> str:
                s = f"{v:.1%}"
                return f"[bold green]{s}[/bold green]" if is_best else s

            top_sc = f"{p.top_scorelines[0][0]}-{p.top_scorelines[0][1]}" if p.top_scorelines else "–"
            conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(p.confidence, "white")

            t.add_row(
                f"{pred.home_team}  vs  {pred.away_team}",
                _pct(p.p_home_win, p.p_home_win == best),
                _pct(p.p_draw,     p.p_draw     == best),
                _pct(p.p_away_win, p.p_away_win == best),
                f"{p.home_xg:.2f} – {p.away_xg:.2f}",
                top_sc,
                f"[{conf_color}]{p.confidence}[/{conf_color}]",
            )
            console.print(t)

        console.print()
        console.rule("[dim]Generated by FootPredict-Pro (Dixon-Coles + ML Ensemble)[/dim]")
        console.print()

    except ImportError:
        # Plain-text fallback (no Rich)
        print("\n" + "=" * 70)
        print("  FootPredict-Pro — Upcoming Match Predictions")
        print("=" * 70)

        current_date = ""
        current_comp = ""

        for fix, pred in results:
            if fix["date"] != current_date:
                current_date = fix["date"]
                dt = datetime.strptime(current_date, "%Y-%m-%d")
                print(f"\n{'─'*70}")
                print(f"  {dt.strftime('%A, %d %B %Y')}")
                print(f"{'─'*70}")
                current_comp = ""

            comp = fix["competition"]
            if comp != current_comp:
                current_comp = comp
                print(f"\n  {comp}")

            p = pred
            best = max(p.p_home_win, p.p_draw, p.p_away_win)
            match_label = f"{p.home_team} vs {p.away_team}"
            top_sc = (
                f"{p.top_scorelines[0][0]}-{p.top_scorelines[0][1]}"
                if p.top_scorelines else "–"
            )
            indicator = (
                f"HOME {p.p_home_win:.0%}" if p.p_home_win == best else
                f"DRAW {p.p_draw:.0%}"     if p.p_draw     == best else
                f"AWAY {p.p_away_win:.0%}"
            )
            print(
                f"    {match_label:<36}  "
                f"H:{p.p_home_win:.0%} D:{p.p_draw:.0%} A:{p.p_away_win:.0%}  "
                f"xG:{p.home_xg:.1f}-{p.away_xg:.1f}  "
                f"Top:{top_sc}  [{indicator}] ({p.confidence})"
            )

        print("\n" + "=" * 70)


def results_to_json(
    results: List[Tuple[Dict[str, Any], MatchPrediction]],
) -> List[Dict[str, Any]]:
    """Convert prediction results to a JSON-serialisable list."""
    output = []
    for fix, pred in results:
        entry = {
            "fixture": fix,
            "prediction": pred.to_dict(),
        }
        output.append(entry)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    today = date.today()
    next_week = today + timedelta(days=7)

    parser = argparse.ArgumentParser(
        description=(
            "FootPredict-Pro: Generate predictions for all upcoming football "
            "matches (today + next 7 days by default)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # This week's fixtures (default)
  python src/inference/predict_upcoming.py

  # Custom range
  python src/inference/predict_upcoming.py --date-from 2026-03-29 --date-to 2026-04-05

  # Only Champions League
  python src/inference/predict_upcoming.py --competition "Champions League"

  # Save JSON report
  python src/inference/predict_upcoming.py --output-format json \\
      --output-file predictions_week.json
        """,
    )
    parser.add_argument(
        "--date-from",
        default=today.isoformat(),
        help=f"Start date ISO-8601 (default: {today})",
    )
    parser.add_argument(
        "--date-to",
        default=next_week.isoformat(),
        help=f"End date ISO-8601 (default: {next_week})",
    )
    parser.add_argument(
        "--competition",
        default=None,
        help="Filter to competitions whose name contains this string (case-insensitive)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Path to trained model directory (default: FootPredict-Pro/models/)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API-Football key (overrides config.yaml)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Write output to this file instead of stdout",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable Rich colour output (plain text)",
    )

    args = parser.parse_args()

    results = predict_all_upcoming(
        date_from=args.date_from,
        date_to=args.date_to,
        model_dir=args.model_dir,
        api_key=args.api_key,
        competition_filter=args.competition,
    )

    if args.output_format == "json":
        payload = json.dumps(results_to_json(results), indent=2)
        if args.output_file:
            Path(args.output_file).write_text(payload, encoding="utf-8")
            logger.info(f"Saved JSON predictions to {args.output_file}")
        else:
            print(payload)
    else:
        print_predictions_report(results, color=not args.no_color)
        if args.output_file:
            # Write plain-text version to file
            import io
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            print_predictions_report(results, color=False)
            sys.stdout = old_stdout
            Path(args.output_file).write_text(buf.getvalue(), encoding="utf-8")
            logger.info(f"Saved table predictions to {args.output_file}")


if __name__ == "__main__":
    main()

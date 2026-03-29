#!/usr/bin/env python3
"""
generate_predictions.py — Scrape upcoming fixtures, run predictions, write txt.

Usage:
    # From the FootPredict-Pro directory
    python generate_predictions.py

    # Custom date range
    python generate_predictions.py --date-from 2026-04-07 --date-to 2026-04-13

    # Save to a specific path
    python generate_predictions.py --output-file /path/to/predictions.txt

    # Dry-run: print to stdout instead of writing file
    python generate_predictions.py --stdout

The script tries live fixture sources (SofaScore → TheSportsDB → BBC Sport)
before falling back to the static hardcoded list in predict_upcoming.py.

Output is written to ``<repo_root>/predictions_<YYYY-MM-DD>.txt`` by default.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the FootPredict-Pro package root is on sys.path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from loguru import logger

from src.inference.predict_upcoming import (
    MatchPrediction,
    predict_all_upcoming,
)


# ---------------------------------------------------------------------------
# Formatted report writer (matches the style of predictions_2026-03-29.txt)
# ---------------------------------------------------------------------------

def _verdict(pred: MatchPrediction) -> str:
    """Return the team name of the most likely outcome."""
    best = max(pred.p_home_win, pred.p_draw, pred.p_away_win)
    if pred.p_home_win == best:
        return pred.home_team
    if pred.p_away_win == best:
        return pred.away_team
    return "Draw"


def _scoreline_str(pred: MatchPrediction) -> str:
    """Top 5 scorelines formatted as ``2-1(8%)  1-1(7%) …``"""
    parts = []
    for h, a, p in pred.top_scorelines[:5]:
        parts.append(f"{h}-{a}({p:.0%})")
    return "  ".join(parts)


def build_report(
    results: List[Tuple[Dict[str, Any], MatchPrediction]],
    date_from: str,
    date_to: str,
    generated_at: Optional[str] = None,
) -> str:
    """
    Build the full predictions report as a plain-text string.

    Args:
        results:      List of (fixture_dict, MatchPrediction) tuples.
        date_from:    Period start (ISO-8601).
        date_to:      Period end   (ISO-8601).
        generated_at: Timestamp string; defaults to current UTC time.

    Returns:
        Multi-line string in the canonical predictions-file format.
    """
    if generated_at is None:
        from datetime import timezone
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    d0 = datetime.strptime(date_from, "%Y-%m-%d")
    d1 = datetime.strptime(date_to,   "%Y-%m-%d")
    period = f"{d0.strftime('%A %d %B %Y')} — {d1.strftime('%A %d %B %Y')}"

    n_fixtures = len(results)
    competitions = sorted({fix["competition"] for fix, _ in results})

    lines: List[str] = []
    SEP = "─" * 72
    WIDE = "=" * 72

    lines += [
        WIDE,
        "  FootPredict-Pro — Football Match Predictions",
        f"  Period: {period}",
        f"  Generated: {generated_at}",
        "  Model: Dixon-Coles Bivariate Poisson (seeded with 2024-25 priors)",
        WIDE,
        "",
        "  Columns: Match | P(Home) | P(Draw) | P(Away) | xG(H-A) |",
        "           Most likely score | Verdict | Confidence",
        "",
    ]

    current_date = ""
    current_comp = ""

    for fix, pred in results:
        # ── Date header ───────────────────────────────────────────────────────
        if fix["date"] != current_date:
            current_date = fix["date"]
            dt = datetime.strptime(current_date, "%Y-%m-%d")
            lines += ["", SEP, f"  {dt.strftime('%A, %d %B %Y').upper()}", SEP, ""]
            current_comp = ""

        # ── Competition header ────────────────────────────────────────────────
        comp = fix["competition"]
        if comp != current_comp:
            current_comp = comp
            lines += [f"  ◆ {comp}"]

        # ── Match row ─────────────────────────────────────────────────────────
        p = pred
        match_label = f"{p.home_team} vs {p.away_team}"
        top_sc = (
            f"{p.top_scorelines[0][0]}-{p.top_scorelines[0][1]}"
            f" ({p.top_scorelines[0][2]:.0%})"
            if p.top_scorelines else "–"
        )
        verdict = _verdict(p)
        conf = p.confidence

        row = (
            f"    {match_label:<40}"
            f"H:{p.p_home_win:.0%}  D:{p.p_draw:.0%}  A:{p.p_away_win:.0%}"
            f"  xG:{p.home_xg:.2f}-{p.away_xg:.2f}"
            f"  Top:{top_sc}"
            f"  → {verdict}  [{conf}]"
        )
        lines.append(row)

        # Scoreline distribution sub-row
        sc_str = _scoreline_str(p)
        if sc_str:
            lines.append(f"    {'':40}Scorelines: {sc_str}")

    # ── High confidence picks ─────────────────────────────────────────────────
    high_picks = [
        (fix, pred) for fix, pred in results if pred.confidence == "high"
    ]
    if high_picks:
        lines += ["", SEP, "  HIGH CONFIDENCE PICKS", SEP]
        max_label_len = max(
            len(f"{pred.home_team} vs {pred.away_team}") for _, pred in high_picks
        )
        for fix, pred in high_picks:
            best = max(pred.p_home_win, pred.p_draw, pred.p_away_win)
            if pred.p_home_win == best:
                outcome = f"HOME WIN  ({pred.p_home_win:.0%})"
            elif pred.p_away_win == best:
                outcome = f"AWAY WIN  ({pred.p_away_win:.0%})"
            else:
                outcome = f"DRAW  ({pred.p_draw:.0%})"
            label = f"{pred.home_team} vs {pred.away_team}"
            lines.append(
                f"  {fix['date']}  {label:<{max_label_len}}  {outcome}"
            )

    # ── Model notes ───────────────────────────────────────────────────────────
    comps_str = ", ".join(competitions) if competitions else "–"
    lines += [
        "",
        SEP,
        "  MODEL NOTES",
        SEP,
        "  Model:    Dixon-Coles bivariate Poisson (Dixon & Coles, 1997)",
        "  Ensemble: When trained .joblib files present ->",
        "            60% Poisson + 40% ML stack",
        "            (XGBoost + LightGBM + CatBoost + Logistic Regression)",
        "  This run: Seeded Dixon-Coles only (no trained models on disk)",
        "  Priors:   Built-in team-strength parameters from 2024-25 season form",
        "  Home adv: +0.30 (log-scale); suppressed for neutral-venue fixtures",
        "  Low-score correction: Dixon-Coles tau term applied to 0-0/1-0/0-1/1-1",
        f"  Fixtures: {n_fixtures} matches — {comps_str}",
        "",
        WIDE,
        "  Generated by FootPredict-Pro",
        "  Repository: https://github.com/Drivax/White-ball-bot-",
        WIDE,
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    today = date.today()
    next_week = today + timedelta(days=7)

    parser = argparse.ArgumentParser(
        description=(
            "Generate football match predictions and write to a txt file. "
            "Scrapes live fixtures from SofaScore / TheSportsDB / BBC Sport "
            "before falling back to the built-in static list."
        )
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
        "--output-file",
        default=None,
        help=(
            "Path to write the predictions txt file.  "
            "Default: <repo_root>/predictions_<date_from>.txt"
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of writing a file",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API-Football key (overrides config.yaml)",
    )
    parser.add_argument(
        "--competition",
        default=None,
        help="Filter competitions (case-insensitive substring)",
    )
    args = parser.parse_args()

    logger.info(
        f"Generating predictions: {args.date_from} → {args.date_to}"
    )

    results = predict_all_upcoming(
        date_from=args.date_from,
        date_to=args.date_to,
        api_key=args.api_key,
        competition_filter=args.competition,
    )

    if not results:
        logger.error(
            "No fixtures found for %s → %s. "
            "Check network access or configure an API key.",
            args.date_from,
            args.date_to,
        )
        sys.exit(1)

    report = build_report(results, args.date_from, args.date_to)

    if args.stdout:
        print(report)
        return

    if args.output_file:
        out_path = Path(args.output_file)
    else:
        repo_root = _HERE.parent
        out_path = repo_root / f"predictions_{args.date_from}.txt"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info(f"Predictions written to {out_path}")
    print(f"✓ Saved predictions to {out_path}")


if __name__ == "__main__":
    main()

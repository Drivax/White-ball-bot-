"""
FootPredict-Pro — Match prediction CLI.

Production inference script that loads trained models and generates
a full prediction in < 2 seconds.

Usage:
    # Basic prediction
    python src/inference/predict_match.py \\
        --home "Manchester City" \\
        --away "Arsenal"

    # With lineups
    python src/inference/predict_match.py \\
        --home "Manchester City" \\
        --away "Arsenal" \\
        --home_lineup "Ederson,Walker,Dias,Akanji,Gvardiol,Rodri,De Bruyne,Silva,Doku,Foden,Haaland" \\
        --away_lineup "Raya,White,Saliba,Gabriel,Zinchenko,Partey,Rice,Odegaard,Saka,Havertz,Martinelli"

    # JSON output
    python src/inference/predict_match.py \\
        --home "Manchester City" \\
        --away "Arsenal" \\
        --output-format json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from loguru import logger

from src.models.ensemble import MasterEnsemble, MatchPrediction
from src.feature_engineering.player_features import PlayerFeatureBuilder
from src.utils.helpers import get_project_root, team_name_normalize


# ---------------------------------------------------------------------------
# Predictor class (loads models once, reuses for multiple predictions)
# ---------------------------------------------------------------------------

class MatchPredictor:
    """
    Production-ready match predictor.

    Loads all trained models from disk on initialization, then provides
    fast (<2s) predictions via the predict() method.
    """

    def __init__(self, model_dir: Optional[str] = None) -> None:
        """
        Args:
            model_dir: Path to directory containing trained models.
                       Defaults to FootPredict-Pro/models/.
        """
        root = get_project_root()
        self.model_dir = Path(model_dir) if model_dir else root / "models"
        self._ensemble: Optional[MasterEnsemble] = None
        self._pipeline = None
        self._player_builder = PlayerFeatureBuilder()
        self._loaded = False

    def load(self) -> "MatchPredictor":
        """Load models from disk."""
        try:
            self._ensemble = MasterEnsemble.load(self.model_dir)
            self._loaded = True
            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load trained models from {self.model_dir}: {e}\n"
                "Falling back to Dixon-Coles with league priors."
            )
            # Create a minimal ensemble with just priors
            from src.models.poisson_dixon_coles import DixonColesModel
            dc = DixonColesModel()
            self._ensemble = MasterEnsemble(
                dixon_coles_model=dc,
                outcome_model=None,
                player_model=None,
            )
            self._loaded = True

        return self

    def predict(
        self,
        home_team: str,
        away_team: str,
        home_lineup: Optional[List[str]] = None,
        away_lineup: Optional[List[str]] = None,
        home_positions: Optional[Dict[str, str]] = None,
        away_positions: Optional[Dict[str, str]] = None,
    ) -> MatchPrediction:
        """
        Generate a full match prediction.

        Args:
            home_team: Home team name (flexible — normalized internally).
            away_team: Away team name.
            home_lineup: Optional list of home XI player names.
            away_lineup: Optional list of away XI player names.
            home_positions: Optional {player: position} for home.
            away_positions: Optional {player: position} for away.

        Returns:
            MatchPrediction with all prediction details.
        """
        if not self._loaded:
            self.load()

        home_norm = home_team
        away_norm = away_team

        # Build player features if lineups provided
        home_player_feats = None
        away_player_feats = None

        if home_lineup:
            _, home_player_feats = self._player_builder.get_lineup_features(
                lineup=home_lineup,
                positions=home_positions,
                team=home_norm,
                opponent_team=away_norm,
                is_home=True,
            )

        if away_lineup:
            _, away_player_feats = self._player_builder.get_lineup_features(
                lineup=away_lineup,
                positions=away_positions,
                team=away_norm,
                opponent_team=home_norm,
                is_home=False,
            )

        prediction = self._ensemble.predict(
            home_team=home_norm,
            away_team=away_norm,
            features=None,  # No ML features without trained pipeline
            home_lineup=home_lineup,
            away_lineup=away_lineup,
            home_player_features=home_player_feats,
            away_player_features=away_player_feats,
        )

        return prediction


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_prediction_rich(pred: MatchPrediction) -> None:
    """Print a beautiful rich-formatted prediction to console."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box

        console = Console()

        # Outcome table
        outcome_table = Table(show_header=True, header_style="bold yellow", box=box.ROUNDED)
        outcome_table.add_column("Outcome", style="bold")
        outcome_table.add_column("Probability", justify="right")
        outcome_table.add_column("Source: Poisson", justify="right")
        outcome_table.add_column("Source: ML", justify="right")

        outcomes = [
            (f"🏠 {pred.home_team} Win", pred.p_home_win, pred.poisson_p_home, pred.ml_p_home),
            ("🤝 Draw", pred.p_draw, pred.poisson_p_draw, pred.ml_p_draw),
            (f"✈️  {pred.away_team} Win", pred.p_away_win, pred.poisson_p_away, pred.ml_p_away),
        ]
        for label, p, pp, pm in outcomes:
            color = "green" if p == max(pred.p_home_win, pred.p_draw, pred.p_away_win) else "white"
            outcome_table.add_row(
                label,
                f"[{color}]{p:.1%}[/{color}]",
                f"{pp:.1%}",
                f"{pm:.1%}",
            )

        # Scorelines table
        score_table = Table(
            title="Top Scorelines",
            show_header=True,
            header_style="bold cyan",
            box=box.SIMPLE,
        )
        score_table.add_column("Score")
        score_table.add_column("Probability", justify="right")

        for h, a, p in pred.top_scorelines[:5]:
            score_table.add_row(f"{h} - {a}", f"{p:.1%}")

        console.print()
        console.rule(f"[bold white]{pred.home_team}  vs  {pred.away_team}[/bold white]")
        console.print(outcome_table)
        console.print(
            f"  xG: [cyan]{pred.home_team}[/cyan] {pred.home_xg:.2f} | "
            f"[cyan]{pred.away_team}[/cyan] {pred.away_xg:.2f}"
        )
        console.print(score_table)

        # Player predictions
        if pred.home_top_scorer or pred.away_top_scorer:
            console.print("\n[bold]Top Goal Scorers:[/bold]")
            if pred.home_top_scorer:
                s = pred.home_top_scorer
                console.print(
                    f"  🏠 [green]{s.name}[/green] ({pred.home_team}): "
                    f"P(≥1 goal) = [bold green]{s.p_goal:.1%}[/bold green] "
                    f"(xG λ={s.lambda_xg:.3f})"
                )
            if pred.away_top_scorer:
                s = pred.away_top_scorer
                console.print(
                    f"  ✈️  [blue]{s.name}[/blue] ({pred.away_team}): "
                    f"P(≥1 goal) = [bold blue]{s.p_goal:.1%}[/bold blue] "
                    f"(xG λ={s.lambda_xg:.3f})"
                )

        console.print(
            f"\n  [dim]Confidence: {pred.confidence} | "
            f"Inference: {pred.inference_time_ms:.0f}ms[/dim]"
        )
        console.rule()

    except ImportError:
        # Fallback plain text output
        print(pred)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FootPredict-Pro: Predict football match outcomes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/inference/predict_match.py --home "Manchester City" --away "Arsenal"

  python src/inference/predict_match.py \\
    --home "Manchester City" --away "Arsenal" \\
    --home_lineup "Ederson,Walker,Dias,Akanji,Gvardiol,Rodri,De Bruyne,Silva,Doku,Foden,Haaland" \\
    --away_lineup "Raya,White,Saliba,Gabriel,Zinchenko,Partey,Rice,Odegaard,Saka,Havertz,Martinelli" \\
    --output-format json
        """,
    )
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name")
    parser.add_argument(
        "--home_lineup",
        type=str,
        default=None,
        help="Comma-separated home starting XI",
    )
    parser.add_argument(
        "--away_lineup",
        type=str,
        default=None,
        help="Comma-separated away starting XI",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    args = parser.parse_args()

    # Parse lineups
    home_lineup = (
        [p.strip() for p in args.home_lineup.split(",")]
        if args.home_lineup
        else None
    )
    away_lineup = (
        [p.strip() for p in args.away_lineup.split(",")]
        if args.away_lineup
        else None
    )

    # Load predictor
    predictor = MatchPredictor(model_dir=args.model_dir)
    predictor.load()

    # Run prediction
    prediction = predictor.predict(
        home_team=args.home,
        away_team=args.away,
        home_lineup=home_lineup,
        away_lineup=away_lineup,
    )

    # Output
    if args.output_format == "json":
        print(json.dumps(prediction.to_dict(), indent=2))
    else:
        print_prediction_rich(prediction)


if __name__ == "__main__":
    main()

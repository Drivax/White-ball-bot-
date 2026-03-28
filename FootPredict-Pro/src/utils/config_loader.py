"""
FootPredict-Pro — Configuration loader.

Loads config.yaml (and optional config.local.yaml override) into a
typed Pydantic model accessible throughout the project.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------

class APIFootballConfig(BaseModel):
    base_url: str = "https://v3.football.api-sports.io"
    key: str = ""
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 5
    rate_limit_per_minute: int = 100


class UnderstatConfig(BaseModel):
    base_url: str = "https://understat.com"
    enabled: bool = True


class FootballDataCoConfig(BaseModel):
    base_url: str = "https://www.football-data.co.uk/mmz4281"
    enabled: bool = True


class APIConfig(BaseModel):
    api_football: APIFootballConfig = APIFootballConfig()
    understat: UnderstatConfig = UnderstatConfig()
    football_data_co: FootballDataCoConfig = FootballDataCoConfig()


class LeagueConfig(BaseModel):
    id: int
    name: str
    country: str
    football_data_code: Optional[str] = None
    understat_name: Optional[str] = None


class PathsConfig(BaseModel):
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    models: str = "models"
    logs: str = "logs"


class PlayerFeatureConfig(BaseModel):
    form_window: int = 5
    min_minutes: int = 45
    position_weights: Dict[str, float] = Field(default_factory=lambda: {
        "striker": 1.0, "forward": 0.85, "midfielder": 0.55,
        "defender": 0.20, "goalkeeper": 0.02
    })


class FeaturesConfig(BaseModel):
    form_window: int = 10
    form_short_window: int = 5
    decay_factor: float = 0.85
    h2h_matches: int = 5
    min_matches_for_rating: int = 5
    home_advantage_factor: float = 0.1
    player: PlayerFeatureConfig = PlayerFeatureConfig()


class XGBoostConfig(BaseModel):
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1


class LightGBMConfig(BaseModel):
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1


class CatBoostConfig(BaseModel):
    iterations: int = 500
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: int = 3
    random_seed: int = 42
    verbose: int = 0


class LogisticConfig(BaseModel):
    C: float = 1.0
    max_iter: int = 1000
    random_state: int = 42


class DixonColesConfig(BaseModel):
    max_goals: int = 10
    xi: float = 0.0018
    min_matches: int = 5


class PlayerXGBConfig(BaseModel):
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = -1


class ModelsConfig(BaseModel):
    xgboost: XGBoostConfig = XGBoostConfig()
    lightgbm: LightGBMConfig = LightGBMConfig()
    catboost: CatBoostConfig = CatBoostConfig()
    logistic_regression: LogisticConfig = LogisticConfig()
    dixon_coles: DixonColesConfig = DixonColesConfig()
    player_xgb: PlayerXGBConfig = PlayerXGBConfig()


class EnsembleConfig(BaseModel):
    outcome: Dict[str, float] = Field(default_factory=lambda: {
        "xgboost": 0.30, "lightgbm": 0.30, "catboost": 0.25, "logistic": 0.15
    })
    scoreline: Dict[str, float] = Field(default_factory=lambda: {
        "dixon_coles": 0.60, "xgboost_goals": 0.40
    })
    calibration: str = "isotonic"


class TrainingConfig(BaseModel):
    test_size: float = 0.15
    min_samples: int = 200
    time_weight: bool = True
    cv_folds: int = 5
    optuna_trials: int = 50
    optuna_timeout: int = 600
    early_stopping_rounds: int = 50


class BacktestConfig(BaseModel):
    min_train_matches: int = 100
    rolling_window: int = 380
    step_size: int = 10


class InferenceConfig(BaseModel):
    max_response_time: float = 2.0
    confidence_threshold: float = 0.45
    default_lineup_size: int = 11


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    rotation: str = "100 MB"
    retention: str = "30 days"


class AppConfig(BaseModel):
    api: APIConfig = APIConfig()
    leagues: List[LeagueConfig] = Field(default_factory=list)
    seasons: Dict[str, Any] = Field(default_factory=dict)
    paths: PathsConfig = PathsConfig()
    features: FeaturesConfig = FeaturesConfig()
    models: ModelsConfig = ModelsConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    training: TrainingConfig = TrainingConfig()
    backtest: BacktestConfig = BacktestConfig()
    inference: InferenceConfig = InferenceConfig()
    logging: LoggingConfig = LoggingConfig()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _find_config_file() -> Path:
    """Locate config.yaml relative to this file or via env var."""
    env_path = os.environ.get("FOOTPREDICT_CONFIG")
    if env_path:
        return Path(env_path)
    # Walk up from src/utils to project root
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "config.yaml not found. Set FOOTPREDICT_CONFIG env var or place "
        "config.yaml in the project root."
    )


@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load and return the application configuration.

    Merges config.yaml with optional config.local.yaml override.
    Results are cached so the file is only read once per process.

    Args:
        config_path: Override path to config file. If None, auto-detected.

    Returns:
        AppConfig: Validated configuration object.
    """
    path = Path(config_path) if config_path else _find_config_file()

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    # Merge local override if it exists
    local_path = path.parent / "config.local.yaml"
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            local_data: Dict[str, Any] = yaml.safe_load(f) or {}
        data = _deep_merge(data, local_data)

    # Allow env var override for API key
    if api_key := os.environ.get("API_FOOTBALL_KEY"):
        data.setdefault("api", {}).setdefault("api_football", {})["key"] = api_key

    return AppConfig(**data)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

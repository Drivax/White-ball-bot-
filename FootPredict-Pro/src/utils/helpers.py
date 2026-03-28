"""
FootPredict-Pro — General utility helpers.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import numpy as np
import pandas as pd

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the project root directory (contains config.yaml)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config.yaml").exists():
            return parent
    return here.parent.parent.parent


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str | Path, metadata: Optional[dict] = None) -> None:
    """
    Save a trained model to disk using joblib.

    Args:
        model: Any sklearn-compatible model.
        path: Output file path (.joblib recommended).
        metadata: Optional dict saved alongside the model (as JSON sidecar).
    """
    import joblib

    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(model, path)

    if metadata:
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)


def load_model(path: str | Path) -> Any:
    """
    Load a model from disk.

    Args:
        path: Path to .joblib file.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def team_name_normalize(name: str) -> str:
    """
    Normalize team names for consistent matching across data sources.

    Strips whitespace, lowercases, and handles common abbreviations.

    Args:
        name: Raw team name string.

    Returns:
        Normalized lowercase string.
    """
    substitutions = {
        "man city": "manchester city",
        "man utd": "manchester united",
        "man united": "manchester united",
        "spurs": "tottenham hotspur",
        "tottenham": "tottenham hotspur",
        "wolves": "wolverhampton wanderers",
        "brentford fc": "brentford",
        "brighton": "brighton & hove albion",
        "west ham": "west ham united",
        "newcastle": "newcastle united",
        "aston villa fc": "aston villa",
        "real madrid cf": "real madrid",
        "fc barcelona": "barcelona",
        "atletico de madrid": "atletico madrid",
        "atletico madrid cf": "atletico madrid",
        "inter milan": "inter",
        "ac milan": "milan",
        "paris saint-germain": "psg",
        "paris sg": "psg",
        "rb leipzig": "rasenballsport leipzig",
        "borussia dortmund": "bvb",
    }
    normalized = name.strip().lower()
    return substitutions.get(normalized, normalized)


def exponential_decay_weights(n: int, decay: float = 0.85) -> np.ndarray:
    """
    Generate exponentially decaying weights for the last N time steps.

    Most recent gets weight 1.0, oldest gets weight decay^(n-1).

    Args:
        n: Number of time steps.
        decay: Decay factor per step (0 < decay < 1).

    Returns:
        Array of weights, length n, summing to 1.0.
    """
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    return weights / weights.sum()


def rolling_weighted_average(
    series: pd.Series,
    window: int,
    decay: float = 0.85,
) -> pd.Series:
    """
    Compute exponentially weighted rolling average over a pandas Series.

    Args:
        series: Input time series.
        window: Rolling window size.
        decay: Exponential decay factor.

    Returns:
        Rolling weighted average series (same index as input).
    """
    result = pd.Series(index=series.index, dtype=float)
    for i in range(len(series)):
        if i < window - 1:
            # Use available data
            sub = series.iloc[max(0, i - window + 1): i + 1].values
        else:
            sub = series.iloc[i - window + 1: i + 1].values

        if len(sub) == 0:
            result.iloc[i] = np.nan
            continue

        w = exponential_decay_weights(len(sub), decay)
        result.iloc[i] = float(np.dot(w, sub))

    return result


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timed(func: F) -> F:
    """Decorator that logs function execution time."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        try:
            from loguru import logger
            logger.debug(f"{func.__qualname__} took {elapsed:.3f}s")
        except ImportError:
            pass
        return result

    return wrapper  # type: ignore

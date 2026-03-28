"""
FootPredict-Pro: Utility helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the FootPredict-Pro project root."""
    return Path(__file__).resolve().parent.parent.parent


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

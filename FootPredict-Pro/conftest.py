"""
Shared pytest configuration for FootPredict-Pro tests.
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

"""Tests import the stage helpers the way the stage scripts do: with pipeline/ on sys.path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

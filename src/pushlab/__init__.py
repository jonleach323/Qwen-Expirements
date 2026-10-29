"""pushlab: push LLM neurons on schedules during generation."""

from importlib.metadata import version
from pathlib import Path

__version__ = version("pushlab")

# The repository checkout pushlab runs from: uv installs the package from
# src/ in editable mode, so experiments/ and uv.lock sit two folders up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

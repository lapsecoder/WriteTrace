"""Phase-1 configuration loader.

Source of truth: ml/configs/phase1.json (free-form JSON, no extra dependencies).
"""
from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path | None = None) -> dict:
    """Load and return the phase-1 configuration dictionary."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "ml" / "configs" / "phase1.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
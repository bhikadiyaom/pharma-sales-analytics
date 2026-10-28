"""Configuration loading and project paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` (cached). Pass a path to use an alternative file."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return ROOT / rel


def results_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    p = resolve(cfg["paths"]["results_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    p = resolve(cfg["paths"]["processed_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def category_assumptions(cfg: dict[str, Any], category: str) -> dict[str, float]:
    """Merge default planning assumptions with per-category overrides."""
    a = cfg["assumptions"]
    out = dict(a["defaults"])
    out.update(a.get("per_category", {}).get(category, {}))
    return out

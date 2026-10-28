"""Persistence: Parquet for analytics, SQLite for the sales table."""

from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd

from src.config import load_config, processed_dir, resolve


def write_sales_sqlite(
    daily: pd.DataFrame, hourly: pd.DataFrame | None = None, cfg: dict[str, Any] | None = None
) -> None:
    """Write the cleaned sales tables into SQLite (``sales_daily`` / ``sales_hourly``)."""
    cfg = cfg or load_config()
    db = resolve(cfg["paths"]["sqlite_db"])
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as con:
        d = daily.assign(date=daily["date"].dt.strftime("%Y-%m-%d"))
        d.to_sql("sales_daily", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_daily ON sales_daily(category, date)")
        if hourly is not None:
            h = hourly.assign(date=hourly["date"].dt.strftime("%Y-%m-%d %H:%M"))
            h.to_sql("sales_hourly", con, if_exists="replace", index=False)


def read_daily(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the cleaned daily table (Parquet)."""
    cfg = cfg or load_config()
    return pd.read_parquet(processed_dir(cfg) / "daily_clean.parquet")


def read_hourly(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_parquet(processed_dir(cfg) / "hourly_clean.parquet")

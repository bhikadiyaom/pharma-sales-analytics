"""Cached data access + global sidebar filters shared by every dashboard page."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, processed_dir, results_dir  # noqa: E402
from src.viz.theme import CSS  # noqa: E402


@st.cache_resource
def cfg() -> dict:
    return load_config()


@st.cache_data(show_spinner=False)
def daily() -> pd.DataFrame:
    return pd.read_parquet(processed_dir(cfg()) / "daily_clean.parquet")


@st.cache_data(show_spinner=False)
def hourly() -> pd.DataFrame:
    return pd.read_parquet(processed_dir(cfg()) / "hourly_clean.parquet")


@st.cache_data(show_spinner=False)
def result(name: str) -> pd.DataFrame:
    p = results_dir(cfg()) / name
    if not p.exists():
        st.error(f"Missing `{name}`. Run `make train` first (see README).")
        st.stop()
    df = pd.read_parquet(p) if name.endswith(".parquet") else pd.read_csv(p)
    for c in ("date", "origin"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


@st.cache_data(show_spinner=False)
def result_json(name: str) -> dict:
    return json.loads((results_dir(cfg()) / name).read_text(encoding="utf-8"))


def page_setup(title: str, icon: str = "💊") -> None:
    """Common page config, CSS and the data-source badge."""
    st.set_page_config(page_title=f"{title} | Pharma Analytics", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def data_badge() -> None:
    real = not cfg()["data"]["is_synthetic"]
    st.markdown(
        (
            '<span class="badge badge-real">REAL DATA · Kaggle Pharma Sales</span> '
            '<span class="badge badge-assume">Prices, costs, lead times = ASSUMPTIONS</span>'
            if real
            else '<span class="badge badge-assume">SYNTHETIC DATA</span>'
        ),
        unsafe_allow_html=True,
    )


def sidebar_filters() -> tuple[pd.DataFrame, list[str]]:
    """Global filters (date range, category/medicine, branch if present) persisted across pages.

    Returns the filtered daily frame and the selected categories.
    """
    df = daily()
    cats_all = sorted(df["category"].unique())
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    ss = st.session_state
    ss.setdefault("f_dates", (dmin, dmax))
    ss.setdefault("f_cats", cats_all)
    labels = cfg()["categories"]

    with st.sidebar:
        st.markdown("### Global filters")
        dates = st.date_input(
            "Date range",
            value=ss["f_dates"],
            min_value=dmin,
            max_value=dmax,
            key="w_dates",
            help="Applies to all pages. Forecasts always start after the last data day.",
        )
        cats = st.multiselect(
            "Medicine category (ATC)",
            cats_all,
            default=ss["f_cats"],
            key="w_cats",
            format_func=lambda c: f"{c} · {labels.get(c, '')}",
            help="In this dataset a 'medicine' is an ATC therapeutic class.",
        )
        if "branch" in df.columns:
            branches = st.multiselect(
                "Branch", sorted(df["branch"].unique()), default=sorted(df["branch"].unique())
            )
            df = df[df["branch"].isin(branches)]
        else:
            st.caption("Branch filter n/a: the dataset has a single sales point.")
        if isinstance(dates, (tuple, list)) and len(dates) == 2:
            ss["f_dates"] = tuple(dates)
        ss["f_cats"] = cats or cats_all

    start, end = ss["f_dates"]
    out = df[
        (df["date"] >= pd.Timestamp(start))
        & (df["date"] <= pd.Timestamp(end))
        & (df["category"].isin(ss["f_cats"]))
    ]
    if out.empty:
        st.warning("No data for the current filters.")
        st.stop()
    return out, list(ss["f_cats"])


def fmt_money(x: float) -> str:
    """Compact Indian-style amount (L = lakh, Cr = crore) with the configured currency."""
    cur = {"INR": "₹"}.get(cfg()["assumptions"]["currency"], cfg()["assumptions"]["currency"] + " ")
    if abs(x) >= 1e7:
        return f"{cur}{x / 1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"{cur}{x / 1e5:,.1f} L"
    return f"{cur}{x:,.0f}"

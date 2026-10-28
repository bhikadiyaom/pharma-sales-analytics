"""Consistent look & feel for the dashboard and report figures."""

from __future__ import annotations

import plotly.graph_objects as go

# Colour-blind-friendly categorical palette (Okabe-Ito based) keyed by ATC class.
CATEGORY_COLORS = {
    "M01AB": "#0072B2",
    "M01AE": "#56B4E9",
    "N02BA": "#009E73",
    "N02BE": "#117733",
    "N05B": "#E69F00",
    "N05C": "#D55E00",
    "R03": "#CC79A7",
    "R06": "#7F7F7F",
}
PRIMARY = "#0F5C8C"
ACCENT = "#E4572E"
GOOD, WARN, BAD = "#2E8B57", "#E69F00", "#C0392B"
STATUS_COLORS = {"OK": GOOD, "REORDER NOW": WARN, "STOCKOUT RISK": BAD, "OVERSTOCK": "#6C5CE7"}

CSS = """
<style>
div[data-testid="stMetric"] {background:#F4F8FB;border:1px solid #D9E4EC;border-radius:10px;padding:12px 14px;}
div[data-testid="stMetric"] label {color:#44566C;}
h1, h2, h3 {color:#0F2F4A;}
.badge {display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.78rem;font-weight:600;}
.badge-real {background:#DFF3E7;color:#1B6B3A;} .badge-assume {background:#FFF1D6;color:#8A5A00;}
.small-note {color:#5C6B7A;font-size:0.85rem;}
</style>
"""


def style_fig(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    """Apply the shared Plotly layout."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top") if title else None,
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#1F2D3D"),
        margin=dict(l=10, r=10, t=85 if title else 30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def color_for(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, PRIMARY)

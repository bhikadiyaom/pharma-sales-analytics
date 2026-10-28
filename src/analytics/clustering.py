"""Demand-class clustering of medicine categories (KMeans + hierarchical)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.analytics.descriptive import rule_based_label
from src.config import load_config

CLUSTER_FEATURES = [
    "mean_daily",
    "cv_weekly",
    "zero_share",
    "weekly_seasonality",
    "annual_seasonality",
    "weekend_ratio",
    "trend_slope_rel",
]


def cluster_demand(feats: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """KMeans (k by silhouette) + Ward hierarchical labels, PCA coordinates, rule label.

    With only a handful of series the clustering is exploratory: it is reported
    next to the interpretable rule-based label rather than instead of it.
    """
    cfg = cfg or load_config()
    X = StandardScaler().fit_transform(np.log1p(feats[CLUSTER_FEATURES].clip(lower=0)))
    k_lo, k_hi = cfg["clustering"]["k_range"]
    k_hi = min(k_hi, len(feats) - 1)
    best_k, best_s = k_lo, -1.0
    for k in range(k_lo, k_hi + 1):
        labels = KMeans(k, n_init=20, random_state=cfg["project"]["seed"]).fit_predict(X)
        s = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
        if s > best_s:
            best_k, best_s = k, s
    km = KMeans(best_k, n_init=20, random_state=cfg["project"]["seed"]).fit_predict(X)
    hc = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    pcs = PCA(2, random_state=cfg["project"]["seed"]).fit_transform(X)
    med = feats["mean_daily"].median()
    out = feats.copy()
    out["kmeans_cluster"] = km
    out["hier_cluster"] = hc
    out["pc1"], out["pc2"] = pcs[:, 0], pcs[:, 1]
    out["demand_label"] = [rule_based_label(r, med) for _, r in feats.iterrows()]
    out.attrs["k"], out.attrs["silhouette"] = best_k, best_s
    return out

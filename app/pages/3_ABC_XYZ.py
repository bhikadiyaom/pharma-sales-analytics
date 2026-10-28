"""ABC-XYZ inventory classification and demand-behaviour clusters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.analytics.descriptive import abc_xyz_matrix  # noqa: E402
from src.viz.app_data import cfg, data_badge, page_setup, result, result_json, sidebar_filters  # noqa: E402
from src.viz.theme import ACCENT, PRIMARY, style_fig  # noqa: E402

page_setup("ABC-XYZ", "🧮")
df, cats = sidebar_filters()
st.title("🧮 ABC-XYZ Analysis & Demand Clusters")
data_badge()

tab1, tab2 = st.tabs(["ABC-XYZ matrix", "Demand-behaviour clusters"])
with tab1:
    if len(cats) < 2:
        st.info("Select at least two categories for a meaningful classification.")
    else:
        m = abc_xyz_matrix(df, cfg())
        st.caption(
            "**ABC** = revenue contribution (Pareto; A ≈ top 80 %, B next 15 %, C rest; revenue uses assumed prices). "
            "**XYZ** = coefficient of variation of *weekly* demand (X stable, Y variable, Z erratic). "
            "Thresholds are in `config.yaml`."
        )
        a, b = st.columns([3, 2])
        with a:
            fig = go.Figure()
            fig.add_bar(x=m["category"], y=m["share"] * 100, name="Revenue share %", marker_color=PRIMARY)
            fig.add_scatter(
                x=m["category"],
                y=m["cum_share"] * 100,
                name="Cumulative %",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color=ACCENT),
            )
            for y_, lbl in [(80, "80 %"), (95, "95 %")]:
                fig.add_hline(y=y_, line_dash="dot", line_color="#888", yref="y2", annotation_text=lbl)
            fig.update_layout(
                yaxis=dict(title="Share %"),
                yaxis2=dict(overlaying="y", side="right", range=[0, 105], title="Cumulative %"),
            )
            st.plotly_chart(style_fig(fig, "Pareto chart (ABC)", 400), width="stretch")
        with b:
            grid = {(a_, x_): [] for a_ in "ABC" for x_ in "XYZ"}
            for r in m.itertuples():
                grid[(r.abc, r.xyz)].append(r.category)
            z = [[len(grid[(a_, x_)]) for x_ in "XYZ"] for a_ in "ABC"]
            text = [["<br>".join(grid[(a_, x_)]) or "–" for x_ in "XYZ"] for a_ in "ABC"]
            fig = go.Figure(
                go.Heatmap(
                    z=z,
                    x=list("XYZ"),
                    y=list("ABC"),
                    text=text,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False,
                    hoverinfo="skip",
                )
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(
                style_fig(fig, "ABC-XYZ matrix", 400).update_layout(hovermode=False), width="stretch"
            )
        st.subheader("Recommended stocking policy")
        show = m[["category", "class", "revenue", "share", "cv", "policy"]].copy()
        show["share"] = (show["share"] * 100).round(1)
        show["cv"] = show["cv"].round(2)
        show = show.rename(columns={"share": "revenue_share_%", "cv": "weekly_CV"})
        st.dataframe(show, width="stretch", hide_index=True)
        st.download_button(
            "⬇ Download classification (CSV)",
            show.to_csv(index=False).encode(),
            file_name="abc_xyz.csv",
            mime="text/csv",
        )
with tab2:
    cl = result("demand_clusters.csv")
    meta = result_json("clustering_meta.json")
    st.caption(
        f"KMeans with k={meta['k']} chosen by silhouette ({meta['silhouette']:.2f}); Ward hierarchical clustering "
        "shown for comparison. With only 8 series this is **exploratory**; the rule-based label "
        "(Erratic / Seasonal / Fast / Slow) is the interpretable result."
    )
    fig = px.scatter(
        cl,
        x="pc1",
        y="pc2",
        color=cl["kmeans_cluster"].astype(str),
        text="category",
        symbol="demand_label",
        labels={"color": "KMeans cluster", "pc1": "PC1", "pc2": "PC2"},
    )
    fig.update_traces(textposition="top center", marker_size=14)
    st.plotly_chart(
        style_fig(fig, "Demand-behaviour map (PCA of standardised features)", 440).update_layout(
            hovermode="closest"
        ),
        width="stretch",
    )
    cols = [
        "category",
        "demand_label",
        "kmeans_cluster",
        "hier_cluster",
        "mean_daily",
        "cv_weekly",
        "zero_share",
        "weekly_seasonality",
        "annual_seasonality",
        "weekend_ratio",
        "trend_slope_rel",
    ]
    st.dataframe(cl[cols].round(3), width="stretch", hide_index=True)

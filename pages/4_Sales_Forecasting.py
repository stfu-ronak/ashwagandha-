from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import load_data
from utils.metrics import compute_seasonal_demand_index
from utils.sidebar import render_sidebar
import utils.charts as charts

st.set_page_config(
    page_title="Sales Forecasting — Ashwagandha",
    page_icon="📈",
    layout="wide",
)

_css = Path(__file__).parent.parent / "assets" / "style.css"
if _css.exists():
    st.markdown(f"<style>{_css.read_text()}</style>", unsafe_allow_html=True)

df_full = load_data()
df = render_sidebar(df_full)

st.title("📈 Sales Forecasting & Seasonality")
st.caption("Revenue trends, ensemble forecasting, seasonal demand patterns, and deal economics")

if df.empty:
    st.warning("No data for current filters. Adjust the sidebar and try again.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Revenue Trend", "🔮 Forecast", "🌡️ Seasonality", "💼 Deal Economics",
])

# ── Tab 1: Revenue Trend ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Monthly Revenue vs Shipment Volume")
    st.plotly_chart(charts.dual_axis_revenue_volume(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Year-over-Year Quarterly Revenue Change")
    st.caption("Waterfall: starts at the earlier year's total, each bar = quarterly YoY delta, ends at the later year's total")
    st.plotly_chart(charts.yoy_waterfall(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Annual Revenue Summary")
    annual = (
        df.groupby("YEAR")
        .agg(Revenue=("USD FOB", "sum"), Shipments=("USD FOB", "count"), Volume_kg=("NET WEIGHT", "sum"))
        .reset_index()
    )
    annual["Revenue ($M)"] = (annual["Revenue"] / 1e6).round(2)
    annual["Volume (t)"] = (annual["Volume_kg"] / 1000).round(0).astype(int)
    annual["Avg Deal ($K)"] = (annual["Revenue"] / annual["Shipments"] / 1e3).round(1)
    annual["YoY Growth (%)"] = (annual["Revenue"].pct_change() * 100).round(1)
    st.dataframe(
        annual[["YEAR", "Revenue ($M)", "Shipments", "Volume (t)", "Avg Deal ($K)", "YoY Growth (%)"]],
        use_container_width=True,
        hide_index=True,
    )

# ── Tab 2: Forecast ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("3-Model Ensemble Revenue Forecast")
    st.caption(
        "**Ensemble forecast** = combining three independent models to produce a more reliable prediction than any single model alone. "
        "The shaded band = **80% confidence interval** (there's an 80% chance the actual monthly revenue will fall inside this range). "
        "The Dec 2025 Rain Nutrience spike ($118.9M) is excluded from model training — it was a one-time bulk pre-buy, not a recurring trend."
    )

    c1, c2, c3 = st.columns(3)
    c1.info(
        "**Model 1 — Holt-Winters (Damped)**\n"
        "A classic time-series method that learns trend + seasonality from past data. "
        "'Damped' means the trend slows down over time instead of shooting to infinity. "
        "Period = 12 months (captures annual cycles)."
    )
    c2.info(
        "**Model 2 — OLS + Fourier**\n"
        "OLS = Ordinary Least Squares regression (a straight-line trend). "
        "Fourier terms = sin/cos waves added to capture repeating seasonal patterns. "
        "2 harmonics means it captures the 2 dominant seasonal cycles per year."
    )
    c3.info(
        "**Model 3 — Naive Seasonal**\n"
        "Simplest model: uses the historical average for each calendar month, "
        "then scales it up/down based on whether recent months were above or below the year-prior. "
        "A useful sanity check against the more complex models."
    )

    with st.spinner("Running ensemble forecast — may take a few seconds..."):
        fig_fc, metrics_df = charts.ensemble_forecast(df)

    st.plotly_chart(fig_fc, use_container_width=True)

    if not metrics_df.empty:
        st.markdown("---")
        col_m, col_warn = st.columns([2, 3])
        with col_m:
            st.subheader("Holdout Evaluation (last 6 months, out-of-sample)")
            st.caption(
                "**Holdout evaluation** = the model was trained on earlier data, then tested on the last 6 months it never saw. "
                "**MAE** (Mean Absolute Error) = average prediction error in dollars per month — lower is better. "
                "**RMSE** (Root Mean Squared Error) = same idea but penalises large errors more heavily. "
                "**Out-of-sample** means these are genuine predictions, not fitted values."
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        with col_warn:
            st.warning(
                "⚠️ **Dec 2025 spike excluded from model fitting.** "
                "Rain Nutrience's $118.9M bulk pre-buy is a one-time anomaly. "
                "All models trained on a winsorized series (95th-pct cap) so this outlier "
                "does not distort seasonality or trend estimates. "
                "The annotation on the chart marks the actual spike."
            )
    else:
        st.info("Not enough data to compute holdout metrics.")

# ── Tab 3: Seasonality ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Monthly Revenue Heatmap — Month × Year")
    st.caption(
        "Colour scale capped at 95th percentile of all monthly values — "
        "prevents the Dec 2025 $118.9M spike from washing out the rest of the heatmap."
    )
    st.plotly_chart(charts.seasonality_heatmap(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Seasonal Demand Index (SDI)")
    st.caption(
        "SDI = (average revenue for that calendar month across all years) ÷ "
        "(average monthly revenue across all months) × 100.  "
        "SDI > 100 → above-average demand.  SDI < 100 → below-average demand."
    )
    sdi_df = compute_seasonal_demand_index(df)
    st.plotly_chart(charts.sdi_bar(sdi_df), use_container_width=True)

    if not sdi_df.empty:
        peak_row = sdi_df.loc[sdi_df["SDI"].idxmax()]
        trough_row = sdi_df.loc[sdi_df["SDI"].idxmin()]
        col_peak, col_trough = st.columns(2)
        with col_peak:
            st.success(
                f"**Peak demand month: {peak_row['Month_Name']}** "
                f"(SDI = {peak_row['SDI']:.0f}) — "
                "Inventory build and supplier pre-orders should land by October to capture Q4 volume."
            )
        with col_trough:
            st.error(
                f"**Demand trough: {trough_row['Month_Name']}** "
                f"(SDI = {trough_row['SDI']:.0f}) — "
                "Universal low season. Optimal window for supplier price negotiations and contract renewals."
            )

        st.markdown("---")
        st.subheader("SDI Reference Table")
        sdi_show = sdi_df[["Month_Name", "SDI"]].copy()
        sdi_show["SDI"] = sdi_show["SDI"].round(1)
        sdi_show["vs Avg"] = sdi_show["SDI"].apply(lambda v: f"+{v-100:.0f}%" if v >= 100 else f"{v-100:.0f}%")
        st.dataframe(sdi_show.rename(columns={"Month_Name": "Month"}), use_container_width=True, hide_index=True)

# ── Tab 4: Deal Economics ─────────────────────────────────────────────────────
with tab4:
    total_rev = df["USD FOB"].sum()
    total_ship = len(df)

    # Pareto stats
    sorted_deals = df.nlargest(51, "USD FOB")
    top51_rev = sorted_deals["USD FOB"].sum()
    top51_pct = top51_rev / total_rev * 100 if total_rev > 0 else 0.0

    st.subheader("Deal Size Distribution")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Shipments", f"{total_ship:,}")
    m2.metric("Total Revenue", f"${total_rev / 1e6:.1f}M")
    m3.metric("Top 51 Shipments Revenue", f"${top51_rev / 1e6:.1f}M")
    m4.metric("Top 51 = % of Revenue", f"{top51_pct:.0f}%")

    st.markdown(
        f"**💡 {len(sorted_deals):,} shipments ({len(sorted_deals) / total_ship * 100:.1f}% of all shipments) "
        f"account for {top51_pct:.0f}% (${top51_rev / 1e6:.1f}M) of total revenue.** "
        "Losing even a single mega-contract has outsized revenue impact."
    )

    st.plotly_chart(charts.deal_histogram(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Deal Size Pareto Table")

    _bins = [0, 1_000, 5_000, 25_000, 100_000, 500_000, float("inf")]
    _labels = ["<$1K", "$1K–5K", "$5K–25K", "$25K–100K", "$100K–500K", ">$500K"]

    pareto_rows = []
    for i, label in enumerate(_labels):
        lo, hi = _bins[i], _bins[i + 1]
        sub = df[(df["USD FOB"] > lo) & (df["USD FOB"] <= hi)]
        rev = sub["USD FOB"].sum()
        pareto_rows.append({
            "Deal Size Bucket": label,
            "Shipments": len(sub),
            "% Shipments": round(len(sub) / total_ship * 100 if total_ship > 0 else 0, 1),
            "Revenue ($M)": round(rev / 1e6, 2),
            "% Revenue": round(rev / total_rev * 100 if total_rev > 0 else 0, 1),
            "Avg Deal ($K)": round(sub["USD FOB"].mean() / 1e3, 1) if not sub.empty else 0.0,
        })

    pareto_df = pd.DataFrame(pareto_rows)
    pareto_df["Cum. % Revenue"] = pareto_df["% Revenue"].cumsum().round(1)
    st.dataframe(pareto_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Largest Individual Deals")
    top_deals = (
        df.nlargest(10, "USD FOB")[
            ["ARRIVAL DATE", "IMPORTER NAME", "IMPORTER COUNTRY", "EXPORTER NAME", "PRODUCT_TYPE", "USD FOB", "NET WEIGHT"]
        ]
        .copy()
    )
    top_deals["USD FOB ($M)"] = (top_deals["USD FOB"] / 1e6).round(2)
    top_deals["NET WEIGHT (t)"] = (top_deals["NET WEIGHT"] / 1000).round(1)
    top_deals["ARRIVAL DATE"] = top_deals["ARRIVAL DATE"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        top_deals.drop(columns=["USD FOB", "NET WEIGHT"]).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

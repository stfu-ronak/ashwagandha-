from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import load_data
from utils.sidebar import render_sidebar
import utils.charts as charts

st.set_page_config(
    page_title="Product Intelligence — Ashwagandha",
    page_icon="🌿",
    layout="wide",
)

_css = Path(__file__).parent.parent / "assets" / "style.css"
if _css.exists():
    st.markdown(f"<style>{_css.read_text()}</style>", unsafe_allow_html=True)

df_full = load_data()
df = render_sidebar(df_full)

st.title("🌿 Product Intelligence")
st.caption("Product mix, pricing dynamics, HS code analysis, and competitive positioning")

if df.empty:
    st.warning("No data for current filters. Adjust the sidebar and try again.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Product Mix", "💰 Pricing", "🔢 HS Codes", "🕸️ Competitive Radar",
])

# ── Tab 1: Product Mix ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Product Category Breakdown")
    st.caption(
        "Products are classified from shipment description text into 7 categories: "
        "**KSM-66 / Sensoril** = premium branded extract (highest price/kg). "
        "**Standardized Extract** = withanolide-standardized products. "
        "**Finished Dosage** = capsules/tablets/softgels ready for retail. "
        "**Organic Powder** = certified-organic bulk powder. "
        "**Raw Powder** = bulk commodity (lowest price/kg). "
        "**Root** = dried root material. "
        "**Other** = uncategorized. "
        "Click any segment to drill into its shipments ↓"
    )
    col_donut, col_bar = st.columns(2)
    with col_donut:
        st.plotly_chart(charts.product_category_donut(df), use_container_width=True)
    with col_bar:
        st.plotly_chart(charts.product_category_bar(df), use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Drill Down — Shipment-Level Data")
    _ptype_opts = ["— Select a product type —"] + sorted(df["PRODUCT_TYPE"].dropna().unique().tolist())
    _selected_ptype = st.selectbox(
        "Choose a product type to see its individual shipments:",
        options=_ptype_opts,
        key="ptype_drill_select",
    )
    if _selected_ptype != "— Select a product type —":
        _drill = df[df["PRODUCT_TYPE"] == _selected_ptype].copy()
        st.info(f"Showing **{len(_drill):,} shipments** for **{_selected_ptype}** — sorted by value (largest first).")
        _drill_display = _drill[[
            "ARRIVAL DATE", "IMPORTER NAME", "IMPORTER COUNTRY",
            "EXPORTER NAME", "USD FOB", "NET WEIGHT", "price_per_kg",
        ]].sort_values("USD FOB", ascending=False).head(300)
        _drill_display = _drill_display.rename(columns={
            "ARRIVAL DATE": "Date", "IMPORTER NAME": "Importer",
            "IMPORTER COUNTRY": "Country", "EXPORTER NAME": "Exporter",
            "USD FOB": "Value ($)", "NET WEIGHT": "Weight (kg)",
            "price_per_kg": "$/kg",
        })
        st.dataframe(_drill_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("KSM-66 vs Generic — Quarterly Trend")
        st.plotly_chart(charts.ksm_vs_generic_area(df), use_container_width=True)
    with col_right:
        st.subheader("Product Share by Top 8 Importers")
        st.plotly_chart(charts.product_share_stacked_bar(df, top_n_countries=8), use_container_width=True)

    st.markdown("---")
    st.subheader("Product Type Summary")
    prod_summary = (
        df.groupby("PRODUCT_TYPE")
        .agg(
            Revenue=("USD FOB", "sum"),
            Shipments=("USD FOB", "count"),
            Countries=("IMPORTER COUNTRY", "nunique"),
            Avg_Deal=("USD FOB", "mean"),
        )
        .reset_index()
        .rename(columns={"PRODUCT_TYPE": "Product Type"})
        .sort_values("Revenue", ascending=False)
    )
    total_rev = prod_summary["Revenue"].sum()
    prod_summary["Revenue ($M)"] = (prod_summary["Revenue"] / 1e6).round(2)
    prod_summary["Revenue Share (%)"] = (prod_summary["Revenue"] / total_rev * 100).round(1)
    prod_summary["Avg Deal ($K)"] = (prod_summary["Avg_Deal"] / 1e3).round(1)
    st.dataframe(
        prod_summary[["Product Type", "Revenue ($M)", "Revenue Share (%)", "Shipments", "Countries", "Avg Deal ($K)"]],
        use_container_width=True,
        hide_index=True,
    )

# ── Tab 2: Pricing ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Price Distribution by Product Type")
    st.caption(
        "**Price per kg** = USD FOB value ÷ net weight in kg. Tells you the unit price of each shipment. "
        "**Violin chart** = shows the full distribution (spread, peaks, tails) not just an average — wide = many shipments at that price. "
        "**$/kg** is a proxy for product quality: KSM-66 commands a premium (~$250–500/kg) vs Raw Powder (~$10–50/kg). "
        "Y-axis capped at $500/kg — outliers shown as individual points above the violin."
    )
    st.plotly_chart(charts.price_violin(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Monthly Price Trend")
    st.plotly_chart(charts.price_trend_line(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Price Summary by Category")
    price_sub = df[df["price_per_kg"].notna() & (df["price_per_kg"] > 0)].copy()
    if not price_sub.empty:
        raw_med = price_sub[price_sub["PRODUCT_TYPE"] == "Raw Powder"]["price_per_kg"].median()
        price_tbl = (
            price_sub.groupby("PRODUCT_TYPE")["price_per_kg"]
            .agg(
                Median="median",
                Mean="mean",
                StdDev="std",
                Count="count",
            )
            .round(1)
            .reset_index()
            .rename(columns={"PRODUCT_TYPE": "Product Type", "Count": "Sample Size"})
            .sort_values("Median", ascending=False)
        )
        if raw_med and raw_med > 0:
            price_tbl["vs Raw Powder (%)"] = ((price_tbl["Median"] / raw_med - 1) * 100).round(1)
        st.dataframe(price_tbl, use_container_width=True, hide_index=True)
    else:
        st.info("No price data available for current filters.")

# ── Tab 3: HS Codes ───────────────────────────────────────────────────────────
with tab3:
    st.caption(
        "**HS Code** (Harmonized System Code) = an internationally standardised 6-digit number used by customs agencies worldwide "
        "to classify traded goods. Every shipment declares an HS code. "
        "Key ashwagandha codes: **1302** = Vegetable Extracts (most ashwagandha extract), "
        "**1211** = Plants/Plant Parts (dried root/raw material), "
        "**0910** = Spices/Roots, **2106** = Food Preparations, **3004** = Medicaments (finished dosage). "
        "The first 4 digits are the 'heading' — companies sometimes choose different headings for the same product."
    )
    col_hs_bar, col_hs_heat = st.columns(2)
    with col_hs_bar:
        st.subheader("Top 10 HS Codes by Revenue")
        st.plotly_chart(charts.hs_bar(df, top_n=10), use_container_width=True)
    with col_hs_heat:
        st.subheader("HS Code × Country Revenue Heatmap")
        st.plotly_chart(charts.hs_country_heatmap(df), use_container_width=True)

    st.markdown("---")
    st.subheader("HS Code Reference Table")
    hs_tbl = (
        df.groupby("HS CODE")
        .agg(
            Label=("HS_GROUP_LABEL", lambda x: x.mode().iloc[0] if not x.empty else "Other"),
            Revenue=("USD FOB", "sum"),
            Shipments=("USD FOB", "count"),
            Countries=("IMPORTER COUNTRY", "nunique"),
            Avg_Deal=("USD FOB", "mean"),
        )
        .reset_index()
        .rename(columns={"HS CODE": "HS Code"})
        .nlargest(25, "Revenue")
    )
    total_hs_rev = hs_tbl["Revenue"].sum()
    hs_tbl["Revenue ($M)"] = (hs_tbl["Revenue"] / 1e6).round(2)
    hs_tbl["Revenue Share (%)"] = (hs_tbl["Revenue"] / total_hs_rev * 100).round(1)
    hs_tbl["Avg Deal ($K)"] = (hs_tbl["Avg_Deal"] / 1e3).round(1)
    hs_tbl = hs_tbl.drop(columns=["Revenue", "Avg_Deal"]).sort_values("Revenue ($M)", ascending=False)
    st.dataframe(hs_tbl, use_container_width=True, hide_index=True)

# ── Tab 4: Competitive Radar ──────────────────────────────────────────────────
with tab4:
    st.subheader("Competitive Radar — Top 5 Exporters")
    st.caption(
        "**Competitive Radar** (spider/radar chart) = each axis represents one dimension of competitiveness, all normalised to 0–1 "
        "(1 = best performer on that dimension). The larger the shaded area, the stronger the overall competitive position. "
        "Dimensions: **Revenue** (total trade value) · **Market Reach** (number of import countries served) · "
        "**Deal Scale** (average shipment size) · **Growth Rate** (**CAGR** = Compound Annual Growth Rate, 2023→2025) · "
        "**Product Diversity** (number of distinct product types) · **Price Premium** (median $/kg — higher = more premium product)."
    )
    n_exporters = min(5, df["EXPORTER NAME"].nunique())
    if n_exporters < 2:
        st.info("Need at least 2 exporters in the filtered data for a radar chart.")
    else:
        st.plotly_chart(charts.competitive_radar(df, top_n=n_exporters), use_container_width=True)

    st.markdown("---")
    st.subheader("Top Exporter Comparison Table")
    top5 = df.groupby("EXPORTER NAME")["USD FOB"].sum().nlargest(5).index.tolist()
    exp_rows = []
    for exp in top5:
        grp = df[df["EXPORTER NAME"] == exp]
        yoy = grp.groupby("YEAR")["USD FOB"].sum().sort_index()
        cagr = 0.0
        if len(yoy) >= 2 and yoy.iloc[0] > 0:
            n_yrs = max(int(yoy.index[-1]) - int(yoy.index[0]), 1)
            cagr = float((yoy.iloc[-1] / yoy.iloc[0]) ** (1 / n_yrs) - 1)
        price_med = grp["price_per_kg"].dropna().median()
        exp_rows.append({
            "Exporter": exp,
            "Country": grp["EXPORTER COUNTRY"].mode().iloc[0] if not grp.empty else "",
            "Revenue ($M)": round(grp["USD FOB"].sum() / 1e6, 2),
            "Shipments": len(grp),
            "Import Countries": grp["IMPORTER COUNTRY"].nunique(),
            "Avg Deal ($K)": round(grp["USD FOB"].mean() / 1e3, 1),
            "Median $/kg": round(float(price_med), 1) if not pd.isna(price_med) else None,
            "CAGR (%)": round(cagr * 100, 1),
            "Product Types": grp["PRODUCT_TYPE"].nunique(),
        })
    st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, hide_index=True)

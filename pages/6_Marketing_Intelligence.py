from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from utils.data_loader import load_data
from utils.sidebar import render_sidebar
import utils.charts as charts

st.set_page_config(
    page_title="Marketing Intelligence — Ashwagandha",
    page_icon="📣",
    layout="wide",
)

_css = Path(__file__).parent.parent / "assets" / "style.css"
if _css.exists():
    st.markdown(f"<style>{_css.read_text()}</style>", unsafe_allow_html=True)

df_full = load_data()
df = render_sidebar(df_full)

st.title("📣 Marketing Intelligence")
st.caption("Brand landscape, channel strategy, transport economics, and competitive positioning")

if df.empty:
    st.warning("No data for current filters. Adjust the sidebar and try again.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🏷️ Brand Landscape",
    "📦 Channel Analysis",
    "🚚 Transport Economics",
    "🥊 Competitive Positioning",
])

# ── Tab 1: Brand Landscape ────────────────────────────────────────────────────
with tab1:
    st.subheader("Buyer Brand Landscape")
    st.caption(
        "Bubble chart shows importer brands (single-entity names, no slash/comma). "
        "X-axis (log): average deal size. Y-axis: shipment frequency. Bubble size: total revenue. "
        "Top-right corner = high-value, repeat buyers."
    )

    st.plotly_chart(charts.brand_bubble(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Top 15 Brands by Revenue")

    brand_tbl = (
        df[~df['IMPORTER NAME'].str.contains('[,/]', na=True)]
        .groupby('IMPORTER NAME')
        .agg(
            Revenue=('USD FOB', 'sum'),
            Shipments=('USD FOB', 'count'),
            Avg_Deal=('USD FOB', 'mean'),
        )
        .nlargest(15, 'Revenue').reset_index()
    )
    total_rev = df['USD FOB'].sum()
    brand_tbl['Revenue ($M)'] = (brand_tbl['Revenue'] / 1e6).round(2)
    brand_tbl['Share (%)'] = (brand_tbl['Revenue'] / total_rev * 100).round(1)
    brand_tbl['Avg Deal ($K)'] = (brand_tbl['Avg_Deal'] / 1e3).round(1)
    brand_tbl['Cum. Share (%)'] = brand_tbl['Share (%)'].cumsum().round(1)
    st.dataframe(
        brand_tbl[['IMPORTER NAME', 'Revenue ($M)', 'Share (%)', 'Cum. Share (%)', 'Shipments', 'Avg Deal ($K)']],
        use_container_width=True, hide_index=True,
    )


# ── Tab 2: Channel Analysis ───────────────────────────────────────────────────
with tab2:
    st.subheader("B2C vs B2B Channel Segmentation")
    st.caption(
        "**B2B** = Business-to-Business (one company selling to another company — e.g., Indian exporter → UK supplement brand). "
        "**B2C** = Business-to-Consumer (selling directly to end consumers — e.g., Amazon FBA shipments). "
        "Channel is inferred from deal size and importer name: "
        "**B2C** = deal < $1K or 'FBA'/'Amazon' in buyer name. "
        "**B2B Retail** = $1K–$25K (small distributor or retail chain). "
        "**B2B Mid** = $25K–$500K (mid-size manufacturer). "
        "**B2B Mega** = > $500K (large industrial contract). "
        "**FOB** (Free On Board) = the declared export price at the port of departure — excludes freight and insurance."
    )

    # Compute channel stats for the callout
    df_ch = charts._add_channel(df)
    ch_rev = df_ch.groupby('CHANNEL')['USD FOB'].sum()
    ch_cnt = df_ch.groupby('CHANNEL').size()
    total_rev_ch = ch_rev.sum()
    total_ship_ch = ch_cnt.sum()
    mega_rev_pct = ch_rev.get('B2B Mega', 0) / total_rev_ch * 100 if total_rev_ch > 0 else 0
    mega_ship_pct = ch_cnt.get('B2B Mega', 0) / total_ship_ch * 100 if total_ship_ch > 0 else 0

    st.success(
        f"💡 **B2B Mega ({mega_ship_pct:.1f}% of shipments) = {mega_rev_pct:.0f}% of revenue.** "
        "A tiny number of mega-contracts dominates total trade value. "
        "Losing one mega buyer has catastrophic revenue impact."
    )

    col_pie, col_bar = st.columns(2)
    with col_pie:
        st.plotly_chart(charts.channel_pie(df), use_container_width=True)
    with col_bar:
        st.plotly_chart(charts.channel_revenue_bar(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Channel Mix Shift by Quarter")
    st.caption("100% stacked area — tracks how revenue mix across channels has evolved over time.")
    st.plotly_chart(charts.channel_stacked_area(df), use_container_width=True)

    # Summary table
    st.markdown("---")
    st.subheader("Channel Summary Table")
    ch_summary = df_ch.groupby('CHANNEL').agg(
        Shipments=('USD FOB', 'count'),
        Revenue=('USD FOB', 'sum'),
        Avg_Deal=('USD FOB', 'mean'),
    ).reset_index()
    ch_summary['Revenue ($M)'] = (ch_summary['Revenue'] / 1e6).round(2)
    ch_summary['Rev Share (%)'] = (ch_summary['Revenue'] / total_rev_ch * 100).round(1)
    ch_summary['Ship Share (%)'] = (ch_summary['Shipments'] / total_ship_ch * 100).round(1)
    ch_summary['Avg Deal ($K)'] = (ch_summary['Avg_Deal'] / 1e3).round(1)
    st.dataframe(
        ch_summary[['CHANNEL', 'Shipments', 'Ship Share (%)', 'Revenue ($M)', 'Rev Share (%)', 'Avg Deal ($K)']],
        use_container_width=True, hide_index=True,
    )


# ── Tab 3: Transport Economics ────────────────────────────────────────────────
with tab3:
    st.subheader("Transport Mode Economics")
    st.caption(
        "Revenue and shipment counts by transport mode. "
        "**Air** = airfreight (fast, expensive, used for high-value/time-sensitive goods like KSM-66 or finished capsules). "
        "**Sea** = ocean freight (slow, cheap per kg, used for bulk raw powder). "
        "**Road** = land transport (common for shipments to neighbouring countries). "
        "**ICD** = Inland Container Depot — a dry port / inland customs clearance facility, common in India for rail-to-sea transfers. "
        "**Click any bar** to see the individual shipments for that mode."
    )

    mode_stats = df.groupby('TRANSPORT_NORM').agg(
        Revenue=('USD FOB', 'sum'),
        Shipments=('USD FOB', 'count'),
        Avg_Deal=('USD FOB', 'mean'),
    ).reset_index()

    m1, m2, m3, m4 = st.columns(4)
    top_mode_rev = mode_stats.loc[mode_stats['Revenue'].idxmax(), 'TRANSPORT_NORM']
    top_mode_ship = mode_stats.loc[mode_stats['Shipments'].idxmax(), 'TRANSPORT_NORM']
    air_row = mode_stats[mode_stats['TRANSPORT_NORM'] == 'Air']
    sea_row = mode_stats[mode_stats['TRANSPORT_NORM'] == 'Sea']
    m1.metric("Top Mode by Revenue", top_mode_rev,
              help="The transport mode (Air/Sea/Road/ICD) with the highest total USD FOB revenue across all filtered shipments.")
    m2.metric("Top Mode by Shipments", top_mode_ship,
              help="The transport mode with the highest number of individual shipment records (bill-of-lading entries).")
    m3.metric("Air Avg Deal ($K)", f"${air_row['Avg_Deal'].values[0] / 1e3:.0f}K" if not air_row.empty else "N/A",
              help="Average USD FOB value per airfreight shipment. Higher average deal = premium products (KSM-66, finished capsules) shipped by air for speed.")
    m4.metric("Sea Avg Deal ($K)", f"${sea_row['Avg_Deal'].values[0] / 1e3:.0f}K" if not sea_row.empty else "N/A",
              help="Average USD FOB value per sea freight shipment. Sea shipments are typically bulk raw powder — large volume, lower price per kg.")

    if not air_row.empty and not sea_row.empty:
        air_avg = float(air_row['Avg_Deal'].values[0])
        sea_avg = float(sea_row['Avg_Deal'].values[0])
        if air_avg > sea_avg:
            st.info(
                f"✈️ **Air shipments average ${air_avg / 1e3:.0f}K vs Sea's ${sea_avg / 1e3:.0f}K.** "
                f"Air premium = {air_avg / sea_avg:.1f}× — consistent with premium branded products "
                "(KSM-66, finished dosage) going by air."
            )

    st.info("💡 Click any bar below to drill into shipments for that transport mode.")
    _transport_fig = charts.transport_mode_bars(df)
    ev_transport = st.plotly_chart(
        _transport_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="transport_bar_select",
    )

    _selected_mode: str | None = None
    if ev_transport and ev_transport.selection and ev_transport.selection.points:
        _pt = ev_transport.selection.points[0]
        _selected_mode = _pt.get("x") or _pt.get("label")

    if _selected_mode:
        _mode_df = df[df["TRANSPORT_NORM"] == _selected_mode].copy()
        st.markdown(f"#### 🔍 Shipments via {_selected_mode} ({len(_mode_df):,} records)")
        _mode_display = _mode_df[[
            "ARRIVAL DATE", "IMPORTER NAME", "IMPORTER COUNTRY",
            "EXPORTER NAME", "USD FOB", "NET WEIGHT", "PRODUCT_TYPE",
        ]].sort_values("USD FOB", ascending=False).head(200)
        _mode_display = _mode_display.rename(columns={
            "ARRIVAL DATE": "Date", "IMPORTER NAME": "Importer",
            "IMPORTER COUNTRY": "Country", "EXPORTER NAME": "Exporter",
            "USD FOB": "Value ($)", "NET WEIGHT": "Weight (kg)",
            "PRODUCT_TYPE": "Product",
        })
        st.dataframe(_mode_display, use_container_width=True, hide_index=True)
        st.markdown("---")

    st.subheader("Revenue Share Shift by Transport Mode (YoY)")
    st.plotly_chart(charts.transport_yoy_bar(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Transport Mode Detail Table")
    mode_stats['Revenue ($M)'] = (mode_stats['Revenue'] / 1e6).round(2)
    mode_stats['Rev Share (%)'] = (mode_stats['Revenue'] / mode_stats['Revenue'].sum() * 100).round(1)
    mode_stats['Avg Deal ($K)'] = (mode_stats['Avg_Deal'] / 1e3).round(1)
    st.dataframe(
        mode_stats[['TRANSPORT_NORM', 'Revenue ($M)', 'Rev Share (%)', 'Shipments', 'Avg Deal ($K)']]
        .rename(columns={'TRANSPORT_NORM': 'Transport Mode'})
        .sort_values('Revenue ($M)', ascending=False),
        use_container_width=True, hide_index=True,
    )


# ── Tab 4: Competitive Positioning ────────────────────────────────────────────
with tab4:
    st.subheader("Exporter Competitive Positioning")
    st.caption(
        "Price/kg (Y) vs total shipment volume (X, log). "
        "Bubble size = total revenue. Color = exporter country. "
        "Dashed lines at median price and median volume divide the market into 4 strategic zones."
    )

    st.plotly_chart(charts.competitive_positioning_scatter(df), use_container_width=True)

    exp_price = df[df['price_per_kg'].notna()].groupby('EXPORTER NAME').agg(
        median_price=('price_per_kg', 'median'),
        total_vol_kg=('NET WEIGHT', 'sum'),
        total_rev=('USD FOB', 'sum'),
        country=('EXPORTER COUNTRY', 'first'),
    ).reset_index()
    if not exp_price.empty:
        premium_exp = exp_price.loc[exp_price['median_price'].idxmax()]
        st.info(
            f"💎 **Highest price/kg exporter: {premium_exp['EXPORTER NAME']} "
            f"({premium_exp['country']})** — "
            f"median ${premium_exp['median_price']:.0f}/kg. "
            "Premium pricing indicates KSM-66 or finished product focus."
        )

    st.markdown("---")
    st.subheader("Exporter Market Share Treemap")
    st.caption("Top 20 exporters by total revenue. Size ∝ revenue share.")
    st.plotly_chart(charts.exporter_share_treemap(df, top_n=20), use_container_width=True)

    st.markdown("---")
    st.subheader("Exporter Competitive Summary (Top 20)")
    exp_summary = (
        df.groupby(['EXPORTER NAME', 'EXPORTER COUNTRY'])
        .agg(
            Revenue=('USD FOB', 'sum'),
            Shipments=('USD FOB', 'count'),
            Avg_Deal=('USD FOB', 'mean'),
            Markets=('IMPORTER COUNTRY', 'nunique'),
        )
        .nlargest(20, 'Revenue').reset_index()
    )
    total_rev = df['USD FOB'].sum()
    exp_summary['Revenue ($M)'] = (exp_summary['Revenue'] / 1e6).round(2)
    exp_summary['Share (%)'] = (exp_summary['Revenue'] / total_rev * 100).round(1)
    exp_summary['Avg Deal ($K)'] = (exp_summary['Avg_Deal'] / 1e3).round(1)

    price_map = (
        df[df['price_per_kg'].notna()]
        .groupby('EXPORTER NAME')['price_per_kg'].median().round(0)
    )
    exp_summary['Med $/kg'] = exp_summary['EXPORTER NAME'].map(price_map)

    st.dataframe(
        exp_summary[['EXPORTER NAME', 'EXPORTER COUNTRY', 'Revenue ($M)', 'Share (%)',
                     'Shipments', 'Avg Deal ($K)', 'Markets', 'Med $/kg']]
        .rename(columns={'EXPORTER NAME': 'Exporter', 'EXPORTER COUNTRY': 'Country'}),
        use_container_width=True, hide_index=True,
    )

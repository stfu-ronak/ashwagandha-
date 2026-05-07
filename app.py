from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_data
from utils.sidebar import render_sidebar
from utils.metrics import (
    compute_cagr,
    compute_concentration_ratio,
    compute_gini,
    compute_hhi,
    hhi_label,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ashwagandha Trade Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
_css_path = Path(__file__).parent / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────────────────────
df_full = load_data()

# ── Sidebar + filters ─────────────────────────────────────────────────────────
df = render_sidebar(df_full)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🌿 Ashwagandha Global Trade Intelligence</h1>
  <p>15,580+ shipments &nbsp;|&nbsp; 2023–2026 &nbsp;|&nbsp; 40+ importing countries &nbsp;|&nbsp; Powered by TradeAtlas</p>
</div>
""", unsafe_allow_html=True)

# ── KPI calculations ─────────────────────────────────────────────────────────
total_rev = df['USD FOB'].sum()

rev_by_year = df_full.groupby('YEAR')['USD FOB'].sum()
rev_2023 = rev_by_year.get(2023, 0)
rev_2025 = rev_by_year.get(2025, 0)
rev_2025_filtered = df[df['YEAR'] == 2025]['USD FOB'].sum()
cagr = compute_cagr(rev_2023, rev_2025, 2) if rev_2023 > 0 else 0.0

total_shipments = len(df)

buyer_rev = df.groupby('IMPORTER NAME')['USD FOB'].sum()
hhi = compute_hhi(buyer_rev)
gini = compute_gini(buyer_rev)

top_importer = (
    df.groupby('IMPORTER COUNTRY')['USD FOB'].sum().idxmax()
    if not df.empty else '—'
)
top_exporter = (
    df.groupby('EXPORTER COUNTRY')['USD FOB'].sum().idxmax()
    if not df.empty else '—'
)

# ── Row 1: Revenue KPIs ──────────────────────────────────────────────────────
st.markdown("#### Revenue Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"${total_rev / 1e6:.1f}M",
          help="Total USD FOB (Free On Board) value of all shipments in the filtered period. FOB excludes freight and insurance costs — it's the ex-factory export price.")
c2.metric("CAGR 2023–2025", f"{cagr * 100:.1f}%", delta="vs flat baseline",
          help="CAGR = Compound Annual Growth Rate. Measures how fast revenue grew per year on average. Formula: (End ÷ Start)^(1 ÷ years) − 1. A 13% CAGR means revenue roughly doubles every 5.5 years.")
c3.metric("2025 Revenue", f"${rev_2025_filtered / 1e6:.1f}M",
          help="Total revenue from shipments arriving in calendar year 2025. Includes the Dec 2025 Rain Nutrience mega-deal (~$119M), which skews this figure significantly upward.")
c4.metric("Total Shipments", f"{total_shipments:,}",
          help="Number of individual bill-of-lading records (shipments) in the filtered dataset. Each row represents one shipment from an exporter to an importer.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: Concentration KPIs ────────────────────────────────────────────────
st.markdown("#### Market Concentration")
c5, c6, c7, c8 = st.columns(4)
c5.metric("Buyer HHI", f"{hhi:,.0f}", delta=hhi_label(hhi), delta_color="off",
          help="HHI = Herfindahl-Hirschman Index. Standard measure of market concentration. Calculated as the sum of (each buyer's market share %)². Range: 0–10,000. Below 1,500 = competitive market; 1,500–2,500 = moderate; above 2,500 = concentrated; above 5,000 = highly concentrated. This market is above 7,000.")
c6.metric("Gini Coefficient", f"{gini:.2f}", delta="1.0 = max inequality", delta_color="off",
          help="Gini Coefficient measures revenue inequality across buyers. 0.0 = every buyer spends the same amount. 1.0 = one buyer accounts for all revenue. 0.92 here means extreme concentration — a handful of mega-buyers control almost all the trade value.")
c7.metric("Top Importer", top_importer[:22] if len(top_importer) > 22 else top_importer,
          help="The importing country with the highest total FOB revenue in the filtered period. 'Importer' = the country receiving the ashwagandha shipment.")
c8.metric("Top Exporter", top_exporter[:22] if len(top_exporter) > 22 else top_exporter,
          help="The exporting country with the highest total FOB revenue in the filtered period. India dominates as the near-exclusive source of ashwagandha globally.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 3: Monthly revenue sparkline ─────────────────────────────────────────
st.markdown("#### Monthly Revenue Trend")
monthly = df.groupby('YM_STR')['USD FOB'].sum().reset_index().sort_values('YM_STR')

fig_spark = go.Figure()
fig_spark.add_trace(go.Bar(
    x=monthly['YM_STR'],
    y=monthly['USD FOB'] / 1e6,
    marker_color='#2e9e5b',
    marker_line_width=0,
    name='Revenue ($M)',
))
fig_spark.update_layout(
    height=220,
    margin=dict(l=0, r=0, t=10, b=40),
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    xaxis=dict(showgrid=False, tickangle=45, tickfont=dict(size=9)),
    yaxis=dict(showgrid=True, gridcolor='#e8f5e9', title='$M'),
    showlegend=False,
)
st.plotly_chart(fig_spark, use_container_width=True)

# ── Row 4: Callout boxes ─────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    top2_share = compute_concentration_ratio(buyer_rev, 2) * 100
    st.markdown(f"""
<div class="callout-alert">
  <div class="callout-title">⚠️ Buyer Concentration Alert</div>
  <strong>Rain Nutrience Ltd (UK)</strong> alone accounts for ~26% of all revenue
  ($102.9M from just 3 shipments, avg $34.3M each).<br>
  Top 2 buyers = <strong>{top2_share:.1f}%</strong> of filtered revenue.
  Gini = <strong>{gini:.2f}</strong> — extreme inequality.
</div>
""", unsafe_allow_html=True)

with col_b:
    india_exp_share = (
        df[df['EXPORTER COUNTRY'] == 'India']['USD FOB'].sum() / total_rev * 100
        if total_rev > 0 else 0
    )
    st.markdown(f"""
<div class="callout-insight">
  <div class="callout-title">💡 Market Insight</div>
  <strong>India</strong> supplies <strong>{india_exp_share:.0f}%</strong> of global ashwagandha
  exports in this dataset — near-total supply-side dominance.<br>
  The UK imports at extraordinary unit values ($~25K/kg avg),
  while the US leads on pure volume.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 5: Navigation guide ───────────────────────────────────────────────────
st.markdown("#### Explore the Dashboard")
nav_cols = st.columns(6)
_nav_items = [
    ("🗺️", "Geographic Intelligence",  "Import/Export maps, port activity, country rankings"),
    ("📊", "Market Concentration",      "HHI, Gini, buyer segmentation, Pareto analysis, risk"),
    ("🧪", "Product Intelligence",      "Product mix, pricing, HS codes, competitive radar"),
    ("📈", "Sales Forecasting",         "Ensemble forecast, seasonality, deal economics"),
    ("🤝", "Business Development",      "Trade flows, opportunity matrix, supply risk, white space"),
    ("📣", "Marketing Intelligence",    "Brand landscape, channel analysis, transport economics"),
]
for col, (icon, title, desc) in zip(nav_cols, _nav_items):
    with col:
        st.markdown(f"""
<div class="nav-card">
  <div class="nav-icon">{icon}</div>
  <div class="nav-title">{title}</div>
  <div class="nav-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Use the sidebar pages (▶ Pages) to navigate. Sidebar filters apply to all pages.")

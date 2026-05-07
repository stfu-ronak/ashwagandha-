from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import load_data
from utils.metrics import (
    compute_concentration_ratio,
    compute_gini,
    compute_hhi,
    hhi_label,
)
from utils.sidebar import render_sidebar
import utils.charts as charts

st.set_page_config(
    page_title="Market Concentration — Ashwagandha",
    page_icon="📊",
    layout="wide",
)

_css = Path(__file__).parent.parent / "assets" / "style.css"
if _css.exists():
    st.markdown(f"<style>{_css.read_text()}</style>", unsafe_allow_html=True)

df_full = load_data()
df = render_sidebar(df_full)

st.title("📊 Market Concentration Intelligence")
st.caption("Buyer & country concentration analysis — HHI, Gini, Lorenz curve, segmentation, revenue at risk")

if df.empty:
    st.warning("No data for current filters. Adjust the sidebar and try again.")
    st.stop()

# ── Core metrics ──────────────────────────────────────────────────────────────
buyer_rev   = df.groupby('IMPORTER NAME')['USD FOB'].sum()
total_rev   = buyer_rev.sum()
hhi         = compute_hhi(buyer_rev)
gini        = compute_gini(buyer_rev)
label       = hhi_label(hhi)
cr2         = compute_concentration_ratio(buyer_rev, 2) * 100

all_sorted  = buyer_rev.sort_values(ascending=False)
cum_pct_arr = (all_sorted.cumsum() / total_rev * 100).values
cross_80    = np.where(cum_pct_arr >= 80)[0]
n_at_80     = int(cross_80[0]) + 1 if len(cross_80) > 0 else len(all_sorted)

# ── ROW 1: Concentration KPIs ─────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    fig_gauge = charts.hhi_gauge(hhi, label)
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.caption("HHI measures buyer concentration (0–10,000). Below 1,500 = competitive; above 2,500 = concentrated; above 5,000 = highly concentrated.")

with col2:
    delta_from_eq = gini - 0.5
    st.metric(
        "Gini Coefficient",
        f"{gini:.2f}",
        delta=f"{delta_from_eq:+.2f} vs perfect equality (0.5)",
        delta_color="inverse",
        help="Gini Coefficient measures revenue inequality across buyers. 0.0 = every buyer spends the same. 1.0 = one buyer has all the revenue. 0.92 = extreme concentration — a tiny number of buyers control almost all trade value.",
    )
    st.caption("1.0 = absolute inequality. 0.92 = extreme buyer concentration.")

with col3:
    st.metric(
        "Buyers at 80% Revenue",
        f"{n_at_80} buyers",
        delta=f"of {len(buyer_rev):,} total buyers",
        delta_color="off",
        help="How many of the top buyers are needed to account for 80% of total revenue. In a healthy market this would be ~20% of all buyers (the 80/20 rule). Here it's far fewer — indicating extreme concentration.",
    )
    st.caption(f"Top {n_at_80} buyers ({n_at_80/len(buyer_rev)*100:.1f}% of all) drive 80% of revenue.")

with col4:
    st.metric(
        "Top 2 Buyers",
        f"{cr2:.1f}% of revenue",
        delta="Rain Nutrience + Munindra",
        delta_color="off",
        help="CR-2 = Concentration Ratio for the top 2 buyers. Shows the combined revenue share of just the two largest buyers. Above 40% is considered dangerously concentrated in most industries.",
    )
    st.caption("Rain Nutrience ($102.9M) + Munindra ($76M) = near-total dependence on 2 entities.")

# ── ROW 2: HHI Trend ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### HHI Trend by Quarter")
st.caption("HHI = Herfindahl-Hirschman Index. Tracks buyer concentration over time. Reference lines: below 1,500 = competitive; above 2,500 = concentrated.")
hhi_rows = []
for yq, grp in df.groupby('YQ'):
    brev_q = grp.groupby('IMPORTER NAME')['USD FOB'].sum()
    hhi_rows.append({'YQ': yq, 'HHI': compute_hhi(brev_q)})
df_hhi = pd.DataFrame(hhi_rows).sort_values('YQ').reset_index(drop=True)
if not df_hhi.empty:
    fig_trend = charts.hhi_trend_line(df_hhi)
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Insufficient quarterly data.")

# ── ROW 3: Buyer Segmentation ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("Buyer Segmentation (K-Means, 3 Clusters)")
st.caption(
    "**K-Means clustering** = a machine-learning algorithm that groups buyers into 3 clusters based on their behaviour. "
    "Here it uses two signals: how large each buyer's average shipment is (deal size) and how often they buy (shipment frequency). "
    "The algorithm finds natural groupings without being told the labels — it discovers them from the data. "
    "Clusters are then labelled by their revenue: **Mega B2B** (highest revenue), **Mid-Market**, **Small/Retail**."
)

buyer_scatter = (
    df.groupby(['IMPORTER NAME', 'BUYER_SEGMENT'])
    .agg(
        avg_deal=('USD FOB', 'mean'),
        freq=('USD FOB', 'count'),
        total_rev=('USD FOB', 'sum'),
    )
    .reset_index()
)

col_scatter, col_seg_table = st.columns([2, 1])
with col_scatter:
    fig_seg = charts.segmentation_scatter(buyer_scatter)
    st.plotly_chart(fig_seg, use_container_width=True)

with col_seg_table:
    seg_summary = (
        buyer_scatter.groupby('BUYER_SEGMENT')
        .agg(
            Buyers=('IMPORTER NAME', 'count'),
            Revenue=('total_rev', 'sum'),
            Avg_Deal=('avg_deal', 'mean'),
        )
        .reset_index()
        .rename(columns={'BUYER_SEGMENT': 'Segment'})
        .sort_values('Revenue', ascending=False)
    )
    seg_summary['Revenue ($M)']  = (seg_summary['Revenue'] / 1e6).round(1)
    seg_summary['Avg Deal ($K)'] = (seg_summary['Avg_Deal'] / 1e3).round(1)
    seg_summary['Rev Share %']   = (seg_summary['Revenue'] / total_rev * 100).round(1)

    st.markdown("**Segment Summary**")
    st.dataframe(
        seg_summary[['Segment', 'Buyers', 'Revenue ($M)', 'Avg Deal ($K)', 'Rev Share %']],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "K-Means on log(avg deal size) and log(shipment frequency). "
        "Clusters labelled by median revenue."
    )

# ── ROW 4: Revenue at Risk Calculator ────────────────────────────────────────
st.markdown("---")
st.subheader("Revenue at Risk — Interactive Scenario")
st.caption(
    "**Revenue at Risk** = how much revenue disappears if the top N buyers stop ordering. "
    "Use the slider to see the worst-case scenario. This metric is critical for assessing supplier/buyer dependency. "
    "In a healthy market no single buyer should account for more than 5–10% of revenue."
)

n_buyers = st.slider("If top N buyers stop purchasing:", min_value=1, max_value=20, value=5)
at_risk  = float(buyer_rev.nlargest(n_buyers).sum())
pct_risk = at_risk / total_rev * 100 if total_rev > 0 else 0

col_warn, col_risk_chart = st.columns([1, 2])
with col_warn:
    st.markdown(f"""
<div class="callout-alert">
  <div class="callout-title">⚠️ Scenario: Top {n_buyers} Buyer{"s" if n_buyers > 1 else ""} Lost</div>
  If top <strong>{n_buyers}</strong> buyer{"s" if n_buyers > 1 else ""} stop purchasing:<br><br>
  <span style="font-size:1.5rem;font-weight:bold;color:#c0392b">
    ${at_risk / 1e6:.1f}M ({pct_risk:.1f}%)
  </span><br>
  of revenue <strong>evaporates overnight</strong>.<br><br>
  Remaining: <strong>${(total_rev - at_risk) / 1e6:.1f}M</strong>
</div>
""", unsafe_allow_html=True)

with col_risk_chart:
    fig_risk = charts.revenue_at_risk_bar(total_rev, at_risk, n_buyers)
    st.plotly_chart(fig_risk, use_container_width=True)

# ── ROW 5: Revenue Treemap ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Revenue Treemap: Country → Buyer")
fig_tree = charts.revenue_treemap(df, top_n=200)
st.plotly_chart(fig_tree, use_container_width=True)
st.caption("Top 200 importers by revenue. Hover over a tile to see exact revenue and share.")

# ── ROW 6: Top 25 Buyers Table ────────────────────────────────────────────────
st.markdown("---")
st.subheader("Top 25 Buyers by Revenue")


def _assign_flag(name: str, revenue: float, shipments: int, share_pct: float) -> str:
    upper = name.upper()
    if 'REFER' in upper or upper.startswith('REFER'):
        return '🔒'
    if revenue > 50_000_000 or (shipments <= 5 and revenue > 20_000_000):
        return '⭐'
    if share_pct > 8:
        return '⚠️'
    return ''


top25 = (
    df.groupby(['IMPORTER NAME', 'IMPORTER COUNTRY', 'BUYER_SEGMENT'])
    .agg(Revenue=('USD FOB', 'sum'), Shipments=('USD FOB', 'count'), Avg_Deal=('USD FOB', 'mean'))
    .reset_index()
    .sort_values('Revenue', ascending=False)
    .head(25)
    .reset_index(drop=True)
)
top25['Rank']          = top25.index + 1
top25['Share_%']       = top25['Revenue'] / total_rev * 100
top25['Cum_%']         = top25['Share_%'].cumsum()
top25['Revenue ($M)']  = (top25['Revenue'] / 1e6).round(2)
top25['Avg Deal ($K)'] = (top25['Avg_Deal'] / 1e3).round(1)
top25['Share (%)']     = top25['Share_%'].round(2)
top25['Cum (%)']       = top25['Cum_%'].round(1)
top25['Flag']          = top25.apply(
    lambda r: _assign_flag(r['IMPORTER NAME'], r['Revenue'], r['Shipments'], r['Share_%']),
    axis=1,
)

display_top25 = top25[[
    'Rank', 'IMPORTER NAME', 'IMPORTER COUNTRY',
    'Revenue ($M)', 'Share (%)', 'Cum (%)',
    'Shipments', 'Avg Deal ($K)', 'BUYER_SEGMENT', 'Flag',
]].rename(columns={
    'IMPORTER NAME': 'Buyer',
    'IMPORTER COUNTRY': 'Country',
    'BUYER_SEGMENT': 'Segment',
})

st.dataframe(
    display_top25,
    use_container_width=True,
    hide_index=True,
    height=640,
    column_config={
        'Revenue ($M)': st.column_config.NumberColumn(format='$%.2f M'),
        'Share (%)':    st.column_config.ProgressColumn(format='%.2f%%', min_value=0, max_value=100),
        'Cum (%)':      st.column_config.ProgressColumn(format='%.1f%%', min_value=0, max_value=100),
        'Avg Deal ($K)': st.column_config.NumberColumn(format='$%.1f K'),
        'Flag': st.column_config.TextColumn(
            help='⭐ Mega-contract  |  🔒 Redacted buyer (US Customs)  |  ⚠️ High concentration risk',
        ),
    },
)

col_leg1, col_leg2, col_leg3 = st.columns(3)
col_leg1.caption("⭐ Mega-contract buyer — >$50M revenue or ≤5 shipments + high value")
col_leg2.caption("🔒 Redacted — 'REFER BUYER' entry from US Customs Bill of Lading")
col_leg3.caption("⚠️ High concentration risk — single buyer >8% of total revenue")

# ── ROW 7: Pareto Chart ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Pareto Analysis: Buyer Revenue Concentration")
st.caption(
    "**Pareto Analysis** = based on the 80/20 rule (Pareto Principle): in most markets, roughly 20% of buyers generate 80% of revenue. "
    "The bars (left axis) show each buyer's revenue, ranked from largest to smallest. "
    "The orange line (right axis) shows the cumulative % of total revenue as you add more buyers. "
    "The red dashed line marks the 80% threshold — the annotation shows how many buyers you need to reach it."
)
fig_pareto = charts.pareto_chart(buyer_rev, top_n=60)
st.plotly_chart(fig_pareto, use_container_width=True)

st.markdown(f"""
<div class="callout-alert">
  <div class="callout-title">⚠️ The 80/20 Rule — Extreme Version</div>
  In a typical healthy market the top 20% of buyers generate 80% of revenue.<br>
  Here, only <strong>{n_at_80} buyers ({n_at_80/len(buyer_rev)*100:.1f}% of {len(buyer_rev):,})</strong>
  drive 80% of revenue — a <strong>Gini of {gini:.2f}</strong> is consulting-grade exceptional concentration.
  Any single buyer exit is a material business event.
</div>
""", unsafe_allow_html=True)

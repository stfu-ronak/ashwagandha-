from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from utils.constants import COUNTRY_ISO3_MAP, POPULATION_M
from utils.data_loader import load_data
from utils.sidebar import render_sidebar
import utils.charts as charts

st.set_page_config(
    page_title="Business Development — Ashwagandha",
    page_icon="🤝",
    layout="wide",
)

_css = Path(__file__).parent.parent / "assets" / "style.css"
if _css.exists():
    st.markdown(f"<style>{_css.read_text()}</style>", unsafe_allow_html=True)

df_full = load_data()
df = render_sidebar(df_full)

st.title("🤝 Business Development Intelligence")
st.caption("Trade flow networks, market opportunities, white space markets, and partner profiles")

if df.empty:
    st.warning("No data for current filters. Adjust the sidebar and try again.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔀 Trade Flows",
    "🎯 Opportunity Matrix",
    "🌍 White Space",
    "🤝 Partner Profiles",
])

# ── Tab 1: Trade Flows ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Ashwagandha Trade Flow Network")
    st.caption(
        "3-level Sankey: Exporter Country → Product Type → Importer Country. "
        "Showing top 5 exporters, top 6 product types, top 8 importers by revenue. "
        "Link width ∝ trade value ($M)."
    )

    m1, m2, m3, m4 = st.columns(4)
    top_exp_name = df.groupby('EXPORTER COUNTRY')['USD FOB'].sum().idxmax()
    top_imp_name = df.groupby('IMPORTER COUNTRY')['USD FOB'].sum().idxmax()
    top_prod_name = df.groupby('PRODUCT_TYPE')['USD FOB'].sum().idxmax()
    m1.metric("Top Export Origin", top_exp_name)
    m2.metric("Top Import Destination", top_imp_name)
    m3.metric("Dominant Product", top_prod_name)
    route_share = (
        df[df['EXPORTER COUNTRY'] == top_exp_name]['USD FOB'].sum() /
        df['USD FOB'].sum() * 100
    )
    m4.metric(f"{top_exp_name} Revenue Share", f"{route_share:.0f}%")

    st.plotly_chart(charts.sankey_trade_flows(df), use_container_width=True)
    st.caption("Note: only top nodes shown. Remaining flows not displayed to preserve readability.")


# ── Tab 2: Opportunity Matrix ─────────────────────────────────────────────────
with tab2:
    st.subheader("Market Opportunity Matrix")
    st.caption(
        "**Opportunity Matrix** = a strategic tool to classify markets into 4 quadrants based on two factors: current size and momentum. "
        "Each bubble = one importer country. Bubble size = total revenue. "
        "X-axis (log scale) = total USD FOB revenue — how big the market already is. "
        "Y-axis = revenue growth from first to last year in the filter — how fast it's growing. "
        "**⭐ Stars** = high revenue + high growth (your best markets). "
        "**💰 Cash Cows** = high revenue but slowing (defend, don't invest heavily). "
        "**🚀 Rising** = small but fast-growing (invest now, cheap entry). "
        "**😴 Dormant** = small and flat (low priority)."
    )

    years_in_data = sorted(df['YEAR'].unique().tolist())
    if len(years_in_data) < 2:
        st.info("Select at least 2 years in the sidebar to enable the opportunity matrix.")
    else:
        st.plotly_chart(charts.opportunity_matrix(df), use_container_width=True)

        st.markdown("---")
        st.subheader("🚀 Rising Markets (High Growth, Below-Median Revenue)")
        st.caption("Primary business development targets — strong momentum, room to grow.")

        yr_s, yr_e = years_in_data[0], years_in_data[-1]
        cy = df.groupby(['IMPORTER COUNTRY', 'YEAR'])['USD FOB'].sum().unstack(fill_value=0)
        ct = df.groupby('IMPORTER COUNTRY')['USD FOB'].sum()

        rising_records = []
        for country in cy.index:
            rev_s = float(cy.loc[country, yr_s]) if yr_s in cy.columns else 0
            rev_e = float(cy.loc[country, yr_e]) if yr_e in cy.columns else 0
            total = float(ct.get(country, 0))
            if total == 0:
                continue
            growth = float(np.clip((rev_e - rev_s) / max(rev_s, 1) * 100, -100, 500))
            rising_records.append({'Country': country, 'Total Revenue ($M)': round(total / 1e6, 2),
                                   'Growth (%)': round(growth, 1)})

        rising_df = pd.DataFrame(rising_records)
        if not rising_df.empty:
            med_rev = rising_df['Total Revenue ($M)'].median()
            rising = rising_df[
                (rising_df['Growth (%)'] > 0) & (rising_df['Total Revenue ($M)'] < med_rev)
            ].sort_values('Growth (%)', ascending=False).head(10).reset_index(drop=True)
            rising.index += 1
            st.dataframe(rising, use_container_width=True)


# ── Tab 3: White Space ────────────────────────────────────────────────────────
with tab3:
    st.subheader("White Space Markets")
    st.caption(
        "**White Space** = markets where there is untapped demand potential but very little current activity. "
        "Here defined as: countries with population > 10M people but ashwagandha import revenue < $1M. "
        "**Opportunity Score** = a composite score combining two factors equally (each 50%): "
        "(1) population size rank — bigger population = more potential buyers; "
        "(2) inverse revenue rank — lower current imports = more room to grow. "
        "Score range: 0 to 1. Higher score = bigger untapped population + almost no current competition."
    )

    # Build white space dataset from POPULATION_M
    country_rev = df.groupby('IMPORTER COUNTRY')['USD FOB'].sum().reset_index()
    country_rev.rename(columns={'USD FOB': 'Revenue'}, inplace=True)

    all_pop = pd.DataFrame({
        'Country': list(POPULATION_M.keys()),
        'Population_M': list(POPULATION_M.values()),
    })
    ws_merged = all_pop.merge(country_rev, left_on='Country', right_on='IMPORTER COUNTRY', how='left')
    ws_merged['Revenue'] = ws_merged['Revenue'].fillna(0)
    ws_merged['ISO3'] = ws_merged['Country'].map(COUNTRY_ISO3_MAP)

    ws = ws_merged[
        (ws_merged['Population_M'] > 10) & (ws_merged['Revenue'] < 1_000_000)
    ].copy()

    if ws.empty:
        st.info("No white space markets found with current filters.")
    else:
        max_pop = ws['Population_M'].max()
        max_rev = ws['Revenue'].max() if ws['Revenue'].max() > 0 else 1.0
        ws['Opportunity_Score'] = (
            0.5 * (ws['Population_M'] / max_pop) +
            0.5 * (1 - ws['Revenue'] / max_rev)
        )
        ws['Revenue_M'] = (ws['Revenue'] / 1e6).round(3)
        ws = ws.sort_values('Opportunity_Score', ascending=False).head(10).reset_index(drop=True)
        ws.index += 1

        m1, m2 = st.columns(2)
        m1.metric("White Space Markets Identified", len(ws))
        m2.metric("Combined Population", f"{ws['Population_M'].sum():,.0f}M")

        col_map, col_tbl = st.columns([3, 2])
        with col_map:
            st.plotly_chart(charts.white_space_choropleth(ws), use_container_width=True)
        with col_tbl:
            st.subheader("Top 10 Opportunity Markets")
            tbl = ws[['Country', 'Population_M', 'Revenue_M', 'Opportunity_Score']].copy()
            tbl['Opportunity_Score'] = tbl['Opportunity_Score'].round(3)
            tbl = tbl.rename(columns={
                'Population_M': 'Pop. (M)', 'Revenue_M': 'Revenue ($M)', 'Opportunity_Score': 'Opp. Score',
            })
            st.dataframe(tbl, use_container_width=True)
            st.info(
                "💡 **Strategy:** These markets have the buying population but not yet the supply "
                "relationships. Target with a local distributor partnership approach."
            )


# ── Tab 4: Partner Profiles ───────────────────────────────────────────────────
with tab4:
    st.subheader("Top Exporter Partner Profiles")
    st.caption("Expand each exporter to view their revenue profile, product mix, and key markets.")

    top10_exporters = (
        df.groupby('EXPORTER NAME')['USD FOB'].sum()
        .nlargest(10).index.tolist()
    )

    for exp_name in top10_exporters:
        exp_data = df[df['EXPORTER NAME'] == exp_name]
        total_rev = exp_data['USD FOB'].sum()
        shipments = len(exp_data)
        avg_deal = exp_data['USD FOB'].mean() if shipments > 0 else 0
        n_countries = exp_data['IMPORTER COUNTRY'].nunique()

        label = (
            f"**{exp_name[:50]}** — ${total_rev / 1e6:.1f}M revenue | "
            f"{shipments:,} shipments | {n_countries} markets"
        )
        with st.expander(label):
            col_m, col_pie = st.columns([2, 2])

            with col_m:
                st.markdown("**Key Metrics**")
                km1, km2 = st.columns(2)
                km1.metric("Total Revenue", f"${total_rev / 1e6:.1f}M",
                           help="Total USD FOB value of all shipments from this exporter in the filtered period.")
                km2.metric("Shipments", f"{shipments:,}",
                           help="Number of individual bill-of-lading shipment records for this exporter.")
                km3, km4 = st.columns(2)
                km3.metric("Avg Deal Size", f"${avg_deal / 1e3:.0f}K",
                           help="Average value per shipment (USD FOB ÷ number of shipments).")
                km4.metric("Import Markets", str(n_countries),
                           help="Number of distinct importing countries this exporter ships to.")

                st.markdown("**Top 3 Importer Markets**")
                top_markets = (
                    exp_data.groupby('IMPORTER COUNTRY')['USD FOB'].sum()
                    .nlargest(3).reset_index()
                )
                top_markets['Revenue ($M)'] = (top_markets['USD FOB'] / 1e6).round(2)
                top_markets['Share (%)'] = (top_markets['USD FOB'] / total_rev * 100).round(1)
                st.dataframe(
                    top_markets[['IMPORTER COUNTRY', 'Revenue ($M)', 'Share (%)']],
                    use_container_width=True, hide_index=True,
                )

            with col_pie:
                st.plotly_chart(
                    charts.partner_product_pie(df, exp_name),
                    use_container_width=True,
                )

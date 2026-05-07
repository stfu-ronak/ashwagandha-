from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from utils.constants import PORT_LATLON_MAP
from utils.data_loader import load_data
from utils.sidebar import render_sidebar
import utils.charts as charts

st.set_page_config(
    page_title="Geographic Intelligence — Ashwagandha",
    page_icon="🗺️",
    layout="wide",
)

_css = Path(__file__).parent.parent / "assets" / "style.css"
if _css.exists():
    st.markdown(f"<style>{_css.read_text()}</style>", unsafe_allow_html=True)

df_full = load_data()
df = render_sidebar(df_full)

st.title("🗺️ Geographic Intelligence")
st.caption("Import/export country flows, port activity, and country performance rankings")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📥 Import Map", "📤 Export Map", "⚓ Port Activity", "🏆 Country Rankings"]
)

# ── Tab 1: Import choropleth ──────────────────────────────────────────────────
with tab1:
    if df.empty:
        st.info("No data for current filters.")
    else:
        df_imp = (
            df.groupby(['IMPORTER COUNTRY', 'IMPORTER_ISO3'])
            .agg(
                Revenue=('USD FOB', 'sum'),
                Shipments=('USD FOB', 'count'),
                Avg_Deal=('USD FOB', 'mean'),
            )
            .reset_index()
            .dropna(subset=['IMPORTER_ISO3'])
            .rename(columns={'IMPORTER COUNTRY': 'Country'})
        )

        top_exp = (
            df.groupby(['IMPORTER COUNTRY', 'EXPORTER COUNTRY'])['USD FOB']
            .sum().reset_index()
            .sort_values('USD FOB', ascending=False)
            .groupby('IMPORTER COUNTRY').first().reset_index()
            [['IMPORTER COUNTRY', 'EXPORTER COUNTRY']]
            .rename(columns={'IMPORTER COUNTRY': 'Country', 'EXPORTER COUNTRY': 'Top_Exporter'})
        )
        df_imp = df_imp.merge(top_exp, on='Country', how='left')

        fig = charts.choropleth_map(
            df_imp,
            title='Ashwagandha Imports by Country — USD FOB',
            iso_col='IMPORTER_ISO3',
            val_col='Revenue',
            name_col='Country',
            hover_cols=['Shipments', 'Avg_Deal', 'Top_Exporter'],
            color_scale='Greens',
        )
        st.plotly_chart(fig, use_container_width=True)

        top3 = df_imp.nlargest(3, 'Revenue')
        c1, c2, c3 = st.columns(3)
        for col, (_, row) in zip([c1, c2, c3], top3.iterrows()):
            col.metric(
                row['Country'],
                f"${row['Revenue'] / 1e6:.1f}M",
                f"{row['Shipments']:,} shipments",
                help="Revenue = total USD FOB (Free On Board) value from this importing country. FOB = the export price at the point of departure, excluding freight and insurance costs.",
            )
        st.caption(
            "Map colour shows import revenue (USD FOB). "
            "**USD FOB** = Free On Board — the ex-factory export price at port of departure, "
            "excluding international freight and insurance. "
            "Colour is capped so that one outlier country doesn't make all others look white."
        )

# ── Tab 2: Export choropleth ──────────────────────────────────────────────────
with tab2:
    if df.empty:
        st.info("No data for current filters.")
    else:
        df_exp = (
            df.groupby(['EXPORTER COUNTRY', 'EXPORTER_ISO3'])
            .agg(
                Revenue=('USD FOB', 'sum'),
                Shipments=('USD FOB', 'count'),
                Avg_Deal=('USD FOB', 'mean'),
            )
            .reset_index()
            .dropna(subset=['EXPORTER_ISO3'])
            .rename(columns={'EXPORTER COUNTRY': 'Country'})
        )

        top_imp = (
            df.groupby(['EXPORTER COUNTRY', 'IMPORTER COUNTRY'])['USD FOB']
            .sum().reset_index()
            .sort_values('USD FOB', ascending=False)
            .groupby('EXPORTER COUNTRY').first().reset_index()
            [['EXPORTER COUNTRY', 'IMPORTER COUNTRY']]
            .rename(columns={'EXPORTER COUNTRY': 'Country', 'IMPORTER COUNTRY': 'Top_Importer'})
        )
        df_exp = df_exp.merge(top_imp, on='Country', how='left')

        fig = charts.choropleth_map(
            df_exp,
            title='Ashwagandha Exports by Country — USD FOB',
            iso_col='EXPORTER_ISO3',
            val_col='Revenue',
            name_col='Country',
            hover_cols=['Shipments', 'Avg_Deal', 'Top_Importer'],
            color_scale='Blues',
        )
        st.plotly_chart(fig, use_container_width=True)

        india_rev = df_exp[df_exp['Country'] == 'India']['Revenue'].sum()
        total_exp = df_exp['Revenue'].sum()
        india_pct = india_rev / total_exp * 100 if total_exp > 0 else 0
        st.markdown(f"""
<div class="callout-insight">
  <div class="callout-title">💡 India's Supply-Side Dominance</div>
  India accounts for <strong>{india_pct:.0f}%</strong> of all export revenue in this dataset —
  a near-total supply-side monopoly for global ashwagandha trade.
</div>
""", unsafe_allow_html=True)

# ── Tab 3: Port Activity Bubble Map ──────────────────────────────────────────
with tab3:
    if df.empty:
        st.info("No data for current filters.")
    else:
        fig = charts.port_bubble_map(
            df, title='Top-30 Ports by Revenue — Arrival Port Activity (bubble = revenue)'
        )
        st.plotly_chart(fig, use_container_width=True)

        port_rev = (
            df.groupby('PORT_ARRIVAL_CLEAN')['USD FOB']
            .sum().sort_values(ascending=False)
        )
        n_matched = sum(1 for p in df['PORT_ARRIVAL_CLEAN'].unique() if p in PORT_LATLON_MAP)
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Top Arrival Port",
            port_rev.index[0] if not port_rev.empty else '—',
            f"${port_rev.iloc[0] / 1e6:.1f}M" if not port_rev.empty else '',
            help="The port of arrival (destination port) with the highest total USD FOB revenue across all filtered shipments.",
        )
        c2.metric("Unique Ports in Data", f"{df['PORT_ARRIVAL_CLEAN'].nunique():,}",
                  help="Number of distinct arrival ports found in the filtered shipment records.")
        c3.metric("Ports Mapped", f"{n_matched} / 30 locations",
                  help="How many arrival ports were matched to a GPS coordinate for the bubble map. Only matched ports appear as bubbles.")
        st.caption("Bubble size ∝ √(revenue). Larger bubble = more revenue through that port. Hover for exact figures. Only ports with known GPS coordinates appear.")

# ── Tab 4: Country Rankings ───────────────────────────────────────────────────
with tab4:
    if df.empty:
        st.info("No data for current filters.")
    else:
        df_rank = (
            df.groupby('IMPORTER COUNTRY')
            .agg(
                Revenue=('USD FOB', 'sum'),
                Shipments=('USD FOB', 'count'),
                Avg_Deal=('USD FOB', 'mean'),
                Volume_kg=('NET WEIGHT', 'sum'),
            )
            .reset_index()
            .rename(columns={'IMPORTER COUNTRY': 'Country'})
            .sort_values('Revenue', ascending=False)
        )
        total_imp_rev = df_rank['Revenue'].sum()
        df_rank['Share_%'] = df_rank['Revenue'] / total_imp_rev * 100
        df_rank['Cum_%'] = df_rank['Share_%'].cumsum()

        col_bar, col_table = st.columns([1, 1])
        with col_bar:
            fig = charts.country_rankings_bar(
                df_rank,
                val_col='Revenue',
                name_col='Country',
                top_n=15,
                title='Top 15 Importers by Revenue',
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown("**Importer Country Rankings**")
            display = df_rank.copy()
            display['Revenue ($M)'] = (display['Revenue'] / 1e6).round(1)
            display['Avg Deal ($K)'] = (display['Avg_Deal'] / 1e3).round(1)
            display['Volume (t)']   = (display['Volume_kg'] / 1e3).round(1)
            display['Share %']      = display['Share_%'].round(1)
            display['Cum %']        = display['Cum_%'].round(1)
            st.dataframe(
                display[[
                    'Country', 'Revenue ($M)', 'Shipments',
                    'Avg Deal ($K)', 'Volume (t)', 'Share %', 'Cum %',
                ]],
                use_container_width=True,
                hide_index=True,
                height=480,
            )

        st.markdown("""
<div class="callout-insight">
  <div class="callout-title">💡 UK Revenue #1 vs US Volume #1 — Two Different Buyer Archetypes</div>
  <strong>United Kingdom</strong> is the top revenue importer yet ordered only ~4,000 kg total —
  Rain Nutrience Ltd placed 3 mega-contracts averaging $34M each ($~25K/kg implied value).<br>
  The <strong>United States</strong> leads on volume (1.99M kg) at conventional bulk pricing.
  These are entirely different buyer profiles requiring distinct commercial strategies.
</div>
""", unsafe_allow_html=True)

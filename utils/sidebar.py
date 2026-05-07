from __future__ import annotations

import streamlit as st

from .data_loader import apply_filters

_DEAL_OPTIONS = [0, 1_000, 5_000, 25_000, 100_000, 500_000]
_DEAL_LABELS  = ['$0', '$1K', '$5K', '$25K', '$100K', '$500K']


def _deal_fmt(v: int) -> str:
    return _DEAL_LABELS[_DEAL_OPTIONS.index(v)] if v in _DEAL_OPTIONS else f'${v:,}'


def render_sidebar(df_full) -> "pd.DataFrame":  # noqa: F821
    """Render global filter sidebar and return filtered DataFrame."""
    with st.sidebar:
        st.markdown("### 🌿 Filters")
        st.markdown("---")

        if st.button("↺ Reset All Filters", key="reset_btn"):
            for k in list(st.session_state.keys()):
                if k != "reset_btn":
                    del st.session_state[k]
            st.rerun()

        year_range = st.slider(
            "Year Range",
            min_value=int(df_full['YEAR'].min()),
            max_value=int(df_full['YEAR'].max()),
            value=(int(df_full['YEAR'].min()), int(df_full['YEAR'].max())),
            key="f_years",
        )
        quarters = st.multiselect(
            "Quarter",
            options=[1, 2, 3, 4],
            format_func=lambda q: f"Q{q}",
            key="f_quarters",
        )
        sorted_imp = sorted(df_full['IMPORTER COUNTRY'].dropna().unique().tolist())
        imp_countries = st.multiselect(
            "Importer Country", sorted_imp, key="f_imp_countries",
        )
        sorted_exp = sorted(df_full['EXPORTER COUNTRY'].dropna().unique().tolist())
        exp_countries = st.multiselect(
            "Exporter Country", sorted_exp, key="f_exp_countries",
        )
        sorted_imp_names = sorted(df_full['IMPORTER NAME'].dropna().unique().tolist())
        imp_names = st.multiselect(
            "Importer Company", sorted_imp_names, key="f_imp_names",
        )
        sorted_exp_names = sorted(df_full['EXPORTER NAME'].dropna().unique().tolist())
        exp_names = st.multiselect(
            "Exporter Company", sorted_exp_names, key="f_exp_names",
        )
        product_types = st.multiselect(
            "Product Type",
            options=sorted(df_full['PRODUCT_TYPE'].dropna().unique().tolist()),
            key="f_product_types",
        )
        transport_modes = st.multiselect(
            "Transport Mode",
            options=sorted(df_full['TRANSPORT_NORM'].dropna().unique().tolist()),
            key="f_transport",
        )
        deal_min = st.select_slider(
            "Min Deal Size",
            options=_DEAL_OPTIONS,
            value=0,
            format_func=_deal_fmt,
            key="f_deal_min",
        )
        st.markdown("---")
        st.caption("Data: TradeAtlas 2023–2026")

    return apply_filters(
        df_full,
        years=year_range,
        quarters=quarters if quarters else None,
        imp_countries=imp_countries if imp_countries else None,
        exp_countries=exp_countries if exp_countries else None,
        imp_names=imp_names if imp_names else None,
        exp_names=exp_names if exp_names else None,
        product_types=product_types if product_types else None,
        transport_modes=transport_modes if transport_modes else None,
        deal_min=deal_min,
    )

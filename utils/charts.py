from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .constants import COLORS, SEQUENTIAL_GREEN, PORT_LATLON_MAP

_LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='#ffffff',
    font=dict(family='system-ui, -apple-system, sans-serif', size=13),
    margin=dict(l=20, r=20, t=45, b=20),
)


# ── Geographic charts ─────────────────────────────────────────────────────────

def choropleth_map(
    df: pd.DataFrame,
    title: str,
    iso_col: str,
    val_col: str,
    name_col: str,
    hover_cols: list[str] | None = None,
    color_scale: str = 'Greens',
) -> go.Figure:
    """px.choropleth with 75th-percentile colour cap to prevent outlier wash-out."""
    cap = float(df[val_col].quantile(0.75)) * 2.5
    hover_data: dict = {iso_col: False}
    if hover_cols:
        for c in hover_cols:
            if c in df.columns:
                hover_data[c] = True
    fig = px.choropleth(
        df,
        locations=iso_col,
        color=val_col,
        hover_name=name_col,
        hover_data=hover_data,
        color_continuous_scale=color_scale,
        range_color=[0, max(cap, 1)],
        projection='natural earth',
        title=title,
        labels={val_col: 'Revenue ($)'},
    )
    fig.update_layout(
        **_LAYOUT,
        height=500,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='#dddddd',
            showland=True,
            landcolor='#f5f5f5',
        ),
        coloraxis_colorbar=dict(title='Revenue ($)'),
    )
    return fig


def port_bubble_map(
    df: pd.DataFrame,
    title: str = 'Port Activity by Revenue',
) -> go.Figure:
    """go.Scattergeo bubble map for top-30 ports joined from PORT_LATLON_MAP."""
    port_rev = (
        df.groupby('PORT_ARRIVAL_CLEAN')['USD FOB']
        .sum()
        .reset_index()
        .rename(columns={'USD FOB': 'Revenue'})
    )
    port_rev['lat'] = port_rev['PORT_ARRIVAL_CLEAN'].map(
        lambda p: PORT_LATLON_MAP.get(p, (None, None))[0]
    )
    port_rev['lon'] = port_rev['PORT_ARRIVAL_CLEAN'].map(
        lambda p: PORT_LATLON_MAP.get(p, (None, None))[1]
    )
    port_rev = port_rev.dropna(subset=['lat', 'lon'])
    port_rev = port_rev[port_rev['Revenue'] > 0].nlargest(30, 'Revenue')

    bubble_sizes = np.sqrt(port_rev['Revenue'] / 1e4).clip(lower=4, upper=55).values

    fig = go.Figure(go.Scattergeo(
        lat=port_rev['lat'].tolist(),
        lon=port_rev['lon'].tolist(),
        text=port_rev['PORT_ARRIVAL_CLEAN'].tolist(),
        customdata=port_rev[['Revenue']].values,
        hovertemplate='<b>%{text}</b><br>Revenue: $%{customdata[0]:,.0f}<extra></extra>',
        mode='markers',
        marker=dict(
            size=bubble_sizes.tolist(),
            color=port_rev['Revenue'].tolist(),
            colorscale='Greens',
            showscale=True,
            colorbar=dict(title='Revenue ($)'),
            line=dict(color='white', width=0.5),
            opacity=0.85,
        ),
        name='Port',
    ))
    fig.update_layout(
        **_LAYOUT,
        title=title,
        height=500,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='#cccccc',
            projection_type='natural earth',
            showland=True,
            landcolor='#f0f0f0',
            showocean=True,
            oceancolor='#e8f4fd',
        ),
    )
    return fig


def country_rankings_bar(
    df: pd.DataFrame,
    val_col: str = 'Revenue',
    name_col: str = 'Country',
    top_n: int = 15,
    title: str = 'Top Countries by Revenue',
) -> go.Figure:
    """Horizontal bar chart of top-N countries."""
    top = df.nlargest(top_n, val_col).copy()
    fig = px.bar(
        top,
        x=val_col,
        y=name_col,
        orientation='h',
        title=title,
        color=val_col,
        color_continuous_scale='Greens',
        labels={val_col: 'Revenue ($)', name_col: ''},
    )
    fig.update_layout(
        **_LAYOUT,
        height=500,
        yaxis=dict(autorange='reversed'),
        coloraxis_showscale=False,
    )
    return fig


# ── Concentration charts ──────────────────────────────────────────────────────

def hhi_gauge(hhi: float, label: str) -> go.Figure:
    """go.Indicator gauge for HHI score with coloured risk bands."""
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=hhi,
        number={'valueformat': ',.0f'},
        gauge={
            'axis': {'range': [0, 10_000], 'tickwidth': 1, 'tickcolor': '#666'},
            'bar': {'color': COLORS['dark_green'], 'thickness': 0.28},
            'bgcolor': 'white',
            'steps': [
                {'range': [0, 1500],   'color': '#d5f5e3'},
                {'range': [1500, 2500], 'color': '#fdebd0'},
                {'range': [2500, 5000], 'color': '#fadbd8'},
                {'range': [5000, 10000], 'color': '#f1948a'},
            ],
            'threshold': {
                'line': {'color': COLORS['red'], 'width': 3},
                'thickness': 0.8,
                'value': hhi,
            },
        },
        title={
            'text': f'HHI Score<br><span style="font-size:13px;color:#666">{label}</span>',
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=70, b=10),
        paper_bgcolor='white',
    )
    return fig


def hhi_trend_line(
    df_hhi: pd.DataFrame,
    yq_col: str = 'YQ',
    hhi_col: str = 'HHI',
) -> go.Figure:
    """Line chart of HHI by quarter with competition-threshold reference lines."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_hhi[yq_col],
        y=df_hhi[hhi_col],
        mode='lines+markers',
        line=dict(color=COLORS['dark_green'], width=2.5),
        marker=dict(size=8, color=COLORS['dark_green']),
        name='HHI',
        hovertemplate='%{x}: HHI = %{y:,.0f}<extra></extra>',
    ))
    for thresh, lbl, color in [
        (1500, 'Competitive (<1,500)', '#2e9e5b'),
        (2500, 'Concentrated (>2,500)', '#e67e22'),
    ]:
        fig.add_hline(
            y=thresh, line_dash='dash', line_color=color, line_width=1.2,
            annotation_text=lbl,
            annotation_position='top right',
            annotation_font_size=10,
        )
    if not df_hhi.empty:
        i_max = df_hhi[hhi_col].idxmax()
        i_min = df_hhi[hhi_col].idxmin()
        for idx, sym in [(i_max, '▲'), (i_min, '▽')]:
            fig.add_annotation(
                x=df_hhi.loc[idx, yq_col],
                y=df_hhi.loc[idx, hhi_col],
                text=f"{sym} {df_hhi.loc[idx, hhi_col]:,.0f}",
                showarrow=True, arrowhead=2, arrowsize=0.8,
                font=dict(size=10, color=COLORS['dark_green']),
                bgcolor='#f0fff4', bordercolor=COLORS['mid_green'],
            )
    fig.update_layout(
        **_LAYOUT,
        title='HHI Trend by Quarter',
        xaxis_title='Quarter',
        yaxis_title='HHI Score',
        height=320,
        showlegend=False,
    )
    return fig


def lorenz_curve(
    x: np.ndarray,
    y: np.ndarray,
    gini: float,
) -> go.Figure:
    """Lorenz curve with equality line, shaded concentration area, and Gini annotation."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='#aaaaaa', dash='dash', width=1.5),
        name='Perfect Equality',
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=x.tolist(), y=y.tolist(),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(26, 107, 60, 0.12)',
        line=dict(color=COLORS['dark_green'], width=2.5),
        name='Lorenz Curve',
        hovertemplate='Buyers: %{x:.1%}<br>Revenue: %{y:.1%}<extra></extra>',
    ))
    fig.add_annotation(
        x=0.25, y=0.72,
        text=f'<b>Gini = {gini:.2f}</b><br>Extreme concentration',
        showarrow=False,
        bgcolor='#fff9e6',
        bordercolor=COLORS['gold'],
        borderwidth=2,
        font=dict(size=12, color=COLORS['dark_green']),
        align='center',
    )
    fig.update_layout(
        **_LAYOUT,
        title='Lorenz Curve — Buyer Revenue Distribution',
        xaxis=dict(title='Cumulative Share of Buyers', tickformat='.0%', range=[0, 1]),
        yaxis=dict(title='Cumulative Share of Revenue', tickformat='.0%', range=[0, 1]),
        height=320,
        legend=dict(x=0.02, y=0.97, bgcolor='rgba(255,255,255,0.8)'),
    )
    return fig


def segmentation_scatter(
    buyer_stats: pd.DataFrame,
    avg_deal_col: str = 'avg_deal',
    freq_col: str = 'freq',
    segment_col: str = 'BUYER_SEGMENT',
    name_col: str = 'IMPORTER NAME',
) -> go.Figure:
    """Log-log scatter of buyer deal size vs shipment frequency, coloured by segment."""
    seg_colors = {
        'Mega B2B':    COLORS['dark_green'],
        'Mid-Market':  COLORS['gold'],
        'Small/Retail': COLORS['blue'],
    }
    fig = go.Figure()
    for seg, color in seg_colors.items():
        sub = buyer_stats[buyer_stats[segment_col] == seg]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[avg_deal_col],
            y=sub[freq_col],
            mode='markers',
            name=seg,
            text=sub[name_col] if name_col in sub.columns else None,
            hovertemplate='<b>%{text}</b><br>Avg Deal: $%{x:,.0f}<br>Shipments: %{y}<extra></extra>',
            marker=dict(color=color, size=7, opacity=0.72,
                        line=dict(color='white', width=0.4)),
        ))
    fig.update_layout(
        **_LAYOUT,
        title='Buyer Segmentation — Deal Size vs Frequency (K-Means, k=3)',
        xaxis=dict(title='Avg Deal Size ($)', type='log'),
        yaxis=dict(title='Shipment Frequency'),
        height=380,
        legend=dict(title='Segment', bgcolor='rgba(255,255,255,0.9)'),
    )
    return fig


def revenue_at_risk_bar(total: float, at_risk: float, n: int) -> go.Figure:
    """Stacked bar showing At Risk vs Remaining revenue for top-N buyer loss scenario."""
    remaining = max(total - at_risk, 0)
    pct_risk   = at_risk / total * 100 if total > 0 else 0
    pct_remain = remaining / total * 100 if total > 0 else 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='⚠️ At Risk',
        x=['Total Revenue'],
        y=[at_risk / 1e6],
        marker_color=COLORS['red'],
        text=[f'${at_risk / 1e6:.1f}M<br>({pct_risk:.1f}%)'],
        textposition='inside',
        textfont=dict(color='white', size=14),
    ))
    fig.add_trace(go.Bar(
        name='✅ Remaining',
        x=['Total Revenue'],
        y=[remaining / 1e6],
        marker_color=COLORS['mid_green'],
        text=[f'${remaining / 1e6:.1f}M<br>({pct_remain:.1f}%)'],
        textposition='inside',
        textfont=dict(color='white', size=14),
    ))
    fig.update_layout(
        **_LAYOUT,
        barmode='stack',
        title=f'Revenue Exposure if Top {n} Buyer{"s" if n > 1 else ""} Stop Purchasing',
        yaxis_title='Revenue ($M)',
        height=300,
        showlegend=True,
        legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center'),
    )
    return fig


def revenue_treemap(df: pd.DataFrame, top_n: int = 200) -> go.Figure:
    """px.treemap Country → Buyer, limited to top-N importers by revenue."""
    top_buyers = (
        df.groupby('IMPORTER NAME')['USD FOB'].sum()
        .nlargest(top_n)
        .index
    )
    df_top = df[df['IMPORTER NAME'].isin(top_buyers)].copy()
    fig = px.treemap(
        df_top,
        path=['IMPORTER COUNTRY', 'IMPORTER NAME'],
        values='USD FOB',
        color='USD FOB',
        color_continuous_scale='Greens',
        title=f'Revenue Treemap — Top {top_n} Importers (Country → Buyer)',
    )
    _treemap_layout = {**_LAYOUT, 'margin': dict(l=10, r=10, t=50, b=10)}
    fig.update_layout(
        **_treemap_layout,
        height=520,
        coloraxis_colorbar=dict(title='Revenue ($)'),
    )
    fig.update_traces(textinfo='label+percent root')
    return fig


def pareto_chart(buyer_rev: pd.Series, top_n: int = 60) -> go.Figure:
    """Dual y-axis Pareto: revenue bars + cumulative % line with 80% threshold."""
    total = buyer_rev.sum()
    all_sorted = buyer_rev.sort_values(ascending=False)
    cum_pct_all = all_sorted.cumsum() / total * 100

    # Find where cumulative crosses 80% in the full distribution
    n_at_80_arr = np.where(cum_pct_all.values >= 80)[0]
    n_at_80 = int(n_at_80_arr[0]) + 1 if len(n_at_80_arr) > 0 else len(all_sorted)

    top = all_sorted.head(top_n).reset_index()
    top.columns = ['Buyer', 'Revenue']
    top['Cumulative_Pct'] = top['Revenue'].cumsum() / total * 100
    ranks = list(range(1, len(top) + 1))

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    fig.add_trace(
        go.Bar(
            x=ranks,
            y=(top['Revenue'] / 1e6).tolist(),
            name='Revenue ($M)',
            marker_color=COLORS['mid_green'],
            marker_line_width=0,
            hovertext=top['Buyer'].tolist(),
            hovertemplate='<b>%{hovertext}</b><br>Rank %{x}<br>$%{y:.2f}M<extra></extra>',
            opacity=0.85,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ranks,
            y=top['Cumulative_Pct'].tolist(),
            name='Cumulative %',
            line=dict(color=COLORS['gold'], width=2.5),
            mode='lines',
            hovertemplate='Rank %{x}: %{y:.1f}% cumulative<extra></extra>',
        ),
        secondary_y=True,
    )

    # 80% reference line on secondary y-axis
    fig.add_shape(
        type='line',
        x0=0, x1=top_n, xref='x',
        y0=80, y1=80, yref='y2',
        line=dict(color=COLORS['red'], dash='dash', width=1.8),
    )
    # Annotation for 80% crossover
    fig.add_annotation(
        x=min(n_at_80 + 1, top_n),
        y=83,
        text=f'<b>{n_at_80} buyers → 80%</b>',
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['red'],
        font=dict(color=COLORS['red'], size=11),
        bgcolor='#fff5f5',
        bordercolor=COLORS['red'],
        borderwidth=1,
        yref='y2',
    )
    fig.update_layout(
        **_LAYOUT,
        title='Pareto Analysis — Buyer Revenue Concentration',
        xaxis_title='Buyer Rank',
        height=400,
        legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
    )
    fig.update_yaxes(title_text='Revenue ($M)', secondary_y=False, showgrid=True, gridcolor='#eeeeee')
    fig.update_yaxes(title_text='Cumulative Revenue %', secondary_y=True,
                     range=[0, 105], showgrid=False)
    return fig


# ── Product charts (Part 2) ───────────────────────────────────────────────────

def product_sunburst(df: pd.DataFrame) -> go.Figure:
    """px.sunburst: HS_GROUP_LABEL → HS CODE → PRODUCT_TYPE by revenue. (legacy — use product_category_donut instead)"""
    fig = px.sunburst(
        df,
        path=['HS_GROUP_LABEL', 'HS CODE', 'PRODUCT_TYPE'],
        values='USD FOB',
        color='USD FOB',
        color_continuous_scale='Greens',
        title='Product Mix — HS Group → HS Code → Product Type',
    )
    fig.update_traces(textinfo='label+percent parent')
    _sb_layout = {**_LAYOUT, 'margin': dict(l=10, r=10, t=50, b=10)}
    fig.update_layout(**_sb_layout, height=520, coloraxis_showscale=False)
    return fig


_PTYPE_COLORS = {
    'KSM-66 / Sensoril':    '#1a6b3c',
    'Standardized Extract':  '#2e9e5b',
    'Finished Dosage':       '#f39c12',
    'Organic Powder':        '#8BC34A',
    'Raw Powder':            '#3498db',
    'Root':                  '#9C27B0',
    'Other':                 '#aaaaaa',
}


def product_category_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart of PRODUCT_TYPE revenue share. Labels rendered outside segments so they're always readable."""
    rev = (
        df.groupby('PRODUCT_TYPE')['USD FOB'].sum()
        .reset_index().sort_values('USD FOB', ascending=False)
    )
    colors = [_PTYPE_COLORS.get(t, '#aaaaaa') for t in rev['PRODUCT_TYPE']]
    fig = go.Figure(go.Pie(
        labels=rev['PRODUCT_TYPE'],
        values=rev['USD FOB'],
        hole=0.45,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='percent+label',
        textposition='outside',
        textfont=dict(size=12, color='#333333'),
        insidetextorientation='horizontal',
        hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>',
        sort=False,
        direction='clockwise',
    ))
    fig.update_layout(**{
        **_LAYOUT,
        'paper_bgcolor': '#ffffff',
        'title': 'Product Type Revenue Share',
        'height': 440,
        'showlegend': False,
        'margin': dict(l=80, r=80, t=55, b=40),
    })
    return fig


def product_category_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of PRODUCT_TYPE by revenue. Labels placed outside bars on white background."""
    rev = (
        df.groupby('PRODUCT_TYPE')['USD FOB'].sum()
        .reset_index().sort_values('USD FOB', ascending=True)
    )
    total = rev['USD FOB'].sum()
    rev['share_pct'] = (rev['USD FOB'] / total * 100).round(1)
    colors = [_PTYPE_COLORS.get(t, '#aaaaaa') for t in rev['PRODUCT_TYPE']]
    max_val = float((rev['USD FOB'] / 1e6).max())
    fig = go.Figure(go.Bar(
        x=rev['USD FOB'] / 1e6,
        y=rev['PRODUCT_TYPE'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=0.5)),
        text=[f'${v:.1f}M  ({p:.0f}%)' for v, p in zip(rev['USD FOB'] / 1e6, rev['share_pct'])],
        textposition='outside',
        textfont=dict(size=11, color='#333333'),
        hovertemplate='<b>%{y}</b><br>Revenue: $%{x:.2f}M<extra></extra>',
    ))
    fig.update_layout(**{
        **_LAYOUT,
        'paper_bgcolor': '#ffffff',
        'title': 'Revenue by Product Type ($M)',
        'xaxis': dict(title='Revenue ($M)', showgrid=True, gridcolor='#eeeeee', range=[0, max_val * 1.5]),
        'yaxis': dict(title=''),
        'height': 440,
        'showlegend': False,
        'margin': dict(l=10, r=20, t=55, b=20),
    })
    return fig


def ksm_vs_generic_area(df: pd.DataFrame) -> go.Figure:
    """Stacked area: KSM-66/Sensoril vs everything else, by quarter."""
    df2 = df.copy()
    df2['is_ksm'] = df2['PRODUCT_TYPE'] == 'KSM-66 / Sensoril'
    qtr = (
        df2.groupby(['YQ', 'is_ksm'])['USD FOB']
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    qtr.columns.name = None
    qtr = qtr.rename(columns={False: 'Generic', True: 'KSM-66 / Sensoril'})
    if 'Generic' not in qtr.columns:
        qtr['Generic'] = 0.0
    if 'KSM-66 / Sensoril' not in qtr.columns:
        qtr['KSM-66 / Sensoril'] = 0.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=qtr['YQ'], y=(qtr['Generic'] / 1e6).tolist(),
        name='Generic', mode='lines', stackgroup='one',
        line=dict(color=COLORS['light_green']),
        fillcolor='rgba(183, 245, 208, 0.7)',
        hovertemplate='%{x}: $%{y:.1f}M<extra>Generic</extra>',
    ))
    fig.add_trace(go.Scatter(
        x=qtr['YQ'], y=(qtr['KSM-66 / Sensoril'] / 1e6).tolist(),
        name='KSM-66 / Sensoril', mode='lines', stackgroup='one',
        line=dict(color=COLORS['dark_green']),
        fillcolor='rgba(26, 107, 60, 0.7)',
        hovertemplate='%{x}: $%{y:.1f}M<extra>KSM-66 / Sensoril</extra>',
    ))
    fig.update_layout(
        **_LAYOUT,
        title='KSM-66 vs Generic Products — Quarterly Revenue',
        xaxis_title='Quarter', yaxis_title='Revenue ($M)',
        height=350,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def product_share_stacked_bar(df: pd.DataFrame, top_n_countries: int = 8) -> go.Figure:
    """100% stacked bar: top-N importer countries × product type revenue share."""
    top_countries = (
        df.groupby('IMPORTER COUNTRY')['USD FOB'].sum()
        .nlargest(top_n_countries).index.tolist()
    )
    sub = df[df['IMPORTER COUNTRY'].isin(top_countries)].copy()
    pivot = (
        sub.groupby(['IMPORTER COUNTRY', 'PRODUCT_TYPE'])['USD FOB']
        .sum().unstack(fill_value=0)
    )
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pct = pct.loc[top_countries[::-1]]  # reverse for horizontal readability

    _ptype_colors = {
        'KSM-66 / Sensoril': COLORS['dark_green'],
        'Standardized Extract': COLORS['mid_green'],
        'Finished Dosage': COLORS['gold'],
        'Organic Powder': '#8BC34A',
        'Raw Powder': COLORS['blue'],
        'Root': '#9C27B0',
        'Other': '#aaaaaa',
    }
    fig = go.Figure()
    for ptype in pct.columns:
        fig.add_trace(go.Bar(
            name=ptype,
            x=pct[ptype].tolist(),
            y=pct.index.tolist(),
            orientation='h',
            marker_color=_ptype_colors.get(ptype, '#aaaaaa'),
            hovertemplate=f'%{{y}}<br>{ptype}: %{{x:.1f}}%<extra></extra>',
        ))
    fig.update_layout(
        **_LAYOUT,
        barmode='stack',
        title=f'Product Mix by Country — Top {top_n_countries} Importers (100% stacked)',
        xaxis=dict(title='Revenue Share (%)', ticksuffix='%', range=[0, 100]),
        yaxis_title='',
        height=380,
        legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center'),
    )
    return fig


def price_violin(df: pd.DataFrame, price_cap: float = 500.0) -> go.Figure:
    """px.violin: PRODUCT_TYPE vs price_per_kg, y-axis capped at price_cap."""
    sub = df[df['price_per_kg'].notna() & (df['price_per_kg'] <= price_cap)].copy()
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT, title='No price data available', height=400)
        return fig
    order = (
        sub.groupby('PRODUCT_TYPE')['price_per_kg']
        .median().sort_values(ascending=False).index.tolist()
    )
    fig = px.violin(
        sub, x='PRODUCT_TYPE', y='price_per_kg',
        box=True, points='outliers',
        color='PRODUCT_TYPE',
        color_discrete_sequence=SEQUENTIAL_GREEN[1:] + [COLORS['gold'], COLORS['red'], '#9C27B0'],
        category_orders={'PRODUCT_TYPE': order},
        title=f'Price Distribution by Product Type (capped at ${price_cap:.0f}/kg)',
        labels={'price_per_kg': 'Price ($/kg)', 'PRODUCT_TYPE': ''},
    )
    fig.update_layout(**_LAYOUT, height=420, yaxis_range=[0, price_cap], showlegend=False)
    return fig


def price_trend_line(df: pd.DataFrame) -> go.Figure:
    """Monthly median price/kg with 3-month rolling mean overlay."""
    sub = df[df['price_per_kg'].notna()].copy()
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT, title='No price data available', height=350)
        return fig
    mp = (
        sub.groupby('YM_STR')['price_per_kg'].median()
        .reset_index().rename(columns={'price_per_kg': 'Median'})
        .sort_values('YM_STR')
    )
    mp['Rolling3M'] = mp['Median'].rolling(3, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mp['YM_STR'], y=mp['Median'], mode='lines+markers',
        name='Monthly Median',
        line=dict(color=COLORS['light_green'], width=1.5),
        marker=dict(size=5, color=COLORS['mid_green']),
        hovertemplate='%{x}: $%{y:.0f}/kg<extra>Monthly Median</extra>',
    ))
    fig.add_trace(go.Scatter(
        x=mp['YM_STR'], y=mp['Rolling3M'], mode='lines',
        name='3-Month Rolling Mean',
        line=dict(color=COLORS['dark_green'], width=2.5),
        hovertemplate='%{x}: $%{y:.0f}/kg<extra>3M Rolling</extra>',
    ))
    fig.update_layout(
        **_LAYOUT,
        title='Monthly Median Price per kg — with 3-Month Rolling Mean',
        xaxis_title='Month', yaxis_title='Median Price ($/kg)',
        height=350,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def hs_bar(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar: top-N HS codes by revenue."""
    hs_rev = (
        df.groupby('HS CODE')['USD FOB'].sum()
        .nlargest(top_n).reset_index()
        .rename(columns={'USD FOB': 'Revenue'})
        .sort_values('Revenue', ascending=True)
    )
    hs_rev['HS CODE'] = hs_rev['HS CODE'].astype(str)
    fig = px.bar(
        hs_rev, x='Revenue', y='HS CODE', orientation='h',
        title=f'Top {top_n} HS Codes by Revenue',
        color='Revenue', color_continuous_scale='Greens',
        labels={'Revenue': 'Revenue ($)', 'HS CODE': 'HS Code'},
    )
    fig.update_layout(
        **_LAYOUT, height=420,
        coloraxis_showscale=False,
        xaxis_tickformat='$,.0f',
    )
    return fig


def hs_country_heatmap(df: pd.DataFrame, top_n_hs: int = 10, top_n_countries: int = 15) -> go.Figure:
    """HS code × Importer Country revenue heatmap ($M)."""
    top_hs = (
        df.groupby('HS CODE')['USD FOB'].sum()
        .nlargest(top_n_hs).index.astype(str).tolist()
    )
    top_countries = (
        df.groupby('IMPORTER COUNTRY')['USD FOB'].sum()
        .nlargest(top_n_countries).index.tolist()
    )
    sub = df[
        df['HS CODE'].astype(str).isin(top_hs) &
        df['IMPORTER COUNTRY'].isin(top_countries)
    ].copy()
    pivot = (
        sub.groupby(['HS CODE', 'IMPORTER COUNTRY'])['USD FOB']
        .sum().unstack(fill_value=0)
    )
    pivot.index = pivot.index.astype(str)
    if pivot.empty:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT, title='No data for heatmap', height=400)
        return fig
    fig = px.imshow(
        pivot.values / 1e6,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        color_continuous_scale='Greens',
        labels={'color': 'Revenue ($M)', 'x': 'Importer Country', 'y': 'HS Code'},
        title='HS Code × Country Revenue Heatmap ($M)',
        aspect='auto',
    )
    fig.update_layout(
        **_LAYOUT, height=420,
        xaxis=dict(tickangle=-45),
        coloraxis_colorbar=dict(title='Revenue ($M)'),
    )
    return fig


def competitive_radar(df: pd.DataFrame, top_n: int = 5) -> go.Figure:
    """go.Scatterpolar radar: top-N exporters across 6 normalized dimensions."""
    top_exporters = (
        df.groupby('EXPORTER NAME')['USD FOB'].sum()
        .nlargest(top_n).index.tolist()
    )
    records: list[dict] = []
    for exp in top_exporters:
        grp = df[df['EXPORTER NAME'] == exp]
        total_rev = grp['USD FOB'].sum()
        n_countries = grp['IMPORTER COUNTRY'].nunique()
        avg_deal = grp['USD FOB'].mean()
        yoy = grp.groupby('YEAR')['USD FOB'].sum().sort_index()
        if len(yoy) >= 2 and yoy.iloc[0] > 0:
            n_yrs = max(int(yoy.index[-1]) - int(yoy.index[0]), 1)
            cagr = float((yoy.iloc[-1] / yoy.iloc[0]) ** (1 / n_yrs) - 1)
            cagr = float(np.clip(cagr, -1.0, 5.0))
        else:
            cagr = 0.0
        n_products = grp['PRODUCT_TYPE'].nunique()
        price_med = grp['price_per_kg'].dropna().median()
        price_med = float(price_med) if not pd.isna(price_med) else 0.0
        records.append({
            'name': str(exp),
            'Revenue': total_rev,
            'Market Reach': float(n_countries),
            'Deal Scale': avg_deal,
            'Growth Rate': max(cagr + 1.0, 0.0),
            'Product Diversity': float(n_products),
            'Price Premium': price_med,
        })

    radar_df = pd.DataFrame(records)
    dims = ['Revenue', 'Market Reach', 'Deal Scale', 'Growth Rate', 'Product Diversity', 'Price Premium']
    cats = [
        'Revenue\n(Normalized)', 'Market Reach\n(Countries)', 'Deal Scale\n(Avg Deal)',
        'Growth Rate\n(2023→2025)', 'Product\nDiversity', 'Price Premium\n($/kg)',
    ]
    for dim in dims:
        mx = radar_df[dim].max()
        radar_df[f'{dim}_n'] = (radar_df[dim] / mx).clip(0, 1) if mx > 0 else 0.0

    _radar_colors = [
        COLORS['dark_green'], COLORS['gold'], COLORS['blue'],
        COLORS['red'], COLORS['mid_green'],
    ]
    fig = go.Figure()
    for idx, (_, row) in enumerate(radar_df.iterrows()):
        vals = [row[f'{d}_n'] for d in dims]
        vals_c = vals + [vals[0]]
        cats_c = cats + [cats[0]]
        hx = _radar_colors[idx % len(_radar_colors)]
        r, g, b = int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)
        fig.add_trace(go.Scatterpolar(
            r=vals_c, theta=cats_c, fill='toself',
            fillcolor=f'rgba({r},{g},{b},0.12)',
            line=dict(color=hx, width=2),
            name=row['name'][:35],
        ))
    fig.update_layout(
        **_LAYOUT,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor='#dddddd'),
            angularaxis=dict(gridcolor='#dddddd'),
        ),
        title='Competitive Radar — Top 5 Exporters (6 Dimensions)',
        height=520,
        legend=dict(x=1.05, y=0.5, bgcolor='rgba(255,255,255,0.9)'),
    )
    return fig


# ── Forecast / time-series charts (Part 2) ────────────────────────────────────

def dual_axis_revenue_volume(df: pd.DataFrame) -> go.Figure:
    """Dual y-axis: monthly revenue bars (primary) + volume in tonnes line (secondary)."""
    monthly = (
        df.groupby('YM_STR')
        .agg(Revenue=('USD FOB', 'sum'), Volume_kg=('NET WEIGHT', 'sum'))
        .reset_index().sort_values('YM_STR')
    )
    monthly['Volume_t'] = monthly['Volume_kg'] / 1000

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(
        go.Bar(
            x=monthly['YM_STR'], y=(monthly['Revenue'] / 1e6).tolist(),
            name='Revenue ($M)', marker_color=COLORS['mid_green'],
            marker_line_width=0, opacity=0.85,
            hovertemplate='%{x}: $%{y:.1f}M<extra>Revenue</extra>',
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly['YM_STR'], y=monthly['Volume_t'].tolist(),
            name='Volume (tonnes)', mode='lines+markers',
            line=dict(color=COLORS['gold'], width=2),
            marker=dict(size=5),
            hovertemplate='%{x}: %{y:,.0f}t<extra>Volume</extra>',
        ),
        secondary_y=True,
    )
    fig.update_layout(
        **_LAYOUT,
        title='Monthly Revenue vs Shipment Volume (Dual Axis)',
        height=420,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)'),
        hovermode='x unified',
        xaxis_title='Month',
    )
    fig.update_yaxes(title_text='Revenue ($M)', secondary_y=False, showgrid=True, gridcolor='#eeeeee')
    fig.update_yaxes(title_text='Volume (tonnes)', secondary_y=True, showgrid=False)
    return fig


def yoy_waterfall(df: pd.DataFrame) -> go.Figure:
    """YoY waterfall: 2024 total → per-quarter deltas → 2025 total."""
    qtr_rev = df.groupby(['YEAR', 'QUARTER'])['USD FOB'].sum().reset_index()
    yrs = sorted(qtr_rev['YEAR'].unique())
    if len(yrs) < 2:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT, title='Not enough year data for waterfall', height=350)
        return fig

    yr_a, yr_b = yrs[-2], yrs[-1]
    rev_a = qtr_rev[qtr_rev['YEAR'] == yr_a].set_index('QUARTER')['USD FOB']
    rev_b = qtr_rev[qtr_rev['YEAR'] == yr_b].set_index('QUARTER')['USD FOB']
    quarters = sorted(set(rev_a.index) & set(rev_b.index))
    total_a = sum(rev_a.get(q, 0) for q in quarters)
    deltas = [(rev_b.get(q, 0) - rev_a.get(q, 0)) / 1e6 for q in quarters]

    measures = ['absolute'] + ['relative'] * len(deltas) + ['total']
    x_labels = [f'{yr_a} Total'] + [f'Q{q} Change' for q in quarters] + [f'{yr_b} Total']
    y_values = [total_a / 1e6] + deltas + [0.0]
    texts = [f'${total_a/1e6:.1f}M'] + [f'${d:+.1f}M' for d in deltas] + ['']

    fig = go.Figure(go.Waterfall(
        measure=measures, x=x_labels, y=y_values,
        text=texts, textposition='outside',
        connector={'line': {'color': '#cccccc', 'dash': 'dot', 'width': 1}},
        increasing={'marker': {'color': COLORS['mid_green']}},
        decreasing={'marker': {'color': COLORS['red']}},
        totals={'marker': {'color': COLORS['dark_green']}},
    ))
    fig.update_layout(
        **_LAYOUT,
        title=f'YoY Quarterly Revenue Change — {yr_a} → {yr_b}',
        yaxis_title='Revenue ($M)',
        height=380, showlegend=False,
    )
    return fig


def ensemble_forecast(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    """3-model ensemble forecast: Holt-Winters + OLS Fourier + Naive Seasonal."""
    import warnings

    monthly = (
        df.groupby('YM_STR')['USD FOB'].sum()
        .reset_index().rename(columns={'YM_STR': 'period', 'USD FOB': 'revenue'})
        .sort_values('period').reset_index(drop=True)
    )
    monthly['period_dt'] = pd.to_datetime(monthly['period'])

    if len(monthly) < 12:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT, title='Insufficient data (need ≥ 12 months)', height=450)
        return fig, pd.DataFrame()

    cap = float(monthly['revenue'].quantile(0.95))
    rev_win = monthly['revenue'].clip(upper=cap).values.astype(float)

    n_total = len(monthly)
    n_holdout = min(6, n_total // 4)
    n_train = n_total - n_holdout
    n_forecast = 12

    train_rev = rev_win[:n_train]
    holdout_rev = monthly['revenue'].iloc[n_train:].values.astype(float)
    last_dt = monthly['period_dt'].iloc[-1]
    future_dates = pd.date_range(last_dt + pd.DateOffset(months=1), periods=n_forecast, freq='MS')
    holdout_dates = monthly['period_dt'].iloc[n_train:].tolist()

    hw_ho: np.ndarray | None = None
    hw_fut: np.ndarray | None = None
    ols_ho: np.ndarray | None = None
    ols_fut: np.ndarray | None = None
    naive_ho: np.ndarray | None = None
    naive_fut: np.ndarray | None = None
    ci_lower: np.ndarray | None = None
    ci_upper: np.ndarray | None = None

    # Model 1: Holt-Winters damped
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as _HW
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            seasonal = 'add' if n_train >= 24 else None
            hw_m = _HW(
                train_rev,
                trend='add', seasonal=seasonal,
                seasonal_periods=12 if seasonal else None,
                damped_trend=True,
            ).fit(optimized=True)
            hw_full = hw_m.forecast(n_holdout + n_forecast)
            hw_ho = hw_full[:n_holdout]
            hw_fut = hw_full[n_holdout:]
            resid_std = float(np.std(train_rev - hw_m.fittedvalues))
            ci_lower = np.clip(hw_fut - 1.28 * resid_std, 0, None)
            ci_upper = hw_fut + 1.28 * resid_std
    except Exception:
        pass

    # Model 2: OLS + Fourier seasonality
    try:
        from statsmodels.regression.linear_model import OLS as _OLS
        from statsmodels.tools import add_constant as _ac

        all_m = np.concatenate([monthly['period_dt'].dt.month.values, future_dates.month.values])
        all_t = np.arange(n_total + n_forecast, dtype=float)

        def _fx(m_arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
            cols = [t_arr]
            for h in range(1, 3):
                cols.append(np.sin(2 * np.pi * h * m_arr / 12))
                cols.append(np.cos(2 * np.pi * h * m_arr / 12))
            return _ac(np.column_stack(cols))

        X_all = _fx(all_m, all_t)
        ols_res = _OLS(train_rev, X_all[:n_train]).fit()
        ols_ho = np.clip(ols_res.predict(X_all[n_train:n_total]), 0, None)
        ols_fut = np.clip(ols_res.predict(X_all[n_total:]), 0, None)
    except Exception:
        pass

    # Model 3: Naive Seasonal (same-month avg × trend multiplier)
    try:
        tr_months = monthly['period_dt'].iloc[:n_train].dt.month.values
        m_avgs = {
            m: float(train_rev[tr_months == m].mean()) if (tr_months == m).any() else float(train_rev.mean())
            for m in range(1, 13)
        }
        trend_mult = 1.0
        if n_train >= 18:
            recent_avg = float(train_rev[n_train - 6:].mean())
            older_avg = float(train_rev[n_train - 18:n_train - 12].mean())
            if older_avg > 0:
                trend_mult = float(np.clip(recent_avg / older_avg, 0.5, 2.0))
        ho_m = monthly['period_dt'].iloc[n_train:].dt.month.values
        naive_ho = np.clip(np.array([m_avgs[m] for m in ho_m]) * trend_mult, 0, None)
        fu_m = future_dates.month.values
        naive_fut = np.clip(np.array([m_avgs[m] for m in fu_m]) * trend_mult, 0, None)
    except Exception:
        pass

    def _ens(preds: list) -> np.ndarray | None:
        valid = [np.asarray(p).ravel() for p in preds if p is not None]
        return np.median(np.vstack(valid), axis=0) if valid else None

    ens_ho = _ens([hw_ho, ols_ho, naive_ho])
    ens_fut = _ens([hw_fut, ols_fut, naive_fut])

    if ens_fut is not None and ci_lower is None:
        spread = np.maximum(ens_fut * 0.15, 1e5)
        ci_lower = np.clip(ens_fut - spread, 0, None)
        ci_upper = ens_fut + spread

    def _mae(a: np.ndarray, p: np.ndarray | None) -> float:
        return float(np.mean(np.abs(a - p))) if p is not None else float('nan')

    def _rmse(a: np.ndarray, p: np.ndarray | None) -> float:
        return float(np.sqrt(np.mean((a - p) ** 2))) if p is not None else float('nan')

    metrics = pd.DataFrame({
        'Model': ['Holt-Winters (Damped)', 'OLS + Fourier', 'Naive Seasonal', 'Ensemble (Median)'],
        'MAE ($M)': [_mae(holdout_rev, hw_ho) / 1e6, _mae(holdout_rev, ols_ho) / 1e6,
                     _mae(holdout_rev, naive_ho) / 1e6, _mae(holdout_rev, ens_ho) / 1e6],
        'RMSE ($M)': [_rmse(holdout_rev, hw_ho) / 1e6, _rmse(holdout_rev, ols_ho) / 1e6,
                      _rmse(holdout_rev, naive_ho) / 1e6, _rmse(holdout_rev, ens_ho) / 1e6],
    }).round(2)

    fig = go.Figure()

    # 80% CI band (future only)
    if ens_fut is not None and ci_lower is not None and ci_upper is not None:
        fut_list = future_dates.tolist()
        fig.add_trace(go.Scatter(
            x=fut_list + fut_list[::-1],
            y=(ci_upper / 1e6).tolist() + (ci_lower / 1e6).tolist()[::-1],
            fill='toself', fillcolor='rgba(46, 158, 91, 0.12)',
            line=dict(color='rgba(0,0,0,0)'),
            name='80% CI Band', hoverinfo='skip',
        ))

    # Historical actual
    fig.add_trace(go.Scatter(
        x=monthly['period_dt'].tolist(),
        y=(monthly['revenue'] / 1e6).tolist(),
        mode='lines', name='Actual Revenue',
        line=dict(color=COLORS['dark_green'], width=2.5),
        hovertemplate='%{x|%b %Y}: $%{y:.1f}M<extra>Actual</extra>',
    ))

    # Individual model lines (holdout + forecast)
    _mspecs = [
        ('Holt-Winters', hw_ho, hw_fut, 'dash', COLORS['blue']),
        ('OLS + Fourier', ols_ho, ols_fut, 'dot', COLORS['gold']),
        ('Naive Seasonal', naive_ho, naive_fut, 'dashdot', COLORS['red']),
    ]
    for name, ho_p, fu_p, dash, color in _mspecs:
        if ho_p is None or fu_p is None:
            continue
        all_dt = holdout_dates + future_dates.tolist()
        all_v = list(ho_p / 1e6) + list(fu_p / 1e6)
        fig.add_trace(go.Scatter(
            x=all_dt, y=all_v, mode='lines', name=name,
            line=dict(color=color, width=1.5, dash=dash),
            hovertemplate=f'%{{x|%b %Y}}: $%{{y:.1f}}M<extra>{name}</extra>',
        ))

    # Ensemble bold line
    if ens_fut is not None:
        ens_ho_list = list(ens_ho / 1e6) if ens_ho is not None else []
        ens_dt = holdout_dates[:len(ens_ho_list)] + future_dates.tolist()
        ens_v = ens_ho_list + list(ens_fut / 1e6)
        fig.add_trace(go.Scatter(
            x=ens_dt, y=ens_v, mode='lines', name='Ensemble (Median)',
            line=dict(color=COLORS['mid_green'], width=3.5),
            hovertemplate='%{x|%b %Y}: $%{y:.1f}M<extra>Ensemble</extra>',
        ))

    # Dec 2025 annotation
    dec25 = pd.Timestamp('2025-12-01')
    dec25_mask = monthly['period_dt'] == dec25
    if dec25_mask.any():
        dec25_rev = float(monthly.loc[dec25_mask, 'revenue'].iloc[0]) / 1e6
        fig.add_annotation(
            x=dec25, y=dec25_rev,
            text='Dec 2025: $118.9M<br>Rain Nutrience bulk pre-buy<br>(smoothed in model fit)',
            showarrow=True, arrowhead=2, arrowcolor=COLORS['red'],
            font=dict(size=10, color=COLORS['red']),
            bgcolor='#fff5f5', bordercolor=COLORS['red'], borderwidth=1,
            ax=55, ay=-65,
        )

    # add_vline hits a pandas 2.x Timestamp arithmetic bug in Plotly; use add_shape instead
    _vl_x = last_dt.strftime('%Y-%m-%d')
    fig.add_shape(
        type='line', xref='x', yref='paper',
        x0=_vl_x, x1=_vl_x, y0=0, y1=1,
        line=dict(color='#aaaaaa', width=1.5, dash='dash'),
    )
    fig.add_annotation(
        xref='x', yref='paper',
        x=_vl_x, y=1.01, text='Forecast →',
        showarrow=False, font=dict(size=10, color='#888888'),
        xanchor='left',
    )
    fig.update_layout(
        **_LAYOUT,
        title='Revenue Forecast — 3-Model Ensemble (to Apr 2027)',
        xaxis_title='Month', yaxis_title='Revenue ($M)',
        height=500,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)'),
        hovermode='x unified',
    )
    return fig, metrics


def seasonality_heatmap(df: pd.DataFrame) -> go.Figure:
    """Month × Year revenue heatmap, zmax capped at 95th percentile."""
    pivot = (
        df.groupby(['YEAR', 'MONTH'])['USD FOB'].sum()
        .unstack(level=0, fill_value=0)
    )
    _mnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot.index = [_mnames[int(m) - 1] for m in pivot.index]
    z = pivot.values / 1e6
    zmax = float(np.nanpercentile(z, 95)) * 1.2 if z.size > 0 else 1.0

    fig = px.imshow(
        z,
        x=[str(c) for c in pivot.columns.tolist()],
        y=pivot.index.tolist(),
        color_continuous_scale='Greens',
        zmin=0, zmax=max(zmax, 0.1),
        labels={'color': 'Revenue ($M)', 'x': 'Year', 'y': 'Month'},
        title='Monthly Revenue Heatmap — Month × Year ($M)',
        aspect='auto',
    )
    fig.update_layout(
        **_LAYOUT, height=400,
        coloraxis_colorbar=dict(title='Revenue ($M)'),
    )
    return fig


def sdi_bar(sdi_df: pd.DataFrame) -> go.Figure:
    """Seasonal Demand Index bar chart: green >100, red <100."""
    colors = [COLORS['mid_green'] if v > 100 else COLORS['red'] for v in sdi_df['SDI']]
    fig = go.Figure(go.Bar(
        x=sdi_df['Month_Name'], y=sdi_df['SDI'],
        marker_color=colors,
        text=[f'{v:.0f}' for v in sdi_df['SDI']],
        textposition='outside',
        hovertemplate='%{x}: SDI = %{y:.0f}<extra></extra>',
    ))
    fig.add_hline(
        y=100, line_dash='dash', line_color='#666666', line_width=1.5,
        annotation_text='Baseline = 100', annotation_position='top right',
        annotation_font_size=10,
    )
    ymax = float(sdi_df['SDI'].max()) * 1.2 if not sdi_df.empty else 150
    fig.update_layout(
        **_LAYOUT,
        title='Seasonal Demand Index (SDI) — Monthly Demand vs Annual Average',
        xaxis_title='Month', yaxis_title='SDI (100 = average)',
        height=380, showlegend=False,
        yaxis=dict(range=[0, max(ymax, 150)]),
    )
    return fig


def deal_histogram(df: pd.DataFrame) -> go.Figure:
    """Log-scale deal size distribution histogram."""
    valid = df[df['USD FOB'] > 0]['USD FOB'].values
    if len(valid) == 0:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT, title='No deal data', height=350)
        return fig
    vmax = float(valid.max())
    bins = np.logspace(0, np.log10(vmax + 1), 50)
    counts, edges = np.histogram(valid, bins=bins)
    mids = (edges[:-1] * edges[1:]) ** 0.5  # geometric midpoints

    fig = go.Figure(go.Bar(
        x=mids, y=counts,
        marker_color=COLORS['mid_green'], marker_line_width=0,
        hovertemplate='Deal size ~$%{x:,.0f}<br>Shipments: %{y}<extra></extra>',
    ))
    fig.update_layout(
        **_LAYOUT,
        title='Deal Size Distribution (Log Scale)',
        xaxis=dict(title='Deal Size (USD FOB)', type='log', tickformat='$,.0f'),
        yaxis_title='Number of Shipments',
        height=360, showlegend=False,
    )
    return fig


# ── Business Development & Marketing charts (Part 3) ─────────────────────────

_CHANNEL_MAP: dict[str, str] = {
    '>$500K':     'B2B Mega',
    '$100K-500K': 'B2B Mid',
    '$25K-100K':  'B2B Mid',
    '$5K-25K':    'B2B Retail',
    '$1K-5K':     'B2B Retail',
    '<$1K':       'B2C',
}
_CHANNEL_COLORS: dict[str, str] = {
    'B2B Mega':   COLORS['dark_green'],
    'B2B Mid':    COLORS['mid_green'],
    'B2B Retail': COLORS['gold'],
    'B2C':        COLORS['blue'],
}


def _add_channel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['CHANNEL'] = out['deal_segment'].astype(str).map(_CHANNEL_MAP).fillna('B2C')
    fba_mask = out['IMPORTER NAME'].str.contains('FBA|Amazon', case=False, na=False)
    out.loc[fba_mask, 'CHANNEL'] = 'B2C'
    return out


def sankey_trade_flows(df: pd.DataFrame) -> go.Figure:
    """go.Sankey: Exporter Country → Product Type → Importer Country (top 5/6/8 nodes)."""
    top_exp  = df.groupby('EXPORTER COUNTRY')['USD FOB'].sum().nlargest(5).index.tolist()
    top_prod = df.groupby('PRODUCT_TYPE')['USD FOB'].sum().nlargest(6).index.tolist()
    top_imp  = df.groupby('IMPORTER COUNTRY')['USD FOB'].sum().nlargest(8).index.tolist()

    sub = df[
        df['EXPORTER COUNTRY'].isin(top_exp) &
        df['PRODUCT_TYPE'].isin(top_prod) &
        df['IMPORTER COUNTRY'].isin(top_imp)
    ]
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT, title='No data for Sankey', height=500)
        return fig

    nodes = top_exp + top_prod + top_imp
    node_idx = {n: i for i, n in enumerate(nodes)}
    node_colors = (
        [COLORS['blue']] * len(top_exp) +
        [COLORS['mid_green']] * len(top_prod) +
        [COLORS['gold']] * len(top_imp)
    )

    ep = sub.groupby(['EXPORTER COUNTRY', 'PRODUCT_TYPE'])['USD FOB'].sum().reset_index()
    ep = ep[ep['USD FOB'] > 0]
    pi = sub.groupby(['PRODUCT_TYPE', 'IMPORTER COUNTRY'])['USD FOB'].sum().reset_index()
    pi = pi[pi['USD FOB'] > 0]

    sources = [node_idx[r] for r in ep['EXPORTER COUNTRY']] + [node_idx[r] for r in pi['PRODUCT_TYPE']]
    targets = [node_idx[r] for r in ep['PRODUCT_TYPE']] + [node_idx[r] for r in pi['IMPORTER COUNTRY']]
    values  = (ep['USD FOB'] / 1e6).tolist() + (pi['USD FOB'] / 1e6).tolist()

    fig = go.Figure(go.Sankey(
        arrangement='freeform',
        node=dict(
            label=nodes,
            color=node_colors,
            pad=25,
            thickness=28,
            line=dict(color='white', width=0.8),
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='rgba(150,150,150,0.20)',
        ),
    ))
    fig.update_layout(**{
        **_LAYOUT,
        'title': 'Trade Flows — Exporter Country → Product Type → Importer Country ($M, top nodes only)',
        'height': 720,
        'font': dict(family='system-ui, -apple-system, sans-serif', size=14),
    })
    return fig


def opportunity_matrix(df: pd.DataFrame) -> go.Figure:
    """Quadrant scatter: importer countries by total revenue (log x) vs YoY growth %."""
    country_year = (
        df.groupby(['IMPORTER COUNTRY', 'YEAR'])['USD FOB']
        .sum().unstack(fill_value=0)
    )
    country_total = df.groupby('IMPORTER COUNTRY')['USD FOB'].sum()
    years = sorted(country_year.columns.tolist())
    if len(years) < 2:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT, title='Need ≥ 2 years for opportunity matrix', height=500)
        return fig

    yr_s, yr_e = years[0], years[-1]
    records = []
    for country in country_year.index:
        rev_s = float(country_year.loc[country, yr_s])
        rev_e = float(country_year.loc[country, yr_e])
        total = float(country_total.get(country, 0))
        if total == 0:
            continue
        growth = float(np.clip((rev_e - rev_s) / max(rev_s, 1) * 100, -100, 500))
        records.append({'Country': country, 'Total_Revenue': total, 'YoY_Growth': growth})

    if not records:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT, title='No data for opportunity matrix', height=500)
        return fig

    opp_df = pd.DataFrame(records)
    med_rev = float(opp_df['Total_Revenue'].median())

    fig = px.scatter(
        opp_df, x='Total_Revenue', y='YoY_Growth', size='Total_Revenue',
        hover_name='Country', color='YoY_Growth',
        color_continuous_scale=[COLORS['red'], COLORS['gold'], COLORS['mid_green']],
        log_x=True, size_max=55,
        title=f'Market Opportunity Matrix — {yr_s}→{yr_e} Revenue Growth vs Total Revenue',
        labels={'Total_Revenue': 'Total Revenue (log scale, $)', 'YoY_Growth': f'Revenue Growth {yr_s}→{yr_e} (%)'},
    )
    fig.add_shape(type='line', xref='x', yref='paper',
                  x0=med_rev, x1=med_rev, y0=0, y1=1,
                  line=dict(color='#aaaaaa', dash='dash', width=1.2))
    fig.add_shape(type='line', xref='paper', yref='y',
                  x0=0, x1=1, y0=0, y1=0,
                  line=dict(color='#aaaaaa', dash='dash', width=1.2))
    for label, xp, yp in [
        ('⭐ Stars', 0.82, 0.93), ('💰 Cash Cows', 0.82, 0.07),
        ('🚀 Rising', 0.04, 0.93), ('😴 Dormant', 0.04, 0.07),
    ]:
        fig.add_annotation(xref='paper', yref='paper', x=xp, y=yp,
                           text=f'<b>{label}</b>', showarrow=False,
                           font=dict(size=11, color='#555555'),
                           bgcolor='rgba(255,255,255,0.75)', bordercolor='#cccccc', borderwidth=1)
    fig.update_layout(**_LAYOUT, height=520, coloraxis_showscale=False, xaxis_tickformat='$,.0f')
    return fig


def supply_risk_bar(risk_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of exporter risk scores, colour-coded by risk level."""
    _risk_colors = {'Low': COLORS['mid_green'], 'Moderate': COLORS['gold'],
                    'High': '#e67e22', 'Critical': COLORS['red']}
    df_plot = risk_df.sort_values('Risk_Score').tail(20).copy()
    df_plot['Short_Name'] = df_plot['EXPORTER NAME'].str[:40]
    colors = [_risk_colors.get(lv, '#aaaaaa') for lv in df_plot['Risk_Level']]

    fig = go.Figure(go.Bar(
        x=df_plot['Risk_Score'], y=df_plot['Short_Name'], orientation='h',
        marker_color=colors,
        text=[f"{s:.0f}" for s in df_plot['Risk_Score']], textposition='outside',
        customdata=df_plot[['Risk_Level', 'Countries']].values,
        hovertemplate='<b>%{y}</b><br>Risk Score: %{x:.0f} (%{customdata[0]})<br>Countries: %{customdata[1]}<extra></extra>',
    ))
    fig.update_layout(
        **_LAYOUT,
        title='Supply Chain Risk Score — Top Exporters (highest risk at top)',
        xaxis=dict(title='Risk Score (0–100)', range=[0, 115]),
        yaxis_title='', height=max(380, len(df_plot) * 28), showlegend=False,
    )
    return fig


def white_space_choropleth(ws_df: pd.DataFrame) -> go.Figure:
    """Choropleth highlighting white space markets using orange scale."""
    plot_df = ws_df.dropna(subset=['ISO3']).copy()
    fig = px.choropleth(
        plot_df, locations='ISO3', color='Opportunity_Score',
        hover_name='Country',
        hover_data={'Revenue_M': ':.3f', 'Population_M': True, 'ISO3': False},
        color_continuous_scale=['#fff3e0', '#ff9800', '#bf360c'],
        projection='natural earth',
        title='White Space Markets — High Population, Low Ashwagandha Revenue',
        labels={'Opportunity_Score': 'Opp. Score', 'Revenue_M': 'Revenue ($M)', 'Population_M': 'Pop. (M)'},
    )
    fig.update_layout(
        **_LAYOUT, height=500,
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor='#dddddd',
                 showland=True, landcolor='#f5f5f5'),
        coloraxis_colorbar=dict(title='Opp.\nScore'),
    )
    return fig


def partner_product_pie(df: pd.DataFrame, exporter_name: str) -> go.Figure:
    """Product mix pie chart for a single exporter."""
    sub = df[df['EXPORTER NAME'] == exporter_name]
    prod_mix = sub.groupby('PRODUCT_TYPE')['USD FOB'].sum().reset_index()
    prod_mix = prod_mix[prod_mix['USD FOB'] > 0]

    if prod_mix.empty:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT, title='No data', height=280)
        return fig

    _ptype_colors = [COLORS['dark_green'], COLORS['mid_green'], COLORS['gold'],
                     COLORS['blue'], '#9C27B0', '#8BC34A', '#aaaaaa']
    fig = px.pie(
        prod_mix, names='PRODUCT_TYPE', values='USD FOB',
        color_discrete_sequence=_ptype_colors,
        title='Product Mix',
    )
    fig.update_traces(textinfo='percent+label', textposition='outside')
    _pie_layout = {**_LAYOUT, 'margin': dict(l=10, r=10, t=40, b=10)}
    fig.update_layout(**_pie_layout, height=300, showlegend=False)
    return fig


def brand_bubble(df: pd.DataFrame) -> go.Figure:
    """Bubble: x=Avg Deal, y=Frequency, size=Revenue for single-brand importers."""
    single = df[~df['IMPORTER NAME'].str.contains('[,/]', na=True)].copy()
    brand_stats = (
        single.groupby('IMPORTER NAME')
        .agg(avg_deal=('USD FOB', 'mean'), freq=('USD FOB', 'count'), total_rev=('USD FOB', 'sum'))
        .reset_index()
    )
    brand_stats = brand_stats[brand_stats['avg_deal'] > 0].nlargest(80, 'total_rev')

    fig = px.scatter(
        brand_stats, x='avg_deal', y='freq', size='total_rev',
        hover_name='IMPORTER NAME', color='total_rev',
        color_continuous_scale='Greens', size_max=60, log_x=True,
        title='Brand Landscape — Avg Deal Size vs Shipment Frequency (size = Total Revenue)',
        labels={'avg_deal': 'Avg Deal ($, log)', 'freq': 'Shipment Frequency', 'total_rev': 'Revenue ($)'},
    )
    fig.update_layout(**_LAYOUT, height=500, coloraxis_showscale=False, xaxis_tickformat='$,.0f')
    return fig


def channel_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart: % of shipments by channel."""
    df2 = _add_channel(df)
    ch = df2.groupby('CHANNEL').size().reset_index(name='Shipments')
    colors = [_CHANNEL_COLORS.get(c, '#aaaaaa') for c in ch['CHANNEL']]

    fig = go.Figure(go.Pie(
        labels=ch['CHANNEL'], values=ch['Shipments'],
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Shipments: %{value:,}<br>%{percent}<extra></extra>',
    ))
    _pie_layout = {**_LAYOUT, 'margin': dict(l=10, r=10, t=50, b=10)}
    fig.update_layout(**_pie_layout, title='Channel Mix — % of Shipments', height=360)
    return fig


def channel_revenue_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: % of revenue by channel."""
    df2 = _add_channel(df)
    ch = df2.groupby('CHANNEL')['USD FOB'].sum().reset_index(name='Revenue')
    total = ch['Revenue'].sum()
    ch['Pct'] = ch['Revenue'] / total * 100 if total > 0 else 0
    ch = ch.sort_values('Pct', ascending=True)
    colors = [_CHANNEL_COLORS.get(c, '#aaaaaa') for c in ch['CHANNEL']]

    fig = go.Figure(go.Bar(
        x=ch['Pct'], y=ch['CHANNEL'], orientation='h',
        marker_color=colors,
        text=[f'{p:.1f}%' for p in ch['Pct']], textposition='outside',
        hovertemplate='<b>%{y}</b><br>Revenue Share: %{x:.1f}%<extra></extra>',
    ))
    fig.update_layout(**_LAYOUT, title='Channel Mix — % of Revenue', height=300,
                      xaxis=dict(title='Revenue Share (%)', range=[0, 80]), yaxis_title='')
    return fig


def channel_stacked_area(df: pd.DataFrame) -> go.Figure:
    """Quarterly 100% stacked area chart by channel."""
    df2 = _add_channel(df)
    pivot = (
        df2.groupby(['YQ', 'CHANNEL'])['USD FOB'].sum()
        .unstack(fill_value=0).reset_index()
    )
    ch_cols = [c for c in pivot.columns if c != 'YQ']
    row_totals = pivot[ch_cols].sum(axis=1).replace(0, 1)
    for c in ch_cols:
        pivot[f'{c}_pct'] = pivot[c] / row_totals * 100

    fig = go.Figure()
    for ch in ['B2B Mega', 'B2B Mid', 'B2B Retail', 'B2C']:
        pct_col = f'{ch}_pct'
        if pct_col not in pivot.columns:
            continue
        hx = _CHANNEL_COLORS.get(ch, '#aaaaaa')
        r, g, b = int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)
        fig.add_trace(go.Scatter(
            x=pivot['YQ'], y=pivot[pct_col].tolist(), name=ch,
            mode='lines', stackgroup='one',
            line=dict(color=hx, width=0), fillcolor=f'rgba({r},{g},{b},0.85)',
            hovertemplate=f'%{{x}}: %{{y:.1f}}%<extra>{ch}</extra>',
        ))
    fig.update_layout(
        **_LAYOUT, title='Channel Revenue Mix by Quarter (100% Stacked)',
        xaxis_title='Quarter', yaxis=dict(title='Revenue Share (%)', range=[0, 100], ticksuffix='%'),
        height=380, legend=dict(x=0.01, y=0.99),
    )
    return fig


def transport_mode_bars(df: pd.DataFrame) -> go.Figure:
    """Side-by-side bar subplots: revenue ($M) and shipment count by transport mode."""
    tm = (
        df.groupby('TRANSPORT_NORM')
        .agg(Revenue=('USD FOB', 'sum'), Shipments=('USD FOB', 'count'))
        .reset_index().sort_values('Revenue', ascending=False)
    )
    _mc = [COLORS['dark_green'], COLORS['gold'], COLORS['blue'], COLORS['mid_green'], '#aaaaaa']

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Revenue by Transport Mode ($M)', 'Shipments by Mode'])
    colors = _mc[:len(tm)]
    fig.add_trace(
        go.Bar(x=tm['TRANSPORT_NORM'], y=(tm['Revenue'] / 1e6).tolist(), marker_color=colors,
               text=[f'${v:.1f}M' for v in tm['Revenue'] / 1e6], textposition='outside',
               hovertemplate='%{x}: $%{y:.1f}M<extra></extra>', name='Revenue'),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=tm['TRANSPORT_NORM'], y=tm['Shipments'].tolist(), marker_color=colors,
               text=[f'{v:,}' for v in tm['Shipments']], textposition='outside',
               hovertemplate='%{x}: %{y:,}<extra></extra>', name='Shipments'),
        row=1, col=2,
    )
    fig.update_layout(**_LAYOUT, title='Transport Mode Economics', height=420, showlegend=False)
    return fig


def transport_avg_deal_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: average deal size ($K) by transport mode."""
    tm = (
        df.groupby('TRANSPORT_NORM')
        .agg(avg_deal=('USD FOB', 'mean'), n=('USD FOB', 'count'))
        .reset_index()
    )
    tm = tm[tm['n'] >= 5].sort_values('avg_deal', ascending=True)
    _mc = [COLORS['dark_green'], COLORS['gold'], COLORS['blue'], COLORS['mid_green'], '#aaaaaa']

    fig = go.Figure(go.Bar(
        x=tm['avg_deal'] / 1e3, y=tm['TRANSPORT_NORM'], orientation='h',
        marker_color=_mc[:len(tm)],
        text=[f'${v:.0f}K' for v in tm['avg_deal'] / 1e3], textposition='outside',
        hovertemplate='%{y}: $%{x:,.0f}K avg deal<extra></extra>',
    ))
    fig.update_layout(**_LAYOUT, title='Average Deal Size by Transport Mode',
                      xaxis_title='Avg Deal ($K)', yaxis_title='', height=300, showlegend=False)
    return fig


def transport_yoy_bar(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: revenue share by transport mode, one bar per year (last 3 years)."""
    years_avail = sorted(df['YEAR'].unique().tolist())[-3:]
    sub = df[df['YEAR'].isin(years_avail)]
    pivot = (
        sub.groupby(['YEAR', 'TRANSPORT_NORM'])['USD FOB'].sum()
        .unstack(fill_value=0).reset_index()
    )
    mode_cols = [c for c in pivot.columns if c != 'YEAR']
    row_totals = pivot[mode_cols].sum(axis=1).replace(0, 1)

    _mode_colors_map = {'Air': COLORS['gold'], 'Sea': COLORS['blue'],
                        'Road': COLORS['mid_green'], 'ICD': '#9C27B0', 'Other': '#aaaaaa'}
    fig = go.Figure()
    for mode in mode_cols:
        pcts = (pivot[mode] / row_totals * 100).tolist()
        color = _mode_colors_map.get(str(mode), '#aaaaaa')
        fig.add_trace(go.Bar(
            name=str(mode), x=pivot['YEAR'].astype(str).tolist(), y=pcts,
            marker_color=color,
            hovertemplate=f'%{{x}}: %{{y:.1f}}%<extra>{mode}</extra>',
        ))
    fig.update_layout(
        **_LAYOUT, barmode='stack',
        title='Transport Mode Revenue Share — Year-over-Year Shift',
        xaxis_title='Year', yaxis=dict(title='Revenue Share (%)', range=[0, 100], ticksuffix='%'),
        height=380, legend=dict(x=1.01, y=0.5),
    )
    return fig


def competitive_positioning_scatter(df: pd.DataFrame) -> go.Figure:
    """Price/kg (y) vs total volume (x, log), size=revenue, color=exporter country."""
    sub = df[df['price_per_kg'].notna() & (df['NET WEIGHT'] > 0) & (df['USD FOB'] > 0)].copy()
    exp_stats = (
        sub.groupby(['EXPORTER NAME', 'EXPORTER COUNTRY'])
        .agg(median_price=('price_per_kg', 'median'),
             total_volume=('NET WEIGHT', 'sum'),
             total_rev=('USD FOB', 'sum'))
        .reset_index()
    )
    exp_stats = exp_stats[
        (exp_stats['median_price'] > 0) & (exp_stats['total_volume'] > 0)
    ].nlargest(40, 'total_rev')

    if exp_stats.empty:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT, title='No data for competitive positioning', height=500)
        return fig

    fig = px.scatter(
        exp_stats, x='total_volume', y='median_price', size='total_rev',
        color='EXPORTER COUNTRY', hover_name='EXPORTER NAME',
        hover_data={'EXPORTER COUNTRY': True, 'total_rev': ':$,.0f', 'median_price': ':.0f'},
        size_max=55, log_x=True,
        title='Competitive Positioning — Price/kg vs Volume (size = Revenue)',
        labels={'total_volume': 'Total Volume kg (log)', 'median_price': 'Median Price ($/kg)',
                'EXPORTER COUNTRY': 'Country'},
    )
    med_vol = float(exp_stats['total_volume'].median())
    med_price = float(exp_stats['median_price'].median())
    fig.add_shape(type='line', xref='x', yref='paper',
                  x0=med_vol, x1=med_vol, y0=0, y1=1,
                  line=dict(color='#aaaaaa', dash='dash', width=1.2))
    fig.add_shape(type='line', xref='paper', yref='y',
                  x0=0, x1=1, y0=med_price, y1=med_price,
                  line=dict(color='#aaaaaa', dash='dash', width=1.2))
    for label, xp, yp in [
        ('Premium\nLow-Volume', 0.04, 0.90), ('Premium\nHigh-Volume', 0.88, 0.90),
        ('Commodity\nHigh-Volume', 0.88, 0.10), ('Commodity\nLow-Volume', 0.04, 0.10),
    ]:
        fig.add_annotation(xref='paper', yref='paper', x=xp, y=yp,
                           text=f'<b>{label}</b>', showarrow=False,
                           font=dict(size=10, color='#555555'),
                           bgcolor='rgba(255,255,255,0.75)')
    fig.update_layout(**_LAYOUT, height=530, xaxis_tickformat=',.0f', yaxis_tickformat='$,.0f')
    return fig


def exporter_share_treemap(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """px.treemap of top-N exporters by revenue."""
    top_exp = (
        df.groupby('EXPORTER NAME')['USD FOB'].sum()
        .nlargest(top_n).reset_index().rename(columns={'USD FOB': 'Revenue'})
    )
    fig = px.treemap(
        top_exp, path=['EXPORTER NAME'], values='Revenue',
        color='Revenue', color_continuous_scale='Greens',
        title=f'Exporter Market Share — Top {top_n} Exporters',
    )
    fig.update_traces(textinfo='label+percent root')
    _tm = {**_LAYOUT, 'margin': dict(l=10, r=10, t=50, b=10)}
    fig.update_layout(**_tm, height=480, coloraxis_showscale=False)
    return fig

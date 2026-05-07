from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def compute_hhi(revenue_series: pd.Series) -> float:
    """Standard HHI × 10,000. Range [0, 10000]."""
    s = revenue_series.dropna()
    if s.empty or s.sum() == 0:
        return 0.0
    shares = s / s.sum()
    return float((shares ** 2).sum() * 10_000)


def compute_gini(revenue_series: pd.Series) -> float:
    """Gini coefficient [0, 1]. Higher = more concentrated."""
    s = revenue_series.dropna().sort_values().values
    if len(s) == 0 or s.sum() == 0:
        return 0.0
    n = len(s)
    cumsum = np.cumsum(s)
    return float((2 * np.sum((np.arange(1, n + 1)) * s) - (n + 1) * s.sum()) / (n * s.sum()))


def compute_lorenz(revenue_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Returns (x_pct_buyers, y_pct_revenue), downsampled to 500 pts."""
    s = revenue_series.dropna().sort_values().values
    if len(s) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    cumrev = np.cumsum(s) / s.sum()
    x = np.linspace(0, 1, len(s) + 1)
    y = np.concatenate([[0.0], cumrev])
    if len(x) > 500:
        idx = np.linspace(0, len(x) - 1, 500, dtype=int)
        x, y = x[idx], y[idx]
    return x, y


def compute_cagr(start: float, end: float, years: float) -> float:
    """Compound Annual Growth Rate."""
    if start <= 0 or years <= 0:
        return 0.0
    return float((end / start) ** (1 / years) - 1)


def hhi_label(hhi: float) -> str:
    """Return concentration label for HHI score."""
    if hhi < 1500:
        return 'Competitive'
    elif hhi < 2500:
        return 'Moderately Concentrated'
    elif hhi < 5000:
        return 'Concentrated'
    return 'Highly Concentrated'


def compute_concentration_ratio(revenue_series: pd.Series, n: int) -> float:
    """CR-N: Revenue share of top-N entities."""
    s = revenue_series.dropna()
    if s.empty or s.sum() == 0:
        return 0.0
    top_n = s.nlargest(n).sum()
    return float(top_n / s.sum())


def compute_seasonal_demand_index(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue / annual average × 100 → SDI per month."""
    monthly = df.groupby('MONTH')['USD FOB'].sum().reset_index()
    annual_avg = monthly['USD FOB'].mean()
    if annual_avg == 0:
        monthly['SDI'] = 100.0
    else:
        monthly['SDI'] = (monthly['USD FOB'] / annual_avg) * 100
    monthly['Month_Name'] = monthly['MONTH'].apply(
        lambda m: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m - 1]
    )
    return monthly


def compute_supply_risk_score(exporter_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each exporter, compute 0–100 risk score from 4 components.
    exporter_df must have columns: EXPORTER NAME, IMPORTER COUNTRY, USD FOB, YEAR.
    """
    records = []
    for name, grp in exporter_df.groupby('EXPORTER NAME'):
        total_rev = grp['USD FOB'].sum()
        if total_rev == 0:
            continue

        # Component 1: customer concentration (CR-3 of importer countries)
        imp_rev = grp.groupby('IMPORTER COUNTRY')['USD FOB'].sum()
        cr3 = compute_concentration_ratio(imp_rev, 3)
        c1 = cr3 * 25

        # Component 2: deal size variance (CV)
        deals = grp['USD FOB']
        cv = (deals.std() / deals.mean()) if deals.mean() > 0 else 0
        c2 = min(cv / 3, 1.0) * 25

        # Component 3: revenue stability (YoY change 2023→2024 or 2024→2025)
        yoy = grp.groupby('YEAR')['USD FOB'].sum().sort_index()
        if len(yoy) >= 2:
            pct_change = abs((yoy.iloc[-1] - yoy.iloc[-2]) / max(yoy.iloc[-2], 1))
            c3 = min(pct_change / 2, 1.0) * 25
        else:
            c3 = 25.0  # no history = max instability

        # Component 4: market breadth (inverse of unique importer countries)
        n_countries = grp['IMPORTER COUNTRY'].nunique()
        c4 = max(0, 1 - n_countries / 20) * 25

        score = c1 + c2 + c3 + c4
        level = (
            'Low' if score < 25 else
            'Moderate' if score < 50 else
            'High' if score < 75 else
            'Critical'
        )
        records.append({
            'EXPORTER NAME': name,
            'Risk_Score': round(score, 1),
            'Risk_Level': level,
            'CR3': round(cr3, 3),
            'Deal_CV': round(cv, 3),
            'Countries': n_countries,
            'Total_Revenue': total_rev,
        })

    return pd.DataFrame(records).sort_values('Risk_Score', ascending=False)


def compute_buyer_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    K-means clustering (k=3) on log(avg_deal_size), log(shipment_frequency).
    Returns df with BUYER_SEGMENT column: 'Mega B2B', 'Mid-Market', 'Small/Retail'.
    """
    buyer_stats = (
        df.groupby('IMPORTER NAME')
        .agg(
            avg_deal=('USD FOB', 'mean'),
            freq=('USD FOB', 'count'),
            total_rev=('USD FOB', 'sum'),
        )
        .reset_index()
    )
    buyer_stats = buyer_stats[buyer_stats['avg_deal'] > 0].copy()

    features = np.column_stack([
        np.log1p(buyer_stats['avg_deal'].clip(upper=buyer_stats['avg_deal'].quantile(0.99))),
        np.log1p(buyer_stats['freq']),
    ])

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    buyer_stats['_cluster'] = labels

    # Label clusters by median total_rev (highest = Mega)
    cluster_rev = buyer_stats.groupby('_cluster')['total_rev'].median().sort_values(ascending=False)
    label_map = {
        cluster_rev.index[0]: 'Mega B2B',
        cluster_rev.index[1]: 'Mid-Market',
        cluster_rev.index[2]: 'Small/Retail',
    }
    buyer_stats['BUYER_SEGMENT'] = buyer_stats['_cluster'].map(label_map)
    return buyer_stats[['IMPORTER NAME', 'BUYER_SEGMENT', 'avg_deal', 'freq', 'total_rev']]

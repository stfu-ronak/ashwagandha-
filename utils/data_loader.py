from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .constants import COUNTRY_ISO3_MAP
from .metrics import compute_buyer_segments

_DATA_PATH = Path(__file__).parent.parent / "data" / "Ashwagandha - Data 2023-26.xlsx"

_PRODUCT_KEYWORDS: list[tuple[str, str]] = [
    ('KSM-66 / Sensoril',    lambda t: 'ksm' in t or 'sensoril' in t),
    ('Standardized Extract', lambda t: 'extract' in t or 'withanolide' in t),
    ('Finished Dosage',      lambda t: any(k in t for k in ('capsule', 'tablet', 'softgel'))),
    ('Organic Powder',       lambda t: 'powder' in t and 'organic' in t),
    ('Raw Powder',           lambda t: 'powder' in t),
    ('Root',                 lambda t: 'root' in t),
]

_TRANSPORT_MAP: dict[str, str] = {
    'air': 'Air', 'courier': 'Air',
    'sea': 'Sea', 'ocean': 'Sea',
    'road': 'Road', 'truck': 'Road', 'land': 'Road',
    'icd': 'ICD',
}

_DEAL_BINS = [0, 1_000, 5_000, 25_000, 100_000, 500_000, float('inf')]
_DEAL_LABELS = ['<$1K', '$1K-5K', '$5K-25K', '$25K-100K', '$100K-500K', '>$500K']

_HS_LABELS: dict[str, str] = {
    '1302': 'Vegetable Extracts',
    '1211': 'Plants / Plant Parts',
    '0910': 'Spices / Roots',
    '2106': 'Food Preparations',
    '2941': 'Antibiotics / Botanicals',
    '3004': 'Medicaments',
    '3203': 'Colouring Matter',
}

_PORT_NORM: dict[str, str] = {
    'new york': 'New York', 'jfk': 'New York', 'newark': 'New York',
    'heathrow': 'London - Heathrow', 'london heathrow': 'London - Heathrow', 'london - heathrow': 'London - Heathrow',
    'los angeles': 'Los Angeles', 'lax': 'Los Angeles',
    'hamburg': 'Hamburg',
    'amsterdam': 'Amsterdam/Schiphol', 'schiphol': 'Amsterdam/Schiphol', 'amsterdam/schiphol': 'Amsterdam/Schiphol',
    'singapore': 'Singapore',
    'sydney': 'Sydney',
    'melbourne': 'Melbourne',
    'toronto': 'Toronto', 'pearson': 'Toronto',
    'houston': 'Houston',
    'frankfurt': 'Frankfurt',
    'warsaw': 'Warsaw',
    'gdynia': 'Gdynia',
    'dubai': 'Dubai',
    'almaty': 'Almaty',
    'mukachevo': 'Mukachevo/Ukraine',
    'delhi': 'Delhi', 'new delhi': 'Delhi', 'indira gandhi': 'Delhi',
    'mumbai': 'Mumbai', 'bombay': 'Mumbai',
    'chennai': 'Chennai', 'madras': 'Chennai',
    'bangalore': 'Bangalore', 'bengaluru': 'Bangalore',
    'hyderabad': 'Hyderabad',
    'mundra': 'Mundra',
    'cochin': 'Cochin', 'kochi': 'Cochin',
    'nhava sheva': 'Nhava Sheva', 'jnpt': 'Nhava Sheva',
    'bucharest': 'Bucharest',
    'dublin': 'Dublin',
    'auckland': 'Auckland',
    'oakland': 'Oakland',
    'seoul': 'Seoul', 'incheon': 'Seoul',
    'cengkareng': 'Cengkareng/Jakarta', 'jakarta': 'Cengkareng/Jakarta',
}


def _classify_product(text: str) -> str:
    t = str(text).lower()
    for label, check in _PRODUCT_KEYWORDS:
        if check(t):
            return label
    return 'Other'


def _normalize_transport(text: str) -> str:
    t = str(text).lower()
    for k, v in _TRANSPORT_MAP.items():
        if k in t:
            return v
    return 'Other'


def _normalize_port(text: str) -> str:
    t = str(text).lower().strip()
    for k, v in _PORT_NORM.items():
        if k in t:
            return v
    return str(text).strip()


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    df = pd.read_excel(_DATA_PATH, sheet_name='TradeAtlas Shipment Records', header=1)

    # Parse date
    df['ARRIVAL DATE'] = pd.to_datetime(df['ARRIVAL DATE'], errors='coerce')
    df = df.dropna(subset=['ARRIVAL DATE'])

    # Temporal columns
    df['YEAR']    = df['ARRIVAL DATE'].dt.year
    df['MONTH']   = df['ARRIVAL DATE'].dt.month
    df['QUARTER'] = df['ARRIVAL DATE'].dt.quarter
    df['YM_STR']  = df['ARRIVAL DATE'].dt.to_period('M').astype(str)
    df['YQ']      = df['YEAR'].astype(str) + 'Q' + df['QUARTER'].astype(str)

    # Numeric safety
    df['USD FOB']    = pd.to_numeric(df['USD FOB'],    errors='coerce').fillna(0)
    df['NET WEIGHT'] = pd.to_numeric(df['NET WEIGHT'], errors='coerce').fillna(0)

    # Price per kg (capped at 5000)
    mask = (df['USD FOB'] > 0) & (df['NET WEIGHT'] > 0)
    df['price_per_kg'] = float('nan')
    df.loc[mask, 'price_per_kg'] = df.loc[mask, 'USD FOB'] / df.loc[mask, 'NET WEIGHT']
    df.loc[df['price_per_kg'] > 5000, 'price_per_kg'] = float('nan')

    # Product type
    df['PRODUCT_TYPE'] = df['PRODUCT DETAILS'].apply(_classify_product)

    # Transport normalised
    df['TRANSPORT_NORM'] = df['TRANSPORT TYPE'].apply(_normalize_transport)

    # Deal segment
    df['deal_segment'] = pd.cut(
        df['USD FOB'],
        bins=_DEAL_BINS,
        labels=_DEAL_LABELS,
        right=True,
    )

    # HS group
    df['HS_GROUP'] = df['HS CODE'].astype(str).str[:4]
    df['HS_GROUP_LABEL'] = df['HS_GROUP'].map(_HS_LABELS).fillna('Other')

    # Country ISO3
    df['IMPORTER_ISO3'] = df['IMPORTER COUNTRY'].map(COUNTRY_ISO3_MAP)
    df['EXPORTER_ISO3'] = df['EXPORTER COUNTRY'].map(COUNTRY_ISO3_MAP)

    # Port normalisation
    df['PORT_ARRIVAL_CLEAN'] = df['PORT OF ARRIVAL'].apply(_normalize_port)

    # Buyer segments
    try:
        seg_df = compute_buyer_segments(df)
        df = df.merge(
            seg_df[['IMPORTER NAME', 'BUYER_SEGMENT']],
            on='IMPORTER NAME',
            how='left',
        )
    except Exception:
        df['BUYER_SEGMENT'] = 'Mid-Market'

    df['BUYER_SEGMENT'] = df['BUYER_SEGMENT'].fillna('Mid-Market')
    return df


def apply_filters(
    df: pd.DataFrame,
    years: tuple[int, int] | None = None,
    quarters: list[int] | None = None,
    imp_countries: list[str] | None = None,
    exp_countries: list[str] | None = None,
    imp_names: list[str] | None = None,
    exp_names: list[str] | None = None,
    product_types: list[str] | None = None,
    transport_modes: list[str] | None = None,
    deal_min: float = 0,
    deal_max: float = float('inf'),
) -> pd.DataFrame:
    out = df.copy()
    if years:
        out = out[(out['YEAR'] >= years[0]) & (out['YEAR'] <= years[1])]
    if quarters:
        out = out[out['QUARTER'].isin(quarters)]
    if imp_countries:
        out = out[out['IMPORTER COUNTRY'].isin(imp_countries)]
    if exp_countries:
        out = out[out['EXPORTER COUNTRY'].isin(exp_countries)]
    if imp_names:
        out = out[out['IMPORTER NAME'].isin(imp_names)]
    if exp_names:
        out = out[out['EXPORTER NAME'].isin(exp_names)]
    if product_types:
        out = out[out['PRODUCT_TYPE'].isin(product_types)]
    if transport_modes:
        out = out[out['TRANSPORT_NORM'].isin(transport_modes)]
    if deal_min > 0:
        out = out[out['USD FOB'] >= deal_min]
    if deal_max < float('inf'):
        out = out[out['USD FOB'] <= deal_max]
    return out

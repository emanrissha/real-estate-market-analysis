"""
Feature engineering and preprocessing for Real Estate dataset
"""

import pandas as pd
import numpy as np
import os


def preprocess(df):
    """Engineer all features from raw loaded DataFrame"""
    df = df.copy()

    # --- Date features ---
    df['year']  = df['transaction_date'].astype(int)
    df['month'] = ((df['transaction_date'] % 1) * 12).round().astype(int).clip(1, 12)
    df['date']  = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str) + '-01'
    )

    # --- MRT distance features ---
    df['log_mrt_distance'] = np.log1p(df['mrt_distance'])
    df['mrt_category'] = pd.cut(
        df['mrt_distance'],
        bins=[0, 500, 1500, 5000, 99999],
        labels=['Very Close', 'Close', 'Far', 'Very Far']
    )

    # --- House age features ---
    df['age_category'] = pd.cut(
        df['house_age'],
        bins=[-1, 5, 15, 30, 999],
        labels=['New', 'Recent', 'Middle-aged', 'Old']
    )

    # --- Price features ---
    df['price_segment'] = pd.qcut(
        df['price_per_unit'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )
    df['price_segment_encoded'] = pd.qcut(
        df['price_per_unit'],
        q=3,
        labels=[0, 1, 2]
    ).astype(int)

    # --- Location features ---
    df['lat_centered'] = df['latitude']  - df['latitude'].mean()
    df['lng_centered'] = df['longitude'] - df['longitude'].mean()
    df['distance_from_center'] = np.sqrt(
        df['lat_centered']**2 + df['lng_centered']**2
    )

    print(f"✅ Preprocessing complete: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    print(f"   Price range: {df['price_per_unit'].min()} - {df['price_per_unit'].max()} per unit")
    print(f"   MRT range:   {df['mrt_distance'].min():.0f} - {df['mrt_distance'].max():.0f} meters")
    print(f"   Age range:   {df['house_age'].min()} - {df['house_age'].max()} years")

    return df


def save(df, path='data/processed/real_estate_clean.csv'):
    """Save processed DataFrame to CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved to {path}")
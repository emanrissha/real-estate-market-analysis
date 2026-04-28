"""
Data loader for Taiwan Real Estate dataset
"""

import pandas as pd
import os


def load_raw(path='data/raw/real_estate.csv'):
    """Load raw CSV and return clean DataFrame with renamed columns"""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found at: {path}")

    df = pd.read_csv(path)

    df = df.rename(columns={
        'No':                                    'id',
        'X1 transaction date':                   'transaction_date',
        'X2 house age':                          'house_age',
        'X3 distance to the nearest MRT station':'mrt_distance',
        'X4 number of convenience stores':       'convenience_stores',
        'X5 latitude':                           'latitude',
        'X6 longitude':                          'longitude',
        'Y house price of unit area':            'price_per_unit'
    })

    print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate(df):
    """Basic validation checks"""
    required = ['id','transaction_date','house_age','mrt_distance',
                'convenience_stores','latitude','longitude','price_per_unit']

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    nulls = df[required].isnull().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        print(f"⚠️  Nulls found:\n{nulls}")

    assert df['price_per_unit'].min() > 0, "Negative prices found"
    assert df['house_age'].min() >= 0,     "Negative house age found"
    assert df['mrt_distance'].min() >= 0,  "Negative MRT distance found"

    print("✅ Validation passed")
    return True
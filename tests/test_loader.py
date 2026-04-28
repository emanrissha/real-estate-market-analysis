"""
Tests for data loader
"""

import pytest
import pandas as pd
from src.data.loader import load_raw, validate


def test_load_raw_returns_dataframe():
    df = load_raw()
    assert isinstance(df, pd.DataFrame)


def test_load_raw_correct_shape():
    df = load_raw()
    assert df.shape[0] == 414
    assert df.shape[1] == 8


def test_load_raw_correct_columns():
    df = load_raw()
    expected = [
        'id', 'transaction_date', 'house_age',
        'mrt_distance', 'convenience_stores',
        'latitude', 'longitude', 'price_per_unit'
    ]
    assert list(df.columns) == expected


def test_load_raw_no_nulls_in_key_columns():
    df = load_raw()
    key_cols = ['price_per_unit', 'mrt_distance', 'house_age']
    assert df[key_cols].isnull().sum().sum() == 0


def test_load_raw_price_positive():
    df = load_raw()
    assert df['price_per_unit'].min() > 0


def test_load_raw_mrt_distance_positive():
    df = load_raw()
    assert df['mrt_distance'].min() >= 0


def test_load_raw_house_age_non_negative():
    df = load_raw()
    assert df['house_age'].min() >= 0


def test_validate_passes():
    df = load_raw()
    assert validate(df) is True


def test_validate_fails_on_missing_column():
    df = load_raw().drop(columns=['price_per_unit'])
    with pytest.raises(ValueError):
        validate(df)
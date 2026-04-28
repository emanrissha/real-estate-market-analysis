"""
Tests for preprocessor
"""

import pytest
import pandas as pd
from src.data.loader import load_raw
from src.data.preprocessor import preprocess


@pytest.fixture
def processed_df():
    return preprocess(load_raw())


def test_preprocess_returns_dataframe(processed_df):
    assert isinstance(processed_df, pd.DataFrame)


def test_preprocess_row_count_unchanged(processed_df):
    assert len(processed_df) == 414


def test_preprocess_adds_columns(processed_df):
    new_cols = [
        'year', 'month', 'date',
        'log_mrt_distance', 'mrt_category',
        'age_category', 'price_segment',
        'price_segment_encoded', 'distance_from_center'
    ]
    for col in new_cols:
        assert col in processed_df.columns, f"Missing column: {col}"


def test_log_mrt_distance_positive(processed_df):
    assert processed_df['log_mrt_distance'].min() >= 0


def test_price_segment_encoded_values(processed_df):
    assert set(processed_df['price_segment_encoded'].unique()) == {0, 1, 2}


def test_price_segment_balanced(processed_df):
    counts = processed_df['price_segment'].value_counts()
    assert counts.min() >= 130


def test_date_column_is_datetime(processed_df):
    assert pd.api.types.is_datetime64_any_dtype(processed_df['date'])


def test_year_range(processed_df):
    assert processed_df['year'].min() >= 2012
    assert processed_df['year'].max() <= 2013


def test_distance_from_center_non_negative(processed_df):
    assert processed_df['distance_from_center'].min() >= 0
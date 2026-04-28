"""
Tests for ML models
"""

import pytest
import pandas as pd
import numpy as np
from src.data.loader import load_raw
from src.data.preprocessor import preprocess
from src.models.price_predictor import PricePredictor
from src.models.price_classifier import PriceClassifier
from src.models.location_segmentation import LocationSegmentation
from src.models.time_series_forecast import TimeSeriesForecast


@pytest.fixture(scope='module')
def df():
    return preprocess(load_raw())


@pytest.fixture(scope='module')
def trained_price_predictor(df):
    model = PricePredictor()
    model.train(df)
    return model


@pytest.fixture(scope='module')
def trained_classifier(df):
    model = PriceClassifier()
    model.train(df)
    return model


@pytest.fixture(scope='module')
def trained_segmentation(df):
    model = LocationSegmentation()
    model.fit(df)
    return model


@pytest.fixture(scope='module')
def trained_forecast(df):
    model = TimeSeriesForecast()
    model.train(df)
    return model


# --- Price Predictor ---

def test_price_predictor_r2(df):
    model = PricePredictor()
    results = model.train(df)
    assert results['r2'] >= 0.70, f"R² too low: {results['r2']}"


def test_price_predictor_cv_r2(df):
    model = PricePredictor()
    results = model.train(df)
    assert results['cv_r2_mean'] >= 0.50, f"CV R² too low: {results['cv_r2_mean']}"


def test_price_predictor_predict_returns_float(trained_price_predictor):
    result = trained_price_predictor.predict(
        house_age=10, mrt_distance=300,
        convenience_stores=5,
        latitude=24.983, longitude=121.540
    )
    assert isinstance(result, float)


def test_price_predictor_predict_positive(trained_price_predictor):
    result = trained_price_predictor.predict(
        house_age=10, mrt_distance=300,
        convenience_stores=5,
        latitude=24.983, longitude=121.540
    )
    assert result > 0


def test_price_predictor_mrt_effect(trained_price_predictor):
    """Closer MRT should predict higher price"""
    close = trained_price_predictor.predict(
        house_age=10, mrt_distance=100,
        convenience_stores=5,
        latitude=24.983, longitude=121.540
    )
    far = trained_price_predictor.predict(
        house_age=10, mrt_distance=3000,
        convenience_stores=5,
        latitude=24.983, longitude=121.540
    )
    assert close > far


# --- Price Classifier ---

def test_price_classifier_accuracy(df):
    model = PriceClassifier()
    results = model.train(df)
    assert results['accuracy'] >= 0.70, f"Accuracy too low: {results['accuracy']}"


def test_price_classifier_cv_accuracy(df):
    model = PriceClassifier()
    results = model.train(df)
    assert results['cv_accuracy_mean'] >= 0.65


def test_price_classifier_predict_valid_label(trained_classifier):
    result = trained_classifier.predict(
        house_age=10, mrt_distance=300,
        convenience_stores=5,
        latitude=24.983, longitude=121.540
    )
    assert result in ['Low', 'Medium', 'High']


# --- Location Segmentation ---

def test_segmentation_silhouette(df):
    model = LocationSegmentation()
    results = model.fit(df)
    assert results['silhouette_score'] >= 0.25


def test_segmentation_correct_n_clusters(df):
    model = LocationSegmentation(n_clusters=4)
    results = model.fit(df)
    assert results['n_clusters'] == 4


def test_segmentation_predict_returns_dict(trained_segmentation):
    result = trained_segmentation.predict(
        latitude=24.983, longitude=121.540,
        mrt_distance=300, convenience_stores=5,
        price_per_unit=40
    )
    assert 'segment' in result
    assert 'label' in result


def test_segmentation_segment_id_valid(trained_segmentation):
    result = trained_segmentation.predict(
        latitude=24.983, longitude=121.540,
        mrt_distance=300, convenience_stores=5,
        price_per_unit=40
    )
    assert result['segment'] in [0, 1, 2, 3]


# --- Time Series Forecast ---

def test_forecast_r2(df):
    model = TimeSeriesForecast()
    results = model.train(df)
    assert results['r2'] >= 0.50


def test_forecast_returns_correct_months(trained_forecast):
    result = trained_forecast.forecast(months_ahead=6)
    assert len(result) == 6


def test_forecast_values_positive(trained_forecast):
    result = trained_forecast.forecast(months_ahead=3)
    assert all(v > 0 for v in result.values())


def test_forecast_raises_before_train():
    model = TimeSeriesForecast()
    with pytest.raises(ValueError):
        model.forecast(3)
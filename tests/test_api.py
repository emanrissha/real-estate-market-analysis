"""
Tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.api.endpoints import load_models

client = TestClient(app)

# Load models once before tests
load_models()


def test_root_returns_200():
    response = client.get('/')
    assert response.status_code == 200


def test_root_contains_name():
    response = client.get('/')
    assert 'Real Estate' in response.json()['name']


def test_health_returns_200():
    response = client.get('/health')
    assert response.status_code == 200


def test_health_status_ok():
    response = client.get('/health')
    assert response.json()['status'] == 'ok'


def test_health_dataset_rows():
    response = client.get('/health')
    assert response.json()['dataset_rows'] == 414


def test_statistics_returns_200():
    response = client.get('/statistics')
    assert response.status_code == 200


def test_statistics_correct_total():
    response = client.get('/statistics')
    assert response.json()['total_properties'] == 414


def test_statistics_avg_price_reasonable():
    response = client.get('/statistics')
    avg = response.json()['avg_price_per_unit']
    assert 30 <= avg <= 50


def test_predict_price_returns_200():
    response = client.post('/predict/price', json={
        'house_age': 10.0,
        'mrt_distance': 300.0,
        'convenience_stores': 5,
        'latitude': 24.983,
        'longitude': 121.540
    })
    assert response.status_code == 200


def test_predict_price_positive_value():
    response = client.post('/predict/price', json={
        'house_age': 10.0,
        'mrt_distance': 300.0,
        'convenience_stores': 5,
        'latitude': 24.983,
        'longitude': 121.540
    })
    assert response.json()['predicted_price_per_unit'] > 0


def test_predict_price_invalid_input():
    response = client.post('/predict/price', json={
        'house_age': -5,
        'mrt_distance': 300.0,
        'convenience_stores': 5,
        'latitude': 24.983,
        'longitude': 121.540
    })
    assert response.status_code == 422


def test_classify_price_returns_200():
    response = client.post('/classify/price', json={
        'house_age': 10.0,
        'mrt_distance': 300.0,
        'convenience_stores': 5,
        'latitude': 24.983,
        'longitude': 121.540
    })
    assert response.status_code == 200


def test_classify_price_valid_segment():
    response = client.post('/classify/price', json={
        'house_age': 10.0,
        'mrt_distance': 300.0,
        'convenience_stores': 5,
        'latitude': 24.983,
        'longitude': 121.540
    })
    assert response.json()['price_segment'] in ['Low', 'Medium', 'High']


def test_forecast_returns_200():
    response = client.get('/forecast?months_ahead=6')
    assert response.status_code == 200


def test_forecast_correct_month_count():
    response = client.get('/forecast?months_ahead=6')
    assert len(response.json()['forecast']) == 6


def test_forecast_invalid_months():
    response = client.get('/forecast?months_ahead=99')
    assert response.status_code == 400


def test_segment_returns_200():
    response = client.post('/predict/segment', json={
        'latitude': 24.983,
        'longitude': 121.540,
        'mrt_distance': 300.0,
        'convenience_stores': 5,
        'price_per_unit': 40.0
    })
    assert response.status_code == 200


def test_segment_valid_id():
    response = client.post('/predict/segment', json={
        'latitude': 24.983,
        'longitude': 121.540,
        'mrt_distance': 300.0,
        'convenience_stores': 5,
        'price_per_unit': 40.0
    })
    assert response.json()['segment_id'] in [0, 1, 2, 3]
"""
FastAPI route handlers
"""

from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import os

from src.api.schemas import (
    PricePredictRequest, PricePredictResponse,
    ClassifyPriceRequest, ClassifyPriceResponse,
    SegmentRequest, SegmentResponse,
    ForecastResponse, StatisticsResponse, HealthResponse
)

router = APIRouter()

# --- Globals (loaded once at startup) ---
_price_predictor   = None
_price_classifier  = None
_segmentation      = None
_forecast          = None
_df                = None


def load_models():
    """Load all models and dataset into memory"""
    global _price_predictor, _price_classifier, _segmentation, _forecast, _df

    from src.models.price_predictor      import PricePredictor
    from src.models.price_classifier     import PriceClassifier
    from src.models.location_segmentation import LocationSegmentation
    from src.models.time_series_forecast  import TimeSeriesForecast

    models_status = {}

    try:
        _price_predictor = PricePredictor()
        _price_predictor.load_model('models/price_predictor.pkl')
        models_status['price_predictor'] = 'loaded'
    except Exception as e:
        models_status['price_predictor'] = f'error: {e}'

    try:
        _price_classifier = PriceClassifier()
        _price_classifier.load_model('models/price_classifier.pkl')
        models_status['price_classifier'] = 'loaded'
    except Exception as e:
        models_status['price_classifier'] = f'error: {e}'

    try:
        _segmentation = LocationSegmentation()
        _segmentation.load_model('models/location_segments.pkl')
        models_status['segmentation'] = 'loaded'
    except Exception as e:
        models_status['segmentation'] = f'error: {e}'

    try:
        _forecast = TimeSeriesForecast()
        _forecast.load_model('models/forecast_model.pkl')
        models_status['forecast'] = 'loaded'
    except Exception as e:
        models_status['forecast'] = f'error: {e}'

    try:
        _df = pd.read_csv('data/processed/real_estate_clean.csv')
        models_status['dataset'] = f'{len(_df)} rows'
    except Exception as e:
        models_status['dataset'] = f'error: {e}'

    return models_status


# --- Routes ---

@router.get('/', tags=['General'])
def root():
    return {
        'name': 'Real Estate Market Analysis API',
        'version': '1.0.0',
        'endpoints': [
            '/health', '/statistics',
            '/predict/price', '/predict/segment',
            '/classify/price', '/forecast'
        ]
    }


@router.get('/health', response_model=HealthResponse, tags=['General'])
def health():
    status = load_models()
    return HealthResponse(
        status='ok',
        models_loaded=status,
        dataset_rows=len(_df) if _df is not None else 0
    )


@router.get('/statistics', response_model=StatisticsResponse, tags=['Analysis'])
def statistics():
    if _df is None:
        raise HTTPException(status_code=503, detail='Dataset not loaded')
    return StatisticsResponse(
        total_properties=len(_df),
        avg_price_per_unit=round(float(_df['price_per_unit'].mean()), 2),
        median_price_per_unit=round(float(_df['price_per_unit'].median()), 2),
        min_price=round(float(_df['price_per_unit'].min()), 2),
        max_price=round(float(_df['price_per_unit'].max()), 2),
        avg_mrt_distance=round(float(_df['mrt_distance'].mean()), 2),
        avg_house_age=round(float(_df['house_age'].mean()), 2),
        avg_convenience_stores=round(float(_df['convenience_stores'].mean()), 2)
    )


@router.post('/predict/price', response_model=PricePredictResponse, tags=['Prediction'])
def predict_price(req: PricePredictRequest):
    if _price_predictor is None:
        raise HTTPException(status_code=503, detail='Price predictor not loaded')
    try:
        dist_center = np.sqrt(
            (req.latitude  - 24.9692) ** 2 +
            (req.longitude - 121.5357) ** 2
        )
        prediction = _price_predictor.predict(
            req.house_age, req.mrt_distance,
            req.convenience_stores, req.latitude,
            req.longitude, dist_center
        )
        return PricePredictResponse(
            predicted_price_per_unit=round(prediction, 2),
            confidence='R² = 0.77 on test set, CV R² = 0.60',
            note='Price per unit area in 10,000 NTD'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/classify/price', response_model=ClassifyPriceResponse, tags=['Prediction'])
def classify_price(req: ClassifyPriceRequest):
    if _price_classifier is None:
        raise HTTPException(status_code=503, detail='Price classifier not loaded')
    try:
        dist_center = np.sqrt(
            (req.latitude  - 24.9692) ** 2 +
            (req.longitude - 121.5357) ** 2
        )
        segment = _price_classifier.predict(
            req.house_age, req.mrt_distance,
            req.convenience_stores, req.latitude,
            req.longitude, dist_center
        )
        return ClassifyPriceResponse(
            price_segment=segment,
            segments=['Low', 'Medium', 'High']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/predict/segment', response_model=SegmentResponse, tags=['Prediction'])
def predict_segment(req: SegmentRequest):
    if _segmentation is None:
        raise HTTPException(status_code=503, detail='Segmentation model not loaded')
    try:
        result = _segmentation.predict(
            req.latitude, req.longitude,
            req.mrt_distance, req.convenience_stores,
            req.price_per_unit
        )
        return SegmentResponse(
            segment_id=result['segment'],
            segment_label=result['label']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/forecast', response_model=ForecastResponse, tags=['Analysis'])
def forecast(months_ahead: int = 6):
    if _forecast is None:
        raise HTTPException(status_code=503, detail='Forecast model not loaded')
    if not 1 <= months_ahead <= 24:
        raise HTTPException(status_code=400, detail='months_ahead must be between 1 and 24')
    try:
        result = _forecast.forecast(months_ahead)
        return ForecastResponse(
            forecast=result,
            months_ahead=months_ahead,
            note='Forecast based on linear trend with cyclical seasonality encoding'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
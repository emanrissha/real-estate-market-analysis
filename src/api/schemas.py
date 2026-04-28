"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional


# --- Request Schemas ---

class PricePredictRequest(BaseModel):
    house_age: float = Field(..., ge=0, le=100, description="Age of house in years")
    mrt_distance: float = Field(..., ge=0, description="Distance to nearest MRT in meters")
    convenience_stores: int = Field(..., ge=0, le=10, description="Number of nearby convenience stores")
    latitude: float = Field(..., description="Property latitude")
    longitude: float = Field(..., description="Property longitude")

    model_config = {
        "json_schema_extra": {
            "example": {
                "house_age": 10.0,
                "mrt_distance": 300.0,
                "convenience_stores": 5,
                "latitude": 24.983,
                "longitude": 121.540
            }
        }
    }


class ClassifyPriceRequest(BaseModel):
    house_age: float = Field(..., ge=0, le=100)
    mrt_distance: float = Field(..., ge=0)
    convenience_stores: int = Field(..., ge=0, le=10)
    latitude: float
    longitude: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "house_age": 10.0,
                "mrt_distance": 300.0,
                "convenience_stores": 5,
                "latitude": 24.983,
                "longitude": 121.540
            }
        }
    }


class SegmentRequest(BaseModel):
    latitude: float
    longitude: float
    mrt_distance: float = Field(..., ge=0)
    convenience_stores: int = Field(..., ge=0, le=10)
    price_per_unit: float = Field(..., ge=0)


# --- Response Schemas ---

class PricePredictResponse(BaseModel):
    predicted_price_per_unit: float
    confidence: str
    note: str


class ClassifyPriceResponse(BaseModel):
    price_segment: str
    segments: list


class SegmentResponse(BaseModel):
    segment_id: int
    segment_label: str


class ForecastResponse(BaseModel):
    forecast: dict
    months_ahead: int
    note: str


class StatisticsResponse(BaseModel):
    total_properties: int
    avg_price_per_unit: float
    median_price_per_unit: float
    min_price: float
    max_price: float
    avg_mrt_distance: float
    avg_house_age: float
    avg_convenience_stores: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    dataset_rows: int
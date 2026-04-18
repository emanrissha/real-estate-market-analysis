"""
FastAPI application for Real Estate Market Analysis
Run with: uvicorn src.api.app:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.price_predictor import PricePredictor
from src.models.customer_segmentation import CustomerSegmentation

# Initialize FastAPI
app = FastAPI(
    title="Real Estate Market Analysis API",
    description="API for property price prediction and customer segmentation",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
price_model = None
segmentation_model = None
df = None

# Request/Response Models
class PricePredictionRequest(BaseModel):
    area: float
    age: int
    mortgage: int

class PricePredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"

class SegmentRequest(BaseModel):
    age: int
    price: float
    satisfaction: int

class SegmentResponse(BaseModel):
    cluster: int
    description: str

class PropertyResponse(BaseModel):
    id: int
    building: str
    area: float
    price: float
    state: str
    country: str

# Startup event
@app.on_event("startup")
async def startup_event():
    global price_model, segmentation_model, df
    
    # Load models
    price_model = PricePredictor()
    try:
        price_model.load_model('models/price_predictor.pkl')
        print("✅ Price model loaded")
    except:
        print("⚠️ Price model not found. Train first with scripts/train_models.py")
    
    segmentation_model = CustomerSegmentation()
    try:
        segmentation_model.load_model('models/customer_segments.pkl')
        print("✅ Segmentation model loaded")
    except:
        print("⚠️ Segmentation model not found")
    
    # Load data
    try:
        df = pd.read_csv('data/processed/merged_real_estate.csv')
        print(f"✅ Data loaded: {len(df)} rows")
    except:
        print("⚠️ Data not found")

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Real Estate Market Analysis API",
        "version": "1.0.0",
        "endpoints": [
            "/predict_price",
            "/predict_segment",
            "/properties",
            "/statistics",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": price_model is not None}

@app.post("/predict_price", response_model=PricePredictionResponse)
async def predict_price(request: PricePredictionRequest):
    if price_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predicted = price_model.predict(request.area, request.age, request.mortgage)
    return PricePredictionResponse(predicted_price=round(predicted, 2))

@app.post("/predict_segment", response_model=SegmentResponse)
async def predict_segment(request: SegmentRequest):
    if segmentation_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    cluster = segmentation_model.predict(request.age, request.price, request.satisfaction)
    
    descriptions = {
        0: "Budget-conscious first-time buyer",
        1: "Luxury property investor",
        2: "Family home seeker",
        3: "Retirement planner"
    }
    
    return SegmentResponse(cluster=int(cluster), description=descriptions.get(cluster, "Standard buyer"))

@app.get("/properties", response_model=List[PropertyResponse])
async def get_properties(limit: int = 10, state: Optional[str] = None):
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    result = df.copy()
    if state:
        result = result[result['state'] == state]
    
    result = result.head(limit)
    
    return [
        PropertyResponse(
            id=row.get('id', i),
            building=str(row.get('building', '')),
            area=float(row.get('area', 0)),
            price=float(row.get('price', 0)),
            state=str(row.get('state', '')),
            country=str(row.get('country', ''))
        )
        for i, row in result.iterrows()
    ]

@app.get("/statistics")
async def get_statistics():
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return {
        "total_properties": len(df),
        "avg_price": float(df['price'].mean()),
        "min_price": float(df['price'].min()),
        "max_price": float(df['price'].max()),
        "avg_satisfaction": float(df['deal_satisfaction'].mean()),
        "top_state": df['state'].value_counts().index[0],
        "top_country": df['country'].value_counts().index[0]
    }
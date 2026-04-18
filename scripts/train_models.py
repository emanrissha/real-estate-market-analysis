#!/usr/bin/env python3
"""
Train all ML models for real estate analysis
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.price_predictor import PricePredictor
from src.models.customer_segmentation import CustomerSegmentation
from src.models.satisfaction_classifier import SatisfactionClassifier
from src.models.time_series_forecast import TimeSeriesForecast

def train_all_models():
    """Train and evaluate all models"""
    print("="*60)
    print("🤖 TRAINING ALL ML MODELS")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/processed/merged_real_estate.csv')
    print(f"✅ Loaded {len(df)} rows")
    
    # 1. Price Prediction
    print("\n" + "-"*40)
    print("1️⃣ Training Price Predictor...")
    price_model = PricePredictor()
    price_results = price_model.train(df)
    price_model.save_model('models/price_predictor.pkl')
    
    # 2. Customer Segmentation
    print("\n" + "-"*40)
    print("2️⃣ Training Customer Segmentation...")
    seg_model = CustomerSegmentation()
    seg_results = seg_model.fit(df)
    seg_model.save_model('models/customer_segments.pkl')
    
    # 3. Satisfaction Classifier
    print("\n" + "-"*40)
    print("3️⃣ Training Satisfaction Classifier...")
    sat_model = SatisfactionClassifier()
    sat_results = sat_model.train(df)
    sat_model.save_model('models/satisfaction_model.pkl')
    
    # 4. Time Series Forecast
    print("\n" + "-"*40)
    print("4️⃣ Training Time Series Forecast...")
    ts_model = TimeSeriesForecast()
    ts_results = ts_model.train(df)
    ts_model.save_model('models/forecast_model.pkl')
    
    print("\n" + "="*60)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    
    return {
        'price': price_results,
        'segmentation': seg_results,
        'satisfaction': sat_results,
        'forecast': ts_results
    }

if __name__ == "__main__":
    results = train_all_models()
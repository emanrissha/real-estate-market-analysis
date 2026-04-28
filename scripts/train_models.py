"""
Train all ML models and save to models/
Run: python scripts/train_models.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data.loader import load_raw, validate
from src.data.preprocessor import preprocess
from src.models.price_predictor import PricePredictor
from src.models.price_classifier import PriceClassifier
from src.models.location_segmentation import LocationSegmentation
from src.models.time_series_forecast import TimeSeriesForecast
from src.explainability.shap_explainer import ShapExplainer

os.makedirs('models', exist_ok=True)


def train_all():
    print("=" * 60)
    print("🤖 TRAINING ALL ML MODELS")
    print("=" * 60)

    # Load data
    df = load_raw()
    validate(df)
    df = preprocess(df)
    print(f"✅ Dataset ready: {len(df)} rows\n")

    # 1 — Price Predictor
    print("-" * 40)
    print("1️⃣  Price Predictor")
    print("-" * 40)
    price_model = PricePredictor()
    price_results = price_model.train(df)
    price_model.save_model('models/price_predictor.pkl')

    # 2 — Price Classifier
    print("\n" + "-" * 40)
    print("2️⃣  Price Classifier")
    print("-" * 40)
    classifier = PriceClassifier()
    clf_results = classifier.train(df)
    classifier.save_model('models/price_classifier.pkl')

    # 3 — Location Segmentation
    print("\n" + "-" * 40)
    print("3️⃣  Location Segmentation")
    print("-" * 40)
    segmentation = LocationSegmentation()
    seg_results = segmentation.fit(df)
    segmentation.save_model('models/location_segments.pkl')

    # 4 — Time Series Forecast
    print("\n" + "-" * 40)
    print("4️⃣  Time Series Forecast")
    print("-" * 40)
    forecast = TimeSeriesForecast()
    ts_results = forecast.train(df)
    forecast.forecast(6)
    forecast.save_model('models/forecast_model.pkl')

    # 5 — SHAP Explainability
    print("\n" + "-" * 40)
    print("5️⃣  SHAP Explainability")
    print("-" * 40)
    features = price_model.features_used
    X = df[features].fillna(df[features].mean())
    shap = ShapExplainer()
    shap.fit(price_model, X)
    shap.bar_plot(X)
    shap.summary_plot(X)
    print("   Top features:", shap.get_top_features())

    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    print(f"\n📊 Results Summary:")
    print(f"   Price Predictor  — R²: {price_results['r2']:.4f}, CV R²: {price_results['cv_r2_mean']:.4f}")
    print(f"   Price Classifier — Accuracy: {clf_results['accuracy']:.4f}, CV: {clf_results['cv_accuracy_mean']:.4f}")
    print(f"   Segmentation     — Silhouette: {seg_results['silhouette_score']:.4f}")
    print(f"   Forecast         — R²: {ts_results['r2']:.4f}")


if __name__ == '__main__':
    train_all()
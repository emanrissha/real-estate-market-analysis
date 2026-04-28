"""
Full end-to-end pipeline
Run: python scripts/run_pipeline.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data.loader import load_raw, validate
from src.data.preprocessor import preprocess, save
from src.analysis.insights import run_all as run_insights
from src.analysis.statistical_tests import run_all as run_tests
from src.visualization.charts import run_all as run_charts
from src.models.time_series_forecast import TimeSeriesForecast


def run_pipeline():
    print("=" * 60)
    print("🚀 REAL ESTATE ANALYSIS PIPELINE")
    print("=" * 60)

    # Step 1 — Load
    print("\n📂 Step 1: Loading data...")
    df = load_raw()
    validate(df)

    # Step 2 — Preprocess
    print("\n🏗️  Step 2: Preprocessing...")
    df = preprocess(df)
    save(df)

    # Step 3 — Insights
    print("\n🔍 Step 3: Running insights...")
    run_insights(df)

    # Step 4 — Statistical tests
    print("\n🧪 Step 4: Statistical tests...")
    run_tests(df)

    # Step 5 — Charts
    print("\n🎨 Step 5: Generating charts...")
    # Get forecast data for chart
    forecast_model = TimeSeriesForecast()
    forecast_model.train(df)
    forecast_dict = forecast_model.forecast(6)

    run_charts(df, forecast_data={
        'monthly_data': forecast_model.monthly_data.to_dict('records'),
        'forecast': forecast_dict
    })

    # Done
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n📁 Outputs:")
    print(f"   data/processed/real_estate_clean.csv")
    print(f"   reports/mrt_impact.csv")
    print(f"   reports/age_price.csv")
    print(f"   reports/store_impact.csv")
    print(f"   reports/figures/*.png")


if __name__ == '__main__':
    run_pipeline()
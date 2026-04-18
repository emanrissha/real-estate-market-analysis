"""
Additional API endpoints for specific analyses
"""

from fastapi import APIRouter, Query
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

router = APIRouter(prefix="/v1", tags=["analysis"])

# Load data
try:
    df = pd.read_csv('data/processed/merged_real_estate.csv')
except:
    df = None

@router.get("/analysis/by-building")
async def analysis_by_building():
    """Get statistics grouped by building type"""
    if df is None:
        return {"error": "Data not available"}
    
    result = df.groupby('building').agg({
        'price': ['mean', 'min', 'max'],
        'deal_satisfaction': 'mean',
        'customerid': 'count'
    }).round(2)
    
    return result.to_dict()

@router.get("/analysis/by-country")
async def analysis_by_country():
    """Get statistics grouped by country"""
    if df is None:
        return {"error": "Data not available"}
    
    result = df.groupby('country').agg({
        'price': 'mean',
        'deal_satisfaction': 'mean',
        'customerid': 'count'
    }).round(2)
    
    result.columns = ['avg_price', 'avg_satisfaction', 'count']
    return result.to_dict()

@router.get("/analysis/top-states")
async def top_states(limit: int = Query(5, ge=1, le=20)):
    """Get top states by number of properties"""
    if df is None:
        return {"error": "Data not available"}
    
    top = df['state'].value_counts().head(limit).reset_index()
    top.columns = ['state', 'count']
    return top.to_dict('records')

@router.get("/analysis/age-distribution")
async def age_distribution():
    """Get age distribution of customers"""
    if df is None or 'age' not in df.columns:
        return {"error": "Age data not available"}
    
    return {
        "min_age": int(df['age'].min()),
        "max_age": int(df['age'].max()),
        "mean_age": float(df['age'].mean()),
        "median_age": float(df['age'].median())
    }

@router.get("/analysis/price-range")
async def price_range():
    """Get price distribution statistics"""
    if df is None:
        return {"error": "Data not available"}
    
    return {
        "min_price": float(df['price'].min()),
        "max_price": float(df['price'].max()),
        "mean_price": float(df['price'].mean()),
        "median_price": float(df['price'].median()),
        "percentile_25": float(df['price'].quantile(0.25)),
        "percentile_75": float(df['price'].quantile(0.75))
    }
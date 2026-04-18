#!/usr/bin/env python3
"""
Main pipeline script - Runs entire ETL + ML pipeline
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.preprocessor import Preprocessor
from src.features.build_features import FeatureBuilder
from src.models.price_predictor import PricePredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_full_pipeline():
    """Execute complete pipeline"""
    logger.info("="*60)
    logger.info("STARTING REAL ESTATE ANALYSIS PIPELINE")
    logger.info("="*60)
    
    # Step 1: Load and Clean
    logger.info("\n📂 Step 1: Loading data...")
    preprocessor = Preprocessor()
    df = preprocessor.run_full_pipeline()
    
    # Step 2: Feature Engineering
    logger.info("\n🏗️ Step 2: Feature engineering...")
    feature_builder = FeatureBuilder()
    df = feature_builder.create_all_features(df)
    
    # Step 3: Train Models
    logger.info("\n🤖 Step 3: Training ML models...")
    predictor = PricePredictor()
    results = predictor.train(df)
    
    # Step 4: Save results
    logger.info("\n💾 Step 4: Saving results...")
    df.to_csv('data/processed/final_with_features.csv', index=False)
    
    logger.info("\n" + "="*60)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("="*60)
    
    return df, results

if __name__ == "__main__":
    df, results = run_full_pipeline()
    print(f"\nFinal dataset: {df.shape[0]} rows, {df.shape[1]} columns")
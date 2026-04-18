"""
Test script for Preprocessor - merges properties and customers
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import Preprocessor

def main():
    print("="*60)
    print("🔗 TESTING PREPROCESSOR (MERGE)")
    print("="*60)
    
    # Run full pipeline
    preprocessor = Preprocessor()
    merged_df = preprocessor.run_full_pipeline()
    
    # Get summary
    summary = preprocessor.get_data_summary(merged_df)
    
    # Show column list
    print("\n📋 ALL COLUMNS IN MERGED DATASET:")
    print("-" * 40)
    for i, col in enumerate(merged_df.columns, 1):
        print(f"   {i:2}. {col} ({merged_df[col].dtype})")
    
    # Check for required columns
    print("\n✅ REQUIRED COLUMNS CHECK:")
    print("-" * 40)
    
    required_cols = ['customerid', 'price', 'area', 'building', 'deal_satisfaction', 'country', 'state']
    for col in required_cols:
        if col in merged_df.columns:
            print(f"   ✅ {col}")
        else:
            print(f"   ❌ {col} MISSING")
    
    # Show sample data
    print("\n📊 SAMPLE DATA (FIRST 5 ROWS):")
    print("-" * 40)
    pd.set_option('display.max_columns', None)
    print(merged_df.head(5))
    
    # Basic statistics
    print("\n📈 BASIC STATISTICS:")
    print("-" * 40)
    
    if 'price' in merged_df.columns:
        print(f"   Price - Min: ${merged_df['price'].min():,.2f}")
        print(f"   Price - Max: ${merged_df['price'].max():,.2f}")
        print(f"   Price - Mean: ${merged_df['price'].mean():,.2f}")
        print(f"   Price - Median: ${merged_df['price'].median():,.2f}")
    
    if 'deal_satisfaction' in merged_df.columns:
        print(f"\n   Deal Satisfaction - Mean: {merged_df['deal_satisfaction'].mean():.2f}")
        print(f"   Deal Satisfaction - Min: {merged_df['deal_satisfaction'].min()}")
        print(f"   Deal Satisfaction - Max: {merged_df['deal_satisfaction'].max()}")
    
    if 'area' in merged_df.columns:
        print(f"\n   Area - Mean: {merged_df['area'].mean():.1f} sq ft")
        print(f"   Area - Min: {merged_df['area'].min():.1f}")
        print(f"   Area - Max: {merged_df['area'].max():.1f}")
    
    print("\n" + "="*60)
    print("✅ TASK 1 COMPLETE!")
    print("="*60)
    print("\n📁 Merged data saved to: data/processed/merged_real_estate.csv")

if __name__ == "__main__":
    import pandas as pd
    pd.set_option('display.max_columns', None)
    main()
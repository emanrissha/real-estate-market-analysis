"""
Test script for DataCleaner
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner

def main():
    print("="*60)
    print("🧹 TESTING DATA CLEANER")
    print("="*60)
    
    # Load data
    print("\n📂 Loading raw data...")
    loader = DataLoader()
    properties, customers = loader.load_both()
    
    print(f"\nBefore cleaning:")
    print(f"   Properties shape: {properties.shape}")
    print(f"   Customers shape: {customers.shape}")
    print(f"   Properties columns: {list(properties.columns)}")
    print(f"   Customers columns: {list(customers.columns)}")
    
    # Clean data
    cleaner = DataCleaner()
    properties_clean, customers_clean = cleaner.clean_all(properties, customers)
    
    print(f"\nAfter cleaning:")
    print(f"   Properties shape: {properties_clean.shape}")
    print(f"   Customers shape: {customers_clean.shape}")
    print(f"   Properties columns: {list(properties_clean.columns)}")
    print(f"   Customers columns: {list(customers_clean.columns)}")
    
    # Check if customer_id column exists now
    print("\n" + "="*60)
    print("🔍 VERIFYING CUSTOMER_ID COLUMN")
    print("="*60)
    
    if 'customerid' in properties_clean.columns:
        print(f"✅ properties has 'customerid' column")
        print(f"   Sample: {properties_clean['customerid'].head(3).tolist()}")
    else:
        print("❌ properties missing customerid")
        
    if 'customerid' in customers_clean.columns:
        print(f"✅ customers has 'customerid' column")
        print(f"   Sample: {customers_clean['customerid'].head(3).tolist()}")
    else:
        print("❌ customers missing customerid")
    
    # Show data types after cleaning
    print("\n" + "="*60)
    print("📊 DATA TYPES AFTER CLEANING")
    print("="*60)
    
    print("\nProperties dtypes:")
    for col, dtype in properties_clean.dtypes.items():
        print(f"   {col}: {dtype}")
    
    print("\nCustomers dtypes:")
    for col, dtype in customers_clean.dtypes.items():
        print(f"   {col}: {dtype}")
    
    # Show sample data
    print("\n" + "="*60)
    print("📋 SAMPLE DATA AFTER CLEANING")
    print("="*60)
    
    print("\nProperties (first 3 rows):")
    print(properties_clean.head(3))
    
    print("\nCustomers (first 3 rows):")
    print(customers_clean.head(3))
    
    # Save cleaned data
    print("\n" + "="*60)
    print("💾 SAVING CLEANED DATA")
    print("="*60)
    
    properties_clean.to_csv("data/processed/properties_cleaned.csv", index=False)
    customers_clean.to_csv("data/processed/customers_cleaned.csv", index=False)
    print("✅ Saved to data/processed/")

if __name__ == "__main__":
    main()
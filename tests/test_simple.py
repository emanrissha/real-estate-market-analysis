"""
Simple test to verify files load without classes
"""

import pandas as pd
from pathlib import Path

print("="*50)
print("SIMPLE DATA LOAD TEST")
print("="*50)

# Check if files exist
raw_path = Path("data/raw/")
customers_file = raw_path / "customers.csv"
properties_file = raw_path / "properties.csv"

print(f"\n📁 Checking files:")
print(f"   Customers exists: {customers_file.exists()}")
print(f"   Properties exists: {properties_file.exists()}")

if customers_file.exists() and properties_file.exists():
    # Load files
    customers = pd.read_csv(customers_file)
    properties = pd.read_csv(properties_file)
    
    print(f"\n✅ Customers loaded: {customers.shape}")
    print(f"✅ Properties loaded: {properties.shape}")
    
    print(f"\n📊 Customers columns ({len(customers.columns)}):")
    for col in customers.columns:
        print(f"   - {col}")
    
    print(f"\n📊 Properties columns ({len(properties.columns)}):")
    for col in properties.columns:
        print(f"   - {col}")
    
    print("\n✅ SUCCESS! Files are ready.")
else:
    print("\n❌ ERROR: CSV files not found in data/raw/")
    print("   Please add customers.csv and properties.csv to data/raw/")
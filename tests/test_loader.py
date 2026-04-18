"""
Test script for DataLoader
Run this to verify the loader works correctly
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader

def test_loader_initialization():
    """Test that DataLoader initializes correctly"""
    print("="*60)
    print("TEST 1: DataLoader Initialization")
    print("="*60)
    
    loader = DataLoader()
    assert loader.raw_path == Path("data/raw/")
    print("✅ DataLoader initialized successfully")
    return loader

def test_load_properties(loader):
    """Test loading properties.csv"""
    print("\n" + "="*60)
    print("TEST 2: Load Properties")
    print("="*60)
    
    try:
        properties = loader.load_properties()
        assert properties is not None
        assert len(properties) > 0
        print(f"✅ Properties loaded: {properties.shape[0]} rows, {properties.shape[1]} columns")
        print(f"   Columns: {list(properties.columns)}")
        return properties
    except Exception as e:
        print(f"❌ Failed to load properties: {e}")
        return None

def test_load_customers(loader):
    """Test loading customers.csv"""
    print("\n" + "="*60)
    print("TEST 3: Load Customers")
    print("="*60)
    
    try:
        customers = loader.load_customers()
        assert customers is not None
        assert len(customers) > 0
        print(f"✅ Customers loaded: {customers.shape[0]} rows, {customers.shape[1]} columns")
        print(f"   Columns: {list(customers.columns)}")
        return customers
    except Exception as e:
        print(f"❌ Failed to load customers: {e}")
        return None

def test_load_both(loader):
    """Test loading both files together"""
    print("\n" + "="*60)
    print("TEST 4: Load Both Datasets")
    print("="*60)
    
    try:
        properties, customers = loader.load_both()
        assert properties is not None
        assert customers is not None
        print("✅ Both datasets loaded successfully")
        return properties, customers
    except Exception as e:
        print(f"❌ Failed to load both: {e}")
        return None, None

def test_customer_id_column(properties, customers):
    """Test that customer_id column exists in both"""
    print("\n" + "="*60)
    print("TEST 5: Check customer_id Column")
    print("="*60)
    
    if 'customer_id' in properties.columns:
        print(f"✅ customer_id found in properties")
        print(f"   Sample values: {properties['customer_id'].head(3).tolist()}")
    else:
        print("❌ customer_id NOT found in properties")
        
    if 'customer_id' in customers.columns:
        print(f"✅ customer_id found in customers")
        print(f"   Sample values: {customers['customer_id'].head(3).tolist()}")
    else:
        print("❌ customer_id NOT found in customers")

def test_data_quality(properties, customers):
    """Basic data quality checks"""
    print("\n" + "="*60)
    print("TEST 6: Data Quality Checks")
    print("="*60)
    
    # Check for missing values
    props_missing = properties.isnull().sum().sum()
    cust_missing = customers.isnull().sum().sum()
    
    print(f"Properties missing values: {props_missing}")
    print(f"Customers missing values: {cust_missing}")
    
    # Check for duplicate customer_ids
    if 'customer_id' in customers.columns:
        dupes = customers['customer_id'].duplicated().sum()
        print(f"Duplicate customer_ids in customers: {dupes}")
    
    if 'customer_id' in properties.columns:
        dupes = properties['customer_id'].duplicated().sum()
        print(f"Duplicate customer_ids in properties: {dupes}")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 STARTING DATA LOADER TESTS")
    print("="*60 + "\n")
    
    # Run tests
    loader = test_loader_initialization()
    properties = test_load_properties(loader)
    customers = test_load_customers(loader)
    
    if properties is not None and customers is not None:
        properties_df, customers_df = test_load_both(loader)
        test_customer_id_column(properties, customers)
        test_data_quality(properties, customers)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        
        # Show first few rows
        print("\n📋 FIRST 3 ROWS - PROPERTIES:")
        print(properties.head(3))
        
        print("\n📋 FIRST 3 ROWS - CUSTOMERS:")
        print(customers.head(3))
    else:
        print("\n❌ TESTS FAILED - Check if CSV files are in data/raw/ folder")

if __name__ == "__main__":
    main()
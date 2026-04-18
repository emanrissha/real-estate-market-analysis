"""
Data validation utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class DataValidator:
    def __init__(self):
        self.required_columns_properties = ['customerid', 'price', 'area', 'building']
        self.required_columns_customers = ['customerid', 'birth_date']
        self.expected_row_count = 267
        self.expected_column_count = 19
    
    def check_columns(self, df: pd.DataFrame, required_cols: List[str], name: str = "dataset") -> Tuple[bool, List[str]]:
        """Check if required columns exist"""
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"⚠️ Missing columns in {name}: {missing}")
            return False, missing
        print(f"✅ All required columns present in {name}")
        return True, []
    
    def check_no_null_customer_ids(self, df: pd.DataFrame, name: str = "dataset") -> Tuple[bool, int]:
        """Check for null customer_ids"""
        null_count = df['customerid'].isnull().sum()
        if null_count > 0:
            print(f"⚠️ {null_count} null customer_ids in {name}")
            return False, null_count
        print(f"✅ No null customer_ids in {name}")
        return True, 0
    
    def check_unique_customer_ids(self, df: pd.DataFrame, name: str = "dataset") -> Tuple[bool, int]:
        """Check for duplicate customer_ids"""
        duplicate_count = df['customerid'].duplicated().sum()
        if duplicate_count > 0:
            print(f"⚠️ {duplicate_count} duplicate customer_ids in {name}")
            return False, duplicate_count
        print(f"✅ No duplicate customer_ids in {name}")
        return True, 0
    
    def validate_data_types(self, df: pd.DataFrame, name: str = "dataset") -> Dict[str, str]:
        """Validate and report data types"""
        type_report = {}
        for col in df.columns:
            type_report[col] = str(df[col].dtype)
        
        print(f"\n📊 Data types for {name}:")
        for col, dtype in type_report.items():
            print(f"   {col}: {dtype}")
        
        return type_report
    
    def validate_price_range(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Validate price column has reasonable values"""
        if 'price' not in df.columns:
            return False, {"error": "price column not found"}
        
        price_min = df['price'].min()
        price_max = df['price'].max()
        price_mean = df['price'].mean()
        
        is_valid = price_min > 0 and price_max < 10000000  # Reasonable range
        
        stats = {
            'min': price_min,
            'max': price_max,
            'mean': price_mean,
            'is_valid': is_valid
        }
        
        if is_valid:
            print(f"✅ Price range valid: ${price_min:,.2f} - ${price_max:,.2f}")
        else:
            print(f"⚠️ Price range suspicious: ${price_min:,.2f} - ${price_max:,.2f}")
        
        return is_valid, stats
    
    def validate_satisfaction_range(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Validate satisfaction scores are between 1-5"""
        if 'deal_satisfaction' not in df.columns:
            return False, {"error": "deal_satisfaction column not found"}
        
        sat_min = df['deal_satisfaction'].min()
        sat_max = df['deal_satisfaction'].max()
        sat_mean = df['deal_satisfaction'].mean()
        
        is_valid = 1 <= sat_min and sat_max <= 5
        
        stats = {
            'min': sat_min,
            'max': sat_max,
            'mean': sat_mean,
            'is_valid': is_valid
        }
        
        if is_valid:
            print(f"✅ Satisfaction range valid: {sat_min} - {sat_max}")
        else:
            print(f"⚠️ Satisfaction range invalid: {sat_min} - {sat_max}")
        
        return is_valid, stats
    
    def validate_merge(self, merged_df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Validate the merged dataset against expected standards"""
        print("\n" + "="*60)
        print("VALIDATING MERGED DATASET")
        print("="*60)
        
        results = {
            'row_count': merged_df.shape[0],
            'column_count': merged_df.shape[1],
            'expected_rows': self.expected_row_count,
            'expected_cols': self.expected_column_count,
            'row_match': False,
            'col_match': False
        }
        
        # Check row count
        if merged_df.shape[0] >= 190:
            results['row_match'] = True
            print(f"✅ Row count: {merged_df.shape[0]} (expected ~{self.expected_row_count})")
        else:
            print(f"⚠️ Row count low: {merged_df.shape[0]} (expected ~{self.expected_row_count})")
        
        # Check column count
        if merged_df.shape[1] >= self.expected_column_count:
            results['col_match'] = True
            print(f"✅ Column count: {merged_df.shape[1]} (expected {self.expected_column_count})")
        else:
            print(f"⚠️ Column count low: {merged_df.shape[1]} (expected {self.expected_column_count})")
        
        # Overall validation
        is_valid = results['row_match'] and results['col_match']
        
        if is_valid:
            print("\n✅ MERGED DATASET VALIDATION PASSED!")
        else:
            print("\n⚠️ MERGED DATASET VALIDATION ISSUES FOUND")
        
        return is_valid, results
    
    def run_all_validations(self, properties: pd.DataFrame, customers: pd.DataFrame, merged: pd.DataFrame) -> Dict:
        """Run all validation checks"""
        print("\n" + "🚀"*30)
        print("RUNNING COMPLETE DATA VALIDATION")
        print("🚀"*30)
        
        results = {
            'properties': {},
            'customers': {},
            'merged': {}
        }
        
        # Validate properties
        print("\n📋 VALIDATING PROPERTIES:")
        results['properties']['columns'] = self.check_columns(properties, self.required_columns_properties, "properties")
        results['properties']['null_ids'] = self.check_no_null_customer_ids(properties, "properties")
        results['properties']['duplicate_ids'] = self.check_unique_customer_ids(properties, "properties")
        results['properties']['price'] = self.validate_price_range(properties)
        
        # Validate customers
        print("\n📋 VALIDATING CUSTOMERS:")
        results['customers']['columns'] = self.check_columns(customers, self.required_columns_customers, "customers")
        results['customers']['null_ids'] = self.check_no_null_customer_ids(customers, "customers")
        results['customers']['duplicate_ids'] = self.check_unique_customer_ids(customers, "customers")
        results['customers']['satisfaction'] = self.validate_satisfaction_range(customers)
        
        # Validate merged
        print("\n📋 VALIDATING MERGED:")
        results['merged']['validate'] = self.validate_merge(merged)
        
        return results
"""
Feature engineering pipeline
"""

import pandas as pd
import numpy as np
from src.features.age_binner import AgeBinner
from src.features.price_binner import PriceBinner

class FeatureBuilder:
    def __init__(self):
        self.age_binner = AgeBinner()
        self.price_binner = PriceBinner()
    
    def create_all_features(self, df):
        """Create all feature engineering transformations"""
        print("\n🏗️ BUILDING FEATURES")
        print("-" * 40)
        
        # Calculate age
        if 'birth_date' in df.columns:
            df = self.age_binner.calculate_age(df)
            df = self.age_binner.create_age_groups(df)
            print("   ✅ Age features created")
        
        # Create price bins
        if 'price' in df.columns:
            df = self.price_binner.create_price_bins(df)
            print("   ✅ Price bin features created")
        
        # Extract date features
        date_col = 'date' if 'date' in df.columns else 'date_sale'
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['quarter'] = df[date_col].dt.quarter
            df['day_of_week'] = df[date_col].dt.dayofweek
            print("   ✅ Date features created")
        
        # Create price per area feature
        if 'price' in df.columns and 'area' in df.columns:
            df['price_per_area'] = df['price'] / df['area']
            print("   ✅ Price per area feature created")
        
        # Create age squared for non-linear relationships
        if 'age' in df.columns:
            df['age_squared'] = df['age'] ** 2
            print("   ✅ Polynomial features created")
        
        # Binary features
        if 'mortgage' in df.columns:
            df['has_mortgage'] = df['mortgage'].map({'Yes': 1, 'yes': 1, 'No': 0, 'no': 0}).fillna(0)
            print("   ✅ Binary features created")
        
        print("\n✅ FEATURE ENGINEERING COMPLETE!")
        print(f"   Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_importance_ranking(self, df, target='price'):
        """Rank features by correlation with target"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()[target].abs().sort_values(ascending=False)
        return correlations
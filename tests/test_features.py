"""
Unit tests for feature engineering modules
Run with: pytest tests/test_features.py -v
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.age_binner import AgeBinner
from src.features.price_binner import PriceBinner
from src.features.build_features import FeatureBuilder

class TestAgeBinner:
    """Test age binning functionality"""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'birth_date': ['1990-01-01', '1985-06-15', '1975-12-20'],
            'date_sale': ['2020-01-01', '2020-06-15', '2020-12-20']
        })
    
    def test_calculate_age(self, sample_df):
        """Test age calculation"""
        binner = AgeBinner()
        result = binner.calculate_age(sample_df)
        
        assert 'age' in result.columns
        assert result['age'].iloc[0] == 30
        assert result['age'].iloc[1] == 35
    
    def test_create_age_groups(self, sample_df):
        """Test age group creation"""
        binner = AgeBinner()
        sample_df['age'] = [25, 35, 45]
        result = binner.create_age_groups(sample_df)
        
        assert 'age_group' in result.columns
        assert result['age_group'].nunique() <= len(binner.labels)
    
    def test_get_age_distribution(self, sample_df):
        """Test age distribution"""
        binner = AgeBinner()
        sample_df['age'] = [25, 35, 45, 25, 35]
        sample_df = binner.create_age_groups(sample_df)
        distribution = binner.get_age_distribution(sample_df)
        
        assert distribution is not None
        assert distribution.sum() == len(sample_df)

class TestPriceBinner:
    """Test price binning functionality"""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'price': [100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000]
        })
    
    def test_create_price_bins(self, sample_df):
        """Test price bin creation"""
        binner = PriceBinner(n_bins=5)
        result = binner.create_price_bins(sample_df)
        
        assert 'price_bin' in result.columns
        assert result['price_bin'].nunique() <= 5
    
    def test_get_price_distribution(self, sample_df):
        """Test price distribution"""
        binner = PriceBinner(n_bins=5)
        sample_df = binner.create_price_bins(sample_df)
        distribution = binner.get_price_distribution(sample_df)
        
        assert distribution is not None
    
    def test_get_price_statistics(self, sample_df):
        """Test price statistics by bin"""
        binner = PriceBinner(n_bins=5)
        sample_df = binner.create_price_bins(sample_df)
        stats = binner.get_price_statistics_by_bin(sample_df)
        
        assert stats is not None
        assert 'count' in stats.columns
        assert 'mean' in stats.columns

class TestFeatureBuilder:
    """Test feature builder pipeline"""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'birth_date': ['1990-01-01', '1985-06-15', '1975-12-20'],
            'date_sale': ['2020-01-01', '2020-06-15', '2020-12-20'],
            'price': [100000, 200000, 300000],
            'area': [500, 600, 700],
            'mortgage': ['Yes', 'No', 'Yes']
        })
    
    def test_create_all_features(self, sample_df):
        """Test full feature engineering pipeline"""
        builder = FeatureBuilder()
        result = builder.create_all_features(sample_df)
        
        assert result is not None
        assert len(result.columns) > len(sample_df.columns)
    
    def test_get_feature_importance(self, sample_df):
        """Test feature importance ranking"""
        builder = FeatureBuilder()
        result = builder.create_all_features(sample_df)
        importance = builder.get_feature_importance_ranking(result)
        
        assert importance is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
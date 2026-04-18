"""
Unit tests for analysis modules
Run with: pytest tests/test_analysis.py -v
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.descriptive_stats import DescriptiveStats
from src.analysis.correlation_analysis import CorrelationAnalysis
from src.analysis.hypothesis_tests import HypothesisTests

class TestDescriptiveStats:
    """Test descriptive statistics functions"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing"""
        return pd.DataFrame({
            'building': ['A', 'B', 'A', 'B', 'C'],
            'sold': [1, 1, 0, 1, 1],
            'mortgage_binary': [0, 1, 0, 1, 0],
            'area': [500, 600, 700, 800, 900],
            'price': [100000, 150000, 200000, 250000, 300000],
            'deal_satisfaction': [3, 4, 5, 4, 3],
            'country': ['USA', 'USA', 'Canada', 'Canada', 'USA'],
            'state': ['CA', 'NY', 'ON', 'BC', 'CA'],
            'customerid': [1, 2, 3, 4, 5]
        })
    
    def test_breakdown_by_building(self, sample_df):
        """Test building type breakdown"""
        stats = DescriptiveStats(sample_df)
        totals, averages = stats.breakdown_by_building()
        
        assert totals is not None
        assert averages is not None
        assert 'sold' in totals.columns
        assert 'mortgage' in totals.columns
        assert 'price' in averages.columns
    
    def test_breakdown_by_country(self, sample_df):
        """Test country breakdown"""
        stats = DescriptiveStats(sample_df)
        country_stats = stats.breakdown_by_country()
        
        assert country_stats is not None
        assert 'frequency' in country_stats.columns
        assert 'avg_price' in country_stats.columns
    
    def test_breakdown_by_state(self, sample_df):
        """Test state breakdown with cumulative frequency"""
        stats = DescriptiveStats(sample_df)
        state_stats = stats.breakdown_by_state()
        
        assert state_stats is not None
        assert 'relative_frequency' in state_stats.columns
        assert 'cumulative_frequency' in state_stats.columns

class TestCorrelationAnalysis:
    """Test correlation analysis functions"""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'age': [25, 35, 45, 55, 65],
            'price': [100000, 150000, 200000, 250000, 300000],
            'area': [500, 600, 700, 800, 900],
            'deal_satisfaction': [3, 4, 5, 4, 3]
        })
    
    def test_compute_correlations(self, sample_df):
        """Test correlation matrix computation"""
        corr_analysis = CorrelationAnalysis(sample_df)
        matrix = corr_analysis.compute_correlations()
        
        assert matrix is not None
        assert matrix.shape[0] == len(sample_df.columns)
    
    def test_find_strong_correlations(self, sample_df):
        """Test finding strong correlations"""
        corr_analysis = CorrelationAnalysis(sample_df)
        strong_pairs = corr_analysis.find_strong_correlations(threshold=0.5)
        
        assert isinstance(strong_pairs, list)
    
    def test_correlation_with_target(self, sample_df):
        """Test correlation with target variable"""
        corr_analysis = CorrelationAnalysis(sample_df)
        correlations = corr_analysis.correlation_with_target(target='price')
        
        assert correlations is not None
        assert 'age' in correlations.index or 'area' in correlations.index

class TestHypothesisTests:
    """Test hypothesis testing functions"""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'building': ['A', 'B', 'A', 'B', 'C'],
            'price': [100000, 150000, 200000, 250000, 300000],
            'deal_satisfaction': [3, 4, 5, 4, 3],
            'age': [25, 35, 45, 55, 65],
            'mortgage_binary': [0, 1, 0, 1, 0],
            'country': ['USA', 'USA', 'Canada', 'Canada', 'USA']
        })
    
    def test_price_by_building_type(self, sample_df):
        """Test ANOVA for price by building type"""
        tests = HypothesisTests(sample_df)
        result = tests.test_price_by_building_type()
        
        assert 'statistic' in result
        assert 'p_value' in result
    
    def test_age_price_correlation(self, sample_df):
        """Test correlation hypothesis"""
        tests = HypothesisTests(sample_df)
        result = tests.test_age_price_correlation()
        
        assert 'correlation' in result
        assert 'p_value' in result
    
    def test_run_all_tests(self, sample_df):
        """Test running all hypothesis tests"""
        tests = HypothesisTests(sample_df)
        results = tests.run_all_tests()
        
        assert len(results) == 4
        assert 'price_by_building' in results

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
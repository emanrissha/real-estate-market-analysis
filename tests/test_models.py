"""
Unit tests for ML models
Run with: pytest tests/test_models.py -v
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.price_predictor import PricePredictor
from src.models.customer_segmentation import CustomerSegmentation

class TestPricePredictor:
    """Test price prediction model"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'area': [500, 600, 700, 800, 900],
            'age': [25, 35, 45, 55, 65],
            'mortgage_binary': [0, 1, 0, 1, 0],
            'price': [100000, 150000, 200000, 250000, 300000]
        })
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        model = PricePredictor()
        assert model is not None
        assert model.model is not None
    
    def test_prepare_features(self, sample_data):
        """Test feature preparation"""
        model = PricePredictor()
        X, y, features = model.prepare_features(sample_data)
        
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        assert len(features) > 0
    
    def test_train_model(self, sample_data):
        """Test model training"""
        model = PricePredictor()
        results = model.train(sample_data)
        
        assert 'mae' in results
        assert 'r2' in results
        assert results['mae'] >= 0
    
    def test_predict(self, sample_data):
        """Test price prediction"""
        model = PricePredictor()
        model.train(sample_data)
        
        prediction = model.predict(area=600, age=30, mortgage=1)
        assert prediction > 0
        assert isinstance(prediction, float)

class TestCustomerSegmentation:
    """Test customer segmentation model"""
    
    @pytest.fixture
    def sample_customers(self):
        """Create sample customer data"""
        return pd.DataFrame({
            'age': [25, 35, 45, 55, 65, 30, 40, 50, 60, 70],
            'price': [100000, 150000, 200000, 250000, 300000,
                      120000, 180000, 220000, 280000, 350000],
            'deal_satisfaction': [3, 4, 5, 4, 3, 4, 5, 4, 3, 5]
        })
    
    def test_model_initialization(self):
        """Test that segmentation model initializes correctly"""
        model = CustomerSegmentation(n_clusters=3)
        assert model is not None
        assert model.n_clusters == 3
    
    def test_fit_model(self, sample_customers):
        """Test model fitting"""
        model = CustomerSegmentation(n_clusters=3)
        results = model.fit(sample_customers)
        
        assert 'n_clusters' in results
        assert results['n_clusters'] == 3
    
    def test_predict_segment(self, sample_customers):
        """Test segment prediction"""
        model = CustomerSegmentation(n_clusters=3)
        model.fit(sample_customers)
        
        segment = model.predict(age=40, price=200000, satisfaction=4)
        assert isinstance(segment, int)
        assert 0 <= segment < model.n_clusters

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
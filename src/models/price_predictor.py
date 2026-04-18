"""
Price prediction model using Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class PricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.features = ['area', 'age', 'mortgage_binary']
        
    def prepare_features(self, df):
        """Prepare features for training"""
        # Ensure required columns exist
        available_features = [f for f in self.features if f in df.columns]
        X = df[available_features].fillna(df[available_features].mean())
        y = df['price']
        return X, y, available_features
    
    def train(self, df):
        """Train the price prediction model"""
        X, y, features = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train
        self.model.fit(X_train, y_train)
        self.features_used = features
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"   MAE: ${mae:,.2f}")
        print(f"   R² Score: {r2:.4f}")
        
        # Feature importance
        importance = dict(zip(features, self.model.feature_importances_))
        print(f"   Feature importance: {importance}")
        
        return {'mae': mae, 'r2': r2, 'importance': importance}
    
    def predict(self, area, age, mortgage):
        """Predict price for new property"""
        features = np.array([[area, age, mortgage]])
        return self.model.predict(features)[0]
    
    def save_model(self, path):
        """Save model to disk"""
        joblib.dump(self.model, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk"""
        self.model = joblib.load(path)
        print(f"✅ Model loaded from {path}")
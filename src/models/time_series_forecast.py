"""
Time series forecasting for revenue prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

class TimeSeriesForecast:
    def __init__(self):
        self.model = LinearRegression()
        
    def prepare_time_series(self, df):
        """Prepare time series data for forecasting"""
        # Get date column
        date_col = 'date' if 'date' in df.columns else 'date_sale'
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Create monthly revenue series
        monthly_revenue = df.groupby(df[date_col].dt.to_period('M'))['price'].sum().reset_index()
        monthly_revenue['date'] = monthly_revenue[date_col].dt.to_timestamp()
        monthly_revenue['revenue'] = monthly_revenue['price']
        
        # Create time features
        monthly_revenue['month_num'] = range(len(monthly_revenue))
        monthly_revenue['month'] = monthly_revenue['date'].dt.month
        monthly_revenue['quarter'] = monthly_revenue['date'].dt.quarter
        monthly_revenue['year'] = monthly_revenue['date'].dt.year
        
        return monthly_revenue
    
    def train(self, df):
        """Train time series forecast model"""
        monthly_data = self.prepare_time_series(df)
        
        # Features for prediction
        features = ['month_num', 'month', 'quarter']
        X = monthly_data[features]
        y = monthly_data['revenue']
        
        # Train model
        self.model.fit(X, y)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Evaluate
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        print(f"\n📈 Time Series Forecast Results:")
        print(f"   MAE: ${mae:,.2f}")
        print(f"   RMSE: ${rmse:,.2f}")
        print(f"   R²: {self.model.score(X, y):.4f}")
        
        # Feature coefficients
        coeffs = dict(zip(features, self.model.coef_))
        print(f"   Coefficients: {coeffs}")
        
        return {'mae': mae, 'rmse': rmse, 'r2': self.model.score(X, y)}
    
    def forecast(self, months_ahead=3):
        """Forecast revenue for future months"""
        last_month = self.monthly_data['month_num'].max()
        
        future_dates = []
        future_features = []
        
        for i in range(1, months_ahead + 1):
            month_num = last_month + i
            # Simple seasonality assumption
            month = ((month_num - 1) % 12) + 1
            quarter = (month - 1) // 3 + 1
            
            future_features.append([month_num, month, quarter])
            future_dates.append(f"Month +{i}")
        
        predictions = self.model.predict(future_features)
        
        print(f"\n🔮 Forecast for next {months_ahead} months:")
        for date, pred in zip(future_dates, predictions):
            print(f"   {date}: ${pred:,.2f}")
        
        return dict(zip(future_dates, predictions))
    
    def save_model(self, path):
        """Save model to disk"""
        joblib.dump(self.model, path)
        print(f"✅ Forecast model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk"""
        self.model = joblib.load(path)
        print(f"✅ Forecast model loaded from {path}")
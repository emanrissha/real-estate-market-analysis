"""
Time series forecasting for monthly revenue trends
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


class TimeSeriesForecast:
    def __init__(self):
        self.model = LinearRegression()
        self.monthly_data = None
        self.features_used = None

    def prepare(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['date', 'price_per_unit'])

        monthly = (
            df.groupby(df['date'].dt.to_period('M'))
            .agg(
                avg_price=('price_per_unit', 'mean'),
                total_sales=('price_per_unit', 'count'),
                total_revenue=('price_per_unit', 'sum')
            )
            .reset_index()
        )
        monthly['date'] = monthly['date'].dt.to_timestamp()
        monthly = monthly.sort_values('date').reset_index(drop=True)

        # Time features
        monthly['month_num']  = range(len(monthly))
        monthly['month']      = monthly['date'].dt.month
        monthly['quarter']    = monthly['date'].dt.quarter
        monthly['month_sin']  = np.sin(2 * np.pi * monthly['month'] / 12)
        monthly['month_cos']  = np.cos(2 * np.pi * monthly['month'] / 12)

        return monthly

    def train(self, df):
        self.monthly_data = self.prepare(df)
        features = ['month_num', 'month_sin', 'month_cos', 'quarter']
        self.features_used = features

        X = self.monthly_data[features]
        y = self.monthly_data['avg_price']

        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        mae  = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2   = r2_score(y, y_pred)

        print(f"\n📈 Time Series Forecast Results:")
        print(f"   Monthly points: {len(self.monthly_data)}")
        print(f"   Date range: {self.monthly_data['date'].min().strftime('%Y-%m')} "
              f"to {self.monthly_data['date'].max().strftime('%Y-%m')}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²:   {r2:.4f}")

        return {'mae': mae, 'rmse': rmse, 'r2': r2,
                'monthly_data': self.monthly_data.to_dict('records')}

    def forecast(self, months_ahead=6):
        if self.monthly_data is None:
            raise ValueError("Call train() first")

        last_num  = self.monthly_data['month_num'].max()
        last_date = self.monthly_data['date'].max()

        rows, labels = [], []
        for i in range(1, months_ahead + 1):
            future = last_date + pd.DateOffset(months=i)
            m = future.month
            q = (m - 1) // 3 + 1
            rows.append([last_num + i,
                         np.sin(2 * np.pi * m / 12),
                         np.cos(2 * np.pi * m / 12),
                         q])
            labels.append(future.strftime('%Y-%m'))

        preds = self.model.predict(rows)

        print(f"\n🔮 Forecast — next {months_ahead} months:")
        for label, pred in zip(labels, preds):
            print(f"   {label}: {pred:.2f} avg price/unit")

        return dict(zip(labels, preds.tolist()))

    def save_model(self, path):
        joblib.dump({
            'model': self.model,
            'monthly_data': self.monthly_data,
            'features_used': self.features_used
        }, path)
        print(f"✅ Forecast model saved to {path}")

    def load_model(self, path):
        saved = joblib.load(path)
        self.model         = saved['model']
        self.monthly_data  = saved['monthly_data']
        self.features_used = saved['features_used']
        print(f"✅ Forecast model loaded from {path}")
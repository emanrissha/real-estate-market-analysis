"""
Location-based property segmentation using KMeans clustering
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib


class LocationSegmentation:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.feature_names = [
            'latitude', 'longitude', 'mrt_distance',
            'convenience_stores', 'price_per_unit'
        ]
        self.segment_profiles = None

    def prepare(self, df):
        available = [f for f in self.feature_names if f in df.columns]
        X = df[available].fillna(df[available].mean())
        return X, available

    def fit(self, df):
        X, available = self.prepare(df)
        self.features_used = available
        X_scaled = self.scaler.fit_transform(X)

        labels = self.model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)

        df = df.copy()
        df['segment'] = labels

        profiles = []
        print(f"\n📊 Location Segments (Silhouette: {sil:.4f}):")
        for i in range(self.n_clusters):
            c = df[df['segment'] == i]
            avg_price  = c['price_per_unit'].mean()
            avg_mrt    = c['mrt_distance'].mean()
            avg_stores = c['convenience_stores'].mean()
            avg_age    = c['house_age'].mean()

            # Auto label
            if avg_price >= df['price_per_unit'].quantile(0.75):
                label = 'Premium'
            elif avg_mrt <= 500:
                label = 'Transit-Hub'
            elif avg_stores >= 7:
                label = 'Urban-Core'
            else:
                label = 'Suburban'

            profiles.append({
                'segment': i,
                'label': label,
                'count': len(c),
                'avg_price': round(avg_price, 2),
                'avg_mrt_distance': round(avg_mrt, 1),
                'avg_stores': round(avg_stores, 1),
                'avg_house_age': round(avg_age, 1)
            })

            print(f"\n   Segment {i} — {label} ({len(c)} properties)")
            print(f"     Avg Price/Unit:  {avg_price:.2f}")
            print(f"     Avg MRT Dist:    {avg_mrt:.0f} m")
            print(f"     Avg Stores:      {avg_stores:.1f}")
            print(f"     Avg House Age:   {avg_age:.1f} yrs")

        self.segment_profiles = profiles
        print(f"\n   Silhouette Score: {sil:.4f}")

        return {
            'n_clusters': self.n_clusters,
            'silhouette_score': sil,
            'inertia': self.model.inertia_,
            'profiles': profiles
        }

    def predict(self, latitude, longitude, mrt_distance,
                convenience_stores, price_per_unit):
        X = self.scaler.transform([[
            latitude, longitude, mrt_distance,
            convenience_stores, price_per_unit
        ]])
        segment = int(self.model.predict(X)[0])
        label = self.segment_profiles[segment]['label'] \
                if self.segment_profiles else str(segment)
        return {'segment': segment, 'label': label}

    def save_model(self, path):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features_used': self.features_used,
            'segment_profiles': self.segment_profiles
        }, path)
        print(f"✅ Segmentation model saved to {path}")

    def load_model(self, path):
        saved = joblib.load(path)
        self.model = saved['model']
        self.scaler = saved['scaler']
        self.features_used = saved['features_used']
        self.segment_profiles = saved['segment_profiles']
        print(f"✅ Segmentation model loaded from {path}")
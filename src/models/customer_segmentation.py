"""
Customer segmentation using K-Means clustering
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

class CustomerSegmentation:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare customer features for clustering"""
        features = df[['age', 'price', 'deal_satisfaction']].copy()
        features = features.dropna()
        return features
    
    def fit(self, df):
        """Fit clustering model"""
        features = self.prepare_features(df)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit model
        self.model.fit(features_scaled)
        
        # Add cluster labels to dataframe
        df['cluster'] = self.model.labels_
        
        # Analyze clusters
        print("\n📊 Customer Segments:")
        for i in range(self.n_clusters):
            cluster_data = df[df['cluster'] == i]
            print(f"\n   Cluster {i}: {len(cluster_data)} customers")
            print(f"     Avg Age: {cluster_data['age'].mean():.1f}")
            print(f"     Avg Price: ${cluster_data['price'].mean():,.2f}")
            print(f"     Avg Satisfaction: {cluster_data['deal_satisfaction'].mean():.2f}")
        
        return {'n_clusters': self.n_clusters, 'inertia': self.model.inertia_}
    
    def predict(self, age, price, satisfaction):
        """Predict segment for new customer"""
        features = self.scaler.transform([[age, price, satisfaction]])
        return self.model.predict(features)[0]
    
    def save_model(self, path):
        """Save model to disk"""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        print(f"✅ Segmentation model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk"""
        saved = joblib.load(path)
        self.model = saved['model']
        self.scaler = saved['scaler']
        print(f"✅ Segmentation model loaded from {path}")
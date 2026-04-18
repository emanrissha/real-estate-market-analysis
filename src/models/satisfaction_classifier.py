"""
Customer satisfaction classification model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

class SatisfactionClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = ['price', 'area', 'age', 'mortgage_binary']
        
    def prepare_features(self, df):
        """Prepare features for satisfaction classification"""
        # Create satisfaction category (1-2: low, 3: medium, 4-5: high)
        df['satisfaction_category'] = pd.cut(
            df['deal_satisfaction'], 
            bins=[0, 2.5, 3.5, 5], 
            labels=[0, 1, 2]  # 0: low, 1: medium, 2: high
        ).astype(int)
        
        # Prepare features
        available_features = [f for f in self.features if f in df.columns]
        X = df[available_features].fillna(df[available_features].mean())
        y = df['satisfaction_category']
        
        return X, y, available_features
    
    def train(self, df):
        """Train satisfaction classification model"""
        X, y, features = self.prepare_features(df)
        self.features_used = features
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Satisfaction Classifier Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Low', 'Medium', 'High']))
        
        # Feature importance
        importance = dict(zip(features, self.model.feature_importances_))
        print(f"\n   Feature Importance: {importance}")
        
        return {
            'accuracy': accuracy,
            'feature_importance': importance,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def predict(self, price, area, age, mortgage):
        """Predict satisfaction category for new customer"""
        features = np.array([[price, area, age, mortgage]])
        prediction = self.model.predict(features)[0]
        
        categories = {0: 'Low Satisfaction', 1: 'Medium Satisfaction', 2: 'High Satisfaction'}
        return categories[prediction]
    
    def save_model(self, path):
        """Save model to disk"""
        joblib.dump(self.model, path)
        print(f"✅ Satisfaction model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk"""
        self.model = joblib.load(path)
        print(f"✅ Satisfaction model loaded from {path}")
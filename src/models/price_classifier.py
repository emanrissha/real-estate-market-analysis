"""
Price segment classifier - predicts Low / Medium / High price tier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


class PriceClassifier:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200, random_state=42, max_depth=3
        )
        self.features = [
            'house_age', 'mrt_distance', 'convenience_stores',
            'latitude', 'longitude', 'log_mrt_distance',
            'distance_from_center'
        ]
        self.features_used = None
        self.labels = ['Low', 'Medium', 'High']

    def prepare(self, df):
        available = [f for f in self.features if f in df.columns]
        X = df[available].fillna(df[available].mean())
        y = df['price_segment_encoded']
        return X, y, available

    def train(self, df):
        X, y, features = self.prepare(df)
        self.features_used = features

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')

        print(f"\n📊 Price Classifier Results:")
        print(f"   Accuracy:    {accuracy:.4f}")
        print(f"   CV Accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})")
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=self.labels))
        print(f"   Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        importance = dict(zip(features, self.model.feature_importances_))
        importance_sorted = dict(sorted(importance.items(),
                                        key=lambda x: x[1], reverse=True))
        print(f"\n   Feature Importance:")
        for f, v in importance_sorted.items():
            print(f"     {f}: {v:.4f}")

        return {
            'accuracy': accuracy,
            'cv_accuracy_mean': cv.mean(),
            'cv_accuracy_std': cv.std(),
            'importance': importance,
            'report': classification_report(y_test, y_pred,
                                            target_names=self.labels,
                                            output_dict=True)
        }

    def predict(self, house_age, mrt_distance, convenience_stores,
                latitude, longitude, distance_from_center=0):
        log_mrt = np.log1p(mrt_distance)
        data = pd.DataFrame([[
            house_age, mrt_distance, convenience_stores,
            latitude, longitude, log_mrt, distance_from_center
        ]], columns=self.features_used)
        pred = self.model.predict(data)[0]
        return self.labels[pred]

    def save_model(self, path):
        joblib.dump({'model': self.model, 'features_used': self.features_used}, path)
        print(f"✅ Price classifier saved to {path}")

    def load_model(self, path):
        saved = joblib.load(path)
        self.model = saved['model']
        self.features_used = saved['features_used']
        print(f"✅ Price classifier loaded from {path}")
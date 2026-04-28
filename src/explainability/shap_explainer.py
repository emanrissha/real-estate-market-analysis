"""
SHAP explainability for the Price Predictor model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class ShapExplainer:
    def __init__(self):
        self.explainer = None
        self.shap_values = None
        self.feature_names = None

    def fit(self, model, X):
        """Fit SHAP explainer on trained model and data"""
        try:
            import shap
        except ImportError:
            raise ImportError("Run: pip install shap")

        self.feature_names = list(X.columns)
        print("⚙️  Fitting SHAP explainer...")
        self.explainer  = shap.TreeExplainer(model.model)
        self.shap_values = self.explainer.shap_values(X)
        print("✅ SHAP explainer fitted")
        return self

    def summary_plot(self, X, save_path='reports/figures/shap_summary.png'):
        """Generate and save SHAP summary plot"""
        try:
            import shap
        except ImportError:
            raise ImportError("Run: pip install shap")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, X,
            feature_names=self.feature_names,
            show=False
        )
        plt.title('SHAP Feature Importance — Price Predictor', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP summary plot saved to {save_path}")

    def bar_plot(self, X, save_path='reports/figures/shap_bar.png'):
        """Generate and save SHAP bar chart"""
        try:
            import shap
        except ImportError:
            raise ImportError("Run: pip install shap")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        mean_shap = np.abs(self.shap_values).mean(axis=0)
        importance = pd.Series(mean_shap, index=self.feature_names)
        importance = importance.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(importance.index, importance.values,
                       color='steelblue', edgecolor='white')
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax.set_title('Feature Importance (SHAP)', fontsize=14)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP bar chart saved to {save_path}")

    def get_top_features(self, n=3):
        """Return top N most impactful features"""
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        importance = pd.Series(mean_shap, index=self.feature_names)
        return importance.sort_values(ascending=False).head(n).to_dict()

    def explain_single(self, X_row):
        """Explain a single prediction"""
        sv = self.explainer.shap_values(X_row)
        explanation = pd.Series(sv[0], index=self.feature_names)
        explanation = explanation.sort_values(key=abs, ascending=False)
        print("\n🔍 Single Prediction Explanation:")
        for feat, val in explanation.items():
            direction = '↑ increases' if val > 0 else '↓ decreases'
            print(f"   {feat}: {direction} price by {abs(val):.3f}")
        return explanation.to_dict()
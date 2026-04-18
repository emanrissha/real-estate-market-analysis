"""
Advanced correlation analysis for real estate data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CorrelationAnalysis:
    def __init__(self, df):
        self.df = df
    
    def compute_correlations(self):
        """Compute correlation matrix for numeric variables"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        print("\n" + "="*60)
        print("CORRELATION MATRIX")
        print("="*60)
        print(correlation_matrix.round(4))
        
        return correlation_matrix
    
    def plot_correlation_heatmap(self, save_path='../reports/correlation_heatmap.png'):
        """Create correlation heatmap visualization"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                    mask=mask,
                    annot=True, 
                    fmt='.2f', 
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Heatmap of Real Estate Variables', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Heatmap saved to {save_path}")
    
    def find_strong_correlations(self, threshold=0.5):
        """Find variable pairs with strong correlations"""
        correlation_matrix = self.compute_correlations()
        
        strong_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    strong_pairs.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr
                    })
        
        print("\n" + "="*60)
        print(f"STRONG CORRELATIONS (|r| >= {threshold})")
        print("="*60)
        
        for pair in sorted(strong_pairs, key=lambda x: abs(x['correlation']), reverse=True):
            direction = "positive" if pair['correlation'] > 0 else "negative"
            print(f"   {pair['var1']} ↔ {pair['var2']}: {pair['correlation']:.4f} ({direction})")
        
        return strong_pairs
    
    def correlation_with_target(self, target='price'):
        """Find correlations of all variables with target variable"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlations = self.df[numeric_cols].corr()[target].sort_values(ascending=False)
        
        print("\n" + "="*60)
        print(f"CORRELATION WITH {target.upper()}")
        print("="*60)
        
        for var, corr in correlations.items():
            if var != target:
                print(f"   {var}: {corr:.4f}")
        
        return correlations
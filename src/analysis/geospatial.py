"""
Geospatial analysis for real estate data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GeospatialAnalysis:
    def __init__(self, df):
        self.df = df
    
    def analyze_by_state(self):
        """Analyze real estate metrics by state"""
        print("\n" + "="*60)
        print("GEOSPATIAL ANALYSIS BY STATE")
        print("="*60)
        
        state_analysis = self.df.groupby('state').agg({
            'price': ['mean', 'median', 'count'],
            'deal_satisfaction': 'mean',
            'area': 'mean'
        }).round(2)
        
        state_analysis.columns = ['avg_price', 'median_price', 'property_count', 'avg_satisfaction', 'avg_area']
        state_analysis = state_analysis.sort_values('property_count', ascending=False)
        
        print("\n" + state_analysis.to_string())
        
        return state_analysis
    
    def plot_state_performance(self, save_path='../reports/state_performance.png'):
        """Create state performance visualization"""
        state_metrics = self.df.groupby('state').agg({
            'price': 'mean',
            'deal_satisfaction': 'mean',
            'customerid': 'count'
        }).round(2)
        
        state_metrics.columns = ['avg_price', 'avg_satisfaction', 'count']
        state_metrics = state_metrics.sort_values('count', ascending=False).head(10)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        # Plot 1: Property count by state
        axes[0].barh(state_metrics.index, state_metrics['count'], color='steelblue')
        axes[0].set_xlabel('Number of Properties')
        axes[0].set_title('Top 10 States by Property Count')
        
        # Plot 2: Average price by state
        axes[1].barh(state_metrics.index, state_metrics['avg_price'], color='coral')
        axes[1].set_xlabel('Average Price ($)')
        axes[1].set_title('Average Price by State')
        
        # Plot 3: Average satisfaction by state
        axes[2].barh(state_metrics.index, state_metrics['avg_satisfaction'], color='green')
        axes[2].set_xlabel('Average Satisfaction (1-5)')
        axes[2].set_title('Customer Satisfaction by State')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ State performance chart saved to {save_path}")
    
    def identify_top_markets(self, n=5):
        """Identify top markets based on multiple criteria"""
        state_metrics = self.df.groupby('state').agg({
            'price': 'mean',
            'deal_satisfaction': 'mean',
            'customerid': 'count'
        }).round(2)
        
        state_metrics.columns = ['avg_price', 'avg_satisfaction', 'sales_volume']
        
        # Normalize metrics for ranking
        state_metrics['price_rank'] = state_metrics['avg_price'].rank(ascending=False)
        state_metrics['satisfaction_rank'] = state_metrics['avg_satisfaction'].rank(ascending=False)
        state_metrics['volume_rank'] = state_metrics['sales_volume'].rank(ascending=False)
        
        # Composite score
        state_metrics['composite_score'] = (
            state_metrics['price_rank'] + 
            state_metrics['satisfaction_rank'] + 
            state_metrics['volume_rank']
        )
        
        top_markets = state_metrics.nsmallest(n, 'composite_score')
        
        print("\n" + "="*60)
        print(f"TOP {n} MARKETS (Composite Score)")
        print("="*60)
        print(top_markets[['avg_price', 'avg_satisfaction', 'sales_volume', 'composite_score']])
        
        return top_markets
    
    def price_map_by_state(self, save_path='../reports/price_map.png'):
        """Create price distribution map (simulated)"""
        state_prices = self.df.groupby('state')['price'].mean().sort_values()
        
        plt.figure(figsize=(12, 8))
        
        # Horizontal bar chart as map alternative
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(state_prices)))
        bars = plt.barh(range(len(state_prices)), state_prices.values, color=colors)
        
        plt.yticks(range(len(state_prices)), state_prices.index)
        plt.xlabel('Average Property Price ($)')
        plt.title('Average Property Price by State')
        
        # Add value labels
        for i, v in enumerate(state_prices.values):
            plt.text(v + 5000, i, f'${v:,.0f}', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Price map saved to {save_path}")
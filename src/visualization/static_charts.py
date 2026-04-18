"""
Static chart generation for reports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class StaticCharts:
    def __init__(self, df):
        self.df = df
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def generate_all_charts(self, save_dir='../reports/'):
        """Generate all static charts for reporting"""
        print("\n" + "="*60)
        print("GENERATING STATIC CHARTS")
        print("="*60)
        
        charts = {
            'price_distribution': self.plot_price_distribution,
            'satisfaction_distribution': self.plot_satisfaction_distribution,
            'building_comparison': self.plot_building_comparison,
            'age_price_scatter': self.plot_age_price_scatter,
            'monthly_trend': self.plot_monthly_trend
        }
        
        for name, chart_func in charts.items():
            print(f"\n📊 Generating {name}...")
            chart_func(save_dir)
        
        print("\n✅ All static charts generated!")
    
    def plot_price_distribution(self, save_dir='../reports/'):
        """Plot price distribution histogram"""
        plt.figure()
        sns.histplot(self.df['price'].dropna(), bins=30, kde=True, color='steelblue')
        plt.title('Property Price Distribution')
        plt.xlabel('Price ($)')
        plt.ylabel('Frequency')
        plt.axvline(self.df['price'].mean(), color='red', linestyle='--', label=f'Mean: ${self.df["price"].mean():,.0f}')
        plt.axvline(self.df['price'].median(), color='green', linestyle='--', label=f'Median: ${self.df["price"].median():,.0f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}price_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_satisfaction_distribution(self, save_dir='../reports/'):
        """Plot satisfaction score distribution"""
        plt.figure()
        satisfaction_counts = self.df['deal_satisfaction'].value_counts().sort_index()
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        plt.bar(satisfaction_counts.index, satisfaction_counts.values, color=colors, edgecolor='black')
        plt.title('Customer Satisfaction Distribution')
        plt.xlabel('Satisfaction Score (1-5)')
        plt.ylabel('Number of Customers')
        plt.xticks(range(1, 6))
        
        # Add value labels
        for i, (score, count) in enumerate(satisfaction_counts.items()):
            plt.text(score, count + 1, str(count), ha='center')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}satisfaction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_building_comparison(self, save_dir='../reports/'):
        """Compare metrics across building types"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Price by building
        building_price = self.df.groupby('building')['price'].mean().sort_values()
        axes[0].barh(building_price.index, building_price.values, color='coral')
        axes[0].set_xlabel('Average Price ($)')
        axes[0].set_title('Average Price by Building Type')
        
        # Satisfaction by building
        building_sat = self.df.groupby('building')['deal_satisfaction'].mean().sort_values()
        axes[1].barh(building_sat.index, building_sat.values, color='steelblue')
        axes[1].set_xlabel('Average Satisfaction (1-5)')
        axes[1].set_title('Satisfaction by Building Type')
        
        # Count by building
        building_count = self.df['building'].value_counts()
        axes[2].barh(building_count.index, building_count.values, color='green')
        axes[2].set_xlabel('Number of Properties')
        axes[2].set_title('Property Count by Building Type')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}building_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_age_price_scatter(self, save_dir='../reports/'):
        """Scatter plot of age vs price"""
        if 'age' not in self.df.columns:
            print("   Age column not available")
            return
        
        plt.figure()
        scatter = plt.scatter(self.df['age'], self.df['price'], 
                            c=self.df['deal_satisfaction'], cmap='RdYlGn', 
                            alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Satisfaction Score')
        plt.xlabel('Customer Age')
        plt.ylabel('Property Price ($)')
        plt.title('Age vs Price (colored by Satisfaction)')
        
        # Add trend line
        z = np.polyfit(self.df['age'].dropna(), self.df['price'].dropna(), 1)
        p = np.poly1d(z)
        plt.plot(self.df['age'].sort_values(), p(self.df['age'].sort_values()), 
                'r--', linewidth=2, label='Trend Line')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}age_price_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_monthly_trend(self, save_dir='../reports/'):
        """Plot monthly revenue trend"""
        date_col = 'date' if 'date' in self.df.columns else 'date_sale'
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        
        monthly = self.df.groupby(self.df[date_col].dt.to_period('M'))['price'].sum()
        
        plt.figure()
        plt.plot(monthly.index.astype(str), monthly.values, marker='o', linewidth=2, markersize=6, color='darkblue')
        plt.xlabel('Month')
        plt.ylabel('Total Revenue ($)')
        plt.title('Monthly Revenue Trend')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        x = range(len(monthly))
        z = np.polyfit(x, monthly.values, 1)
        p = np.poly1d(z)
        plt.plot(monthly.index.astype(str), p(x), 'r--', linewidth=2, label='Trend')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}monthly_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
"""
Geospatial visualizations for real estate data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class MapVisualizations:
    def __init__(self, df):
        self.df = df
    
    def create_state_heatmap(self, save_path='../reports/state_heatmap.png'):
        """Create a heatmap-style visualization of state performance"""
        
        # Aggregate data by state
        state_data = self.df.groupby('state').agg({
            'price': 'mean',
            'deal_satisfaction': 'mean',
            'customerid': 'count'
        }).round(2)
        
        state_data.columns = ['avg_price', 'avg_satisfaction', 'property_count']
        state_data = state_data.sort_values('property_count', ascending=False)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Heatmap 1: Property Count
        colors1 = plt.cm.Blues(np.linspace(0.3, 0.9, len(state_data)))
        axes[0].barh(range(len(state_data)), state_data['property_count'], color=colors1)
        axes[0].set_yticks(range(len(state_data)))
        axes[0].set_yticklabels(state_data.index)
        axes[0].set_xlabel('Number of Properties')
        axes[0].set_title('Property Count by State')
        
        # Add value labels
        for i, v in enumerate(state_data['property_count']):
            axes[0].text(v + 0.5, i, str(int(v)), va='center')
        
        # Heatmap 2: Average Price
        colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(state_data)))
        axes[1].barh(range(len(state_data)), state_data['avg_price'], color=colors2)
        axes[1].set_yticks(range(len(state_data)))
        axes[1].set_yticklabels(state_data.index)
        axes[1].set_xlabel('Average Price ($)')
        axes[1].set_title('Average Property Price by State')
        
        for i, v in enumerate(state_data['avg_price']):
            axes[1].text(v + 5000, i, f'${v:,.0f}', va='center')
        
        # Heatmap 3: Average Satisfaction
        colors3 = plt.cm.Greens(np.linspace(0.3, 0.9, len(state_data)))
        axes[2].barh(range(len(state_data)), state_data['avg_satisfaction'], color=colors3)
        axes[2].set_yticks(range(len(state_data)))
        axes[2].set_yticklabels(state_data.index)
        axes[2].set_xlabel('Average Satisfaction (1-5)')
        axes[2].set_title('Customer Satisfaction by State')
        
        for i, v in enumerate(state_data['avg_satisfaction']):
            axes[2].text(v + 0.05, i, f'{v:.2f}', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ State heatmap saved to {save_path}")
        
        return state_data
    
    def create_bubble_map(self, save_path='../reports/bubble_map.png'):
        """Create bubble chart for state performance"""
        
        state_data = self.df.groupby('state').agg({
            'price': 'mean',
            'deal_satisfaction': 'mean',
            'customerid': 'count'
        }).round(2)
        
        plt.figure(figsize=(12, 8))
        
        # Create bubble chart
        scatter = plt.scatter(
            state_data['avg_price'], 
            state_data['avg_satisfaction'],
            s=state_data['customerid'] * 50,  # Bubble size
            c=state_data['customerid'],
            cmap='viridis',
            alpha=0.6,
            edgecolors='black',
            linewidth=1
        )
        
        # Add state labels
        for state, row in state_data.iterrows():
            plt.annotate(state, (row['avg_price'], row['avg_satisfaction']), 
                        fontsize=9, ha='center', va='bottom')
        
        plt.colorbar(scatter, label='Number of Properties')
        plt.xlabel('Average Price ($)')
        plt.ylabel('Average Satisfaction (1-5)')
        plt.title('State Performance Bubble Chart\n(Bubble size = number of properties)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Bubble map saved to {save_path}")
    
    def generate_all_maps(self):
        """Generate all geospatial visualizations"""
        print("\n" + "="*60)
        print("🗺️ GENERATING GEOSPATIAL VISUALIZATIONS")
        print("="*60)
        
        self.create_state_heatmap()
        self.create_bubble_map()
        
        print("\n✅ All maps generated!")
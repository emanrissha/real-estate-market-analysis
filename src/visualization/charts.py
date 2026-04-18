import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class RealEstateVisualizer:
    def __init__(self, df):
        """
        Initialize with merged dataframe
        """
        self.df = df
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_satisfaction_by_country(self):
        """
        Visualization 1: Average deal satisfaction by country
        Bar chart
        """
        # Calculate average satisfaction by country
        satisfaction = self.df.groupby('country')['deal_satisfaction'].mean().sort_values()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(satisfaction)), satisfaction.values, color='steelblue')
        
        # Customize
        plt.yticks(range(len(satisfaction)), satisfaction.index)
        plt.xlabel('Average Deal Satisfaction (1-5)')
        plt.title('Average Deal Satisfaction by Country')
        
        # Add value labels
        for i, v in enumerate(satisfaction.values):
            plt.text(v + 0.05, i, f'{v:.2f}', va='center')
        
        plt.tight_layout()
        
        # Save
        plt.savefig('../reports/satisfaction_by_country.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: reports/satisfaction_by_country.png")
        
        return satisfaction
    
    def plot_monthly_revenue(self):
        """
        Visualization 2: Monthly revenue over time
        Time-series line graph
        """
        # Extract month and year
        df_copy = self.df.copy()
        date_col = 'date' if 'date' in df_copy.columns else 'date_sale'
        df_copy['month'] = pd.to_datetime(df_copy[date_col]).dt.to_period('M')
        
        # Calculate monthly revenue
        monthly_revenue = df_copy.groupby('month')['price'].sum()
        
        # Create figure
        plt.figure(figsize=(12, 5))
        plt.plot(monthly_revenue.index.astype(str), monthly_revenue.values, 
                 marker='o', linewidth=2, markersize=6, color='darkgreen')
        
        # Customize
        plt.xlabel('Month')
        plt.ylabel('Total Revenue ($)')
        plt.title('Monthly Revenue Over Time')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        from matplotlib.ticker import FuncFormatter
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        # Save
        plt.savefig('../reports/monthly_revenue.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: reports/monthly_revenue.png")
        
        return monthly_revenue
    
    def plot_apartments_by_state_pareto(self):
        """
        Visualization 3: Apartments sold by state - Pareto Chart
        Bars for frequency, line for cumulative percentage
        """
        # Filter for apartments only
        apartments = self.df[self.df['building'].astype(str).str.lower() == 'apartment']
        
        # Count by state
        state_counts = apartments['state'].value_counts()
        
        # Calculate cumulative percentage
        cumulative_pct = state_counts.cumsum() / state_counts.sum() * 100
        
        # Create figure with twin axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Bars
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(state_counts)))
        bars = ax1.bar(range(len(state_counts)), state_counts.values, color=colors, alpha=0.8)
        ax1.set_xticks(range(len(state_counts)))
        ax1.set_xticklabels(state_counts.index, rotation=45, ha='right')
        ax1.set_ylabel('Number of Apartments Sold')
        ax1.set_xlabel('State')
        
        # Line for cumulative percentage
        ax2 = ax1.twinx()
        ax2.plot(range(len(cumulative_pct)), cumulative_pct.values, 
                 'r-o', linewidth=2, markersize=6)
        ax2.set_ylabel('Cumulative Percentage (%)')
        ax2.set_ylim(0, 105)
        
        # Add 80% line
        ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.7)
        
        plt.title('Apartments Sold by State (Pareto Chart)')
        
        # Add value labels on bars
        for i, (state, count) in enumerate(state_counts.items()):
            ax1.text(i, count + 0.5, str(count), ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save
        plt.savefig('../reports/apartments_by_state_pareto.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: reports/apartments_by_state_pareto.png")
        
        return state_counts, cumulative_pct
    
    def plot_age_distribution(self):
        """
        Visualization 4: Age distribution histogram
        """
        plt.figure(figsize=(10, 6))
        
        # Histogram
        n, bins, patches = plt.hist(self.df['age'].dropna(), bins=15, 
                                     edgecolor='black', alpha=0.7, color='coral')
        
        # Add mean and median lines
        mean_age = self.df['age'].mean()
        median_age = self.df['age'].median()
        
        plt.axvline(mean_age, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_age:.1f}')
        plt.axvline(median_age, color='green', linestyle='--', linewidth=2, label=f'Median: {median_age:.1f}')
        
        # Customize
        plt.xlabel('Age')
        plt.ylabel('Number of Customers')
        plt.title('Age Distribution of Customers')
        plt.legend()
        
        plt.tight_layout()
        
        # Save
        plt.savefig('../reports/age_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: reports/age_distribution.png")
        
        return mean_age, median_age
    
    def plot_yearly_sales_by_building(self):
        """
        Visualization 5: Yearly sales by building type
        Stacked bar chart
        """
        # Extract year
        df_copy = self.df.copy()
        date_col = 'date' if 'date' in df_copy.columns else 'date_sale'
        df_copy['year'] = pd.to_datetime(df_copy[date_col]).dt.year
        
        # Group by year and building
        yearly_sales = df_copy.groupby(['year', 'building'])['sold'].sum().unstack(fill_value=0)
        
        # Create stacked bar chart
        ax = yearly_sales.plot(kind='bar', stacked=True, figsize=(12, 6), 
                                colormap='Set2', edgecolor='black')
        
        # Customize
        plt.xlabel('Year')
        plt.ylabel('Number of Properties Sold')
        plt.title('Yearly Sales by Building Type')
        plt.legend(title='Building Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for c in ax.containers:
            ax.bar_label(c, label_type='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save
        plt.savefig('../reports/yearly_sales_by_building.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: reports/yearly_sales_by_building.png")
        
        return yearly_sales
    
    def run_all_visualizations(self):
        """
        Run all 5 visualizations
        """
        print("\n" + "="*60)
        print("🎨 TASK 4: GENERATING VISUALIZATIONS")
        print("="*60)
        
        results = {}
        
        print("\n1️⃣ Deal Satisfaction by Country...")
        results['satisfaction'] = self.plot_satisfaction_by_country()
        
        print("\n2️⃣ Monthly Revenue...")
        results['revenue'] = self.plot_monthly_revenue()
        
        print("\n3️⃣ Apartments Sold by State (Pareto)...")
        results['apartments'], results['cumulative'] = self.plot_apartments_by_state_pareto()
        
        print("\n4️⃣ Age Distribution...")
        results['age_mean'], results['age_median'] = self.plot_age_distribution()
        
        print("\n5️⃣ Yearly Sales by Building...")
        results['yearly_sales'] = self.plot_yearly_sales_by_building()
        
        print("\n" + "="*60)
        print("✅ TASK 4 COMPLETE! All visualizations saved to reports/")
        print("="*60)
        
        return results
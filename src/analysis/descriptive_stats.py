import pandas as pd
import numpy as np

class DescriptiveStats:
    def __init__(self, df):
        """
        Initialize with merged dataframe
        """
        self.df = df
    
    def breakdown_by_building(self):
        """
        TASK 2.1: Breakdown by building type
        - Count building types
        - Totals: sold, mortgage
        - Averages: area, price, deal_satisfaction
        """
        print("\n" + "="*60)
        print("BREAKDOWN BY BUILDING TYPE")
        print("="*60)
        
        # How many building types
        n_buildings = self.df['building'].nunique()
        building_types = self.df['building'].unique().tolist()
        
        print(f"\nNumber of building types: {n_buildings}")
        print(f"Building types: {building_types}")
        
        # TOTALS by building type
        totals = self.df.groupby('building').agg({
            'sold': 'sum',
            'mortgage_binary': 'sum'
        }).rename(columns={'mortgage_binary': 'mortgage'})
        
        print("\n" + "-"*40)
        print("TOTALS by building type:")
        print("-"*40)
        print(totals)
        
        # AVERAGES by building type
        averages = self.df.groupby('building').agg({
            'area': 'mean',
            'price': 'mean',
            'deal_satisfaction': 'mean'
        }).round(2)
        
        print("\n" + "-"*40)
        print("AVERAGES by building type:")
        print("-"*40)
        print(averages)
        
        return totals, averages
    
    def breakdown_by_country(self):
        """
        TASK 2.2: Breakdown by country
        - Frequency distribution
        - Totals and averages
        """
        print("\n" + "="*60)
        print("BREAKDOWN BY COUNTRY")
        print("="*60)
        
        # Frequency and statistics by country
        country_stats = self.df.groupby('country').agg({
            'customerid': 'count',
            'sold': 'sum',
            'price': 'mean',
            'deal_satisfaction': 'mean'
        }).round(2)
        
        country_stats.columns = ['frequency', 'total_sold', 'avg_price', 'avg_satisfaction']
        
        # Add percentage
        country_stats['percentage'] = (country_stats['frequency'] / country_stats['frequency'].sum() * 100).round(1)
        
        print("\n" + country_stats.to_string())
        
        return country_stats
    
    def breakdown_by_state(self):
        """
        TASK 2.3: Breakdown by state
        - Frequency distribution
        - Relative frequency
        - Cumulative frequency
        """
        print("\n" + "="*60)
        print("BREAKDOWN BY STATE")
        print("="*60)
        
        # Frequency distribution
        state_freq = self.df['state'].value_counts().reset_index()
        state_freq.columns = ['state', 'frequency']
        
        # Sort by frequency descending
        state_freq = state_freq.sort_values('frequency', ascending=False)
        
        # Relative frequency
        total = state_freq['frequency'].sum()
        state_freq['relative_frequency'] = (state_freq['frequency'] / total * 100).round(2)
        
        # Cumulative frequency
        state_freq['cumulative_frequency'] = state_freq['relative_frequency'].cumsum().round(2)
        
        print("\n" + state_freq.to_string(index=False))
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Total states: {len(state_freq)}")
        print(f"   Most frequent: {state_freq.iloc[0]['state']} ({state_freq.iloc[0]['frequency']} properties)")
        print(f"   Top 3 states: {state_freq.iloc[0]['state']}, {state_freq.iloc[1]['state']}, {state_freq.iloc[2]['state']}")
        print(f"   Top 3 cumulative: {state_freq.iloc[2]['cumulative_frequency']}%")
        
        return state_freq
    
    def run_all(self):
        """
        Run all Task 2 analyses
        """
        print("\n" + "🚀"*30)
        print("RUNNING TASK 2: DESCRIPTIVE STATISTICS")
        print("🚀"*30)
        
        results = {}
        
        results['building_totals'], results['building_averages'] = self.breakdown_by_building()
        results['country_stats'] = self.breakdown_by_country()
        results['state_stats'] = self.breakdown_by_state()
        
        print("\n" + "="*60)
        print("✅ TASK 2 COMPLETE!")
        print("="*60)
        
        return results
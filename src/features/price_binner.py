import pandas as pd
import numpy as np

class PriceBinner:
    def __init__(self, n_bins=10):
        """
        Initialize PriceBinner with number of bins
        """
        self.n_bins = n_bins
    
    def create_price_bins(self, df):
        """
        Create equal frequency price bins (10 bins)
        """
        # Remove any null prices
        df_clean = df.dropna(subset=['price'])
        
        # Create equal frequency bins
        df['price_bin'] = pd.qcut(df['price'], q=self.n_bins, duplicates='drop')
        
        print(f"✅ Price bins created: {df['price_bin'].nunique()} bins")
        print(f"   Price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
        
        return df
    
    def get_price_distribution(self, df):
        """
        Get distribution of properties by price bin
        """
        distribution = df['price_bin'].value_counts().sort_index()
        print("\n📊 Properties sold by price interval:")
        print(distribution)
        return distribution
    
    def get_price_statistics_by_bin(self, df):
        """
        Get statistics for each price bin
        """
        stats = df.groupby('price_bin')['price'].agg(['count', 'min', 'max', 'mean']).round(2)
        print("\n📊 Price bin statistics:")
        print(stats)
        return stats
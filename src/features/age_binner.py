import pandas as pd
import numpy as np

class AgeBinner:
    def __init__(self, bins=None, labels=None):
        """
        Initialize AgeBinner with custom bins
        
        Default bins: [19, 25, 31, 36, 42, 48, 54, 59, 65, 71, 76]
        Default labels: ['19-25', '25-31', '31-36', '36-42', '42-48', 
                         '48-54', '54-59', '59-65', '65-71', '71-76']
        """
        self.bins = bins or [19, 25, 31, 36, 42, 48, 54, 59, 65, 71, 76]
        self.labels = labels or ['19-25', '25-31', '31-36', '36-42', '42-48', 
                                  '48-54', '54-59', '59-65', '65-71', '71-76']
    
    def calculate_age(self, df):
        """
        Calculate age at purchase from birth_date and sale date
        """
        # Ensure dates are datetime
        df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
        
        # Find date column (could be 'date' or 'date_sale')
        date_col = None
        if 'date' in df.columns:
            date_col = 'date'
        elif 'date_sale' in df.columns:
            date_col = 'date_sale'
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            # Calculate age
            df['age'] = df[date_col].dt.year - df['birth_date'].dt.year
            # Adjust for birthday not yet occurred in the sale year
            df.loc[df[date_col].dt.month < df['birth_date'].dt.month, 'age'] -= 1
        
        print(f"✅ Age calculated. Age range: {df['age'].min():.0f} - {df['age'].max():.0f}")
        return df
    
    def create_age_groups(self, df):
        """
        Create age interval categories
        """
        df['age_group'] = pd.cut(
            df['age'], 
            bins=self.bins, 
            labels=self.labels, 
            right=True
        )
        
        print(f"✅ Age groups created: {df['age_group'].nunique()} groups")
        return df
    
    def get_age_distribution(self, df):
        """
        Get number of properties sold by age interval
        """
        distribution = df['age_group'].value_counts().sort_index()
        print("\n📊 Properties sold by age interval:")
        print(distribution)
        return distribution
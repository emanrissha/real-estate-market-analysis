import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self):
        pass
    
    def clean_column_names(self, df):
        cleaned_columns = []
        for col in df.columns:
            col = col.replace('\ufeff', '')
            col = col.lower().strip()
            col = col.replace(' ', '_')
            cleaned_columns.append(col)
        df.columns = cleaned_columns
        return df
    
    def handle_missing_values(self, df):
        df = df.replace(['NA', 'N/A', 'null', '', 'NULL', 'None'], np.nan)
        return df
    
    def fix_customer_id(self, df):
        if 'customerid' in df.columns:
            df['customerid'] = df['customerid'].astype(str).str.strip()
            df['customerid'] = df['customerid'].replace('nan', np.nan)
        return df
    
    def convert_data_types(self, properties, customers):
        # Properties
        if 'price' in properties.columns:
            properties['price'] = properties['price'].astype(str).str.replace('$', '')
            properties['price'] = properties['price'].astype(str).str.replace(',', '')
            properties['price'] = pd.to_numeric(properties['price'], errors='coerce')
        
        if 'area' in properties.columns:
            properties['area'] = pd.to_numeric(properties['area'], errors='coerce')
        
        if 'status' in properties.columns:
            properties['sold'] = properties['status'].map({'Sold': 1}).fillna(0)
        
        # Customers
        if 'birth_date' in customers.columns:
            customers['birth_date'] = pd.to_datetime(customers['birth_date'], errors='coerce')
        
        if 'mortgage' in customers.columns:
            customers['mortgage_binary'] = customers['mortgage'].map({'Yes': 1, 'yes': 1, 'No': 0, 'no': 0}).fillna(0)
        
        if 'deal_satisfaction' in customers.columns:
            customers['deal_satisfaction'] = pd.to_numeric(customers['deal_satisfaction'], errors='coerce')
        
        return properties, customers
    
    def clean_all(self, properties, customers):
        print("\n🧹 STARTING DATA CLEANING...")
        
        properties = self.clean_column_names(properties)
        customers = self.clean_column_names(customers)
        
        properties = self.handle_missing_values(properties)
        customers = self.handle_missing_values(customers)
        
        properties = self.fix_customer_id(properties)
        customers = self.fix_customer_id(customers)
        
        properties, customers = self.convert_data_types(properties, customers)
        
        print("✅ DATA CLEANING COMPLETE!")
        return properties, customers
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, raw_path="data/raw/"):
        base_path = Path('/workspaces/real-estate-market-analysis')
        self.raw_path = base_path / raw_path
        
    def load_customers(self):
        file_path = self.raw_path / "customers.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ Loaded customers: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def load_properties(self):
        file_path = self.raw_path / "properties.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ Loaded properties: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def load_both(self):
        print("📂 Loading data files...")
        properties = self.load_properties()
        customers = self.load_customers()
        return properties, customers
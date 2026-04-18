import pandas as pd
from pathlib import Path
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner

class Preprocessor:
    def __init__(self):
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
    
    def run_full_pipeline(self):
        print("\n" + "="*60)
        print("🚀 RUNNING TASK 1: DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load
        print("\n📂 STEP 1: Loading raw data...")
        properties, customers = self.loader.load_both()
        
        # Step 2: Clean
        print("\n🧹 STEP 2: Cleaning data...")
        properties, customers = self.cleaner.clean_all(properties, customers)
        
        # Step 3: Merge
        print("\n🔗 STEP 3: Merging datasets...")
        merged = pd.merge(properties, customers, on='customerid', how='inner')
        print(f"   Merged shape: {merged.shape}")
        
        # Step 4: Remove unnamed columns
        unnamed_cols = [col for col in merged.columns if 'unnamed' in col.lower()]
        if unnamed_cols:
            merged = merged.drop(columns=unnamed_cols)
            print(f"   Dropped {len(unnamed_cols)} unnamed columns")
        
        # Step 5: Save
        print("\n💾 STEP 4: Saving merged dataset...")
        output_path = Path("data/processed/merged_real_estate.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"   Saved to {output_path}")
        
        print("\n" + "="*60)
        print(f"✅ TASK 1 COMPLETE! Final dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
        print("="*60)
        
        return merged
"""
Statistical hypothesis testing for real estate data
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, pearsonr, chi2_contingency

class HypothesisTests:
    def __init__(self, df):
        self.df = df
    
    def test_price_by_building_type(self):
        """
        H0: Average price is the same across all building types
        H1: At least one building type has different average price
        """
        print("\n" + "="*60)
        print("HYPOTHESIS TEST: Price by Building Type")
        print("="*60)
        
        groups = [self.df[self.df['building'] == bt]['price'].dropna() 
                  for bt in self.df['building'].unique()]
        
        f_stat, p_value = f_oneway(*groups)
        
        print(f"ANOVA F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("✅ Reject H0: Building type significantly affects price")
        else:
            print("❌ Fail to reject H0: No significant difference in price")
        
        return {'test': 'ANOVA', 'statistic': f_stat, 'p_value': p_value}
    
    def test_satisfaction_by_country(self):
        """
        H0: Satisfaction levels are equal across countries
        H1: Satisfaction differs by country
        """
        print("\n" + "="*60)
        print("HYPOTHESIS TEST: Satisfaction by Country")
        print("="*60)
        
        groups = [self.df[self.df['country'] == c]['deal_satisfaction'].dropna() 
                  for c in self.df['country'].unique()]
        
        f_stat, p_value = f_oneway(*groups)
        
        print(f"ANOVA F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("✅ Reject H0: Country significantly affects satisfaction")
        else:
            print("❌ Fail to reject H0: No significant difference in satisfaction")
        
        return {'test': 'ANOVA', 'statistic': f_stat, 'p_value': p_value}
    
    def test_age_price_correlation(self):
        """
        H0: No correlation between age and price (ρ = 0)
        H1: Correlation exists between age and price (ρ ≠ 0)
        """
        print("\n" + "="*60)
        print("HYPOTHESIS TEST: Age-Price Correlation")
        print("="*60)
        
        age_clean = self.df['age'].dropna()
        price_clean = self.df.loc[age_clean.index, 'price'].dropna()
        
        corr, p_value = pearsonr(age_clean, price_clean)
        
        print(f"Pearson Correlation: {corr:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"✅ Reject H0: Significant correlation exists (r = {corr:.4f})")
        else:
            print("❌ Fail to reject H0: No significant correlation")
        
        return {'test': 'Pearson', 'correlation': corr, 'p_value': p_value}
    
    def test_mortgage_by_building(self):
        """
        H0: Mortgage usage is independent of building type
        H1: Mortgage usage depends on building type
        """
        print("\n" + "="*60)
        print("HYPOTHESIS TEST: Mortgage Usage by Building Type")
        print("="*60)
        
        contingency = pd.crosstab(self.df['building'], self.df['mortgage_binary'])
        
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Degrees of freedom: {dof}")
        
        if p_value < 0.05:
            print("✅ Reject H0: Building type affects mortgage usage")
        else:
            print("❌ Fail to reject H0: No association between building and mortgage")
        
        return {'test': 'Chi-square', 'statistic': chi2, 'p_value': p_value}
    
    def run_all_tests(self):
        """Run all hypothesis tests"""
        print("\n" + "🚀"*30)
        print("RUNNING ALL HYPOTHESIS TESTS")
        print("🚀"*30)
        
        results = {
            'price_by_building': self.test_price_by_building_type(),
            'satisfaction_by_country': self.test_satisfaction_by_country(),
            'age_price_correlation': self.test_age_price_correlation(),
            'mortgage_by_building': self.test_mortgage_by_building()
        }
        
        return results
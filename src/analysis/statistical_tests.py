"""
Statistical hypothesis tests for real estate insights
"""

import pandas as pd
import numpy as np
from scipy import stats


def test_mrt_price_difference(df):
    """
    H0: No difference in price between properties close vs far from MRT
    H1: Properties close to MRT have significantly higher prices
    """
    close = df[df['mrt_distance'] <= 500]['price_per_unit']
    far   = df[df['mrt_distance'] >  500]['price_per_unit']

    t_stat, p_value = stats.ttest_ind(close, far)

    print(f"\n📊 T-Test: Close vs Far MRT Price Difference")
    print(f"   Close MRT (n={len(close)}): mean={close.mean():.2f}")
    print(f"   Far MRT   (n={len(far)}):   mean={far.mean():.2f}")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value:     {p_value:.6f}")
    print(f"   Result: {'✅ Significant' if p_value < 0.05 else '❌ Not significant'} at α=0.05")

    return {'test': 'ttest_mrt', 't_stat': t_stat, 'p_value': p_value,
            'significant': p_value < 0.05,
            'close_mean': close.mean(), 'far_mean': far.mean()}


def test_age_groups_price(df):
    """
    H0: No difference in price across house age groups
    H1: Price differs significantly across age groups (ANOVA)
    """
    groups = [
        df[df['age_category'] == cat]['price_per_unit'].dropna()
        for cat in ['New', 'Recent', 'Middle-aged', 'Old']
    ]
    groups = [g for g in groups if len(g) > 0]

    f_stat, p_value = stats.f_oneway(*groups)

    print(f"\n📊 ANOVA: Price Across House Age Groups")
    for cat in ['New', 'Recent', 'Middle-aged', 'Old']:
        g = df[df['age_category'] == cat]['price_per_unit']
        if len(g) > 0:
            print(f"   {cat:15s} (n={len(g):3d}): mean={g.mean():.2f}")
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   p-value:     {p_value:.6f}")
    print(f"   Result: {'✅ Significant' if p_value < 0.05 else '❌ Not significant'} at α=0.05")

    return {'test': 'anova_age', 'f_stat': f_stat, 'p_value': p_value,
            'significant': p_value < 0.05}


def test_stores_correlation(df):
    """
    H0: No correlation between convenience stores and price
    H1: Significant correlation exists
    """
    r, p_value = stats.pearsonr(df['convenience_stores'], df['price_per_unit'])

    print(f"\n📊 Pearson Correlation: Convenience Stores vs Price")
    print(f"   r:       {r:.4f}")
    print(f"   p-value: {p_value:.6f}")
    print(f"   Result: {'✅ Significant' if p_value < 0.05 else '❌ Not significant'} at α=0.05")
    if abs(r) >= 0.5:
        print(f"   Strength: Strong correlation")
    elif abs(r) >= 0.3:
        print(f"   Strength: Moderate correlation")
    else:
        print(f"   Strength: Weak correlation")

    return {'test': 'pearson_stores', 'r': r, 'p_value': p_value,
            'significant': p_value < 0.05}


def test_segment_price_difference(df):
    """
    H0: No price difference between location segments
    H1: Significant price difference exists across segments (ANOVA)
    """
    if 'segment' not in df.columns:
        print("⚠️  No segment column found — run LocationSegmentation first")
        return None

    groups = [
        df[df['segment'] == s]['price_per_unit']
        for s in df['segment'].unique()
    ]

    f_stat, p_value = stats.f_oneway(*groups)

    print(f"\n📊 ANOVA: Price Across Location Segments")
    for s in sorted(df['segment'].unique()):
        g = df[df['segment'] == s]['price_per_unit']
        print(f"   Segment {s} (n={len(g):3d}): mean={g.mean():.2f}")
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   p-value:     {p_value:.6f}")
    print(f"   Result: {'✅ Significant' if p_value < 0.05 else '❌ Not significant'} at α=0.05")

    return {'test': 'anova_segments', 'f_stat': f_stat, 'p_value': p_value,
            'significant': p_value < 0.05}


def run_all(df):
    """Run all statistical tests"""
    print("=" * 55)
    print("STATISTICAL HYPOTHESIS TESTS")
    print("=" * 55)

    results = {
        'mrt':     test_mrt_price_difference(df),
        'age':     test_age_groups_price(df),
        'stores':  test_stores_correlation(df),
        'segments': test_segment_price_difference(df)
    }

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    for name, r in results.items():
        if r:
            sig = '✅' if r['significant'] else '❌'
            print(f"   {sig} {name}: p={r['p_value']:.4f}")

    return results
"""
Key business insights from the real estate dataset
"""

import pandas as pd
import numpy as np
from scipy import stats
import os


def mrt_impact(df):
    """Analyze MRT distance impact on price"""
    result = (
        df.groupby('mrt_category')['price_per_unit']
        .agg(['mean', 'median', 'std', 'count'])
        .round(2)
        .reset_index()
    )
    result.columns = ['mrt_category', 'avg_price', 'median_price', 'std_price', 'count']

    print("\n📊 MRT Distance Impact on Price:")
    print(result.to_string(index=False))

    return result


def age_price_correlation(df):
    """Correlation between house age and price"""
    r, p = stats.pearsonr(df['house_age'], df['price_per_unit'])

    buckets = (
        df.groupby('age_category')['price_per_unit']
        .agg(['mean', 'median', 'count'])
        .round(2)
        .reset_index()
    )

    print(f"\n📊 House Age vs Price:")
    print(f"   Pearson r: {r:.4f}, p-value: {p:.4f}")
    if p < 0.05:
        direction = 'negative' if r < 0 else 'positive'
        print(f"   ✅ Statistically significant {direction} correlation")
    else:
        print(f"   ⚠️  No statistically significant correlation")
    print(buckets.to_string(index=False))

    return {'r': r, 'p': p, 'buckets': buckets}


def convenience_store_impact(df):
    """Impact of nearby convenience stores on price"""
    r, p = stats.pearsonr(df['convenience_stores'], df['price_per_unit'])

    result = (
        df.groupby('convenience_stores')['price_per_unit']
        .agg(['mean', 'count'])
        .round(2)
        .reset_index()
    )

    print(f"\n📊 Convenience Stores vs Price:")
    print(f"   Pearson r: {r:.4f}, p-value: {p:.4f}")
    if p < 0.05:
        print(f"   ✅ Statistically significant correlation")
    print(result.to_string(index=False))

    return {'r': r, 'p': p, 'by_store_count': result}


def price_summary(df):
    """Overall price summary statistics"""
    summary = df['price_per_unit'].describe().round(2)

    segment_counts = df['price_segment'].value_counts()

    print(f"\n📊 Price Summary:")
    print(summary)
    print(f"\n   Segment distribution:")
    print(segment_counts)

    return {
        'summary': summary.to_dict(),
        'segments': segment_counts.to_dict()
    }


def run_all(df, save_dir='reports'):
    """Run all insights and save CSVs"""
    os.makedirs(save_dir, exist_ok=True)

    mrt     = mrt_impact(df)
    age     = age_price_correlation(df)
    stores  = convenience_store_impact(df)
    summary = price_summary(df)

    # Save reports
    mrt.to_csv(f'{save_dir}/mrt_impact.csv', index=False)
    age['buckets'].to_csv(f'{save_dir}/age_price.csv', index=False)
    stores['by_store_count'].to_csv(f'{save_dir}/store_impact.csv', index=False)

    print(f"\n✅ Reports saved to {save_dir}/")

    return {
        'mrt': mrt,
        'age_correlation': age,
        'store_correlation': stores,
        'price_summary': summary
    }
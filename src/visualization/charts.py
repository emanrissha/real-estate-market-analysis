"""
Chart generation for Real Estate analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

SAVE_DIR = 'reports/figures'
os.makedirs(SAVE_DIR, exist_ok=True)


def _save(fig, name):
    path = f"{SAVE_DIR}/{name}"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {path}")
    return path


def price_distribution(df):
    """Histogram of price per unit area"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df['price_per_unit'], bins=30, color='steelblue', edgecolor='white')
    axes[0].set_title('Price Per Unit Area — Distribution')
    axes[0].set_xlabel('Price per Unit')
    axes[0].set_ylabel('Count')
    axes[0].axvline(df['price_per_unit'].mean(), color='red',
                    linestyle='--', label=f"Mean: {df['price_per_unit'].mean():.1f}")
    axes[0].legend()

    axes[1].hist(df['price_per_unit'], bins=30, color='steelblue',
                 edgecolor='white', cumulative=True, density=True)
    axes[1].set_title('Price Per Unit Area — Cumulative')
    axes[1].set_xlabel('Price per Unit')
    axes[1].set_ylabel('Cumulative Proportion')

    fig.suptitle('Price Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save(fig, 'price_distribution.png')


def mrt_vs_price(df):
    """Scatter plot of MRT distance vs price"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(df['mrt_distance'], df['price_per_unit'],
                    alpha=0.4, color='steelblue', s=20)
    axes[0].set_title('MRT Distance vs Price')
    axes[0].set_xlabel('MRT Distance (meters)')
    axes[0].set_ylabel('Price per Unit')

    axes[1].scatter(df['log_mrt_distance'], df['price_per_unit'],
                    alpha=0.4, color='darkorange', s=20)
    z = np.polyfit(df['log_mrt_distance'], df['price_per_unit'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['log_mrt_distance'].min(),
                          df['log_mrt_distance'].max(), 100)
    axes[1].plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
    axes[1].set_title('Log MRT Distance vs Price (with trend)')
    axes[1].set_xlabel('Log MRT Distance')
    axes[1].set_ylabel('Price per Unit')
    axes[1].legend()

    fig.suptitle('MRT Distance Impact on Price', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save(fig, 'mrt_vs_price.png')


def age_vs_price(df):
    """Box plot of price by age category"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    order = ['New', 'Recent', 'Middle-aged', 'Old']
    data  = [df[df['age_category'] == cat]['price_per_unit'].dropna()
             for cat in order]

    axes[0].boxplot(data, labels=order, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6))
    axes[0].set_title('Price by House Age Category')
    axes[0].set_xlabel('Age Category')
    axes[0].set_ylabel('Price per Unit')

    axes[1].scatter(df['house_age'], df['price_per_unit'],
                    alpha=0.4, color='steelblue', s=20)
    z = np.polyfit(df['house_age'], df['price_per_unit'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['house_age'].min(), df['house_age'].max(), 100)
    axes[1].plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
    axes[1].set_title('House Age vs Price (with trend)')
    axes[1].set_xlabel('House Age (years)')
    axes[1].set_ylabel('Price per Unit')
    axes[1].legend()

    fig.suptitle('House Age Impact on Price', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save(fig, 'age_vs_price.png')


def convenience_stores_vs_price(df):
    """Bar chart of avg price by number of convenience stores"""
    fig, ax = plt.subplots(figsize=(10, 5))

    avg = df.groupby('convenience_stores')['price_per_unit'].mean()
    bars = ax.bar(avg.index, avg.values, color='steelblue', edgecolor='white')
    ax.set_title('Convenience Stores vs Avg Price per Unit', fontsize=14)
    ax.set_xlabel('Number of Nearby Convenience Stores')
    ax.set_ylabel('Avg Price per Unit')
    ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)

    plt.tight_layout()
    return _save(fig, 'stores_vs_price.png')


def location_map(df):
    """Scatter map of properties colored by price"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sc = axes[0].scatter(
        df['longitude'], df['latitude'],
        c=df['price_per_unit'], cmap='RdYlGn',
        alpha=0.6, s=30
    )
    plt.colorbar(sc, ax=axes[0], label='Price per Unit')
    axes[0].set_title('Property Locations — Colored by Price')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')

    sc2 = axes[1].scatter(
        df['longitude'], df['latitude'],
        c=df['mrt_distance'], cmap='RdYlGn_r',
        alpha=0.6, s=30
    )
    plt.colorbar(sc2, ax=axes[1], label='MRT Distance (m)')
    axes[1].set_title('Property Locations — Colored by MRT Distance')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')

    fig.suptitle('Property Location Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save(fig, 'location_map.png')


def forecast_chart(monthly_data, forecast_dict):
    """Time series + forecast chart"""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Handle both list of dicts and DataFrame
    if isinstance(monthly_data, list):
        monthly_data = pd.DataFrame(monthly_data)

    dates  = pd.to_datetime(monthly_data['date'])
    prices = monthly_data['avg_price']

    ax.plot(dates, prices, 'o-', color='steelblue',
            linewidth=2, markersize=5, label='Historical')

    f_dates  = pd.to_datetime(list(forecast_dict.keys()))
    f_prices = list(forecast_dict.values())
    ax.plot(f_dates, f_prices, 's--', color='darkorange',
            linewidth=2, markersize=5, label='Forecast')

    ax.axvline(dates.max(), color='gray', linestyle=':', alpha=0.7)
    ax.set_title('Monthly Avg Price per Unit — Historical + Forecast',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Avg Price per Unit')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return _save(fig, 'forecast.png')


def run_all(df, forecast_data=None):
    """Generate all charts"""
    print("\n🎨 Generating charts...")
    price_distribution(df)
    mrt_vs_price(df)
    age_vs_price(df)
    convenience_stores_vs_price(df)
    location_map(df)
    if forecast_data:
        forecast_chart(
            forecast_data['monthly_data'],
            forecast_data['forecast']
        )
    print("✅ All charts generated")
#!/usr/bin/env python3
"""Debug the price coverage regression from 2,783 to 376 markets."""

import pandas as pd
import numpy as np

print('=== INVESTIGATING PRICE DATA COVERAGE REGRESSION ===')

# Check current price data
try:
    prices = pd.read_parquet('data/processed/prices.parquet')
    markets_with_prices = prices['market_id'].nunique()
    print(f'Current price coverage: {markets_with_prices} unique markets')
    print(f'Total price data points: {len(prices)}')
    
    # Check date range
    prices['date'] = pd.to_datetime(prices['date'])
    print(f'Price date range: {prices["date"].min()} to {prices["date"].max()}')
    
    # Platform breakdown for markets with prices
    price_platforms = prices.merge(
        pd.read_parquet('data/processed/markets.parquet')[['market_id', 'platform']], 
        on='market_id', how='left'
    )['platform'].value_counts()
    print(f'Platform breakdown of markets with prices:')
    for platform, count in price_platforms.items():
        print(f'  {platform}: {count} markets')
    
except Exception as e:
    print(f'Error loading prices: {e}')

# Check original markets data
markets = pd.read_parquet('data/processed/markets.parquet')
print(f'\nTotal markets ingested: {len(markets)}')

# Check volume distribution to understand the threshold impact
print(f'\nVolume analysis:')
vol_500k = len(markets[markets['volume'] >= 500000])
vol_100k = len(markets[markets['volume'] >= 100000])
vol_50k = len(markets[markets['volume'] >= 50000])
vol_10k = len(markets[markets['volume'] >= 10000]) 
vol_1k = len(markets[markets['volume'] >= 1000])
vol_any = len(markets[markets['volume'] > 0])

print(f'Markets with volume >= 500K: {vol_500k}')
print(f'Markets with volume >= 100K: {vol_100k}')
print(f'Markets with volume >= 50K: {vol_50k}')
print(f'Markets with volume >= 10K: {vol_10k}')
print(f'Markets with volume >= 1K: {vol_1k}')
print(f'Markets with volume > 0: {vol_any}')

# Volume percentiles
vol_data = markets['volume']
print(f'\nVolume percentiles:')
print(f'50th percentile: ${vol_data.quantile(0.5):,.0f}')
print(f'75th percentile: ${vol_data.quantile(0.75):,.0f}')
print(f'90th percentile: ${vol_data.quantile(0.9):,.0f}')
print(f'95th percentile: ${vol_data.quantile(0.95):,.0f}')
print(f'99th percentile: ${vol_data.quantile(0.99):,.0f}')

print(f'\n=== PLATFORM ANALYSIS ===')
platform_stats = markets.groupby('platform').agg({
    'volume': ['count', 'mean', 'median', lambda x: (x >= 10000).sum(), lambda x: (x >= 500000).sum()]
}).round(0)
platform_stats.columns = ['total_markets', 'avg_volume', 'median_volume', 'above_10k', 'above_500k']
print(platform_stats)

print(f'\n=== ROOT CAUSE HYPOTHESIS ===')
print('1. The 500K volume threshold is WAY too aggressive')
print('2. Most markets are small volume, so we filtered out the majority')
print('3. We need to revert to 10K threshold AND expand price data sources')

# Check if price data exists for lower volume markets
if 'prices' in locals():
    print(f'\n=== PRICE DATA vs VOLUME ANALYSIS ===')
    # Get volume for markets with prices
    markets_with_vol = markets[['market_id', 'volume']]
    price_vol_analysis = markets_with_vol[markets_with_vol['market_id'].isin(prices['market_id'])]
    
    print(f'Markets with prices and volume >= 500K: {len(price_vol_analysis[price_vol_analysis["volume"] >= 500000])}')
    print(f'Markets with prices and volume >= 10K: {len(price_vol_analysis[price_vol_analysis["volume"] >= 10000])}')
    print(f'Markets with prices and volume >= 1K: {len(price_vol_analysis[price_vol_analysis["volume"] >= 1000])}')
    
    print(f'\nPrice data volume distribution:')
    print(f'Min volume with prices: ${price_vol_analysis["volume"].min():,.0f}')
    print(f'Max volume with prices: ${price_vol_analysis["volume"].max():,.0f}')
    print(f'Median volume with prices: ${price_vol_analysis["volume"].median():,.0f}')
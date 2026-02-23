#!/usr/bin/env python3

import pandas as pd
import numpy as np

print("=== INSPECTING MARKETS_FILTERED.PARQUET ===")
markets_df = pd.read_parquet('data/processed/markets_filtered.parquet')
print(f"Shape: {markets_df.shape}")
print(f"Columns: {markets_df.columns.tolist()}")
print("\nFirst few rows:")
print(markets_df.head())
print(f"\nIndex: {markets_df.index.name}")
print(f"Index type: {type(markets_df.index)}")

print("\n=== INSPECTING PRICES.PARQUET ===")
prices_df = pd.read_parquet('data/processed/prices.parquet')
print(f"Shape: {prices_df.shape}")
print(f"Columns: {prices_df.columns.tolist()}")
print("\nFirst few rows:")
print(prices_df.head())
print(f"\nIndex: {prices_df.index.name}")
print(f"Index type: {type(prices_df.index)}")

print("\n=== FINDING JOIN KEY ===")
# Check if any column in markets_df matches column names in prices_df
common_cols = set(markets_df.columns) & set(prices_df.columns)
print(f"Common column names: {common_cols}")

# Check if index matches
if hasattr(markets_df.index, 'name') and hasattr(prices_df.index, 'name'):
    print(f"Markets index name: {markets_df.index.name}")
    print(f"Prices index name: {prices_df.index.name}")

# Check if any markets_df columns match prices_df columns
print(f"\nMarkets columns that could be market IDs:")
for col in markets_df.columns:
    if any(keyword in col.lower() for keyword in ['id', 'market', 'uuid', 'key']):
        print(f"  {col}: {markets_df[col].dtype}, sample values: {markets_df[col].head(3).tolist()}")

print(f"\nPrices columns that could be market IDs:")
for col in prices_df.columns:
    if any(keyword in col.lower() for keyword in ['id', 'market', 'uuid', 'key']):
        print(f"  {col}: {prices_df[col].dtype}, sample values: {prices_df[col].head(3).tolist()}")

# Check if prices columns match any sample market IDs
if len(markets_df) > 0 and len(prices_df.columns) > 10:
    print(f"\nSample market IDs from markets_filtered:")
    if 'id' in markets_df.columns:
        sample_ids = markets_df['id'].head(5).tolist()
    elif 'market_id' in markets_df.columns:
        sample_ids = markets_df['market_id'].head(5).tolist()
    else:
        sample_ids = markets_df.index[:5].tolist()
    
    print(f"Sample market IDs: {sample_ids}")
    
    matching_cols = []
    for sample_id in sample_ids[:3]:  # Check first 3 sample IDs
        if str(sample_id) in prices_df.columns:
            matching_cols.append(str(sample_id))
    
    print(f"Market IDs found as price columns: {len(matching_cols)} out of 3 checked")
    if matching_cols:
        print(f"Examples: {matching_cols[:5]}")
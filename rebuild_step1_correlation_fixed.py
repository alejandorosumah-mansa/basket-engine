#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rebuild_correlation_matrix():
    """Rebuild correlation matrix from scratch with filtered markets"""
    
    print("=== STEP 1: REBUILD CORRELATION MATRIX (FIXED) ===")
    
    # Load filtered markets
    print("Loading markets_filtered.parquet...")
    markets_df = pd.read_parquet('data/processed/markets_filtered.parquet')
    print(f"Filtered markets shape: {markets_df.shape}")
    print(f"Unique market_ids: {markets_df['market_id'].nunique()}")
    
    # Load prices
    print("Loading prices.parquet...")
    prices_df = pd.read_parquet('data/processed/prices.parquet')
    print(f"Prices shape: {prices_df.shape}")
    print(f"Unique market_ids in prices: {prices_df['market_id'].nunique()}")
    
    # Filter prices to only markets that exist in markets_filtered
    print("Filtering prices to filtered markets...")
    filtered_market_ids = set(markets_df['market_id'])
    prices_filtered = prices_df[prices_df['market_id'].isin(filtered_market_ids)].copy()
    print(f"Filtered prices shape: {prices_filtered.shape}")
    print(f"Unique market_ids after filter: {prices_filtered['market_id'].nunique()}")
    
    # Convert to date index format for correlation calculation
    print("Converting to pivot table format...")
    prices_filtered['date'] = pd.to_datetime(prices_filtered['date'])
    
    # Pivot to have market_id as columns, date as index, close_price as values
    price_matrix = prices_filtered.pivot(index='date', columns='market_id', values='close_price')
    print(f"Price matrix shape: {price_matrix.shape}")
    
    # Sort by date to ensure proper ordering
    price_matrix = price_matrix.sort_index()
    
    # Compute daily returns using .diff() as specified
    print("Computing daily returns using .diff()...")
    returns = price_matrix.diff()
    
    # Drop the first row (all NaN from diff)
    returns = returns.dropna(how='all')
    print(f"Returns shape after dropping first row: {returns.shape}")
    
    # Count non-NaN observations per market
    print("Counting observations per market...")
    obs_count = returns.count()
    print(f"Observation counts - min: {obs_count.min()}, max: {obs_count.max()}, mean: {obs_count.mean():.1f}")
    
    # Drop markets with <30 observations
    print("Dropping markets with <30 observations...")
    valid_markets = obs_count[obs_count >= 30]
    returns_filtered = returns[valid_markets.index]
    print(f"Markets after filtering: {len(valid_markets)}")
    print(f"Final returns shape: {returns_filtered.shape}")
    
    # Compute pairwise correlation matrix with minimum observations requirement
    print("Computing correlation matrix with minimum periods=20...")
    correlation_matrix = returns_filtered.corr(min_periods=20)
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    
    # Check for any issues
    nan_count = correlation_matrix.isna().sum().sum()
    print(f"Correlation matrix NaN count: {nan_count}")
    
    # For diagonal, only check non-NaN values
    diagonal = correlation_matrix.values.diagonal()
    non_nan_diagonal = diagonal[~np.isnan(diagonal)]
    print(f"Non-NaN diagonal values: {len(non_nan_diagonal)} out of {len(diagonal)}")
    if len(non_nan_diagonal) > 0:
        print(f"Diagonal range: min={non_nan_diagonal.min():.3f}, max={non_nan_diagonal.max():.3f}")
    
    # Get upper triangle without diagonal for stats
    upper_triangle = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
    valid_corrs = upper_triangle[~np.isnan(upper_triangle)]
    print(f"Valid correlation pairs: {len(valid_corrs)} out of {len(upper_triangle)}")
    
    if len(valid_corrs) > 0:
        print(f"Correlation stats - mean: {valid_corrs.mean():.4f}, std: {valid_corrs.std():.4f}")
        print(f"Correlation range: min={valid_corrs.min():.4f}, max={valid_corrs.max():.4f}")
    
    # Save correlation matrix
    output_path = Path('data/processed/correlation_matrix.parquet')
    print(f"Saving correlation matrix to {output_path}...")
    correlation_matrix.to_parquet(output_path)
    
    print(f"✅ Correlation matrix rebuilt successfully!")
    print(f"Summary:")
    print(f"  - Total markets: {len(correlation_matrix)}")
    print(f"  - Valid correlation pairs: {len(valid_corrs)}")
    print(f"  - NaN pairs: {len(upper_triangle) - len(valid_corrs)}")
    
    return correlation_matrix

if __name__ == "__main__":
    correlation_matrix = rebuild_correlation_matrix()
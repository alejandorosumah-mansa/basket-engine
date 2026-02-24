#!/usr/bin/env python3
"""
Rebuild correlation matrix using only filtered markets (no sports/entertainment).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.analysis.correlation_clustering import load_prices, compute_price_changes, build_correlation_matrix

def main():
    print("=== Rebuilding Correlation Matrix with Filtered Markets ===")
    
    # Load filtered markets
    markets_filtered = pd.read_parquet("data/processed/markets_filtered.parquet")
    valid_market_ids = set(markets_filtered["market_id"].tolist())
    print(f"Loaded {len(valid_market_ids):,} filtered markets (no sports/entertainment)")
    
    # Load prices
    prices = load_prices("data/processed/prices.parquet")
    print(f"Loaded prices for {prices['market_id'].nunique():,} markets")
    
    # Filter prices to only include valid markets
    prices_filtered = prices[prices["market_id"].isin(valid_market_ids)]
    print(f"Filtered prices to {prices_filtered['market_id'].nunique():,} markets")
    
    # Compute price changes
    price_changes = compute_price_changes(prices_filtered)
    print(f"Computed price changes for {price_changes['market_id'].nunique():,} markets")
    
    # Build correlation matrix with stricter overlap requirements
    corr_matrix, market_stats = build_correlation_matrix(
        price_changes,
        min_overlapping_days=30,  # Require 30+ overlapping days for correlation
        min_days_per_market=30
    )
    
    print(f"Built correlation matrix: {corr_matrix.shape[0]}x{corr_matrix.shape[1]}")
    
    # Save correlation matrix (overwrite existing)
    output_path = "data/processed/correlation_matrix.parquet"
    corr_matrix.to_parquet(output_path)
    print(f"Saved correlation matrix to {output_path}")
    
    # Print summary stats
    print(f"\nCorrelation Matrix Summary:")
    print(f"  Markets: {len(corr_matrix)}")
    print(f"  Non-zero correlations: {(corr_matrix != 0).sum().sum() - len(corr_matrix):,}")
    print(f"  Mean |correlation|: {np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]).mean():.3f}")
    
    print("Done!")

if __name__ == "__main__":
    main()
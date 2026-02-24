#!/usr/bin/env python3
"""
Rebuild correlation matrix with proper data quality filters - efficient version.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import gc

def apply_data_quality_filters_batch(
    prices: pd.DataFrame,
    markets: pd.DataFrame,
    min_volume_threshold: float = None,
    min_price_variance: float = 0.005,
    min_unique_prices: int = 10,
    min_active_days: int = 30
) -> tuple[pd.DataFrame, dict]:
    """
    Apply data quality filters efficiently.
    """
    print("\n=== Applying Data Quality Filters ===")
    
    # Set volume threshold at 25th percentile if not specified
    if min_volume_threshold is None:
        min_volume_threshold = markets['volume'].quantile(0.25)
    
    print(f"Volume threshold (25th percentile): {min_volume_threshold:,.0f}")
    
    # Filter 1: Volume filter
    print("Filter 1: Volume filter...")
    volume_pass = set(markets[markets['volume'] >= min_volume_threshold]['market_id'])
    initial_count = len(markets)
    print(f"  Initial: {initial_count:,} markets")
    print(f"  Volume pass: {len(volume_pass):,} markets")
    print(f"  Volume filtered: {initial_count - len(volume_pass):,} markets")
    
    # Filter prices to volume-passing markets
    filtered_prices = prices[prices['market_id'].isin(volume_pass)].copy()
    print(f"  Price records after volume filter: {len(filtered_prices):,}")
    
    # Prepare for batch processing
    print("\\nApplying variance, unique prices, and active days filters...")
    
    # Convert dates and sort
    filtered_prices['date'] = pd.to_datetime(filtered_prices['date'])
    filtered_prices = filtered_prices.sort_values(['market_id', 'date'])
    
    # Compute price changes once
    filtered_prices['price_change'] = filtered_prices.groupby('market_id')['close_price'].diff()
    
    # Group by market for efficient processing
    market_groups = filtered_prices.groupby('market_id')
    
    results = []
    processed = 0
    
    for market_id, group in market_groups:
        processed += 1
        if processed % 1000 == 0:
            print(f"  Processed {processed:,} markets...")
        
        prices_series = group['close_price']
        price_changes = group['price_change'].dropna()
        
        # Filter 2: Variance
        if len(price_changes) == 0:
            continue
        variance = price_changes.std()
        if variance < min_price_variance:
            continue
        
        # Filter 3: Unique prices
        unique_prices = len(prices_series.unique())
        if unique_prices < min_unique_prices:
            continue
        
        # Filter 4: Active days
        active_days = (price_changes != 0).sum()
        if active_days < min_active_days:
            continue
        
        # Market passed all filters
        results.append({
            'market_id': market_id,
            'variance': variance,
            'unique_prices': unique_prices,
            'active_days': active_days,
            'total_days': len(prices_series)
        })
    
    # Create final filtered dataset
    passed_markets = set(r['market_id'] for r in results)
    final_prices = filtered_prices[filtered_prices['market_id'].isin(passed_markets)]
    
    # Statistics
    filter_stats = {
        'initial_markets': initial_count,
        'after_volume': len(volume_pass),
        'final_markets': len(passed_markets),
        'total_filtered': initial_count - len(passed_markets),
        'filter_rate': (initial_count - len(passed_markets)) / initial_count,
        'min_volume_threshold': min_volume_threshold,
        'min_price_variance': min_price_variance,
        'min_unique_prices': min_unique_prices,
        'min_active_days': min_active_days
    }
    
    # Compute filter-level stats
    if results:
        variances = [r['variance'] for r in results]
        unique_counts = [r['unique_prices'] for r in results]
        active_counts = [r['active_days'] for r in results]
        
        filter_stats['final_stats'] = {
            'variance': {'min': min(variances), 'median': np.median(variances), 'max': max(variances)},
            'unique_prices': {'min': min(unique_counts), 'median': np.median(unique_counts), 'max': max(unique_counts)},
            'active_days': {'min': min(active_counts), 'median': np.median(active_counts), 'max': max(active_counts)}
        }
    
    print(f"\\n=== Filter Results ===")
    print(f"Initial markets: {filter_stats['initial_markets']:,}")
    print(f"After volume filter: {filter_stats['after_volume']:,}")
    print(f"Final markets: {filter_stats['final_markets']:,}")
    print(f"Total filtered: {filter_stats['total_filtered']:,} ({filter_stats['filter_rate']:.1%})")
    
    return final_prices, filter_stats

def build_correlation_matrix_efficient(
    prices: pd.DataFrame,
    min_overlapping_days: int = 30,
    min_days_per_market: int = 30
) -> tuple[pd.DataFrame, dict]:
    """
    Build correlation matrix efficiently.
    """
    print(f"\\nBuilding correlation matrix...")
    
    # Ensure price changes are computed
    if 'price_change' not in prices.columns:
        print("Computing price changes...")
        prices = prices.sort_values(['market_id', 'date'])
        prices['price_change'] = prices.groupby('market_id')['close_price'].diff()
    
    # Filter markets with sufficient history
    market_counts = prices.groupby('market_id').size()
    eligible_markets = market_counts[market_counts >= min_days_per_market].index
    print(f"Markets with ≥{min_days_per_market} days: {len(eligible_markets):,}")
    
    # Create pivot table
    print("Creating price change matrix...")
    eligible_data = prices[prices['market_id'].isin(eligible_markets)]
    eligible_data = eligible_data.dropna(subset=['price_change'])
    
    pivot = eligible_data.pivot_table(
        index='date',
        columns='market_id', 
        values='price_change',
        aggfunc='first'  # In case of duplicates
    )
    
    print(f"Price change matrix: {pivot.shape[0]} days × {pivot.shape[1]} markets")
    
    # Compute correlation matrix
    print("Computing correlations...")
    corr_matrix = pivot.corr(min_periods=min_overlapping_days)
    
    # Basic statistics
    n_markets = len(corr_matrix)
    valid_corrs = (~corr_matrix.isna()).sum().sum() - n_markets  # Exclude diagonal
    total_possible = n_markets * (n_markets - 1)
    
    # Get correlation values
    upper_tri_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_values = corr_matrix.values[upper_tri_mask]
    valid_corr_values = corr_values[~np.isnan(corr_values)]
    
    diagnostics = {
        'n_markets': n_markets,
        'valid_correlations': int(valid_corrs),
        'total_possible': int(total_possible),
        'coverage': float(valid_corrs / total_possible) if total_possible > 0 else 0.0,
        'correlation_stats': {
            'count': len(valid_corr_values),
            'mean': float(np.mean(valid_corr_values)) if len(valid_corr_values) > 0 else 0.0,
            'std': float(np.std(valid_corr_values)) if len(valid_corr_values) > 0 else 0.0,
            'min': float(np.min(valid_corr_values)) if len(valid_corr_values) > 0 else 0.0,
            'max': float(np.max(valid_corr_values)) if len(valid_corr_values) > 0 else 0.0,
            'mean_abs': float(np.mean(np.abs(valid_corr_values))) if len(valid_corr_values) > 0 else 0.0,
        }
    }
    
    print(f"Correlation diagnostics:")
    print(f"  Markets: {diagnostics['n_markets']:,}")
    print(f"  Valid correlations: {diagnostics['valid_correlations']:,} / {diagnostics['total_possible']:,} ({diagnostics['coverage']:.1%})")
    print(f"  Mean |correlation|: {diagnostics['correlation_stats']['mean_abs']:.4f}")
    
    return corr_matrix, diagnostics

def get_top_correlations(corr_matrix: pd.DataFrame, markets: pd.DataFrame, top_n: int = 20) -> list:
    """Get top correlated pairs with titles."""
    print(f"\\nFinding top {top_n} correlated pairs...")
    
    markets_indexed = markets.set_index('market_id')
    correlations = []
    
    # Get upper triangle correlations
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            market1 = corr_matrix.index[i]
            market2 = corr_matrix.index[j]
            corr = corr_matrix.iloc[i, j]
            
            if not np.isnan(corr):
                correlations.append((abs(corr), corr, market1, market2))
    
    # Sort and get top N
    correlations.sort(reverse=True)
    
    top_pairs = []
    for abs_corr, corr, market1, market2 in correlations[:top_n]:
        title1 = markets_indexed.loc[market1, 'title'] if market1 in markets_indexed.index else market1[:50]
        title2 = markets_indexed.loc[market2, 'title'] if market2 in markets_indexed.index else market2[:50]
        
        top_pairs.append({
            'correlation': float(corr),
            'abs_correlation': float(abs_corr),
            'market1': market1,
            'market2': market2,
            'title1': title1[:80],  # Truncate for readability
            'title2': title2[:80]
        })
    
    return top_pairs

def main():
    print("=== Rebuilding Correlation Matrix with Data Quality Filters ===")
    
    # Load data
    print("Loading data...")
    markets = pd.read_parquet("data/processed/markets.parquet")
    prices = pd.read_parquet("data/processed/prices.parquet")
    
    print(f"Loaded {len(markets):,} markets and {len(prices):,} price records")
    
    # Apply filters
    filtered_prices, filter_stats = apply_data_quality_filters_batch(
        prices,
        markets,
        min_volume_threshold=None,  # Use 25th percentile
        min_price_variance=0.005,
        min_unique_prices=10,
        min_active_days=30
    )
    
    # Clean up
    del prices
    gc.collect()
    
    # Build correlation matrix
    corr_matrix, corr_diagnostics = build_correlation_matrix_efficient(
        filtered_prices,
        min_overlapping_days=30,
        min_days_per_market=30
    )
    
    # Get top correlations
    top_pairs = get_top_correlations(corr_matrix, markets, top_n=20)
    
    # Save results
    print(f"\\nSaving results...")
    
    # Save correlation matrix
    output_path = "data/processed/correlation_matrix_filtered.parquet"
    corr_matrix.to_parquet(output_path)
    print(f"Saved filtered correlation matrix to {output_path}")
    
    # Save diagnostics
    all_diagnostics = {
        'timestamp': datetime.now().isoformat(),
        'filter_stats': filter_stats,
        'correlation_diagnostics': corr_diagnostics,
        'top_correlated_pairs': top_pairs
    }
    
    with open("data/processed/correlation_diagnostics.json", "w") as f:
        json.dump(all_diagnostics, f, indent=2)
    
    print(f"Saved diagnostics to data/processed/correlation_diagnostics.json")
    
    # Display results
    print(f"\\n=== TOP 20 CORRELATED PAIRS (Sanity Check) ===")
    for i, pair in enumerate(top_pairs, 1):
        sign = "+" if pair['correlation'] > 0 else "-"
        print(f"{i:2d}. {sign}{pair['abs_correlation']:.4f}: ")
        print(f"     {pair['title1']}")
        print(f"     {pair['title2']}")
    
    # Summary
    print(f"\\n=== FINAL SUMMARY ===")
    print(f"Original markets: {filter_stats['initial_markets']:,}")
    print(f"Markets after filters: {filter_stats['final_markets']:,}")
    print(f"Filter rate: {filter_stats['filter_rate']:.1%}")
    print(f"Correlation matrix: {len(corr_matrix)}×{len(corr_matrix)}")
    print(f"Valid correlations: {corr_diagnostics['valid_correlations']:,}")
    print(f"Coverage: {corr_diagnostics['coverage']:.1%}")
    print(f"Mean |correlation|: {corr_diagnostics['correlation_stats']['mean_abs']:.4f}")
    
    return corr_matrix, all_diagnostics

if __name__ == "__main__":
    main()
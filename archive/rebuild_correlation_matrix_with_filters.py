#!/usr/bin/env python3
"""
Rebuild correlation matrix with proper data quality filters.

Fixes the issue where dead/illiquid markets create fake correlations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def apply_data_quality_filters(
    prices: pd.DataFrame,
    markets: pd.DataFrame,
    min_volume_threshold: float = None,
    min_price_variance: float = 0.005,
    min_unique_prices: int = 10,
    min_active_days: int = 30
) -> tuple[pd.DataFrame, dict]:
    """
    Apply data quality filters to remove dead/illiquid markets.
    
    Returns:
        filtered_prices: DataFrame with only good quality markets
        filter_stats: Dict with statistics about what was filtered
    """
    print("\n=== Applying Data Quality Filters ===")
    
    # Start with all markets
    all_market_ids = set(prices['market_id'].unique())
    initial_count = len(all_market_ids)
    print(f"Initial markets: {initial_count:,}")
    
    # Convert volume column to match if needed
    if min_volume_threshold is None:
        # Set threshold at 25th percentile to filter out bottom 25% by volume
        min_volume_threshold = markets['volume'].quantile(0.25)
    
    filter_stats = {
        'initial_markets': initial_count,
        'min_volume_threshold': min_volume_threshold,
        'min_price_variance': min_price_variance,
        'min_unique_prices': min_unique_prices,
        'min_active_days': min_active_days,
        'filters': {}
    }
    
    # Filter 1: Minimum volume filter
    print(f"\nFilter 1: Minimum volume >= {min_volume_threshold:,.0f}")
    volume_pass = set(markets[markets['volume'] >= min_volume_threshold]['market_id'])
    volume_fail = all_market_ids - volume_pass
    print(f"  Passed: {len(volume_pass):,} markets")
    print(f"  Failed: {len(volume_fail):,} markets")
    filter_stats['filters']['volume'] = {'passed': len(volume_pass), 'failed': len(volume_fail)}
    
    # Continue with markets that passed volume filter
    surviving_markets = volume_pass.copy()
    
    # Filter 2: Minimum variance filter
    print(f"\nFilter 2: Price variance >= {min_price_variance}")
    variance_stats = []
    variance_pass = set()
    
    for market_id in surviving_markets:
        market_prices = prices[prices['market_id'] == market_id]['close_price']
        if len(market_prices) < 2:
            continue
        
        price_changes = market_prices.diff().dropna()
        if len(price_changes) == 0:
            continue
            
        variance = price_changes.std()
        variance_stats.append(variance)
        
        if variance >= min_price_variance:
            variance_pass.add(market_id)
    
    variance_fail = surviving_markets - variance_pass
    print(f"  Passed: {len(variance_pass):,} markets")
    print(f"  Failed: {len(variance_fail):,} markets")
    if variance_stats:
        print(f"  Variance distribution: min={min(variance_stats):.6f}, "
              f"median={np.median(variance_stats):.6f}, "
              f"max={max(variance_stats):.6f}")
    filter_stats['filters']['variance'] = {'passed': len(variance_pass), 'failed': len(variance_fail)}
    
    # Update surviving markets
    surviving_markets = variance_pass.copy()
    
    # Filter 3: Minimum unique price levels
    print(f"\nFilter 3: Unique prices >= {min_unique_prices}")
    unique_pass = set()
    unique_stats = []
    
    for market_id in surviving_markets:
        market_prices = prices[prices['market_id'] == market_id]['close_price']
        unique_prices = len(market_prices.unique())
        unique_stats.append(unique_prices)
        
        if unique_prices >= min_unique_prices:
            unique_pass.add(market_id)
    
    unique_fail = surviving_markets - unique_pass
    print(f"  Passed: {len(unique_pass):,} markets")
    print(f"  Failed: {len(unique_fail):,} markets")
    if unique_stats:
        print(f"  Unique prices distribution: min={min(unique_stats)}, "
              f"median={int(np.median(unique_stats))}, "
              f"max={max(unique_stats)}")
    filter_stats['filters']['unique_prices'] = {'passed': len(unique_pass), 'failed': len(unique_fail)}
    
    # Update surviving markets
    surviving_markets = unique_pass.copy()
    
    # Filter 4: Active trading days
    print(f"\nFilter 4: Active days (price changed) >= {min_active_days}")
    active_pass = set()
    active_stats = []
    
    for market_id in surviving_markets:
        market_prices = prices[prices['market_id'] == market_id]['close_price']
        if len(market_prices) < 2:
            continue
            
        # Count days where price actually changed
        price_changes = market_prices.diff().dropna()
        active_days = (price_changes != 0).sum()
        active_stats.append(active_days)
        
        if active_days >= min_active_days:
            active_pass.add(market_id)
    
    active_fail = surviving_markets - active_pass
    print(f"  Passed: {len(active_pass):,} markets")
    print(f"  Failed: {len(active_fail):,} markets")
    if active_stats:
        print(f"  Active days distribution: min={min(active_stats)}, "
              f"median={int(np.median(active_stats))}, "
              f"max={max(active_stats)}")
    filter_stats['filters']['active_days'] = {'passed': len(active_pass), 'failed': len(active_fail)}
    
    # Final surviving markets
    final_markets = active_pass
    filter_stats['final_markets'] = len(final_markets)
    filter_stats['total_filtered'] = initial_count - len(final_markets)
    filter_stats['filter_rate'] = (initial_count - len(final_markets)) / initial_count
    
    print(f"\n=== Filter Summary ===")
    print(f"Initial markets: {initial_count:,}")
    print(f"Final markets: {len(final_markets):,}")
    print(f"Filtered out: {filter_stats['total_filtered']:,} ({filter_stats['filter_rate']:.1%})")
    
    # Return filtered prices
    filtered_prices = prices[prices['market_id'].isin(final_markets)]
    
    return filtered_prices, filter_stats

def compute_price_changes_with_stats(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily price changes with additional statistics."""
    print("Computing daily price changes...")
    
    changes = prices.copy()
    changes['date'] = pd.to_datetime(changes['date'])
    changes = changes.sort_values(['market_id', 'date'])
    changes['price_change'] = changes.groupby('market_id')['close_price'].diff()
    changes = changes.dropna(subset=['price_change'])
    
    print(f"Computed changes for {changes['market_id'].nunique()} markets")
    print(f"Total observations: {len(changes):,}")
    
    return changes

def build_correlation_matrix_with_diagnostics(
    price_changes: pd.DataFrame,
    markets: pd.DataFrame,
    min_overlapping_days: int = 30,
    min_days_per_market: int = 30
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Build correlation matrix with enhanced diagnostics.
    """
    print(f"\nBuilding correlation matrix (≥{min_overlapping_days} overlapping days)...")
    
    # Filter markets with sufficient history
    market_counts = price_changes.groupby('market_id').size()
    eligible_markets = set(market_counts[market_counts >= min_days_per_market].index)
    
    print(f"Markets with ≥{min_days_per_market} days: {len(eligible_markets)}")
    
    # Create price change matrix
    eligible_changes = price_changes[price_changes['market_id'].isin(eligible_markets)]
    pivot = eligible_changes.pivot(
        index='date', columns='market_id', values='price_change'
    )
    
    print(f"Price change matrix: {pivot.shape[0]} days × {pivot.shape[1]} markets")
    
    # Compute correlation matrix
    corr_matrix = pivot.corr(min_periods=min_overlapping_days)
    
    # Correlation diagnostics
    valid_corrs = (~corr_matrix.isna()).sum().sum() - len(corr_matrix)  # Exclude diagonal
    total_possible = len(corr_matrix) * (len(corr_matrix) - 1)
    
    print(f"Valid correlations: {valid_corrs:,} / {total_possible:,} ({valid_corrs/total_possible:.1%})")
    
    # Get correlation values (upper triangle, excluding diagonal)
    upper_tri_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_values = corr_matrix.values[upper_tri_mask]
    valid_corr_values = corr_values[~np.isnan(corr_values)]
    
    corr_diagnostics = {
        'n_markets': len(corr_matrix),
        'total_possible_pairs': total_possible,
        'valid_correlations': valid_corrs,
        'coverage': valid_corrs / total_possible if total_possible > 0 else 0,
        'correlation_stats': {
            'count': len(valid_corr_values),
            'mean': float(np.mean(valid_corr_values)) if len(valid_corr_values) > 0 else 0,
            'std': float(np.std(valid_corr_values)) if len(valid_corr_values) > 0 else 0,
            'min': float(np.min(valid_corr_values)) if len(valid_corr_values) > 0 else 0,
            'max': float(np.max(valid_corr_values)) if len(valid_corr_values) > 0 else 0,
            'mean_abs': float(np.mean(np.abs(valid_corr_values))) if len(valid_corr_values) > 0 else 0,
        }
    }
    
    print(f"Correlation statistics:")
    print(f"  Mean: {corr_diagnostics['correlation_stats']['mean']:.4f}")
    print(f"  Std:  {corr_diagnostics['correlation_stats']['std']:.4f}")
    print(f"  |Mean|: {corr_diagnostics['correlation_stats']['mean_abs']:.4f}")
    print(f"  Range: [{corr_diagnostics['correlation_stats']['min']:.4f}, {corr_diagnostics['correlation_stats']['max']:.4f}]")
    
    # Top correlated pairs for sanity check
    top_pairs = []
    markets_subset = markets.set_index('market_id')
    
    # Find highest correlations
    corr_flat = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            market1 = corr_matrix.index[i]
            market2 = corr_matrix.index[j]
            corr = corr_matrix.iloc[i, j]
            if not np.isnan(corr):
                corr_flat.append((abs(corr), corr, market1, market2))
    
    # Sort by absolute correlation and take top 20
    corr_flat.sort(reverse=True)
    for abs_corr, corr, market1, market2 in corr_flat[:20]:
        title1 = markets_subset.loc[market1, 'title'] if market1 in markets_subset.index else market1
        title2 = markets_subset.loc[market2, 'title'] if market2 in markets_subset.index else market2
        
        top_pairs.append({
            'correlation': corr,
            'abs_correlation': abs_corr,
            'market1': market1,
            'market2': market2,
            'title1': title1[:100],  # Truncate for readability
            'title2': title2[:100]
        })
    
    return corr_matrix, corr_diagnostics, top_pairs

def main():
    print("=== Rebuilding Correlation Matrix with Data Quality Filters ===")
    
    # Load data
    print("Loading data...")
    markets = pd.read_parquet("data/processed/markets.parquet")
    prices = pd.read_parquet("data/processed/prices.parquet")
    
    print(f"Loaded {len(markets):,} markets and {len(prices):,} price records")
    
    # Apply data quality filters
    filtered_prices, filter_stats = apply_data_quality_filters(
        prices,
        markets,
        min_volume_threshold=None,  # Will use 25th percentile
        min_price_variance=0.005,
        min_unique_prices=10,
        min_active_days=30
    )
    
    # Compute price changes
    price_changes = compute_price_changes_with_stats(filtered_prices)
    
    # Build correlation matrix with diagnostics
    corr_matrix, corr_diagnostics, top_pairs = build_correlation_matrix_with_diagnostics(
        price_changes,
        markets,
        min_overlapping_days=30,
        min_days_per_market=30
    )
    
    # Save results
    print(f"\nSaving results...")
    
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
    
    # Print top correlated pairs for sanity check
    print(f"\n=== TOP 20 CORRELATED PAIRS (Sanity Check) ===")
    for i, pair in enumerate(top_pairs, 1):
        sign = "+" if pair['correlation'] > 0 else "-"
        print(f"{i:2d}. {sign}{pair['abs_correlation']:.4f}: ")
        print(f"     {pair['title1']}")
        print(f"     {pair['title2']}")
    
    # Final summary
    print(f"\n=== SUMMARY ===")
    print(f"Original markets: {filter_stats['initial_markets']:,}")
    print(f"Filtered markets: {filter_stats['final_markets']:,}")
    print(f"Filter rate: {filter_stats['filter_rate']:.1%}")
    print(f"Correlation matrix size: {len(corr_matrix)}×{len(corr_matrix)}")
    print(f"Valid correlations: {corr_diagnostics['valid_correlations']:,}")
    print(f"Mean |correlation|: {corr_diagnostics['correlation_stats']['mean_abs']:.4f}")
    
    return corr_matrix, all_diagnostics

if __name__ == "__main__":
    main()
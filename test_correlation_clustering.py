"""
Test script for correlation clustering on the existing factor-clustered markets.
This ensures fair comparison and manageable computation.
"""

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
from pathlib import Path
import json
from datetime import datetime
import os

def load_existing_clustered_markets():
    """Load the markets that are currently being factor-clustered."""
    clusters = pd.read_parquet('data/processed/cluster_assignments.parquet')
    return list(clusters.index)

def compute_correlation_matrix_efficient(prices_df, market_ids, min_days=20):
    """Efficiently compute correlation matrix for specified markets."""
    print(f"Computing correlations for {len(market_ids)} markets...")
    
    # Filter to specified markets and compute changes
    market_prices = prices_df[prices_df['market_id'].isin(market_ids)].copy()
    market_prices['date'] = pd.to_datetime(market_prices['date'])
    market_prices = market_prices.sort_values(['market_id', 'date'])
    
    # Compute price changes
    market_prices['price_change'] = market_prices.groupby('market_id')['close_price'].diff()
    market_prices = market_prices.dropna(subset=['price_change'])
    
    # Pivot to matrix format for efficient correlation computation
    pivot = market_prices.pivot(index='date', columns='market_id', values='price_change')
    
    print(f"Price change matrix: {pivot.shape[0]} days x {pivot.shape[1]} markets")
    
    # Filter markets with sufficient data
    market_counts = pivot.count()
    valid_markets = market_counts[market_counts >= min_days].index
    pivot_filtered = pivot[valid_markets]
    
    print(f"Markets with ≥{min_days} days: {len(valid_markets)}")
    
    # Compute correlation matrix using pandas (much faster)
    print("Computing correlation matrix...")
    corr_matrix = pivot_filtered.corr()
    
    # Count valid correlations
    valid_corrs = (~corr_matrix.isna()).sum().sum() - len(corr_matrix)  # Exclude diagonal
    total_possible = len(corr_matrix) * (len(corr_matrix) - 1)
    print(f"Valid correlations: {valid_corrs:,} / {total_possible:,} ({valid_corrs/total_possible:.1%})")
    
    return corr_matrix.fillna(0)

def run_test_correlation_clustering():
    """Test correlation clustering on existing clustered markets."""
    print("=== Testing Correlation Clustering ===")
    
    # Load data
    prices = pd.read_parquet('data/processed/prices.parquet')
    markets = pd.read_parquet('data/processed/markets.parquet')
    existing_clusters = pd.read_parquet('data/processed/cluster_assignments.parquet')
    
    print(f"Existing factor clusters: {existing_clusters['cluster'].value_counts().sort_index().to_dict()}")
    
    # Focus on existing clustered markets
    market_ids = list(existing_clusters.index)
    
    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix_efficient(prices, market_ids, min_days=20)
    
    # Build correlation graph
    threshold = 0.3
    print(f"Building graph with correlation threshold {threshold}...")
    
    G = nx.Graph()
    G.add_nodes_from(corr_matrix.index)
    
    edges_added = 0
    for i, market1 in enumerate(corr_matrix.index):
        for j, market2 in enumerate(corr_matrix.columns):
            if i < j:
                corr = corr_matrix.loc[market1, market2]
                if abs(corr) > threshold:
                    G.add_edge(market1, market2, weight=abs(corr))
                    edges_added += 1
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    if G.number_of_edges() == 0:
        print("Warning: No edges in graph! Try lower threshold.")
        return
    
    # Run Louvain community detection
    print("Running Louvain community detection...")
    partition = community_louvain.best_partition(G, weight='weight', random_state=42)
    
    # Analyze results
    community_sizes = {}
    for market, comm in partition.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    print(f"\\nCorrelation-based communities found: {len(community_sizes)}")
    for comm, size in sorted(community_sizes.items(), key=lambda x: -x[1]):
        print(f"  Community {comm}: {size} markets ({size/len(partition):.1%})")
    
    # Compare with existing factor clusters
    print("\\nComparison with existing factor clusters:")
    comparison_df = pd.DataFrame({
        'market_id': list(partition.keys()),
        'correlation_community': list(partition.values()),
        'factor_cluster': [existing_clusters.loc[m, 'cluster'] for m in partition.keys()]
    }).set_index('market_id')
    
    # Cross-tabulation
    crosstab = pd.crosstab(comparison_df['correlation_community'], 
                          comparison_df['factor_cluster'], 
                          margins=True)
    print(crosstab)
    
    # Calculate agreement metrics
    print("\\nCommunity vs Cluster Analysis:")
    for corr_comm in sorted(community_sizes.keys()):
        comm_markets = comparison_df[comparison_df['correlation_community'] == corr_comm]
        dominant_cluster = comm_markets['factor_cluster'].mode().iloc[0]
        agreement_pct = (comm_markets['factor_cluster'] == dominant_cluster).mean()
        print(f"  Correlation community {corr_comm} (n={len(comm_markets)}): "
              f"mostly factor cluster {dominant_cluster} ({agreement_pct:.1%} agreement)")
    
    # Save test results
    Path('data/processed').mkdir(exist_ok=True)
    
    # Save correlation matrix
    corr_matrix.to_parquet('data/processed/correlation_matrix_test.parquet')
    
    # Save community assignments
    comm_df = pd.DataFrame([
        {'market_id': market, 'community': comm} 
        for market, comm in partition.items()
    ]).set_index('market_id')
    comm_df.to_parquet('data/processed/community_assignments_test.parquet')
    
    # Save comparison
    comparison_df.to_parquet('data/processed/correlation_vs_factor_comparison.parquet')
    
    results = {
        'n_markets': len(partition),
        'n_correlation_communities': len(community_sizes),
        'correlation_community_sizes': community_sizes,
        'correlation_threshold': threshold,
        'comparison_crosstab': crosstab.to_dict(),
        'created_at': datetime.now().isoformat()
    }
    
    with open('data/processed/correlation_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\nSaved test results to data/processed/")
    
    return results

if __name__ == "__main__":
    run_test_correlation_clustering()
#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
import community as community_louvain
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_louvain_clustering():
    """Run Louvain clustering at resolution 5.0"""
    
    print("=== STEP 2: LOUVAIN CLUSTERING AT RESOLUTION 5.0 ===")
    
    # Load correlation matrix
    print("Loading correlation matrix...")
    correlation_matrix = pd.read_parquet('data/processed/correlation_matrix.parquet')
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    
    # Build graph from correlation matrix (threshold: keep edges with |corr| > 0.05)
    print("Building graph from correlation matrix (threshold: |corr| > 0.05)...")
    
    # Get absolute correlation values
    abs_corr = correlation_matrix.abs()
    
    # Create graph
    G = nx.Graph()
    
    # Add all nodes
    nodes = correlation_matrix.index.tolist()
    G.add_nodes_from(nodes)
    print(f"Added {len(nodes)} nodes to graph")
    
    # Add edges where |correlation| > 0.05
    edge_count = 0
    total_pairs = 0
    
    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            total_pairs += 1
            market_i = correlation_matrix.index[i]
            market_j = correlation_matrix.index[j]
            
            # Get absolute correlation
            corr_value = abs_corr.iloc[i, j]
            
            # Skip NaN correlations
            if pd.isna(corr_value):
                continue
                
            # Add edge if absolute correlation > 0.05
            if corr_value > 0.05:
                # Use the raw correlation as weight (not absolute)
                raw_corr = correlation_matrix.iloc[i, j]
                G.add_edge(market_i, market_j, weight=raw_corr)
                edge_count += 1
    
    print(f"Added {edge_count} edges out of {total_pairs} possible pairs")
    print(f"Graph density: {edge_count / total_pairs:.4f}")
    print(f"Graph connected components: {nx.number_connected_components(G)}")
    
    # Run Louvain community detection at resolution 5.0
    print("Running Louvain community detection at resolution=5.0...")
    partition = community_louvain.best_partition(G, resolution=5.0, random_state=42)
    
    # Convert to DataFrame
    community_assignments = pd.DataFrame.from_dict(partition, orient='index', columns=['community'])
    community_assignments.index.name = 'market_id'
    
    print(f"Found {community_assignments['community'].nunique()} communities")
    print(f"Community size distribution:")
    community_sizes = community_assignments['community'].value_counts().sort_values(ascending=False)
    print(f"  Largest: {community_sizes.iloc[0]} markets")
    print(f"  Median: {community_sizes.median():.1f} markets")
    print(f"  Smallest: {community_sizes.iloc[-1]} markets")
    print(f"  Communities with 1 market: {(community_sizes == 1).sum()}")
    print(f"  Communities with >10 markets: {(community_sizes > 10).sum()}")
    
    # Save community assignments
    output_path = Path('data/processed/community_assignments.parquet')
    print(f"Saving community assignments to {output_path}...")
    community_assignments.to_parquet(output_path)
    
    print(f"✅ Louvain clustering completed successfully!")
    print(f"Summary:")
    print(f"  - Markets clustered: {len(community_assignments)}")
    print(f"  - Communities found: {community_assignments['community'].nunique()}")
    print(f"  - Graph edges: {edge_count}")
    
    return community_assignments, G

if __name__ == "__main__":
    community_assignments, graph = run_louvain_clustering()
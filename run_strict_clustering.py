#!/usr/bin/env python3
"""
Run strict clustering with higher thresholds and quality filters.
- Correlation threshold: 0.5 minimum 
- Minimum community size: 10 markets
- Exclude isolated markets (no edges above threshold)
"""

import pandas as pd
import networkx as nx
import community as community_louvain
from src.analysis.correlation_clustering import build_correlation_graph

def main():
    print("=== Running Strict Quality Clustering ===")
    
    # Load the filtered correlation matrix
    corr_matrix = pd.read_parquet("data/processed/correlation_matrix.parquet")
    print(f"Loaded correlation matrix: {corr_matrix.shape[0]}x{corr_matrix.shape[1]} markets")
    
    # Build graph with strict threshold
    correlation_threshold = 0.5
    print(f"Building graph with correlation threshold >= {correlation_threshold}")
    
    # Build the correlation graph 
    G = build_correlation_graph(corr_matrix, threshold=correlation_threshold)
    
    print(f"Graph after threshold filtering:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    if G.number_of_edges() == 0:
        print("ERROR: No edges above threshold! Try lower threshold.")
        return
    
    # Remove isolated nodes (markets with no connections above threshold)
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} isolated markets")
    print(f"Connected graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Run Louvain clustering
    print("Running Louvain community detection...")
    partition = community_louvain.best_partition(
        G, weight='weight', random_state=42
    )
    
    # Filter communities by minimum size
    min_community_size = 10
    print(f"Filtering communities with minimum size {min_community_size}")
    
    community_sizes = {}
    for node, comm in partition.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    # Keep only large communities
    large_communities = {comm: size for comm, size in community_sizes.items() if size >= min_community_size}
    
    # Filter partition to only include markets in large communities
    filtered_partition = {
        market: comm for market, comm in partition.items() 
        if comm in large_communities
    }
    
    print(f"\nCommunity filtering results:")
    print(f"  Original communities: {len(community_sizes)}")
    print(f"  Communities ≥{min_community_size}: {len(large_communities)}")
    print(f"  Markets in large communities: {len(filtered_partition)}")
    print(f"  Markets excluded (small communities): {len(partition) - len(filtered_partition)}")
    
    # Save community assignments
    assignments_df = pd.DataFrame([
        {"market_id": market, "community": comm} 
        for market, comm in filtered_partition.items()
    ]).set_index("market_id")
    
    output_path = "data/processed/strict_community_assignments.parquet"
    assignments_df.to_parquet(output_path)
    print(f"Saved strict community assignments to {output_path}")
    
    # Print final summary
    print(f"\nFinal Clustering Summary:")
    print(f"  Total markets clustered: {len(filtered_partition)}")
    print(f"  Communities: {len(large_communities)}")
    
    print(f"\nCommunity sizes:")
    for comm, size in sorted(large_communities.items(), key=lambda x: -x[1]):
        print(f"  Community {comm}: {size} markets")
    
    # Calculate clustering quality metrics
    if len(large_communities) > 0:
        mean_size = sum(large_communities.values()) / len(large_communities)
        print(f"\nQuality metrics:")
        print(f"  Mean community size: {mean_size:.1f}")
        print(f"  Largest community: {max(large_communities.values())} markets")
        print(f"  Smallest community: {min(large_communities.values())} markets")
    
    print("Done!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Run Louvain clustering at resolution 5.0 on the filtered correlation matrix.
"""

import pandas as pd
from src.analysis.correlation_clustering import run_clustering_at_resolution

def main():
    print("=== Running Louvain Clustering at Resolution 5.0 ===")
    
    # Load the filtered correlation matrix
    corr_matrix = pd.read_parquet("data/processed/correlation_matrix.parquet")
    print(f"Loaded correlation matrix: {corr_matrix.shape[0]}x{corr_matrix.shape[1]} markets")
    
    # Run clustering at resolution 5.0
    resolution = 5.0
    partition = run_clustering_at_resolution(corr_matrix, resolution)
    
    # Save community assignments
    assignments_df = pd.DataFrame([
        {"market_id": market, "community": comm} 
        for market, comm in partition.items()
    ]).set_index("market_id")
    
    output_path = "data/processed/community_assignments.parquet"
    assignments_df.to_parquet(output_path)
    print(f"Saved community assignments to {output_path}")
    
    # Print summary
    community_sizes = {}
    for market, comm in partition.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    print(f"\nClustering Summary:")
    print(f"  Total markets: {len(partition)}")
    print(f"  Communities found: {len(community_sizes)}")
    print(f"  Largest community: {max(community_sizes.values())} markets")
    print(f"  Smallest community: {min(community_sizes.values())} markets")
    print(f"  Median community size: {sorted(community_sizes.values())[len(community_sizes)//2]}")
    
    print("\nTop 10 largest communities:")
    for i, (comm, size) in enumerate(sorted(community_sizes.items(), key=lambda x: -x[1])[:10]):
        print(f"  Community {comm}: {size} markets")
    
    print("Done!")

if __name__ == "__main__":
    main()
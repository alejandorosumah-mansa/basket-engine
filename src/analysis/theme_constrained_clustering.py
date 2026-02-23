"""
Theme-Constrained Correlation Clustering

Instead of clustering all markets together (which mixes themes),
run Louvain WITHIN each theme separately. This guarantees
thematic purity: a Fed basket will never contain crypto markets.
"""
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
from pathlib import Path
from collections import Counter


def cluster_within_themes(
    correlation_matrix: pd.DataFrame,
    theme_assignments: pd.Series,  # market_id -> theme
    resolution: float = 2.0,
    min_corr: float = 0.05,
    min_community_size: int = 3,
) -> pd.DataFrame:
    """
    Run Louvain clustering within each theme separately.
    
    Returns DataFrame with columns: market_id, theme, community, community_id
    where community_id is globally unique (theme_communityN)
    """
    results = []
    
    # Get markets that are in both correlation matrix and theme assignments
    common_markets = set(correlation_matrix.index) & set(theme_assignments.index)
    print(f"Markets in correlation matrix: {len(correlation_matrix)}")
    print(f"Markets with themes: {len(theme_assignments)}")
    print(f"Common: {len(common_markets)}")
    
    theme_counts = Counter(theme_assignments[list(common_markets)])
    print(f"\nThemes to cluster:")
    for theme, count in theme_counts.most_common():
        print(f"  {theme}: {count} markets")
    
    global_community_id = 0
    
    for theme in sorted(theme_counts.keys()):
        # Get markets in this theme
        theme_markets = [m for m in common_markets if theme_assignments[m] == theme]
        
        if len(theme_markets) < min_community_size:
            # Too small to cluster, put all in one community
            for m in theme_markets:
                results.append({
                    'market_id': m,
                    'theme': theme,
                    'community': 0,
                    'community_id': f"{theme}_0",
                    'global_community': global_community_id,
                })
            global_community_id += 1
            continue
        
        # Build sub-correlation matrix
        sub_corr = correlation_matrix.loc[theme_markets, theme_markets]
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(theme_markets)
        
        for i, m1 in enumerate(theme_markets):
            for j in range(i + 1, len(theme_markets)):
                m2 = theme_markets[j]
                corr = sub_corr.loc[m1, m2]
                if not np.isnan(corr) and abs(corr) > min_corr:
                    G.add_edge(m1, m2, weight=max(corr, 0))  # Louvain needs positive weights
        
        # Run Louvain
        if G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)
        else:
            partition = {m: 0 for m in theme_markets}
        
        n_communities = len(set(partition.values()))
        print(f"\n  {theme}: {len(theme_markets)} markets -> {n_communities} communities")
        
        # Map local communities to global
        local_to_global = {}
        for local_id in sorted(set(partition.values())):
            local_to_global[local_id] = global_community_id
            global_community_id += 1
        
        for m in theme_markets:
            local_id = partition.get(m, 0)
            results.append({
                'market_id': m,
                'theme': theme,
                'community': local_id,
                'community_id': f"{theme}_{local_id}",
                'global_community': local_to_global[local_id],
            })
    
    df = pd.DataFrame(results)
    print(f"\nTotal communities: {df['global_community'].nunique()}")
    print(f"Total markets assigned: {len(df)}")
    
    return df


if __name__ == '__main__':
    from src.classification.theme_classifier import classify_markets
    
    # Load data
    markets = pd.read_parquet('data/processed/markets_filtered.parquet')
    corr_matrix = pd.read_parquet('data/processed/correlation_matrix.parquet')
    
    print(f"Filtered markets: {len(markets)}")
    print(f"Correlation matrix: {corr_matrix.shape}")
    
    # Classify
    markets = classify_markets(markets)
    
    # Build theme_assignments Series: market_id -> theme
    # Need to map market_id to correlation matrix index
    # Check what IDs the correlation matrix uses
    print(f"\nCorrelation matrix index sample: {list(corr_matrix.index[:5])}")
    print(f"Markets market_id sample: {markets['market_id'].head(5).tolist()}")
    
    # Create mapping
    theme_map = pd.Series(
        markets['theme'].values,
        index=markets['market_id'].values
    )
    
    # Cluster within themes
    assignments = cluster_within_themes(
        corr_matrix, 
        theme_map,
        resolution=2.0,
        min_corr=0.05,
        min_community_size=3,
    )
    
    # Save
    assignments.to_parquet('data/processed/theme_community_assignments.parquet', index=False)
    print(f"\nSaved to data/processed/theme_community_assignments.parquet")
    
    # Summary
    print("\n=== COMMUNITY SUMMARY ===")
    for theme in sorted(assignments['theme'].unique()):
        theme_data = assignments[assignments['theme'] == theme]
        n_comms = theme_data['global_community'].nunique()
        print(f"\n{theme}: {len(theme_data)} markets, {n_comms} communities")
        for gc in sorted(theme_data['global_community'].unique()):
            size = len(theme_data[theme_data['global_community'] == gc])
            print(f"  Community {gc}: {size} markets")

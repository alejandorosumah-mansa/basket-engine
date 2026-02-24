#!/usr/bin/env python3
"""
Optimized Weighted Hybrid Clustering Pipeline

Uses vectorized operations and efficient processing for large correlation matrices.
"""

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
from pathlib import Path
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Import existing modules
import sys
sys.path.append('src')
from analysis.correlation_clustering import (
    compute_explainability_metrics,
    generate_llm_community_names_and_critiques,
    characterize_communities_with_factors
)


def load_theme_classifications(
    classifications_path: str = "data/processed/market_classifications.parquet"
) -> pd.Series:
    """Load LLM theme classifications as market_id -> theme mapping."""
    print("Loading LLM theme classifications...")
    
    df = pd.read_parquet(classifications_path)
    theme_counts = df['category'].value_counts()
    print(f"Loaded {len(df)} classifications with {len(theme_counts)} themes")
    
    return pd.Series(df['category'].values, index=df['market_id'].values)


def build_optimized_weighted_hybrid_graph(
    corr_matrix: pd.DataFrame,
    theme_map: pd.Series,
    correlation_threshold: float = 0.3,
    intra_theme_multiplier: float = 4.0,
    cross_theme_min_corr: float = 0.5
) -> nx.Graph:
    """
    Build weighted graph with optimized processing for large matrices.
    """
    print("Building optimized weighted hybrid graph...")
    print(f"  Correlation threshold: {correlation_threshold}")
    print(f"  Intra-theme multiplier: {intra_theme_multiplier}x")
    print(f"  Cross-theme min correlation: {cross_theme_min_corr}")
    
    # Get common markets
    common_markets = list(set(corr_matrix.index) & set(theme_map.index))
    print(f"  Common markets: {len(common_markets)}")
    
    # Filter correlation matrix and theme map to common markets
    filtered_corr = corr_matrix.loc[common_markets, common_markets]
    filtered_themes = theme_map[common_markets]
    
    print(f"  Building theme matrix...")
    # Create theme comparison matrix (vectorized)
    theme_values = filtered_themes.values
    theme_matrix = np.equal.outer(theme_values, theme_values)  # True if same theme
    
    # Get absolute correlation matrix
    abs_corr_values = np.abs(filtered_corr.values)
    
    # Create edge masks
    print(f"  Creating edge criteria...")
    # For same theme: use regular threshold
    intra_theme_mask = theme_matrix & (abs_corr_values > correlation_threshold)
    # For different themes: use higher threshold  
    cross_theme_mask = ~theme_matrix & (abs_corr_values > cross_theme_min_corr)
    # Combined edge mask
    edge_mask = intra_theme_mask | cross_theme_mask
    
    # Remove diagonal (self-loops)
    np.fill_diagonal(edge_mask, False)
    
    # Calculate weights (vectorized)
    weights = np.where(
        theme_matrix,  # If same theme
        abs_corr_values * intra_theme_multiplier,  # Apply multiplier
        abs_corr_values  # Otherwise use raw correlation
    )
    
    # Build graph
    print(f"  Creating NetworkX graph...")
    G = nx.Graph()
    
    # Add nodes with theme attributes
    for market in common_markets:
        G.add_node(market, theme=filtered_themes[market])
    
    # Add edges efficiently
    edges_to_add = []
    edge_indices = np.where(edge_mask)
    
    print(f"  Processing {len(edge_indices[0])} edges...")
    
    for idx in range(len(edge_indices[0])):
        i, j = edge_indices[0][idx], edge_indices[1][idx]
        if i < j:  # Only add upper triangle to avoid duplicates
            market1 = common_markets[i]
            market2 = common_markets[j]
            
            weight = weights[i, j]
            correlation = filtered_corr.iloc[i, j]
            abs_correlation = abs_corr_values[i, j]
            same_theme = theme_matrix[i, j]
            
            edges_to_add.append((
                market1, market2, {
                    'weight': weight,
                    'correlation': correlation,
                    'abs_correlation': abs_correlation,
                    'same_theme': same_theme
                }
            ))
    
    # Add all edges at once
    G.add_edges_from(edges_to_add)
    
    # Statistics
    intra_theme_edges = np.sum(intra_theme_mask) // 2  # Divide by 2 for upper triangle
    cross_theme_edges = np.sum(cross_theme_mask) // 2
    total_edges = G.number_of_edges()
    
    print(f"  Graph created: {G.number_of_nodes():,} nodes, {total_edges:,} edges")
    print(f"    Intra-theme edges: {intra_theme_edges:,} ({intra_theme_edges/total_edges:.1%})")
    print(f"    Cross-theme edges: {cross_theme_edges:,} ({cross_theme_edges/total_edges:.1%})")
    
    # Edge density
    max_edges = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
    density = G.number_of_edges() / max_edges if max_edges > 0 else 0
    print(f"    Edge density: {density:.4f}")
    
    return G


def run_weighted_hybrid_clustering(
    G: nx.Graph,
    resolution: float = 2.0
) -> Dict[str, int]:
    """Run Louvain clustering on the weighted hybrid graph."""
    print(f"Running Louvain clustering (resolution={resolution})...")
    
    if G.number_of_edges() == 0:
        print("Warning: Graph has no edges, creating singleton communities")
        return {node: i for i, node in enumerate(G.nodes())}
    
    # Run Louvain with weights
    partition = community_louvain.best_partition(
        G, weight='weight', resolution=resolution, random_state=42
    )
    
    # Compute modularity
    modularity = community_louvain.modularity(partition, G, weight='weight')
    print(f"Modularity (weighted): {modularity:.4f}")
    
    # Community statistics
    community_sizes = {}
    community_themes = {}
    
    for market, comm_id in partition.items():
        theme = G.nodes[market]['theme']
        
        # Count community sizes
        community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
        
        # Track theme purity
        if comm_id not in community_themes:
            community_themes[comm_id] = {}
        community_themes[comm_id][theme] = community_themes[comm_id].get(theme, 0) + 1
    
    print(f"Found {len(community_sizes)} communities:")
    for comm_id in sorted(community_sizes.keys(), key=lambda x: -community_sizes[x])[:10]:
        size = community_sizes[comm_id]
        themes = community_themes[comm_id]
        
        # Find dominant theme
        dominant_theme = max(themes.items(), key=lambda x: x[1])
        purity = dominant_theme[1] / size
        
        print(f"  Community {comm_id}: {size} markets, "
              f"dominant theme: {dominant_theme[0]} ({purity:.1%})")
        
        # Show theme breakdown for diverse communities
        if purity < 0.8 and len(themes) > 1:
            theme_strs = [f"{theme}({count})" for theme, count in 
                         sorted(themes.items(), key=lambda x: -x[1])[:3]]
            print(f"    Mixed themes: {', '.join(theme_strs)}")
    
    if len(community_sizes) > 10:
        print(f"  ... and {len(community_sizes) - 10} more communities")
    
    return partition


def analyze_edge_contributions(G: nx.Graph, partition: Dict[str, int]) -> Dict:
    """Analyze how intra-theme vs cross-theme edges contributed to clustering."""
    print("Analyzing edge contributions...")
    
    intra_theme_edges = 0
    cross_theme_edges = 0
    intra_community_edges = 0
    intra_theme_intra_community = 0
    cross_theme_intra_community = 0
    
    for edge in G.edges(data=True):
        market1, market2, data = edge
        same_theme = data['same_theme']
        same_community = (partition[market1] == partition[market2])
        
        # Count edge types
        if same_theme:
            intra_theme_edges += 1
        else:
            cross_theme_edges += 1
            
        if same_community:
            intra_community_edges += 1
            if same_theme:
                intra_theme_intra_community += 1
            else:
                cross_theme_intra_community += 1
    
    total_edges = G.number_of_edges()
    
    # Compute alignment metrics
    theme_purity = intra_theme_intra_community / intra_community_edges if intra_community_edges > 0 else 0
    theme_cohesion = intra_theme_intra_community / intra_theme_edges if intra_theme_edges > 0 else 0
    
    edge_stats = {
        'total_edges': total_edges,
        'intra_theme_edges': intra_theme_edges,
        'cross_theme_edges': cross_theme_edges,
        'intra_community_edges': intra_community_edges,
        'cross_community_edges': total_edges - intra_community_edges,
        'intra_theme_intra_community': intra_theme_intra_community,
        'cross_theme_intra_community': cross_theme_intra_community,
        'theme_community_alignment': {
            'intra_community_theme_purity': theme_purity,
            'theme_cohesion': theme_cohesion
        }
    }
    
    print(f"Edge analysis:")
    print(f"  Intra-theme edges: {intra_theme_edges:,} ({intra_theme_edges/total_edges:.1%})")
    print(f"  Cross-theme edges: {cross_theme_edges:,} ({cross_theme_edges/total_edges:.1%})")
    print(f"  Theme purity within communities: {theme_purity:.1%}")
    print(f"  Theme cohesion (same theme kept together): {theme_cohesion:.1%}")
    
    return edge_stats


def save_hybrid_results(
    partition: Dict[str, int],
    G: nx.Graph,
    theme_map: pd.Series,
    edge_stats: Dict,
    parameters: Dict,
    output_dir: str = "data/processed"
) -> Dict:
    """Save results with comprehensive analysis."""
    print("Saving weighted hybrid clustering results...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load markets for LLM labeling
    try:
        markets_df = pd.read_parquet("data/processed/markets_filtered.parquet")
        print("Generating LLM community names and critiques...")
        community_info = generate_llm_community_names_and_critiques(
            partition, markets_df, model="gpt-4o-mini"
        )
    except Exception as e:
        print(f"Warning: Could not generate LLM labels: {e}")
        community_info = {}
        for comm_id in set(partition.values()):
            community_info[comm_id] = {
                "name": f"Community_{comm_id}",
                "critique": "LLM labeling failed"
            }
    
    # Characterize with factors
    try:
        community_factors = characterize_communities_with_factors(partition)
    except Exception as e:
        print(f"Warning: Could not characterize with factors: {e}")
        community_factors = {}
    
    # Compute explainability metrics
    try:
        corr_matrix = pd.read_parquet("data/processed/correlation_matrix.parquet")
        common_markets = set(partition.keys()) & set(corr_matrix.index)
        filtered_partition = {k: v for k, v in partition.items() if k in common_markets}
        filtered_corr = corr_matrix.loc[list(common_markets), list(common_markets)]
        explainability = compute_explainability_metrics(filtered_corr, filtered_partition)
    except Exception as e:
        print(f"Warning: Could not compute explainability metrics: {e}")
        explainability = {}
    
    # Build comprehensive results
    community_sizes = {}
    community_theme_breakdown = {}
    for market, comm_id in partition.items():
        # Sizes
        community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
        
        # Theme breakdown
        if comm_id not in community_theme_breakdown:
            community_theme_breakdown[comm_id] = {}
        theme = theme_map.get(market, 'unknown')
        community_theme_breakdown[comm_id][theme] = \
            community_theme_breakdown[comm_id].get(theme, 0) + 1
    
    results = {
        "method": "weighted_hybrid_clustering_optimized",
        "parameters": parameters,
        "created_at": datetime.now().isoformat(),
        "graph_stats": {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "edge_density": G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2) if G.number_of_nodes() > 1 else 0,
            "modularity": community_louvain.modularity(partition, G, weight='weight') if G.number_of_edges() > 0 else 0,
        },
        "edge_analysis": edge_stats,
        "clustering_results": {
            "n_communities": len(community_sizes),
            "n_markets": len(partition),
            "community_sizes": community_sizes,
            "community_theme_breakdown": community_theme_breakdown,
        },
        "explainability_metrics": explainability,
        "community_assignments": partition,
        "community_info": community_info,
        "community_factors": community_factors,
    }
    
    # Save files
    # 1. Community assignments (replaces the old ones)
    assignments_df = pd.DataFrame([
        {"market_id": market, "community": comm} 
        for market, comm in partition.items()
    ]).set_index("market_id")
    assignments_df.to_parquet(f"{output_dir}/community_assignments.parquet")
    print(f"  Saved assignments: {output_dir}/community_assignments.parquet")
    
    # 2. Hybrid-specific files
    with open(f"{output_dir}/hybrid_clustering_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved results: {output_dir}/hybrid_clustering_results.json")
    
    with open(f"{output_dir}/hybrid_community_info.json", "w") as f:
        json.dump(community_info, f, indent=2)
    print(f"  Saved community info: {output_dir}/hybrid_community_info.json")
    
    return results


def run_optimized_hybrid_pipeline(
    correlation_threshold: float = 0.3,
    intra_theme_multiplier: float = 4.0,
    cross_theme_min_corr: float = 0.5,
    resolution: float = 2.0,
    output_dir: str = "data/processed"
) -> Dict:
    """
    Run the optimized weighted hybrid clustering pipeline.
    """
    print("=== Optimized Weighted Hybrid Clustering ===")
    print(f"Intra-theme weight multiplier: {intra_theme_multiplier}x")
    print(f"Cross-theme minimum correlation: {cross_theme_min_corr}")
    print(f"Base correlation threshold: {correlation_threshold}")
    print(f"Louvain resolution: {resolution}\n")
    
    parameters = {
        "correlation_threshold": correlation_threshold,
        "intra_theme_multiplier": intra_theme_multiplier,
        "cross_theme_min_corr": cross_theme_min_corr,
        "resolution": resolution
    }
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    corr_matrix = pd.read_parquet("data/processed/correlation_matrix.parquet")
    theme_map = load_theme_classifications()
    print(f"Correlation matrix: {corr_matrix.shape}")
    print(f"Theme classifications: {len(theme_map)}")
    
    # Step 2: Build weighted hybrid graph (optimized)
    print("\nStep 2: Building weighted hybrid graph...")
    G = build_optimized_weighted_hybrid_graph(
        corr_matrix, theme_map, correlation_threshold, 
        intra_theme_multiplier, cross_theme_min_corr
    )
    
    # Step 3: Run clustering
    print("\nStep 3: Running clustering...")
    partition = run_weighted_hybrid_clustering(G, resolution)
    
    # Step 4: Analyze results
    print("\nStep 4: Analyzing results...")
    edge_stats = analyze_edge_contributions(G, partition)
    
    # Step 5: Save results
    print("\nStep 5: Saving results...")
    results = save_hybrid_results(
        partition, G, theme_map, edge_stats, 
        parameters, output_dir
    )
    
    # Final summary
    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"Communities: {results['clustering_results']['n_communities']}")
    print(f"Markets clustered: {results['clustering_results']['n_markets']:,}")
    print(f"Modularity: {results['graph_stats']['modularity']:.4f}")
    
    if 'explainability_score' in results['explainability_metrics']:
        print(f"Explainability score: {results['explainability_metrics']['explainability_score']:.3f}")
    
    theme_align = results['edge_analysis']['theme_community_alignment']
    print(f"Theme purity: {theme_align['intra_community_theme_purity']:.1%}")
    print(f"Theme cohesion: {theme_align['theme_cohesion']:.1%}")
    
    return results


if __name__ == "__main__":
    # Install dependencies
    try:
        import community
        import networkx
    except ImportError:
        print("Installing required dependencies...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-louvain", "networkx"])
        import community
        import networkx
    
    # Run optimized pipeline
    results = run_optimized_hybrid_pipeline(
        correlation_threshold=0.3,
        intra_theme_multiplier=4.0,
        cross_theme_min_corr=0.5,
        resolution=2.0
    )
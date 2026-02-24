#!/usr/bin/env python3
"""
Weighted Hybrid Clustering Pipeline

Combines correlation-based edges with LLM theme categories:
1. Build correlation graph (edges where |ρ| > threshold)  
2. Apply theme-based weight boost: intra-theme edges get 3-5x multiplier
3. Cross-theme edges allowed if correlation is overwhelming (>0.5)
4. Run Louvain community detection on weighted graph
5. LLM labeling and characterization
6. Save results and generate reports

This balances data-driven correlations with semantic theme structure.
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
    print(f"Loaded {len(df)} classifications")
    print("Theme distribution:")
    theme_counts = df['category'].value_counts()
    for theme, count in theme_counts.head(10).items():
        print(f"  {theme}: {count:,}")
    if len(theme_counts) > 10:
        print(f"  ... and {len(theme_counts) - 10} more themes")
    
    return pd.Series(df['category'].values, index=df['market_id'].values)


def build_weighted_hybrid_graph(
    corr_matrix: pd.DataFrame,
    theme_map: pd.Series,
    correlation_threshold: float = 0.3,
    intra_theme_multiplier: float = 4.0,
    cross_theme_min_corr: float = 0.5
) -> nx.Graph:
    """
    Build weighted graph combining correlation and theme information.
    
    Args:
        corr_matrix: Pairwise correlation matrix
        theme_map: market_id -> theme mapping
        correlation_threshold: Minimum |correlation| for edges
        intra_theme_multiplier: Weight multiplier for same-theme markets
        cross_theme_min_corr: Minimum correlation for cross-theme edges
        
    Returns:
        NetworkX graph with weighted edges
    """
    print("Building weighted hybrid graph...")
    print(f"  Correlation threshold: {correlation_threshold}")
    print(f"  Intra-theme multiplier: {intra_theme_multiplier}x")
    print(f"  Cross-theme min correlation: {cross_theme_min_corr}")
    
    G = nx.Graph()
    
    # Get common markets (in both correlation matrix and theme map)
    common_markets = set(corr_matrix.index) & set(theme_map.index)
    print(f"  Markets in correlation matrix: {len(corr_matrix)}")
    print(f"  Markets with themes: {len(theme_map)}")
    print(f"  Common markets: {len(common_markets)}")
    
    # Add nodes with theme attributes
    for market in common_markets:
        theme = theme_map[market]
        G.add_node(market, theme=theme)
    
    # Build weighted edges
    added_edges = 0
    intra_theme_edges = 0
    cross_theme_edges = 0
    
    markets_list = list(common_markets)
    for i, market1 in enumerate(markets_list):
        for j in range(i + 1, len(markets_list)):
            market2 = markets_list[j]
            
            # Get correlation
            if market1 in corr_matrix.index and market2 in corr_matrix.columns:
                corr = corr_matrix.loc[market1, market2]
            else:
                continue
                
            if np.isnan(corr):
                continue
                
            abs_corr = abs(corr)
            
            # Get themes
            theme1 = theme_map[market1] 
            theme2 = theme_map[market2]
            same_theme = (theme1 == theme2)
            
            # Decide whether to add edge
            add_edge = False
            weight = abs_corr
            
            if same_theme:
                # Intra-theme: use regular threshold with multiplier
                if abs_corr > correlation_threshold:
                    weight = abs_corr * intra_theme_multiplier
                    add_edge = True
                    intra_theme_edges += 1
            else:
                # Cross-theme: higher threshold, no multiplier
                if abs_corr > cross_theme_min_corr:
                    weight = abs_corr
                    add_edge = True
                    cross_theme_edges += 1
            
            if add_edge:
                G.add_edge(market1, market2, 
                          weight=weight,
                          correlation=corr,
                          abs_correlation=abs_corr,
                          same_theme=same_theme,
                          themes=f"{theme1}-{theme2}" if not same_theme else theme1)
                added_edges += 1
    
    print(f"  Total edges added: {added_edges:,}")
    print(f"    Intra-theme edges: {intra_theme_edges:,} ({intra_theme_edges/added_edges:.1%})")
    print(f"    Cross-theme edges: {cross_theme_edges:,} ({cross_theme_edges/added_edges:.1%})")
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Edge density
    max_edges = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
    if max_edges > 0:
        density = G.number_of_edges() / max_edges
        print(f"  Edge density: {density:.4f}")
    
    # Connected components
    if G.number_of_edges() > 0:
        components = list(nx.connected_components(G))
        print(f"  Connected components: {len(components)}")
        if len(components) > 1:
            largest = max(components, key=len)
            print(f"    Largest component: {len(largest)} nodes ({len(largest)/G.number_of_nodes():.1%})")
    
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
    for comm_id in sorted(community_sizes.keys(), key=lambda x: -community_sizes[x]):
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
    
    return partition


def analyze_edge_contributions(G: nx.Graph, partition: Dict[str, int]) -> Dict:
    """Analyze how intra-theme vs cross-theme edges contributed to clustering."""
    print("Analyzing edge contributions...")
    
    edge_stats = {
        'total_edges': G.number_of_edges(),
        'intra_theme_edges': 0,
        'cross_theme_edges': 0,
        'intra_community_edges': 0,
        'cross_community_edges': 0,
        'intra_theme_intra_community': 0,
        'cross_theme_intra_community': 0,
        'theme_community_alignment': {}
    }
    
    for edge in G.edges(data=True):
        market1, market2, data = edge
        same_theme = data['same_theme']
        same_community = (partition[market1] == partition[market2])
        
        # Count edge types
        if same_theme:
            edge_stats['intra_theme_edges'] += 1
        else:
            edge_stats['cross_theme_edges'] += 1
            
        if same_community:
            edge_stats['intra_community_edges'] += 1
            if same_theme:
                edge_stats['intra_theme_intra_community'] += 1
            else:
                edge_stats['cross_theme_intra_community'] += 1
        else:
            edge_stats['cross_community_edges'] += 1
    
    # Compute alignment percentages
    if edge_stats['intra_community_edges'] > 0:
        theme_comm_align = edge_stats['intra_theme_intra_community'] / edge_stats['intra_community_edges']
        edge_stats['theme_community_alignment']['intra_community_theme_purity'] = theme_comm_align
    
    if edge_stats['intra_theme_edges'] > 0:
        theme_kept_together = edge_stats['intra_theme_intra_community'] / edge_stats['intra_theme_edges']
        edge_stats['theme_community_alignment']['theme_cohesion'] = theme_kept_together
    
    print(f"Edge analysis:")
    print(f"  Intra-theme edges: {edge_stats['intra_theme_edges']:,} "
          f"({edge_stats['intra_theme_edges']/edge_stats['total_edges']:.1%})")
    print(f"  Cross-theme edges: {edge_stats['cross_theme_edges']:,} "
          f"({edge_stats['cross_theme_edges']/edge_stats['total_edges']:.1%})")
    print(f"  Intra-community edges: {edge_stats['intra_community_edges']:,}")
    print(f"    Same theme: {edge_stats['intra_theme_intra_community']:,}")
    print(f"    Cross theme: {edge_stats['cross_theme_intra_community']:,}")
    
    align = edge_stats['theme_community_alignment']
    if 'intra_community_theme_purity' in align:
        print(f"  Theme purity within communities: {align['intra_community_theme_purity']:.1%}")
    if 'theme_cohesion' in align:
        print(f"  Theme cohesion (same theme kept together): {align['theme_cohesion']:.1%}")
    
    return edge_stats


def save_weighted_hybrid_results(
    partition: Dict[str, int],
    G: nx.Graph,
    theme_map: pd.Series,
    edge_stats: Dict,
    markets_df: pd.DataFrame,
    parameters: Dict,
    output_dir: str = "data/processed"
) -> Dict:
    """Save all results and generate comprehensive report."""
    print("Saving weighted hybrid clustering results...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate LLM community names and critiques
    community_info = generate_llm_community_names_and_critiques(
        partition, markets_df, model="gpt-4o-mini"
    )
    
    # Characterize with factors
    community_factors = characterize_communities_with_factors(partition)
    
    # Compute explainability metrics (on unweighted correlations for fair comparison)
    corr_matrix = pd.read_parquet("data/processed/correlation_matrix.parquet")
    common_markets = set(partition.keys()) & set(corr_matrix.index)
    filtered_partition = {k: v for k, v in partition.items() if k in common_markets}
    filtered_corr = corr_matrix.loc[list(common_markets), list(common_markets)]
    explainability = compute_explainability_metrics(filtered_corr, filtered_partition)
    
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
        "method": "weighted_hybrid_clustering",
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
    
    # Save individual files
    # 1. Community assignments (for downstream use)
    assignments_df = pd.DataFrame([
        {"market_id": market, "community": comm} 
        for market, comm in partition.items()
    ]).set_index("market_id")
    assignments_df.to_parquet(f"{output_dir}/hybrid_community_assignments.parquet")
    print(f"  Saved assignments: {output_dir}/hybrid_community_assignments.parquet")
    
    # 2. Community labels and info
    with open(f"{output_dir}/hybrid_community_info.json", "w") as f:
        json.dump(community_info, f, indent=2)
    print(f"  Saved community info: {output_dir}/hybrid_community_info.json")
    
    # 3. Full results
    with open(f"{output_dir}/hybrid_clustering_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved full results: {output_dir}/hybrid_clustering_results.json")
    
    # 4. Human-readable summary
    summary_path = f"{output_dir}/hybrid_clustering_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Weighted Hybrid Clustering Results\n\n")
        f.write(f"Generated: {results['created_at']}\n\n")
        
        f.write("## Parameters\n")
        for key, value in parameters.items():
            f.write(f"- {key}: {value}\n")
        
        f.write("\n## Graph Statistics\n")
        gs = results['graph_stats']
        f.write(f"- Nodes: {gs['n_nodes']:,}\n")
        f.write(f"- Edges: {gs['n_edges']:,}\n") 
        f.write(f"- Density: {gs['edge_density']:.4f}\n")
        f.write(f"- Modularity: {gs['modularity']:.4f}\n")
        
        f.write("\n## Edge Analysis\n")
        ea = results['edge_analysis']
        f.write(f"- Intra-theme edges: {ea['intra_theme_edges']:,} ({ea['intra_theme_edges']/ea['total_edges']:.1%})\n")
        f.write(f"- Cross-theme edges: {ea['cross_theme_edges']:,} ({ea['cross_theme_edges']/ea['total_edges']:.1%})\n")
        align = ea['theme_community_alignment']
        if 'intra_community_theme_purity' in align:
            f.write(f"- Theme purity within communities: {align['intra_community_theme_purity']:.1%}\n")
        if 'theme_cohesion' in align:
            f.write(f"- Theme cohesion: {align['theme_cohesion']:.1%}\n")
        
        f.write("\n## Communities\n")
        f.write(f"Total: {results['clustering_results']['n_communities']} communities, {results['clustering_results']['n_markets']:,} markets\n\n")
        
        # Sort communities by size
        sorted_communities = sorted(community_sizes.items(), key=lambda x: -x[1])
        for comm_id, size in sorted_communities:
            name = community_info.get(comm_id, {}).get('name', f'Community_{comm_id}')
            f.write(f"### Community {comm_id}: {name}\n")
            f.write(f"- Size: {size:,} markets\n")
            
            # Theme breakdown
            themes = community_theme_breakdown.get(comm_id, {})
            if themes:
                f.write("- Themes:\n")
                for theme, count in sorted(themes.items(), key=lambda x: -x[1]):
                    pct = count / size
                    f.write(f"  - {theme}: {count} ({pct:.1%})\n")
            
            # Critique if available
            critique = community_info.get(comm_id, {}).get('critique', '')
            if critique:
                f.write(f"- Issues: {critique}\n")
            
            f.write("\n")
        
        f.write("\n## Explainability Metrics\n")
        exp = results['explainability_metrics']
        f.write(f"- Intra-coherence: {exp['intra_coherence']:.3f}\n")
        f.write(f"- Inter-separation: {exp['inter_separation']:.3f}\n")
        f.write(f"- Explainability score: {exp['explainability_score']:.3f}\n")
        f.write(f"- Median community size: {exp['median_community_size']:.0f}\n")
    
    print(f"  Saved summary: {summary_path}")
    
    return results


def run_full_weighted_hybrid_pipeline(
    correlation_threshold: float = 0.3,
    intra_theme_multiplier: float = 4.0,
    cross_theme_min_corr: float = 0.5,
    resolution: float = 2.0,
    output_dir: str = "data/processed"
) -> Dict:
    """
    Run the complete weighted hybrid clustering pipeline.
    
    This combines correlation-based clustering with LLM theme categories
    to create semantically meaningful and data-driven market communities.
    """
    print("=== Weighted Hybrid Clustering Pipeline ===")
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
    print("Loading data...")
    corr_matrix = pd.read_parquet("data/processed/correlation_matrix.parquet")
    theme_map = load_theme_classifications()
    markets_df = pd.read_parquet("data/processed/markets_filtered.parquet")
    
    print(f"Correlation matrix: {corr_matrix.shape}")
    print(f"Markets with themes: {len(theme_map)}")
    print(f"Filtered markets: {len(markets_df)}")
    
    # Step 2: Build weighted hybrid graph
    G = build_weighted_hybrid_graph(
        corr_matrix, theme_map, correlation_threshold, 
        intra_theme_multiplier, cross_theme_min_corr
    )
    
    # Step 3: Run clustering
    partition = run_weighted_hybrid_clustering(G, resolution)
    
    # Step 4: Analyze results
    edge_stats = analyze_edge_contributions(G, partition)
    
    # Step 5: Save comprehensive results
    results = save_weighted_hybrid_results(
        partition, G, theme_map, edge_stats, 
        markets_df, parameters, output_dir
    )
    
    # Final summary
    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"Communities: {results['clustering_results']['n_communities']}")
    print(f"Markets clustered: {results['clustering_results']['n_markets']:,}")
    print(f"Modularity: {results['graph_stats']['modularity']:.4f}")
    print(f"Explainability score: {results['explainability_metrics']['explainability_score']:.3f}")
    
    edge_analysis = results['edge_analysis']
    theme_align = edge_analysis['theme_community_alignment']
    if 'intra_community_theme_purity' in theme_align:
        print(f"Theme purity: {theme_align['intra_community_theme_purity']:.1%}")
    
    print(f"\nResults saved to: {output_dir}/")
    print("  - hybrid_community_assignments.parquet")  
    print("  - hybrid_community_info.json")
    print("  - hybrid_clustering_results.json")
    print("  - hybrid_clustering_summary.md")
    
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
    
    # Run pipeline with default parameters
    results = run_full_weighted_hybrid_pipeline(
        correlation_threshold=0.3,
        intra_theme_multiplier=4.0,  # 4x weight boost for same-theme edges
        cross_theme_min_corr=0.5,   # Higher bar for cross-theme connections
        resolution=2.0
    )
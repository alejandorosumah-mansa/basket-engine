#!/usr/bin/env python3
"""
Run strict clustering using the filtered correlation matrix.
Uses the high-quality correlation matrix to do proper community detection.
"""

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import json
import os
import openai
from pathlib import Path
from datetime import datetime
from typing import Dict

def build_correlation_graph(corr_matrix: pd.DataFrame, threshold: float = 0.5) -> nx.Graph:
    """
    Build graph from correlation matrix with strict threshold.
    """
    print(f"Building correlation graph with threshold {threshold}...")
    
    G = nx.Graph()
    G.add_nodes_from(corr_matrix.index)
    
    edges_added = 0
    for i, market1 in enumerate(corr_matrix.index):
        for j, market2 in enumerate(corr_matrix.columns):
            if i < j:  # Upper triangle only
                corr = corr_matrix.loc[market1, market2]
                if pd.notna(corr) and abs(corr) > threshold:
                    G.add_edge(market1, market2, weight=abs(corr), correlation=corr)
                    edges_added += 1
    
    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Graph statistics
    if G.number_of_edges() > 0:
        components = list(nx.connected_components(G))
        print(f"Connected components: {len(components)}")
        if components:
            largest_component = max(components, key=len)
            print(f"Largest component: {len(largest_component)} nodes")
    else:
        print("Warning: No edges found with this threshold")
    
    return G

def run_strict_clustering(
    G: nx.Graph,
    min_community_size: int = 10
) -> Dict[str, int]:
    """
    Run Louvain clustering with strict requirements.
    """
    print(f"Running strict clustering (min community size: {min_community_size})...")
    
    if G.number_of_edges() == 0:
        print("No edges found - creating singleton communities")
        return {node: i for i, node in enumerate(G.nodes())}
    
    # Run Louvain clustering
    partition = community_louvain.best_partition(G, weight='weight', random_state=42)
    
    # Get community sizes
    community_sizes = {}
    for node, comm in partition.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    # Filter small communities - assign them to -1 (unassigned) 
    filtered_partition = {}
    large_communities = {comm for comm, size in community_sizes.items() if size >= min_community_size}
    
    for market, comm in partition.items():
        if comm in large_communities:
            filtered_partition[market] = comm
        # Note: we do NOT force assignment - small communities are dropped
    
    # Get final community sizes
    final_sizes = {}
    for comm in filtered_partition.values():
        final_sizes[comm] = final_sizes.get(comm, 0) + 1
    
    print(f"Initial communities: {len(community_sizes)}")
    print(f"Communities ≥{min_community_size} markets: {len(final_sizes)}")
    print(f"Markets in large communities: {len(filtered_partition)}")
    print(f"Markets dropped (small communities): {len(partition) - len(filtered_partition)}")
    
    for comm, size in sorted(final_sizes.items(), key=lambda x: -x[1]):
        print(f"  Community {comm}: {size} markets")
    
    return filtered_partition

def generate_llm_community_names(
    partition: Dict[str, int],
    markets_df: pd.DataFrame,
    model: str = "gpt-4o-mini"
) -> Dict[int, str]:
    """
    Generate community names using LLM based on top markets.
    """
    print("Generating LLM community names...")
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found, using placeholder names")
        communities = {}
        for market, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(market)
        
        return {comm_id: f"Community_{comm_id}" for comm_id in communities.keys()}
    
    client = openai.OpenAI(api_key=api_key)
    
    # Group markets by community
    communities = {}
    for market, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(market)
    
    community_names = {}
    markets_indexed = markets_df.set_index('market_id')
    
    for comm_id, market_ids in communities.items():
        print(f"Naming community {comm_id} ({len(market_ids)} markets)...")
        
        # Get market titles
        comm_markets = []
        for market_id in market_ids:
            if market_id in markets_indexed.index:
                title = markets_indexed.loc[market_id, 'title']
                volume = markets_indexed.loc[market_id, 'volume']
                comm_markets.append((title, volume))
        
        if not comm_markets:
            community_names[comm_id] = f"Community_{comm_id}"
            continue
        
        # Sort by volume and take top markets
        comm_markets.sort(key=lambda x: x[1], reverse=True)
        top_titles = [title for title, _ in comm_markets[:15]]  # Top 15 by volume
        
        titles_text = "\n".join([f"- {title}" for title in top_titles])
        
        prompt = f"""What theme connects these prediction markets? Give a short, investable basket name (e.g., 'US Elections', 'European Politics', 'FIFA World Cup', 'Tech Stocks').

Markets:
{titles_text}

Respond with just the basket name, maximum 4 words:"""
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            name = response.choices[0].message.content.strip()
            name = name.replace('"', '').replace("'", '').strip()
            community_names[comm_id] = name
            
            print(f"  Community {comm_id}: '{name}'")
            
        except Exception as e:
            print(f"  Error naming community {comm_id}: {e}")
            community_names[comm_id] = f"Community_{comm_id}"
    
    return community_names

def save_clustering_results(
    partition: Dict[str, int],
    community_names: Dict[int, str],
    corr_matrix: pd.DataFrame,
    markets_df: pd.DataFrame,
    threshold: float,
    min_community_size: int
):
    """
    Save clustering results in multiple formats.
    """
    print("\nSaving clustering results...")
    
    # Create community assignments DataFrame
    assignments_df = pd.DataFrame([
        {"market_id": market, "community": comm}
        for market, comm in partition.items()
    ]).set_index("market_id")
    
    assignments_df.to_parquet("data/processed/community_assignments_strict.parquet")
    print("Saved community assignments to community_assignments_strict.parquet")
    
    # Save community names
    with open("data/processed/community_names_strict.json", "w") as f:
        json.dump(community_names, f, indent=2)
    print("Saved community names to community_names_strict.json")
    
    # Create detailed results
    communities = {}
    for market, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(market)
    
    markets_indexed = markets_df.set_index('market_id')
    
    detailed_results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "correlation_threshold": threshold,
            "min_community_size": min_community_size,
            "total_markets_in_correlation_matrix": len(corr_matrix),
            "markets_clustered": len(partition)
        },
        "summary": {
            "n_communities": len(communities),
            "total_markets_clustered": len(partition),
            "largest_community_size": max(len(markets) for markets in communities.values()),
            "smallest_community_size": min(len(markets) for markets in communities.values()),
        },
        "communities": {}
    }
    
    for comm_id, market_ids in communities.items():
        comm_name = community_names.get(comm_id, f"Community_{comm_id}")
        
        # Get market details
        market_details = []
        total_volume = 0
        for market_id in market_ids:
            if market_id in markets_indexed.index:
                row = markets_indexed.loc[market_id]
                total_volume += row['volume']
                market_details.append({
                    "market_id": market_id,
                    "title": row['title'],
                    "volume": float(row['volume']),
                    "platform": row['platform']
                })
        
        # Sort by volume
        market_details.sort(key=lambda x: x['volume'], reverse=True)
        
        detailed_results["communities"][str(comm_id)] = {
            "name": comm_name,
            "size": len(market_ids),
            "total_volume": float(total_volume),
            "markets": market_details
        }
    
    with open("data/processed/strict_clustering_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    print("Saved detailed results to strict_clustering_results.json")
    
    return detailed_results

def update_communities_md(detailed_results: dict):
    """
    Update COMMUNITIES.md with the new clustering results.
    """
    print("\nUpdating COMMUNITIES.md...")
    
    content = f"""# Market Communities (Correlation-Based Clustering)

Generated on: {detailed_results['timestamp'][:19]}

## Method
- **Data Quality Filters Applied**: Volume ≥25th percentile, price variance ≥0.005, ≥10 unique prices, ≥30 active trading days
- **Correlation Threshold**: {detailed_results['parameters']['correlation_threshold']} 
- **Min Community Size**: {detailed_results['parameters']['min_community_size']} markets
- **No Forced Assignment**: Markets in small communities are excluded

## Summary
- **Markets in Correlation Matrix**: {detailed_results['parameters']['total_markets_in_correlation_matrix']:,}
- **Markets Successfully Clustered**: {detailed_results['parameters']['markets_clustered']:,}
- **Communities Found**: {detailed_results['summary']['n_communities']}
- **Community Size Range**: {detailed_results['summary']['smallest_community_size']}-{detailed_results['summary']['largest_community_size']} markets

## Communities

"""
    
    # Sort communities by size (largest first)
    communities_sorted = sorted(
        detailed_results['communities'].items(),
        key=lambda x: x[1]['size'],
        reverse=True
    )
    
    for comm_id, comm_data in communities_sorted:
        content += f"### {comm_data['name']}\n"
        content += f"**Size**: {comm_data['size']} markets  \n"
        content += f"**Total Volume**: ${comm_data['total_volume']:,.0f}\n\n"
        
        content += "**Top Markets by Volume**:\n"
        for i, market in enumerate(comm_data['markets'][:10], 1):
            content += f"{i}. {market['title']} (${market['volume']:,.0f})\n"
        
        if len(comm_data['markets']) > 10:
            content += f"... and {len(comm_data['markets']) - 10} more markets\n"
        
        content += "\n"
    
    content += f"""
## Data Quality Impact

The new data quality filters removed {20180 - detailed_results['parameters']['total_markets_in_correlation_matrix']:,} markets ({100 * (20180 - detailed_results['parameters']['total_markets_in_correlation_matrix']) / 20180:.1f}%) that were:
- Low volume (below 25th percentile)
- Flatlined (very low price variance)
- Limited price levels (< 10 unique prices)
- Inactive (< 30 days of actual price changes)

This eliminated fake correlations between unrelated markets and produced much cleaner communities.
"""
    
    with open("COMMUNITIES.md", "w") as f:
        f.write(content)
    
    print("Updated COMMUNITIES.md")

def main():
    print("=== Strict Clustering with Filtered Correlation Matrix ===")
    
    # Load filtered correlation matrix
    print("Loading filtered correlation matrix...")
    corr_matrix = pd.read_parquet("data/processed/correlation_matrix_filtered.parquet")
    print(f"Correlation matrix: {corr_matrix.shape[0]}×{corr_matrix.shape[1]}")
    
    # Load markets data
    markets = pd.read_parquet("data/processed/markets.parquet")
    
    # Build graph with strict threshold
    G = build_correlation_graph(corr_matrix, threshold=0.5)
    
    # Run strict clustering
    partition = run_strict_clustering(G, min_community_size=10)
    
    # Generate community names
    community_names = generate_llm_community_names(partition, markets)
    
    # Save results
    detailed_results = save_clustering_results(
        partition, community_names, corr_matrix, markets,
        threshold=0.5, min_community_size=10
    )
    
    # Update COMMUNITIES.md
    update_communities_md(detailed_results)
    
    # Final summary
    print(f"\n=== FINAL RESULTS ===")
    print(f"Communities found: {len(set(partition.values()))}")
    print(f"Markets clustered: {len(partition):,}")
    print(f"Largest community: {max(detailed_results['communities'].values(), key=lambda x: x['size'])['name']} ({detailed_results['summary']['largest_community_size']} markets)")
    
    print("\nCommunity summary:")
    communities_sorted = sorted(
        detailed_results['communities'].items(),
        key=lambda x: x[1]['size'],
        reverse=True
    )
    
    for comm_id, comm_data in communities_sorted:
        print(f"  {comm_data['name']}: {comm_data['size']} markets")

if __name__ == "__main__":
    main()
"""
Correlation-Based Community Detection

Cluster markets by their RETURN CORRELATIONS instead of factor loadings.
Markets that move together belong in the same basket.

Process:
1. Load prices and compute daily price changes (diff)  
2. Build pairwise correlation matrix (≥20 overlapping days)
3. Use graph-based community detection (Louvain) to find natural clusters
4. LLM labeling of communities based on top markets
5. Characterize communities with factor loadings
6. Output basket definitions for downstream use

This approach clusters by actual co-movement, then uses factors to describe
the baskets rather than construct them.
"""

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import warnings
from typing import Dict, List, Tuple, Optional
import openai
import os
from datetime import datetime

warnings.filterwarnings("ignore")


def compute_explainability_metrics(
    corr_matrix: pd.DataFrame,
    partition: Dict[str, int]
) -> Dict[str, float]:
    """Compute explainability metrics for a given community assignment.
    
    Args:
        corr_matrix: Square correlation matrix
        partition: Dict mapping market_id to community_id
    
    Returns:
        Dict with explainability metrics
    """
    # Group markets by community
    communities = {}
    for market, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(market)
    
    intra_coherences = []
    inter_separations = []
    
    # Compute intra-community coherence (average correlation within communities)
    for comm_id, markets in communities.items():
        if len(markets) < 2:
            continue
        
        # Get correlations within this community
        comm_corr = corr_matrix.loc[markets, markets]
        
        # Get upper triangle (excluding diagonal)
        upper_tri = np.triu(comm_corr.values, k=1)
        valid_corrs = upper_tri[upper_tri != 0]
        
        if len(valid_corrs) > 0:
            intra_coherences.extend(valid_corrs)
    
    # Compute inter-community separation (average correlation between communities)
    community_ids = list(communities.keys())
    for i, comm1 in enumerate(community_ids):
        for j, comm2 in enumerate(community_ids):
            if i < j:
                markets1 = communities[comm1]
                markets2 = communities[comm2]
                
                # Get correlations between communities
                inter_corr = corr_matrix.loc[markets1, markets2]
                valid_corrs = inter_corr.values.flatten()
                valid_corrs = valid_corrs[~np.isnan(valid_corrs)]
                
                if len(valid_corrs) > 0:
                    inter_separations.extend(valid_corrs)
    
    # Compute averages
    avg_intra_coherence = np.mean(np.abs(intra_coherences)) if intra_coherences else 0.0
    avg_inter_separation = np.mean(np.abs(inter_separations)) if inter_separations else 0.0
    
    # Explainability score (higher is better)
    explainability_score = avg_intra_coherence - avg_inter_separation
    
    # Community size statistics
    community_sizes = [len(markets) for markets in communities.values()]
    
    return {
        "intra_coherence": avg_intra_coherence,
        "inter_separation": avg_inter_separation,
        "explainability_score": explainability_score,
        "n_communities": len(communities),
        "min_community_size": min(community_sizes) if community_sizes else 0,
        "max_community_size": max(community_sizes) if community_sizes else 0,
        "median_community_size": np.median(community_sizes) if community_sizes else 0,
    }


def explainability_sweep(
    corr_matrix: pd.DataFrame,
    resolution_range: Tuple[float, float, float] = (0.5, 5.0, 0.25)
) -> pd.DataFrame:
    """Sweep Louvain resolution parameter and compute explainability metrics.
    
    Args:
        corr_matrix: Square correlation matrix
        resolution_range: (min_res, max_res, step) for resolution sweep
    
    Returns:
        DataFrame with resolution and metrics for each tested value
    """
    print("Running explainability sweep...")
    
    # Build graph from correlation matrix
    threshold = 0.3  # Use same threshold as main clustering
    G = build_correlation_graph(corr_matrix, threshold)
    
    if G.number_of_edges() == 0:
        print("Warning: Graph has no edges, cannot perform resolution sweep")
        return pd.DataFrame()
    
    min_res, max_res, step = resolution_range
    resolutions = np.arange(min_res, max_res + step, step)
    
    results = []
    
    for resolution in resolutions:
        print(f"  Testing resolution {resolution:.2f}")
        
        try:
            # Run Louvain with specific resolution
            partition = community_louvain.best_partition(
                G, weight='weight', resolution=resolution, random_state=42
            )
            
            # Compute modularity
            modularity = community_louvain.modularity(partition, G, weight='weight')
            
            # Compute explainability metrics
            explainability_metrics = compute_explainability_metrics(corr_matrix, partition)
            
            # Combine results
            result = {
                "resolution": resolution,
                "modularity": modularity,
                **explainability_metrics
            }
            results.append(result)
            
            print(f"    Communities: {result['n_communities']}, "
                  f"Modularity: {modularity:.3f}, "
                  f"Explainability: {result['explainability_score']:.3f}")
            
        except Exception as e:
            print(f"    Error at resolution {resolution}: {e}")
            continue
    
    return pd.DataFrame(results)


def find_optimal_resolution(
    sweep_results: pd.DataFrame,
    min_communities: int = 20
) -> float:
    """Find optimal resolution that maximizes explainability with minimum communities.
    
    Args:
        sweep_results: DataFrame from explainability_sweep
        min_communities: Minimum number of communities required
    
    Returns:
        Optimal resolution value
    """
    # Filter results with minimum communities
    valid_results = sweep_results[sweep_results['n_communities'] >= min_communities]
    
    if len(valid_results) == 0:
        print(f"Warning: No resolution found with {min_communities}+ communities")
        print(f"Available community counts: {sweep_results['n_communities'].tolist()}")
        # Fall back to maximum communities
        optimal_idx = sweep_results['n_communities'].idxmax()
        return sweep_results.loc[optimal_idx, 'resolution']
    
    # Find resolution with maximum explainability score
    optimal_idx = valid_results['explainability_score'].idxmax()
    optimal_resolution = valid_results.loc[optimal_idx, 'resolution']
    
    optimal_row = valid_results.loc[optimal_idx]
    print(f"Optimal resolution: {optimal_resolution:.2f}")
    print(f"  Communities: {optimal_row['n_communities']}")
    print(f"  Modularity: {optimal_row['modularity']:.3f}")
    print(f"  Explainability: {optimal_row['explainability_score']:.3f}")
    print(f"  Intra-coherence: {optimal_row['intra_coherence']:.3f}")
    print(f"  Inter-separation: {optimal_row['inter_separation']:.3f}")
    
    return optimal_resolution


def run_clustering_at_resolution(
    corr_matrix: pd.DataFrame,
    resolution: float
) -> Dict[str, int]:
    """Run Louvain clustering at specific resolution."""
    print(f"Running clustering at resolution {resolution:.2f}")
    
    # Build graph
    G = build_correlation_graph(corr_matrix, threshold=0.3)
    
    if G.number_of_edges() == 0:
        print("Warning: Graph has no edges, creating singleton communities")
        return {node: i for i, node in enumerate(G.nodes())}
    
    # Run Louvain with specific resolution
    partition = community_louvain.best_partition(
        G, weight='weight', resolution=resolution, random_state=42
    )
    
    # Get community sizes
    community_sizes = {}
    for node, comm in partition.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    print(f"Found {len(community_sizes)} communities:")
    for comm, size in sorted(community_sizes.items(), key=lambda x: -x[1])[:10]:
        print(f"  Community {comm}: {size} markets")
    
    if len(community_sizes) > 10:
        print(f"  ... and {len(community_sizes) - 10} more")
    
    return partition


def generate_llm_community_names_and_critiques(
    partition: Dict[str, int],
    markets_df: pd.DataFrame,
    model: str = "gpt-4o-mini"
) -> Dict[int, Dict[str, str]]:
    """Generate LLM names and critiques for communities."""
    print("Generating LLM community names and critiques...")
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found, using placeholder names")
        communities = {}
        for market, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(market)
        
        return {
            comm_id: {
                "name": f"Community_{comm_id}",
                "critique": "LLM critique not available (missing API key)"
            }
            for comm_id in communities.keys()
        }
    
    client = openai.OpenAI(api_key=api_key)
    
    # Group markets by community
    communities = {}
    for market, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(market)
    
    community_info = {}
    
    for comm_id, market_ids in communities.items():
        print(f"Processing community {comm_id} ({len(market_ids)} markets)...")
        
        # Get market titles
        comm_markets = markets_df[markets_df["market_id"].isin(market_ids)]
        
        if len(comm_markets) == 0:
            community_info[comm_id] = {
                "name": f"Community_{comm_id}",
                "critique": "No market data available"
            }
            continue
        
        # Get all titles for this community
        market_titles = comm_markets["title"].tolist()
        titles_text = "\n".join([f"- {title}" for title in market_titles])
        
        # Create prompt for naming and critique
        prompt = f"""Here are the prediction markets in this basket. Give it a short descriptive name (max 5 words) and list 2-3 potential problems with grouping these markets together (e.g. unrelated markets, missing key markets, temporal mismatch, etc.)

Markets:
{titles_text}

Respond in this exact format:
NAME: [short descriptive name, max 5 words]
PROBLEMS:
- [problem 1]
- [problem 2]
- [problem 3 if applicable]"""
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse response
            lines = response_text.split('\n')
            name = "Community_" + str(comm_id)
            problems = []
            
            for line in lines:
                line = line.strip()
                if line.startswith("NAME:"):
                    name = line.replace("NAME:", "").strip()
                elif line.startswith("-"):
                    problems.append(line[1:].strip())
            
            critique = "\n".join([f"- {problem}" for problem in problems])
            
            community_info[comm_id] = {
                "name": name,
                "critique": critique
            }
            
            print(f"  Community {comm_id}: '{name}'")
            
        except Exception as e:
            print(f"  Error processing community {comm_id}: {e}")
            community_info[comm_id] = {
                "name": f"Community_{comm_id}",
                "critique": f"Error generating critique: {str(e)}"
            }
    
    return community_info


def load_prices(path: str = "data/processed/prices.parquet") -> pd.DataFrame:
    """Load and clean price data."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["market_id", "date"])
    return df


def compute_price_changes(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily price changes (diff, not pct_change) for each market.
    
    Since these are probabilities (0-1), we use absolute differences.
    """
    print("Computing daily price changes...")
    
    changes = prices.copy()
    changes["price_change"] = changes.groupby("market_id")["close_price"].diff()
    changes = changes.dropna(subset=["price_change"])
    
    print(f"Computed changes for {changes['market_id'].nunique()} markets")
    print(f"Total observations: {len(changes):,}")
    
    return changes


def build_correlation_matrix(
    price_changes: pd.DataFrame,
    min_overlapping_days: int = 20,
    min_days_per_market: int = 30,
    target_markets: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Build pairwise correlation matrix for markets with sufficient overlap.
    
    Uses efficient pandas correlation computation instead of pairwise loops.
    
    Args:
        price_changes: DataFrame with market_id, date, price_change
        min_overlapping_days: Minimum overlapping observations for correlation (legacy, ignored for now)
        min_days_per_market: Minimum total days per market to include
        target_markets: Optional list of market_ids to focus on (for efficiency)
        
    Returns:
        correlation_matrix: Square DataFrame with market correlations
        market_stats: Dict with per-market statistics
    """
    print("Building pairwise correlation matrix (efficient method)...")
    
    # Focus on target markets if specified (e.g., existing clustered markets)
    if target_markets:
        price_changes = price_changes[price_changes["market_id"].isin(target_markets)]
        print(f"Focusing on {len(target_markets)} target markets")
    
    # Filter markets with sufficient history
    market_counts = price_changes.groupby("market_id").size()
    eligible_markets = set(market_counts[market_counts >= min_days_per_market].index)
    
    print(f"Eligible markets (≥{min_days_per_market} days): {len(eligible_markets)}")
    
    # Pivot to market × date matrix  
    pivot = price_changes[price_changes["market_id"].isin(eligible_markets)].pivot(
        index="date", columns="market_id", values="price_change"
    )
    
    print(f"Price change matrix: {pivot.shape[0]} days × {pivot.shape[1]} markets")
    
    # Compute correlation matrix using pandas (much faster than pairwise loops)
    print(f"Computing correlation matrix with min_periods={min_overlapping_days}...")
    corr_matrix = pivot.corr(min_periods=min_overlapping_days)
    
    # Keep NaN values as NaN (no data = no relationship, no edge in graph)
    # Do NOT fill with 0 - that would create spurious connections
    
    # Count valid correlations (those with sufficient overlap)
    valid_corrs = (~corr_matrix.isna()).sum().sum() - len(corr_matrix)  # Exclude diagonal
    total_possible = len(corr_matrix) * (len(corr_matrix) - 1)
    if total_possible > 0:
        print(f"Valid correlations (≥{min_overlapping_days} days overlap): {valid_corrs:,} / {total_possible:,} ({valid_corrs/total_possible:.1%})")
    
    # Market statistics
    market_stats = {}
    for market in pivot.columns:
        market_data = pivot[market].dropna()
        market_stats[market] = {
            "n_observations": len(market_data),
            "mean_change": float(market_data.mean()),
            "std_change": float(market_data.std()),
            "valid_correlations": int((~corr_matrix[market].isna()).sum() - 1)  # Exclude self
        }
    
    return corr_matrix, market_stats


def build_correlation_graph(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.3
) -> nx.Graph:
    """Build graph from correlation matrix.
    
    Nodes = markets, edges = |correlation| > threshold
    """
    print(f"Building correlation graph (threshold = {threshold})...")
    
    G = nx.Graph()
    
    # Add all markets as nodes
    G.add_nodes_from(corr_matrix.index)
    
    # Add edges for strong correlations (skip NaN correlations - insufficient overlap)
    added_edges = 0
    for i, market1 in enumerate(corr_matrix.index):
        for j, market2 in enumerate(corr_matrix.columns):
            if i < j:  # Only upper triangle
                corr = corr_matrix.loc[market1, market2]
                if pd.notna(corr) and abs(corr) > threshold:
                    G.add_edge(market1, market2, weight=abs(corr), correlation=corr)
                    added_edges += 1
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Edge density: {G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2):.4f}")
    
    # Basic graph statistics
    if G.number_of_edges() > 0:
        components = list(nx.connected_components(G))
        print(f"Connected components: {len(components)}")
        largest_component = max(components, key=len)
        print(f"Largest component: {len(largest_component)} nodes")
    
    return G


def find_communities_louvain(G: nx.Graph) -> Dict[str, int]:
    """Find communities using Louvain algorithm."""
    print("Running Louvain community detection...")
    
    if G.number_of_edges() == 0:
        print("Warning: Graph has no edges, creating singleton communities")
        return {node: i for i, node in enumerate(G.nodes())}
    
    # Run Louvain community detection
    partition = community_louvain.best_partition(G, weight='weight', random_state=42)
    
    # Get community sizes
    community_sizes = {}
    for node, comm in partition.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    print(f"Found {len(community_sizes)} communities:")
    for comm, size in sorted(community_sizes.items(), key=lambda x: -x[1]):
        print(f"  Community {comm}: {size} markets")
    
    # Check modularity
    modularity = community_louvain.modularity(partition, G, weight='weight')
    print(f"Modularity: {modularity:.4f}")
    
    return partition


def find_communities_spectral(
    corr_matrix: pd.DataFrame,
    n_clusters: Optional[int] = None
) -> Dict[str, int]:
    """Alternative: spectral clustering on correlation matrix."""
    print("Running spectral clustering...")
    
    # Use absolute correlation as similarity
    similarity = corr_matrix.abs().fillna(0)
    
    # Estimate number of clusters if not provided
    if n_clusters is None:
        # Simple heuristic: sqrt(n_markets)
        n_clusters = max(5, min(50, int(np.sqrt(len(corr_matrix)))))
        print(f"Auto-selected {n_clusters} clusters")
    
    # Run spectral clustering
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='discretize'
    )
    
    labels = spectral.fit_predict(similarity.values)
    
    # Convert to partition dict
    partition = dict(zip(corr_matrix.index, labels))
    
    # Get community sizes
    community_sizes = {}
    for comm in labels:
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    print(f"Spectral clustering: {len(community_sizes)} communities:")
    for comm, size in sorted(community_sizes.items(), key=lambda x: -x[1]):
        print(f"  Community {comm}: {size} markets")
    
    return partition


def filter_small_communities(
    partition: Dict[str, int],
    min_size: int = 5
) -> Dict[str, int]:
    """Merge small communities into 'Other/Miscellaneous'."""
    print(f"Filtering communities smaller than {min_size} markets...")
    
    # Count community sizes
    community_sizes = {}
    for market, comm in partition.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    # Find small communities
    small_communities = {comm for comm, size in community_sizes.items() if size < min_size}
    
    if small_communities:
        print(f"Merging {len(small_communities)} small communities into 'Other'")
        
        # Find or create "Other" community ID
        used_ids = set(partition.values())
        other_id = max(used_ids) + 1 if used_ids else 0
        
        # Reassign small communities
        filtered_partition = {}
        for market, comm in partition.items():
            if comm in small_communities:
                filtered_partition[market] = other_id
            else:
                filtered_partition[market] = comm
        
        return filtered_partition
    
    return partition


def label_communities_with_llm(
    partition: Dict[str, int],
    markets_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    top_n: int = 10
) -> Dict[int, str]:
    """Use LLM to label communities based on top markets by volume."""
    print("Labeling communities with LLM...")
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found, skipping LLM labeling")
        return {comm_id: f"Community_{comm_id}" for comm_id in set(partition.values())}
    
    client = openai.OpenAI(api_key=api_key)
    
    # Group markets by community
    communities = {}
    for market, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(market)
    
    community_labels = {}
    
    for comm_id, market_ids in communities.items():
        print(f"Labeling community {comm_id} ({len(market_ids)} markets)...")
        
        # Get top markets by volume
        comm_markets = markets_df[markets_df["market_id"].isin(market_ids)]
        top_markets = comm_markets.nlargest(top_n, "volume")
        
        if len(top_markets) == 0:
            community_labels[comm_id] = f"Community_{comm_id}"
            continue
        
        # Create prompt
        market_titles = "\n".join([f"- {title}" for title in top_markets["title"]])
        
        prompt = f"""
        What theme connects these prediction markets? Give a short, investable basket name (e.g., 'Middle East Risk', 'Fed Policy', 'US Election 2024').
        
        Markets:
        {market_titles}
        
        Respond with just the basket name, maximum 4 words:
        """
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            label = response.choices[0].message.content.strip()
            # Clean up the label
            label = label.replace('"', '').replace("'", '').strip()
            community_labels[comm_id] = label
            
            print(f"  Community {comm_id}: '{label}'")
            
        except Exception as e:
            print(f"  Error labeling community {comm_id}: {e}")
            community_labels[comm_id] = f"Community_{comm_id}"
    
    return community_labels


def characterize_communities_with_factors(
    partition: Dict[str, int],
    factor_loadings_path: str = "data/processed/factor_loadings.parquet"
) -> Dict[int, Dict]:
    """Characterize communities using existing factor loadings."""
    print("Characterizing communities with factor loadings...")
    
    if not Path(factor_loadings_path).exists():
        print(f"Warning: {factor_loadings_path} not found, skipping factor characterization")
        return {}
    
    factor_loadings = pd.read_parquet(factor_loadings_path)
    
    # Group by community
    communities = {}
    for market, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(market)
    
    community_factors = {}
    
    beta_cols = [col for col in factor_loadings.columns if col.startswith("beta_")]
    other_cols = ["R2", "idio_vol", "total_vol"]
    
    for comm_id, market_ids in communities.items():
        # Get factor loadings for markets in this community
        comm_loadings = factor_loadings[factor_loadings.index.isin(market_ids)]
        
        if len(comm_loadings) == 0:
            continue
        
        # Compute mean factor exposures
        mean_betas = comm_loadings[beta_cols].mean()
        mean_stats = comm_loadings[other_cols].mean()
        
        # Find top factor exposures
        top_factors = mean_betas.abs().nlargest(5)
        factor_signature = []
        for factor_name in top_factors.index:
            clean_name = factor_name.replace("beta_", "")
            direction = "+" if mean_betas[factor_name] > 0 else "-"
            factor_signature.append(f"{direction}{clean_name}")
        
        community_factors[comm_id] = {
            "n_markets_with_factors": len(comm_loadings),
            "mean_betas": {col.replace("beta_", ""): float(mean_betas[col]) for col in beta_cols},
            "factor_signature": factor_signature,
            "mean_R2": float(mean_stats["R2"]) if "R2" in mean_stats else None,
            "mean_idio_vol": float(mean_stats["idio_vol"]) if "idio_vol" in mean_stats else None,
            "mean_total_vol": float(mean_stats["total_vol"]) if "total_vol" in mean_stats else None,
        }
    
    return community_factors


def run_correlation_clustering(
    prices_path: str = "data/processed/prices.parquet",
    markets_path: str = "data/processed/markets.parquet", 
    existing_clusters_path: str = "data/processed/cluster_assignments.parquet",
    correlation_threshold: float = 0.3,
    min_overlapping_days: int = 20,
    min_days_per_market: int = 30,
    min_community_size: int = 5,
    method: str = "louvain",  # "louvain" or "spectral"
    output_dir: str = "data/processed",
    use_existing_markets: bool = True
) -> Dict:
    """Full correlation clustering pipeline."""
    
    print("=== Correlation-Based Community Detection ===")
    print(f"Method: {method}")
    print(f"Correlation threshold: {correlation_threshold}")
    print(f"Min overlapping days: {min_overlapping_days}")
    print(f"Min days per market: {min_days_per_market}")
    print(f"Min community size: {min_community_size}")
    
    # Step 1: Load prices and compute changes
    prices = load_prices(prices_path)
    price_changes = compute_price_changes(prices)
    
    # Focus on existing clustered markets for efficiency and comparison
    target_markets = None
    if use_existing_markets and Path(existing_clusters_path).exists():
        existing_clusters = pd.read_parquet(existing_clusters_path)
        target_markets = list(existing_clusters.index)
        print(f"Using existing clustered markets: {len(target_markets)}")
        
        # Show comparison with current factor clustering
        factor_sizes = existing_clusters['cluster'].value_counts().sort_index()
        print("Current factor cluster sizes:")
        for cluster, size in factor_sizes.items():
            pct = size / len(existing_clusters)
            print(f"  Cluster {cluster}: {size} markets ({pct:.1%})")
    else:
        print("Using all markets (this may take a while...)")
    
    # Step 2: Build correlation matrix
    corr_matrix, market_stats = build_correlation_matrix(
        price_changes, 
        min_overlapping_days=min_overlapping_days,
        min_days_per_market=min_days_per_market,
        target_markets=target_markets
    )
    
    # Step 3: Community detection
    if method == "louvain":
        # Build graph and use Louvain
        G = build_correlation_graph(corr_matrix, correlation_threshold)
        partition = find_communities_louvain(G)
    elif method == "spectral":
        # Use spectral clustering
        partition = find_communities_spectral(corr_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Filter small communities
    partition = filter_small_communities(partition, min_community_size)
    
    print(f"\nFinal communities after filtering:")
    community_sizes = {}
    for market, comm in partition.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    for comm, size in sorted(community_sizes.items(), key=lambda x: -x[1]):
        print(f"  Community {comm}: {size} markets")
    
    # Step 4: LLM labeling
    markets_df = pd.read_parquet(markets_path)
    community_labels = label_communities_with_llm(partition, markets_df)
    
    # Step 5: Factor characterization
    community_factors = characterize_communities_with_factors(partition)
    
    # Create final results
    results = {
        "method": method,
        "parameters": {
            "correlation_threshold": correlation_threshold,
            "min_overlapping_days": min_overlapping_days,  
            "min_days_per_market": min_days_per_market,
            "min_community_size": min_community_size
        },
        "n_communities": len(community_sizes),
        "n_markets": len(partition),
        "community_assignments": partition,
        "community_labels": community_labels,
        "community_factors": community_factors,
        "community_sizes": community_sizes,
        "market_stats": market_stats,
        "created_at": datetime.now().isoformat()
    }
    
    # Save outputs
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save correlation matrix
    corr_matrix.to_parquet(f"{output_dir}/correlation_matrix.parquet")
    
    # Save community assignments
    assignments_df = pd.DataFrame([
        {"market_id": market, "community": comm} 
        for market, comm in partition.items()
    ]).set_index("market_id")
    assignments_df.to_parquet(f"{output_dir}/community_assignments.parquet")
    
    # Save community labels
    with open(f"{output_dir}/community_labels.json", "w") as f:
        json.dump(community_labels, f, indent=2)
    
    # Save full results
    with open(f"{output_dir}/correlation_clustering_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nSaved results to {output_dir}/:")
    print(f"  - correlation_matrix.parquet")
    print(f"  - community_assignments.parquet") 
    print(f"  - community_labels.json")
    print(f"  - correlation_clustering_results.json")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Communities found: {len(community_sizes)}")
    print(f"Markets clustered: {len(partition)}")
    print(f"\nCommunity names and sizes:")
    for comm_id in sorted(community_sizes.keys(), key=lambda x: -community_sizes[x]):
        label = community_labels.get(comm_id, f"Community_{comm_id}")
        size = community_sizes[comm_id]
        print(f"  {label}: {size} markets")
    
    if community_factors:
        print(f"\nTop factor exposures by community:")
        for comm_id in sorted(community_factors.keys()):
            label = community_labels.get(comm_id, f"Community_{comm_id}")
            factors = community_factors[comm_id]
            if factors.get("factor_signature"):
                sig = ", ".join(factors["factor_signature"][:3])
                print(f"  {label}: {sig}")
    
    return results


if __name__ == "__main__":
    # Install dependencies first
    import subprocess
    import sys
    
    try:
        import community
        import networkx
    except ImportError:
        print("Installing required dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-louvain", "networkx"])
        import community
        import networkx
    
    run_correlation_clustering()
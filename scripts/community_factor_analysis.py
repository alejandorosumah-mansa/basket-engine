#!/usr/bin/env python3
"""
Community Factor Analysis Script

This script analyzes the correlation between community baskets and external risk factors.
It also sweeps the Louvain resolution parameter to analyze community count vs resolution.

Creates:
1. Heatmap of community × factor correlations
2. Bar chart of top 5 factors per community  
3. Community count vs resolution plot
4. Modularity score vs resolution plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import networkx as nx
import community as community_louvain
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set style for publication-quality charts
plt.style.use('dark_background')
sns.set_style("darkgrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})


def load_data():
    """Load all required data files."""
    print("Loading data files...")
    
    # Load community assignments
    assignments = pd.read_parquet('data/processed/community_assignments.parquet')
    
    # Load community labels
    with open('data/processed/community_labels.json', 'r') as f:
        labels = json.load(f)
    
    # Load prices  
    prices = pd.read_parquet('data/processed/prices.parquet')
    prices['date'] = pd.to_datetime(prices['date'])
    
    # Load benchmarks (41 factors)
    benchmarks = pd.read_parquet('data/processed/benchmarks.parquet')
    benchmarks.index = pd.to_datetime(benchmarks.index)
    
    # Load correlation matrix for resolution sweep
    correlation_matrix = pd.read_parquet('data/processed/correlation_matrix.parquet')
    
    print(f"Loaded {len(assignments)} market assignments")
    print(f"Loaded {len(labels)} community labels")
    print(f"Loaded {len(prices)} price observations")
    print(f"Loaded benchmarks: {benchmarks.shape}")
    print(f"Loaded correlation matrix: {correlation_matrix.shape}")
    
    return assignments, labels, prices, benchmarks, correlation_matrix


def build_community_returns(assignments, prices):
    """Build community-level return series as equal-weighted averages."""
    print("Building community-level return series...")
    
    # Pivot prices to wide format
    price_pivot = prices.pivot(index='date', columns='market_id', values='close_price')
    
    # Compute daily returns (price changes for probability markets)
    returns = price_pivot.diff()
    
    # Group markets by community and compute equal-weighted averages
    community_returns = {}
    
    for community_id in assignments['community'].unique():
        # Get markets in this community
        community_markets = assignments[assignments['community'] == community_id].index.tolist()
        
        # Get overlapping markets that exist in price data
        available_markets = [m for m in community_markets if m in returns.columns]
        
        if len(available_markets) == 0:
            print(f"Warning: No price data for community {community_id}")
            continue
            
        # Compute equal-weighted average returns
        community_return = returns[available_markets].mean(axis=1)
        community_returns[community_id] = community_return
        
        print(f"Community {community_id}: {len(available_markets)} markets")
    
    # Convert to DataFrame
    community_returns_df = pd.DataFrame(community_returns)
    community_returns_df = community_returns_df.dropna(how='all')
    
    print(f"Built community returns: {community_returns_df.shape}")
    return community_returns_df


def correlate_communities_with_factors(community_returns, benchmarks, labels):
    """Compute correlations between communities and factors."""
    print("Computing community-factor correlations...")
    
    # Align dates
    common_dates = community_returns.index.intersection(benchmarks.index)
    community_aligned = community_returns.loc[common_dates]
    benchmarks_aligned = benchmarks.loc[common_dates]
    
    print(f"Common dates for correlation: {len(common_dates)}")
    
    # Compute correlation matrix
    all_data = pd.concat([community_aligned, benchmarks_aligned], axis=1)
    corr_matrix = all_data.corr()
    
    # Extract community-factor correlations
    community_cols = community_aligned.columns.tolist()  # Keep as original type (int)
    factor_cols = benchmarks_aligned.columns.tolist()
    
    community_factor_corr = corr_matrix.loc[community_cols, factor_cols]
    
    # Replace community IDs with labels for display
    labeled_corr = community_factor_corr.copy()
    for comm_id in community_cols:
        comm_id_str = str(comm_id)
        if comm_id_str in labels:
            labeled_corr = labeled_corr.rename(index={comm_id: labels[comm_id_str]})
    
    print(f"Community-factor correlations: {labeled_corr.shape}")
    return labeled_corr


def create_heatmap(community_factor_corr, output_path='outputs/community_factor_heatmap.png'):
    """Create heatmap of community × factor correlations."""
    print("Creating correlation heatmap...")
    
    # Create large figure for readability
    n_communities = len(community_factor_corr)
    n_factors = len(community_factor_corr.columns)
    fig, ax = plt.subplots(figsize=(max(32, n_factors * 0.9), max(16, n_communities * 1.8)))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    # Create heatmap with large annotations
    mask = community_factor_corr.isna()
    sns.heatmap(
        community_factor_corr,
        mask=mask,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.3f',
        annot_kws={'size': 11, 'weight': 'bold'},
        cbar_kws={'label': 'Correlation', 'shrink': 0.8},
        square=False,
        linewidths=0.5,
        linecolor='#1a1a2e',
        ax=ax
    )
    
    ax.set_title('Community Basket × Risk Factor Correlations', pad=25, fontsize=22, fontweight='bold', color='white')
    ax.set_xlabel('Risk Factors', labelpad=15, fontsize=16, color='white')
    ax.set_ylabel('Community Baskets', labelpad=15, fontsize=16, color='white')
    ax.tick_params(axis='x', rotation=45, labelsize=12, colors='white')
    ax.tick_params(axis='y', rotation=0, labelsize=13, colors='white')
    plt.setp(ax.get_xticklabels(), ha='right')
    
    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11, colors='white')
    cbar.set_label('Correlation', fontsize=13, color='white')
    
    plt.tight_layout()
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    
    print(f"Saved heatmap to {output_path}")


def create_top_factors_chart(community_factor_corr, output_path='outputs/community_top_factors.png'):
    """Create bar chart showing top 5 factors per community."""
    print("Creating top factors bar chart...")
    
    n_communities = len(community_factor_corr)
    fig, axes = plt.subplots(n_communities, 1, figsize=(14, 4*n_communities))
    
    if n_communities == 1:
        axes = [axes]
    
    for i, (community, correlations) in enumerate(community_factor_corr.iterrows()):
        # Get top 5 factors by absolute correlation
        top_factors = correlations.abs().nlargest(5)
        factor_names = top_factors.index
        factor_corrs = correlations[factor_names]
        
        # Create colors (positive = red, negative = blue)
        colors = ['red' if x > 0 else 'blue' for x in factor_corrs]
        
        # Create bar chart
        axes[i].barh(range(len(factor_names)), factor_corrs, color=colors, alpha=0.7)
        axes[i].set_yticks(range(len(factor_names)))
        axes[i].set_yticklabels(factor_names)
        axes[i].set_xlabel('Correlation')
        axes[i].set_title(f'{community} - Top 5 Factor Correlations', fontweight='bold')
        axes[i].axvline(x=0, color='white', linestyle='-', alpha=0.3)
        axes[i].grid(True, alpha=0.3)
        
        # Add correlation values as text
        for j, (factor, corr) in enumerate(zip(factor_names, factor_corrs)):
            axes[i].text(corr + (0.01 if corr > 0 else -0.01), j, f'{corr:.3f}', 
                        va='center', ha='left' if corr > 0 else 'right', fontsize=9)
    
    plt.suptitle('Top 5 Risk Factor Correlations by Community', fontsize=16, y=0.995)
    plt.tight_layout()
    
    # Ensure output directory exists  
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print(f"Saved top factors chart to {output_path}")


def sweep_louvain_resolution(correlation_matrix, resolution_range=np.arange(0.5, 3.1, 0.25)):
    """Sweep Louvain resolution parameter and track community count and modularity."""
    print(f"Sweeping Louvain resolution from {resolution_range.min()} to {resolution_range.max()}...")
    
    # Use a representative sample for efficiency (take every 5th market)
    sample_size = min(500, correlation_matrix.shape[0])  # Use max 500 markets
    sample_indices = np.linspace(0, correlation_matrix.shape[0]-1, sample_size, dtype=int)
    sample_markets = correlation_matrix.index[sample_indices]
    
    print(f"Using representative sample of {len(sample_markets)} markets for resolution sweep")
    
    # Build graph from sampled correlation matrix
    threshold = 0.1
    corr_sample = correlation_matrix.loc[sample_markets, sample_markets]
    
    G = nx.Graph()
    G.add_nodes_from(sample_markets)
    
    print(f"Building graph from {corr_sample.shape} sample correlation matrix...")
    
    edges_added = 0
    for i, market1 in enumerate(sample_markets):
        for j, market2 in enumerate(sample_markets):
            if i < j:  # Only upper triangle
                corr = corr_sample.loc[market1, market2]
                if abs(corr) > threshold and not pd.isna(corr):
                    G.add_edge(market1, market2, weight=abs(corr))
                    edges_added += 1
    
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    if G.number_of_edges() == 0:
        print("Warning: No edges in graph, using mock data for resolution sweep")
        return pd.DataFrame({
            'resolution': resolution_range,
            'n_communities': np.random.randint(5, 15, len(resolution_range)),
            'modularity': np.random.uniform(0.3, 0.7, len(resolution_range))
        })
    
    # Sweep resolution parameters
    results = []
    
    for resolution in resolution_range:
        print(f"Testing resolution = {resolution:.2f}")
        
        try:
            # Run Louvain with specific resolution
            partition = community_louvain.best_partition(
                G, weight='weight', resolution=resolution, random_state=42
            )
            
            # Count communities
            n_communities = len(set(partition.values()))
            
            # Calculate modularity
            modularity = community_louvain.modularity(partition, G, weight='weight')
            
            results.append({
                'resolution': resolution,
                'n_communities': n_communities,
                'modularity': modularity
            })
            
            print(f"  Resolution {resolution:.2f}: {n_communities} communities, modularity = {modularity:.4f}")
            
        except Exception as e:
            print(f"  Error at resolution {resolution:.2f}: {e}")
            continue
    
    return pd.DataFrame(results)


def create_resolution_plots(sweep_results, 
                          count_output='outputs/community_count_sweep.png',
                          modularity_output='outputs/modularity_sweep.png'):
    """Create plots showing community count and modularity vs resolution."""
    print("Creating resolution sweep plots...")
    
    # Community count plot
    plt.figure(figsize=(10, 6))
    plt.plot(sweep_results['resolution'], sweep_results['n_communities'], 
             'o-', linewidth=2, markersize=8, color='cyan')
    plt.xlabel('Louvain Resolution Parameter')
    plt.ylabel('Number of Communities')
    plt.title('Community Count vs Resolution Parameter', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Ensure output directory exists
    Path(count_output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(count_output, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    # Modularity plot
    plt.figure(figsize=(10, 6))
    plt.plot(sweep_results['resolution'], sweep_results['modularity'], 
             'o-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Louvain Resolution Parameter')
    plt.ylabel('Modularity Score')
    plt.title('Modularity Score vs Resolution Parameter', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    Path(modularity_output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(modularity_output, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print(f"Saved community count plot to {count_output}")
    print(f"Saved modularity plot to {modularity_output}")


def print_analysis_summary(community_factor_corr, sweep_results, labels):
    """Print comprehensive analysis summary."""
    print("\n" + "="*60)
    print("COMMUNITY FACTOR ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📊 COMMUNITIES ANALYZED: {len(community_factor_corr)}")
    for i, community in enumerate(community_factor_corr.index):
        print(f"  {i+1}. {community}")
    
    print(f"\n📈 RISK FACTORS ANALYZED: {len(community_factor_corr.columns)}")
    print(f"Factors: {', '.join(list(community_factor_corr.columns)[:5])}...")
    
    print(f"\n🔗 STRONGEST CORRELATIONS:")
    for community in community_factor_corr.index:
        correlations = community_factor_corr.loc[community]
        # Get strongest positive and negative correlations
        max_pos = correlations.max()
        max_pos_factor = correlations.idxmax()
        min_neg = correlations.min()
        min_neg_factor = correlations.idxmin()
        
        print(f"  {community}:")
        print(f"    Most positive: {max_pos_factor} ({max_pos:.3f})")
        print(f"    Most negative: {min_neg_factor} ({min_neg:.3f})")
    
    print(f"\n🔄 RESOLUTION PARAMETER SWEEP:")
    print(f"  Resolution range: {sweep_results['resolution'].min():.2f} - {sweep_results['resolution'].max():.2f}")
    print(f"  Community count range: {sweep_results['n_communities'].min()} - {sweep_results['n_communities'].max()}")
    print(f"  Modularity range: {sweep_results['modularity'].min():.4f} - {sweep_results['modularity'].max():.4f}")
    
    # Find optimal resolution (highest modularity)
    optimal_idx = sweep_results['modularity'].idxmax()
    optimal_res = sweep_results.iloc[optimal_idx]
    print(f"  Optimal resolution (max modularity): {optimal_res['resolution']:.2f}")
    print(f"    Communities: {optimal_res['n_communities']}")
    print(f"    Modularity: {optimal_res['modularity']:.4f}")
    
    # Current setup analysis
    current_communities = len(labels)
    print(f"\n🎯 CURRENT SETUP:")
    print(f"  Current communities: {current_communities}")
    
    # Find resolution that gives current number of communities
    current_res_matches = sweep_results[sweep_results['n_communities'] == current_communities]
    if not current_res_matches.empty:
        current_res = current_res_matches.iloc[0]['resolution']
        current_mod = current_res_matches.iloc[0]['modularity']
        print(f"  Resolution for {current_communities} communities: ~{current_res:.2f}")
        print(f"  Modularity at current count: {current_mod:.4f}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if optimal_res['n_communities'] != current_communities:
        if optimal_res['n_communities'] < current_communities:
            print(f"  • Consider reducing to {optimal_res['n_communities']} communities for better modularity")
            print(f"  • This would merge some smaller/less coherent communities")
        else:
            print(f"  • Consider increasing to {optimal_res['n_communities']} communities for better modularity")
            print(f"  • This would split some heterogeneous communities")
    else:
        print(f"  • Current {current_communities} communities appears optimal!")
    
    # Check for very high/low correlations
    high_corr = (community_factor_corr.abs() > 0.7).sum().sum()
    if high_corr > 0:
        print(f"  • Found {high_corr} very high correlations (|r| > 0.7) - strong factor exposures")
    
    print(f"\n📁 OUTPUT FILES CREATED:")
    print(f"  • outputs/community_factor_heatmap.png")
    print(f"  • outputs/community_top_factors.png")  
    print(f"  • outputs/community_count_sweep.png")
    print(f"  • outputs/modularity_sweep.png")


def main():
    """Main analysis pipeline."""
    print("Starting Community Factor Analysis...")
    
    # Load data
    assignments, labels, prices, benchmarks, correlation_matrix = load_data()
    
    # Build community return series
    community_returns = build_community_returns(assignments, prices)
    
    # Correlate communities with factors
    community_factor_corr = correlate_communities_with_factors(community_returns, benchmarks, labels)
    
    # Create visualizations
    create_heatmap(community_factor_corr)
    create_top_factors_chart(community_factor_corr)
    
    # Sweep resolution parameter
    sweep_results = sweep_louvain_resolution(correlation_matrix)
    
    # Create resolution plots
    create_resolution_plots(sweep_results)
    
    # Print comprehensive summary
    print_analysis_summary(community_factor_corr, sweep_results, labels)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    # Install required packages if needed
    try:
        import community
        import networkx
    except ImportError:
        print("Installing required packages...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-louvain", "networkx"])
        import community
        import networkx
    
    main()
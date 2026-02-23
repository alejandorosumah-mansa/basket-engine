#!/usr/bin/env python3
"""
Explainability Sweep Analysis for Community Detection

This script performs a comprehensive analysis of community detection
with explainability metrics and generates publication-quality visualizations.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json

from analysis.correlation_clustering import (
    load_prices, compute_price_changes, build_correlation_matrix,
    explainability_sweep, find_optimal_resolution, run_clustering_at_resolution,
    generate_llm_community_names_and_critiques
)

# Set style for publication-quality plots
plt.style.use('dark_background')
sns.set_palette("viridis")


def create_explainability_sweep_plot(sweep_results: pd.DataFrame, output_path: str):
    """Create multi-panel plot showing sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Community Detection Explainability Sweep', fontsize=16, color='white')
    
    # Plot 1: Number of communities vs resolution
    axes[0, 0].plot(sweep_results['resolution'], sweep_results['n_communities'], 
                   'o-', linewidth=2, markersize=6, color='cyan')
    axes[0, 0].set_xlabel('Louvain Resolution', fontsize=12)
    axes[0, 0].set_ylabel('Number of Communities', fontsize=12)
    axes[0, 0].set_title('Communities vs Resolution', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Min Communities (20)')
    axes[0, 0].legend()
    
    # Plot 2: Modularity vs resolution
    axes[0, 1].plot(sweep_results['resolution'], sweep_results['modularity'], 
                   'o-', linewidth=2, markersize=6, color='orange')
    axes[0, 1].set_xlabel('Louvain Resolution', fontsize=12)
    axes[0, 1].set_ylabel('Modularity Score', fontsize=12)
    axes[0, 1].set_title('Modularity vs Resolution', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Explainability components
    axes[1, 0].plot(sweep_results['resolution'], sweep_results['intra_coherence'], 
                   'o-', linewidth=2, markersize=6, color='lightgreen', label='Intra-Community Coherence')
    axes[1, 0].plot(sweep_results['resolution'], sweep_results['inter_separation'], 
                   'o-', linewidth=2, markersize=6, color='lightcoral', label='Inter-Community Separation')
    axes[1, 0].set_xlabel('Louvain Resolution', fontsize=12)
    axes[1, 0].set_ylabel('Average |Correlation|', fontsize=12)
    axes[1, 0].set_title('Explainability Components', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Overall explainability score
    axes[1, 1].plot(sweep_results['resolution'], sweep_results['explainability_score'], 
                   'o-', linewidth=2, markersize=6, color='gold')
    axes[1, 1].set_xlabel('Louvain Resolution', fontsize=12)
    axes[1, 1].set_ylabel('Explainability Score', fontsize=12)
    axes[1, 1].set_title('Explainability Score (Intra - Inter)', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Highlight optimal resolution if available
    if 'optimal' in sweep_results.columns:
        optimal_rows = sweep_results[sweep_results['optimal'] == True]
        if len(optimal_rows) > 0:
            opt_res = optimal_rows.iloc[0]['resolution']
            for ax in axes.flat:
                ax.axvline(x=opt_res, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Saved explainability sweep plot to {output_path}")


def create_community_sizes_plot(partition: Dict[str, int], community_info: Dict, output_path: str):
    """Create bar chart of community sizes at optimal resolution."""
    # Calculate community sizes
    community_sizes = {}
    for market, comm_id in partition.items():
        community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
    
    # Sort by size
    sorted_communities = sorted(community_sizes.items(), key=lambda x: -x[1])
    
    # Get community names
    comm_ids = [comm for comm, size in sorted_communities]
    sizes = [size for comm, size in sorted_communities]
    names = [community_info.get(comm, {}).get("name", f"Community_{comm}") for comm in comm_ids]
    
    # Limit to top 20 communities for readability
    if len(names) > 20:
        names = names[:20]
        sizes = sizes[:20]
        comm_ids = comm_ids[:20]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 8))
    bars = ax.barh(range(len(names)), sizes, color='lightblue', alpha=0.8)
    
    # Customize plot
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Number of Markets', fontsize=12)
    ax.set_title('Community Sizes at Optimal Resolution', fontsize=14, color='white')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add size labels on bars
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                str(size), va='center', fontsize=9)
    
    # Invert y-axis to show largest at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Saved community sizes plot to {output_path}")


def create_community_report(
    partition: Dict[str, int], 
    community_info: Dict, 
    markets_df: pd.DataFrame,
    output_path: str
):
    """Create markdown report with community names, sizes, and critiques."""
    
    # Calculate community sizes
    community_sizes = {}
    for market, comm_id in partition.items():
        community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
    
    # Sort by size
    sorted_communities = sorted(community_sizes.items(), key=lambda x: -x[1])
    
    # Create report
    report_lines = [
        "# Community Analysis Report",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total Communities: {len(community_sizes)}",
        f"Total Markets: {len(partition)}",
        "",
        "## Communities by Size",
        ""
    ]
    
    for comm_id, size in sorted_communities:
        info = community_info.get(comm_id, {})
        name = info.get("name", f"Community_{comm_id}")
        critique = info.get("critique", "No critique available")
        
        report_lines.extend([
            f"### {name} (Community {comm_id})",
            f"**Size:** {size} markets",
            "",
            "**Top 10 Market Titles:**"
        ])
        
        # Get markets in this community
        community_markets = [market for market, c_id in partition.items() if c_id == comm_id]
        comm_market_data = markets_df[markets_df["market_id"].isin(community_markets)]
        
        if len(comm_market_data) > 0:
            # Sort by volume if available, otherwise just take first 10
            if "volume" in comm_market_data.columns:
                top_markets = comm_market_data.nlargest(10, "volume")
            else:
                top_markets = comm_market_data.head(10)
            
            for _, market in top_markets.iterrows():
                report_lines.append(f"- {market['title']}")
        else:
            report_lines.append("- No market data available")
        
        report_lines.extend([
            "",
            "**Potential Issues:**",
            critique,
            "",
            "---",
            ""
        ])
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Saved community report to {output_path}")


def main():
    """Run the complete explainability sweep analysis."""
    print("=== Comprehensive Community Analysis with Explainability Sweep ===")
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    prices_path = base_dir / "data/processed/prices.parquet"
    markets_path = base_dir / "data/processed/markets.parquet"
    corr_matrix_path = base_dir / "data/processed/correlation_matrix.parquet"
    
    markets_df = pd.read_parquet(markets_path)
    print(f"Loaded {len(markets_df)} markets")
    
    # Load or compute correlation matrix
    if corr_matrix_path.exists():
        print("Loading existing correlation matrix...")
        corr_matrix = pd.read_parquet(corr_matrix_path)
        print(f"Correlation matrix: {corr_matrix.shape}")
    else:
        print("Computing correlation matrix...")
        prices = load_prices(str(prices_path))
        price_changes = compute_price_changes(prices)
        corr_matrix, _ = build_correlation_matrix(price_changes)
        corr_matrix.to_parquet(corr_matrix_path)
        print(f"Saved correlation matrix: {corr_matrix.shape}")
    
    # Step 2: Run explainability sweep
    print("\n2. Running explainability sweep...")
    sweep_results = explainability_sweep(
        corr_matrix, 
        resolution_range=(0.5, 5.0, 0.25)
    )
    
    if len(sweep_results) == 0:
        print("Error: No sweep results generated")
        return
    
    print(f"Tested {len(sweep_results)} resolution values")
    
    # Step 3: Find optimal resolution
    print("\n3. Finding optimal resolution...")
    optimal_resolution = find_optimal_resolution(sweep_results, min_communities=20)
    
    # Mark optimal resolution in results
    sweep_results['optimal'] = sweep_results['resolution'] == optimal_resolution
    
    # Step 4: Run clustering at optimal resolution
    print("\n4. Running clustering at optimal resolution...")
    optimal_partition = run_clustering_at_resolution(corr_matrix, optimal_resolution)
    
    # Step 5: Generate LLM names and critiques
    print("\n5. Generating LLM community names and critiques...")
    community_info = generate_llm_community_names_and_critiques(
        optimal_partition, markets_df
    )
    
    # Step 6: Create visualizations
    print("\n6. Creating visualizations...")
    
    # Explainability sweep plot
    create_explainability_sweep_plot(
        sweep_results, 
        str(output_dir / "explainability_sweep.png")
    )
    
    # Community sizes plot
    create_community_sizes_plot(
        optimal_partition, 
        community_info,
        str(output_dir / "optimal_communities.png")
    )
    
    # Community report
    create_community_report(
        optimal_partition,
        community_info, 
        markets_df,
        str(output_dir / "community_names_critique.md")
    )
    
    # Step 7: Save final results
    print("\n7. Saving results...")
    
    # Save sweep results
    sweep_results.to_csv(output_dir / "explainability_sweep_results.csv", index=False)
    
    # Save optimal partition
    optimal_assignments = pd.DataFrame([
        {"market_id": market, "community": comm} 
        for market, comm in optimal_partition.items()
    ]).set_index("market_id")
    optimal_assignments.to_parquet(output_dir / "optimal_community_assignments.parquet")
    
    # Save community info
    with open(output_dir / "optimal_community_info.json", "w") as f:
        json.dump(community_info, f, indent=2)
    
    # Step 8: Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Find optimal row
    optimal_row = sweep_results[sweep_results['optimal']].iloc[0]
    
    print(f"Optimal Resolution: {optimal_resolution:.2f}")
    print(f"Reason: Maximizes explainability with 20+ communities")
    print()
    print(f"Number of Communities: {optimal_row['n_communities']}")
    print(f"Modularity Score: {optimal_row['modularity']:.3f}")
    print(f"Explainability Score: {optimal_row['explainability_score']:.3f}")
    print(f"  - Intra-community coherence: {optimal_row['intra_coherence']:.3f}")
    print(f"  - Inter-community separation: {optimal_row['inter_separation']:.3f}")
    print()
    
    # Community size statistics
    community_sizes = {}
    for market, comm_id in optimal_partition.items():
        community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
    
    sorted_communities = sorted(community_sizes.items(), key=lambda x: -x[1])
    
    print("Top 5 Largest Communities:")
    for i, (comm_id, size) in enumerate(sorted_communities[:5]):
        name = community_info.get(comm_id, {}).get("name", f"Community_{comm_id}")
        print(f"  {i+1}. {name}: {size} markets")
    
    print()
    print("Top 5 Smallest Communities:")
    for i, (comm_id, size) in enumerate(sorted_communities[-5:]):
        name = community_info.get(comm_id, {}).get("name", f"Community_{comm_id}")
        print(f"  {i+1}. {name}: {size} markets")
    
    print()
    print(f"Median Community Size: {optimal_row['median_community_size']:.1f} markets")
    print(f"Size Range: {optimal_row['min_community_size']} - {optimal_row['max_community_size']} markets")
    
    print()
    print("Output Files Generated:")
    print(f"  - {output_dir}/explainability_sweep.png")
    print(f"  - {output_dir}/optimal_communities.png") 
    print(f"  - {output_dir}/community_names_critique.md")
    print(f"  - {output_dir}/explainability_sweep_results.csv")
    print(f"  - {output_dir}/optimal_community_assignments.parquet")
    print(f"  - {output_dir}/optimal_community_info.json")
    
    print("\n✅ Comprehensive community analysis complete!")


if __name__ == "__main__":
    main()
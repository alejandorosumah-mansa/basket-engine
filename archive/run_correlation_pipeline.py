#!/usr/bin/env python3
"""
Full correlation-based basket construction pipeline.

Replaces the factor-loading clustering approach with correlation-based community detection.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis.correlation_clustering import run_correlation_clustering
from construction.correlation_baskets import compare_basket_methods


def run_full_pipeline():
    """Run the complete correlation-based basket construction pipeline."""
    
    print("="*80)
    print("CORRELATION-BASED PREDICTION MARKET BASKET CONSTRUCTION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Run correlation clustering
    print("STEP 1: Running correlation-based community detection...")
    print("-" * 60)
    
    clustering_results = run_correlation_clustering(
        correlation_threshold=0.3,
        min_overlapping_days=20,
        min_days_per_market=30,
        min_community_size=5,
        method="louvain"
    )
    
    print("\n" + "="*60)
    print("CORRELATION CLUSTERING RESULTS")
    print("="*60)
    
    print(f"Communities found: {clustering_results['n_communities']}")
    print(f"Markets clustered: {clustering_results['n_markets']}")
    print(f"Method: {clustering_results['method']}")
    print(f"Parameters: {clustering_results['parameters']}")
    
    print("\nCommunity Summary:")
    community_labels = clustering_results.get('community_labels', {})
    community_sizes = clustering_results.get('community_sizes', {})
    
    # Sort by size (descending)
    for comm_id in sorted(community_sizes.keys(), key=lambda x: -community_sizes[x]):
        label = community_labels.get(comm_id, f"Community_{comm_id}")
        size = community_sizes[comm_id]
        pct = size / clustering_results['n_markets'] * 100
        print(f"  {label}: {size} markets ({pct:.1f}%)")
    
    # Step 2: Run basket construction comparison
    print("\n" + "="*60)
    print("STEP 2: Basket construction and comparison...")
    print("-" * 60)
    
    correlation_results, factor_results = compare_basket_methods()
    
    # Step 3: Generate final summary
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    
    if correlation_results and factor_results:
        corr_metrics = correlation_results["metrics"]
        factor_metrics = factor_results["metrics"]
        
        print("\nCORRELATION-BASED BASKETS:")
        print(f"  Baskets: {corr_metrics['n_baskets']}")
        print(f"  Annual Return: {corr_metrics['annual_return']:.2%}")
        print(f"  Annual Volatility: {corr_metrics['annual_vol']:.2%}")
        print(f"  Sharpe Ratio: {corr_metrics['sharpe']:.3f}")
        print(f"  Max Drawdown: {corr_metrics['max_drawdown']:.2%}")
        print(f"  Max Basket Correlation: {corr_metrics['max_basket_correlation']:.3f}")
        print(f"  Mean Basket Correlation: {corr_metrics['mean_basket_correlation']:.3f}")
        
        print("\nFACTOR-BASED BASKETS:")
        print(f"  Baskets: {factor_metrics['n_baskets']}")
        print(f"  Annual Return: {factor_metrics['annual_return']:.2%}")
        print(f"  Annual Volatility: {factor_metrics['annual_vol']:.2%}")
        print(f"  Sharpe Ratio: {factor_metrics['sharpe']:.3f}")
        print(f"  Max Drawdown: {factor_metrics['max_drawdown']:.2%}")
        print(f"  Max Basket Correlation: {factor_metrics['max_basket_correlation']:.3f}")
        print(f"  Mean Basket Correlation: {factor_metrics['mean_basket_correlation']:.3f}")
        
        print("\nIMPROVEMENTS (Correlation vs Factor):")
        vol_improvement = (factor_metrics['annual_vol'] - corr_metrics['annual_vol']) / factor_metrics['annual_vol']
        dd_improvement = (factor_metrics['max_drawdown'] - corr_metrics['max_drawdown']) / abs(factor_metrics['max_drawdown'])
        
        print(f"  Volatility Reduction: {vol_improvement:.1%}")
        print(f"  Max Drawdown Reduction: {dd_improvement:.1%}")
        print(f"  Sharpe Difference: {corr_metrics['sharpe'] - factor_metrics['sharpe']:.3f}")
        
        if dd_improvement > 0.5:
            print(f"  🎯 SUCCESS: >50% drawdown reduction achieved!")
        if vol_improvement > 0.3:
            print(f"  🎯 SUCCESS: >30% volatility reduction achieved!")
    
    # Final outputs summary
    print("\n" + "="*60)
    print("OUTPUT FILES GENERATED")
    print("="*60)
    
    output_files = [
        "data/processed/correlation_matrix.parquet",
        "data/processed/community_assignments.parquet", 
        "data/processed/community_labels.json",
        "data/processed/correlation_clustering_results.json",
        "data/outputs/basket_returns.csv",
        "data/outputs/basket_correlations.csv",
        "data/outputs/basket_compositions.json",
        "data/outputs/portfolio_nav.csv"
    ]
    
    for file_path in output_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return clustering_results, correlation_results, factor_results


if __name__ == "__main__":
    results = run_full_pipeline()
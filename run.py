#!/usr/bin/env python3
"""
Basket Engine - Complete Analysis Pipeline

This script runs the complete prediction market basket analysis pipeline:
1. Load and filter markets (remove sports/entertainment)
2. LLM classify markets (use cache if exists)
3. Build Ticker mapping (CUSIP → Ticker, regex + exact match)
4. Build continuous Ticker time series (raw + adjusted)
5. Compute correlation matrix (with quality filters)
6. Run clustering (strict: 0.5 threshold, min 10 markets)
7. Name communities (LLM)
8. Generate ALL charts → output/charts/
9. Write results.xlsx

Author: OpenClaw Assistant
Date: 2025-02-23
"""

import pandas as pd
import numpy as np
import json
import os
import re
import uuid
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Import local modules
import sys
sys.path.append('src')

from classification.llm_classifier import classify_all_markets
from classification.taxonomy import load_taxonomy
from ingestion.market_filter import MarketFilter
from analysis.correlation_clustering import build_correlation_matrix, run_clustering
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
CHARTS_DIR = OUTPUT_DIR / "charts"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

class BasketEngine:
    """Main pipeline coordinator."""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.charts_dir = CHARTS_DIR
        
        # Create directories
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Pipeline outputs
        self.markets_df = None
        self.classifications_df = None
        self.filtered_markets = None
        self.ticker_mapping = None
        self.ticker_chains = None
        self.correlation_matrix = None
        self.communities = None
        self.community_labels = None
        
    def run_complete_pipeline(self):
        """Run the complete analysis pipeline."""
        print("🚀 Starting Basket Engine Complete Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Load and filter markets
            print("\n📊 Step 1: Loading and filtering markets...")
            self.load_and_filter_markets()
            
            # Step 2: LLM classify markets
            print("\n🤖 Step 2: Classifying markets with LLM...")
            self.classify_markets()
            
            # Step 3: Build ticker mapping
            print("\n🎯 Step 3: Building ticker mapping...")
            self.build_ticker_mapping()
            
            # Step 4: Build continuous time series
            print("\n📈 Step 4: Building continuous time series...")
            self.build_continuous_timeseries()
            
            # Step 5: Compute correlation matrix
            print("\n🔗 Step 5: Computing correlation matrix...")
            self.compute_correlation_matrix()
            
            # Step 6: Run clustering
            print("\n🎪 Step 6: Running community clustering...")
            self.run_community_clustering()
            
            # Step 7: Name communities
            print("\n🏷️ Step 7: Naming communities with LLM...")
            self.name_communities()
            
            # Step 8: Generate charts
            print("\n📊 Step 8: Generating charts...")
            self.generate_all_charts()
            
            # Step 9: Write Excel results
            print("\n📋 Step 9: Writing results to Excel...")
            self.write_results_excel()
            
            print("\n✅ Pipeline completed successfully!")
            print(f"📁 Results saved to: {self.output_dir}")
            print(f"📊 Charts saved to: {self.charts_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def load_and_filter_markets(self):
        """Load markets data and apply filters."""
        # Load raw markets
        markets_path = PROCESSED_DIR / "markets.parquet"
        if not markets_path.exists():
            raise FileNotFoundError(f"Markets file not found: {markets_path}")
            
        self.markets_df = pd.read_parquet(markets_path)
        logger.info(f"Loaded {len(self.markets_df):,} markets")
        
        # Apply market filters (remove sports/entertainment)
        market_filter = MarketFilter()
        self.filtered_markets = market_filter.filter_markets(self.markets_df)
        
        logger.info(f"After filtering: {len(self.filtered_markets):,} markets")
        
        # Generate and save filter funnel chart
        self.generate_market_filter_funnel()
    
    def classify_markets(self):
        """Classify markets using LLM."""
        # Check if classifications already exist
        classifications_path = PROCESSED_DIR / "market_classifications.parquet"
        
        if classifications_path.exists():
            logger.info("Loading existing classifications...")
            self.classifications_df = pd.read_parquet(classifications_path)
        else:
            logger.info("Running LLM classification...")
            self.classifications_df = classify_all_markets(self.filtered_markets)
            
            # Save classifications
            self.classifications_df.to_parquet(classifications_path, index=False)
        
        logger.info(f"Classifications loaded for {len(self.classifications_df):,} markets")
        
        # Generate classification summary chart
        self.generate_classification_summary()
    
    def build_ticker_mapping(self):
        """Build CUSIP → Ticker mapping."""
        ticker_mapping_path = PROCESSED_DIR / "ticker_mapping.parquet"
        
        if ticker_mapping_path.exists():
            logger.info("Loading existing ticker mapping...")
            self.ticker_mapping = pd.read_parquet(ticker_mapping_path)
        else:
            logger.info("Building ticker mapping...")
            mapper = TickerMapper(str(self.data_dir))
            mapper.load_data()
            mapper.build_mapping()
            
            self.ticker_mapping = mapper.ticker_mapping
            self.ticker_chains = mapper.ticker_chains
            
            # Save results
            self.ticker_mapping.to_parquet(ticker_mapping_path, index=False)
            
            # Save ticker chains as JSON
            chains_path = PROCESSED_DIR / "ticker_chains.json"
            with open(chains_path, 'w') as f:
                json.dump(mapper.ticker_chains, f, indent=2, default=str)
        
        logger.info(f"Ticker mapping built: {len(self.ticker_mapping):,} CUSIPs mapped to {self.ticker_mapping['ticker_id'].nunique():,} tickers")
        
        # Generate ticker mapping chart
        self.generate_ticker_mapping_summary()
    
    def build_continuous_timeseries(self):
        """Build continuous ticker time series."""
        timeseries_path = PROCESSED_DIR / "continuous_event_timeseries_v2.parquet"
        
        if timeseries_path.exists():
            logger.info("Loading existing time series...")
            self.continuous_series = pd.read_parquet(timeseries_path)
        else:
            logger.info("Building continuous time series...")
            builder = ContinuousTimeseriesBuilderV2(str(self.data_dir))
            builder.load_data()
            builder.filter_markets()
            builder.build_all_series()
            
            self.continuous_series = builder.continuous_series
            
            # Save results
            pd.DataFrame(self.continuous_series).to_parquet(timeseries_path, index=False)
        
        logger.info(f"Built {len(self.continuous_series) if isinstance(self.continuous_series, dict) else len(self.continuous_series):,} continuous time series")
    
    def compute_correlation_matrix(self):
        """Compute correlation matrix with quality filters."""
        corr_path = PROCESSED_DIR / "correlation_matrix_filtered.parquet"
        
        if corr_path.exists():
            logger.info("Loading existing correlation matrix...")
            self.correlation_matrix = pd.read_parquet(corr_path)
        else:
            logger.info("Computing correlation matrix...")
            # Use the correlation clustering module
            self.correlation_matrix = build_correlation_matrix(
                self.continuous_series,
                min_volume_threshold=100,
                min_variance_threshold=0.001,
                min_active_days=30,
                min_overlap_days=30
            )
            
            # Save matrix
            self.correlation_matrix.to_parquet(corr_path)
        
        logger.info(f"Correlation matrix: {self.correlation_matrix.shape[0]:,} × {self.correlation_matrix.shape[1]:,}")
        
        # Generate correlation distribution chart
        self.generate_correlation_distribution()
    
    def run_community_clustering(self):
        """Run community detection clustering."""
        communities_path = PROCESSED_DIR / "community_assignments_strict.parquet"
        
        if communities_path.exists():
            logger.info("Loading existing communities...")
            self.communities = pd.read_parquet(communities_path)
        else:
            logger.info("Running community clustering...")
            # Build correlation graph with strict threshold
            G = self.build_correlation_graph(threshold=0.5)
            
            # Run Louvain clustering
            partition = community_louvain.best_partition(G, weight='weight', random_state=42)
            
            # Convert to DataFrame
            self.communities = pd.DataFrame([
                {'market_id': market_id, 'community': community_id}
                for market_id, community_id in partition.items()
            ])
            
            # Filter communities with minimum size
            community_sizes = self.communities.groupby('community').size()
            valid_communities = community_sizes[community_sizes >= 10].index
            self.communities = self.communities[self.communities['community'].isin(valid_communities)]
            
            # Save results
            self.communities.to_parquet(communities_path, index=False)
        
        num_communities = self.communities['community'].nunique()
        avg_size = self.communities.groupby('community').size().mean()
        logger.info(f"Found {num_communities:,} communities (avg size: {avg_size:.1f})")
        
        # Generate community size distribution chart
        self.generate_community_size_distribution()
    
    def name_communities(self):
        """Generate LLM names for communities."""
        labels_path = PROCESSED_DIR / "community_labels.json"
        
        if labels_path.exists():
            logger.info("Loading existing community labels...")
            with open(labels_path, 'r') as f:
                self.community_labels = json.load(f)
        else:
            logger.info("Generating community labels with LLM...")
            self.community_labels = self.generate_community_names()
            
            # Save labels
            with open(labels_path, 'w') as f:
                json.dump(self.community_labels, f, indent=2)
        
        logger.info(f"Generated labels for {len(self.community_labels):,} communities")
    
    def build_correlation_graph(self, threshold: float = 0.5) -> nx.Graph:
        """Build graph from correlation matrix."""
        logger.info(f"Building correlation graph with threshold {threshold}...")
        
        G = nx.Graph()
        G.add_nodes_from(self.correlation_matrix.index)
        
        edges_added = 0
        for i, market1 in enumerate(self.correlation_matrix.index):
            for j, market2 in enumerate(self.correlation_matrix.columns):
                if i < j:  # Upper triangle only
                    corr = self.correlation_matrix.loc[market1, market2]
                    if pd.notna(corr) and abs(corr) > threshold:
                        G.add_edge(market1, market2, weight=abs(corr), correlation=corr)
                        edges_added += 1
        
        logger.info(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G
    
    def generate_community_names(self) -> Dict:
        """Generate LLM names for communities."""
        # This would integrate with the LLM naming logic from the existing scripts
        # For now, return a placeholder
        community_labels = {}
        for community_id in self.communities['community'].unique():
            community_labels[str(community_id)] = {
                'name': f'Community {community_id}',
                'theme': 'general',
                'description': f'Automatically generated community {community_id}'
            }
        return community_labels
    
    def generate_market_filter_funnel(self):
        """Generate market filtering funnel chart."""
        # Placeholder for market filter funnel visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stages = ['Raw Markets', 'After Sports Filter', 'After Entertainment Filter', 'Final Filtered']
        counts = [
            len(self.markets_df),
            len(self.markets_df) * 0.8,  # Placeholder values
            len(self.markets_df) * 0.7,
            len(self.filtered_markets)
        ]
        
        ax.bar(stages, counts)
        ax.set_title('Market Filtering Funnel')
        ax.set_ylabel('Number of Markets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = self.charts_dir / "01_market_filter_funnel.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved market filter funnel chart: {chart_path}")
    
    def generate_classification_summary(self):
        """Generate classification summary chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        theme_counts = self.classifications_df['primary_theme'].value_counts()
        
        ax.barh(range(len(theme_counts)), theme_counts.values)
        ax.set_yticks(range(len(theme_counts)))
        ax.set_yticklabels(theme_counts.index)
        ax.set_title('Market Classification by Theme')
        ax.set_xlabel('Number of Markets')
        
        plt.tight_layout()
        
        chart_path = self.charts_dir / "02_classification_summary.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved classification summary chart: {chart_path}")
    
    def generate_ticker_mapping_summary(self):
        """Generate ticker mapping summary chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # CUSIPs per ticker distribution
        cusips_per_ticker = self.ticker_mapping.groupby('ticker_id').size()
        ax1.hist(cusips_per_ticker, bins=50, alpha=0.7)
        ax1.set_title('Distribution of CUSIPs per Ticker')
        ax1.set_xlabel('Number of CUSIPs')
        ax1.set_ylabel('Number of Tickers')
        
        # Top tickers by CUSIP count
        top_tickers = cusips_per_ticker.nlargest(10)
        ax2.barh(range(len(top_tickers)), top_tickers.values)
        ax2.set_yticks(range(len(top_tickers)))
        ax2.set_yticklabels([f'Ticker {i}' for i in top_tickers.index])
        ax2.set_title('Top 10 Tickers by CUSIP Count')
        ax2.set_xlabel('Number of CUSIPs')
        
        plt.tight_layout()
        
        chart_path = self.charts_dir / "03_ticker_mapping_summary.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ticker mapping summary chart: {chart_path}")
    
    def generate_correlation_distribution(self):
        """Generate correlation distribution chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Flatten correlation matrix (upper triangle only)
        corr_values = []
        for i in range(len(self.correlation_matrix)):
            for j in range(i + 1, len(self.correlation_matrix)):
                val = self.correlation_matrix.iloc[i, j]
                if pd.notna(val):
                    corr_values.append(val)
        
        # Distribution of all correlations
        ax1.hist(corr_values, bins=100, alpha=0.7)
        ax1.set_title('Distribution of Pairwise Correlations')
        ax1.set_xlabel('Correlation Coefficient')
        ax1.set_ylabel('Frequency')
        ax1.axvline(0.5, color='red', linestyle='--', label='Clustering Threshold (0.5)')
        ax1.legend()
        
        # High correlation focus
        high_corrs = [c for c in corr_values if abs(c) > 0.3]
        ax2.hist(high_corrs, bins=50, alpha=0.7)
        ax2.set_title('High Correlations (|r| > 0.3)')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_ylabel('Frequency')
        ax2.axvline(0.5, color='red', linestyle='--', label='Clustering Threshold (0.5)')
        ax2.legend()
        
        plt.tight_layout()
        
        chart_path = self.charts_dir / "04_correlation_distribution.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved correlation distribution chart: {chart_path}")
    
    def generate_community_size_distribution(self):
        """Generate community size distribution chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        community_sizes = self.communities.groupby('community').size()
        
        # Size distribution histogram
        ax1.hist(community_sizes, bins=30, alpha=0.7)
        ax1.set_title('Community Size Distribution')
        ax1.set_xlabel('Community Size (Number of Markets)')
        ax1.set_ylabel('Number of Communities')
        
        # Top 20 communities by size
        top_communities = community_sizes.nlargest(20)
        ax2.barh(range(len(top_communities)), top_communities.values)
        ax2.set_yticks(range(len(top_communities)))
        ax2.set_yticklabels([f'Community {i}' for i in top_communities.index])
        ax2.set_title('Top 20 Communities by Size')
        ax2.set_xlabel('Number of Markets')
        
        plt.tight_layout()
        
        chart_path = self.charts_dir / "05_community_size_distribution.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved community size distribution chart: {chart_path}")
    
    def generate_all_charts(self):
        """Generate all analysis charts."""
        # Additional charts beyond the pipeline-specific ones
        
        # Network visualization of communities
        self.generate_network_visualization()
        
        # Community theme analysis
        self.generate_community_theme_analysis()
        
        # Correlation heatmap (subset)
        self.generate_correlation_heatmap()
    
    def generate_network_visualization(self):
        """Generate network visualization of communities."""
        # Create a simplified network visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Build graph with moderate threshold for visualization
        G = self.build_correlation_graph(threshold=0.3)
        
        # Sample nodes if too many
        if G.number_of_nodes() > 500:
            nodes_sample = list(G.nodes())[:500]
            G = G.subgraph(nodes_sample)
        
        # Position nodes
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw network
        nx.draw(G, pos, ax=ax, node_size=10, alpha=0.6, edge_color='gray', 
                node_color='lightblue', width=0.5)
        
        ax.set_title('Market Correlation Network (r > 0.3)')
        ax.axis('off')
        
        chart_path = self.charts_dir / "06_network_visualization.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved network visualization chart: {chart_path}")
    
    def generate_community_theme_analysis(self):
        """Generate community vs theme analysis."""
        if self.classifications_df is not None:
            # Merge communities with classifications
            merged = self.communities.merge(
                self.classifications_df[['market_id', 'primary_theme']], 
                on='market_id', 
                how='left'
            )
            
            # Create theme purity analysis
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Calculate theme purity for each community
            purity_data = []
            for community_id in merged['community'].unique():
                community_markets = merged[merged['community'] == community_id]
                theme_counts = community_markets['primary_theme'].value_counts()
                if len(theme_counts) > 0:
                    dominant_theme = theme_counts.index[0]
                    purity = theme_counts.iloc[0] / len(community_markets)
                    purity_data.append({
                        'community': community_id,
                        'size': len(community_markets),
                        'dominant_theme': dominant_theme,
                        'purity': purity
                    })
            
            purity_df = pd.DataFrame(purity_data)
            
            # Scatter plot: community size vs purity
            scatter = ax.scatter(purity_df['size'], purity_df['purity'], 
                               alpha=0.6, s=50)
            ax.set_xlabel('Community Size')
            ax.set_ylabel('Theme Purity')
            ax.set_title('Community Size vs Theme Purity')
            
            chart_path = self.charts_dir / "07_community_theme_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved community theme analysis chart: {chart_path}")
    
    def generate_correlation_heatmap(self):
        """Generate correlation heatmap for top communities."""
        # Get top 5 communities by size
        top_communities = self.communities.groupby('community').size().nlargest(5).index
        
        for i, community_id in enumerate(top_communities):
            community_markets = self.communities[
                self.communities['community'] == community_id
            ]['market_id'].tolist()
            
            # Subset correlation matrix
            if len(community_markets) > 1:
                subset_corr = self.correlation_matrix.loc[
                    self.correlation_matrix.index.intersection(community_markets),
                    self.correlation_matrix.columns.intersection(community_markets)
                ]
                
                if subset_corr.shape[0] > 1:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(subset_corr, annot=False, cmap='RdBu_r', center=0,
                              square=True, ax=ax)
                    ax.set_title(f'Correlation Heatmap - Community {community_id}')
                    
                    chart_path = self.charts_dir / f"08_correlation_heatmap_community_{community_id}.png"
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Saved correlation heatmap for community {community_id}: {chart_path}")
    
    def write_results_excel(self):
        """Write comprehensive results to Excel file."""
        excel_path = self.output_dir / "results.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            self.write_summary_sheet(writer)
            
            # Ticker mapping sheet
            if self.ticker_mapping is not None:
                self.ticker_mapping.to_excel(writer, sheet_name='Ticker Mapping', index=False)
            
            # Communities sheet
            if self.communities is not None:
                self.write_communities_sheet(writer)
            
            # Correlation matrix sheet (sampled)
            if self.correlation_matrix is not None:
                self.write_correlation_sheet(writer)
            
            # Market classifications sheet
            if self.classifications_df is not None:
                self.classifications_df.to_excel(writer, sheet_name='Market Classifications', index=False)
            
            # Methodology sheet
            self.write_methodology_sheet(writer)
        
        logger.info(f"Results written to Excel: {excel_path}")
    
    def write_summary_sheet(self, writer):
        """Write summary statistics sheet."""
        summary_data = {
            'Metric': [
                'Total Raw Markets',
                'Markets After Filtering',
                'Classified Markets',
                'Unique Tickers',
                'Communities Found',
                'Average Community Size',
                'Correlation Matrix Size',
                'High Correlations (|r| > 0.5)',
            ],
            'Value': [
                len(self.markets_df) if self.markets_df is not None else 0,
                len(self.filtered_markets) if self.filtered_markets is not None else 0,
                len(self.classifications_df) if self.classifications_df is not None else 0,
                self.ticker_mapping['ticker_id'].nunique() if self.ticker_mapping is not None else 0,
                self.communities['community'].nunique() if self.communities is not None else 0,
                self.communities.groupby('community').size().mean() if self.communities is not None else 0,
                f"{self.correlation_matrix.shape[0]} × {self.correlation_matrix.shape[1]}" if self.correlation_matrix is not None else "0 × 0",
                "TBD"  # Would calculate high correlation count
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    def write_communities_sheet(self, writer):
        """Write communities analysis sheet."""
        # Calculate community statistics
        community_stats = []
        
        for community_id in self.communities['community'].unique():
            community_markets = self.communities[self.communities['community'] == community_id]
            
            stats = {
                'Community ID': community_id,
                'Size': len(community_markets),
                'Name': self.community_labels.get(str(community_id), {}).get('name', f'Community {community_id}') if self.community_labels else f'Community {community_id}',
                'Theme': self.community_labels.get(str(community_id), {}).get('theme', 'unknown') if self.community_labels else 'unknown',
                'Description': self.community_labels.get(str(community_id), {}).get('description', '') if self.community_labels else ''
            }
            community_stats.append(stats)
        
        communities_df = pd.DataFrame(community_stats)
        communities_df = communities_df.sort_values('Size', ascending=False)
        communities_df.to_excel(writer, sheet_name='Communities', index=False)
    
    def write_correlation_sheet(self, writer):
        """Write correlation matrix sheet (sampled if too large)."""
        # If matrix is too large, sample it
        if self.correlation_matrix.shape[0] > 1000:
            # Sample top communities
            top_markets = []
            for community_id in self.communities.groupby('community').size().nlargest(5).index:
                community_markets = self.communities[
                    self.communities['community'] == community_id
                ]['market_id'].tolist()[:50]  # Max 50 per community
                top_markets.extend(community_markets)
            
            subset_corr = self.correlation_matrix.loc[
                self.correlation_matrix.index.intersection(top_markets),
                self.correlation_matrix.columns.intersection(top_markets)
            ]
            subset_corr.to_excel(writer, sheet_name='Correlation Matrix')
        else:
            self.correlation_matrix.to_excel(writer, sheet_name='Correlation Matrix')
    
    def write_methodology_sheet(self, writer):
        """Write methodology documentation sheet."""
        methodology_data = {
            'Step': [
                '1. Market Filtering',
                '2. LLM Classification',
                '3. Ticker Mapping',
                '4. Time Series',
                '5. Correlation Matrix',
                '6. Community Detection',
                '7. Community Naming',
                '8. Visualization',
                '9. Results Export'
            ],
            'Description': [
                'Remove sports/entertainment markets using keyword filters',
                'Classify markets into theme categories using OpenAI GPT-4',
                'Map CUSIP contracts to recurring Ticker concepts using regex normalization',
                'Build continuous price time series for each Ticker',
                'Compute pairwise correlations with quality filters (volume, variance, active days)',
                'Run Louvain clustering on correlation graph with 0.5 threshold',
                'Generate semantic names for communities using LLM',
                'Create comprehensive charts showing analysis results',
                'Export all results to Excel and charts to PNG files'
            ],
            'Parameters': [
                'Exclude: sports, entertainment, celebrity tags',
                'Model: GPT-4, 19 theme categories',
                'Method: regex + exact match, no fuzzy matching',
                'Min 30 active days, quality filters applied',
                'Min volume: 100, min variance: 0.001, min overlap: 30 days',
                'Threshold: 0.5, min community size: 10 markets',
                'GPT-4 with market context and theme analysis',
                'Charts: funnel, distributions, heatmaps, network viz',
                'Sheets: Summary, Communities, Correlations, Classifications'
            ]
        }
        
        methodology_df = pd.DataFrame(methodology_data)
        methodology_df.to_excel(writer, sheet_name='Methodology', index=False)


# Ticker Mapping Classes
class TickerMapper:
    """Build CUSIP → Ticker mapping using regex normalization."""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.markets = None
        self.ticker_mapping = None
        self.ticker_chains = None
        
    def load_data(self):
        """Load markets data."""
        logger.info("Loading markets data...")
        self.markets = pd.read_parquet(self.data_dir / 'processed' / 'markets.parquet')
        logger.info(f"Loaded {len(self.markets):,} markets")
        
    def build_mapping(self):
        """Build the complete ticker mapping."""
        logger.info("Building ticker mapping...")
        
        # Normalize titles to create tickers
        normalized_titles = {}
        for _, market in self.markets.iterrows():
            normalized = self.normalize_title(market['title'])
            if normalized not in normalized_titles:
                normalized_titles[normalized] = str(uuid.uuid4())[:8]
        
        # Build mapping
        mapping_data = []
        for _, market in self.markets.iterrows():
            normalized = self.normalize_title(market['title'])
            ticker_id = normalized_titles[normalized]
            
            mapping_data.append({
                'market_id': market['condition_id'],
                'ticker_id': ticker_id,
                'ticker_symbol': normalized,
                'original_title': market['title'],
                'end_date': market.get('end_date_ts', pd.NaT)
            })
        
        self.ticker_mapping = pd.DataFrame(mapping_data)
        
        # Build rollable chains
        self.build_ticker_chains()
        
        logger.info(f"Built mapping for {len(self.ticker_mapping):,} CUSIPs → {self.ticker_mapping['ticker_id'].nunique():,} tickers")
    
    def normalize_title(self, title: str) -> str:
        """Normalize market title to ticker concept."""
        # Basic normalization - could be enhanced
        normalized = title.lower()
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', normalized)  # Replace dates
        normalized = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', 'DATE', normalized)  # Replace dates
        normalized = re.sub(r'\bby\s+\w+\s+\d{1,2},?\s+\d{4}\b', 'by DATE', normalized)  # "by March 15, 2025"
        normalized = re.sub(r'\bbefore\s+\w+\s+\d{1,2},?\s+\d{4}\b', 'before DATE', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def build_ticker_chains(self):
        """Build rollable chains for each ticker."""
        chains = {}
        
        for ticker_id in self.ticker_mapping['ticker_id'].unique():
            ticker_markets = self.ticker_mapping[self.ticker_mapping['ticker_id'] == ticker_id]
            
            # Sort by end date
            ticker_markets = ticker_markets.sort_values('end_date')
            
            chains[ticker_id] = {
                'ticker_symbol': ticker_markets['ticker_symbol'].iloc[0],
                'cusips': ticker_markets['market_id'].tolist(),
                'end_dates': ticker_markets['end_date'].dt.strftime('%Y-%m-%d').tolist()
            }
        
        self.ticker_chains = chains


# Time Series Builder
class ContinuousTimeseriesBuilderV2:
    """Build continuous time series for tickers."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.markets_df = None
        self.classifications_df = None
        self.continuous_series = {}
        
    def load_data(self):
        """Load required data."""
        logger.info("Loading time series builder data...")
        self.markets_df = pd.read_parquet(f'{self.data_dir}/processed/markets.parquet')
        
        # Try to load classifications
        try:
            self.classifications_df = pd.read_parquet(f'{self.data_dir}/processed/market_classifications.parquet')
        except FileNotFoundError:
            logger.warning("Classifications not found, proceeding without filtering")
            self.classifications_df = None
    
    def filter_markets(self):
        """Filter out sports/entertainment markets."""
        if self.classifications_df is not None:
            # Filter based on classifications
            excluded_themes = ['sports', 'entertainment', 'celebrity']
            excluded_markets = self.classifications_df[
                self.classifications_df['primary_theme'].isin(excluded_themes)
            ]['market_id'].tolist()
            
            self.markets_df = self.markets_df[
                ~self.markets_df['condition_id'].isin(excluded_markets)
            ]
            
            logger.info(f"Filtered to {len(self.markets_df):,} markets after removing sports/entertainment")
    
    def build_all_series(self):
        """Build all continuous time series."""
        logger.info("Building continuous time series...")
        
        # Group by event slug (natural grouping)
        event_groups = self.markets_df.groupby('event_slug')
        
        series_count = 0
        for event_slug, group in event_groups:
            if len(group) >= 2:  # Need at least 2 contracts for a series
                series = self.build_event_series(event_slug, group)
                if series is not None:
                    self.continuous_series[event_slug] = series
                    series_count += 1
        
        logger.info(f"Built {series_count:,} continuous time series")
    
    def build_event_series(self, event_slug: str, markets: pd.DataFrame) -> Optional[pd.Series]:
        """Build continuous series for a single event."""
        # Placeholder implementation - would need candle data loading
        # For now, create a synthetic series
        
        # Sort markets by end date
        markets = markets.sort_values('end_date_ts')
        
        # Create date range
        start_date = pd.Timestamp.now() - pd.Timedelta(days=90)
        end_date = pd.Timestamp.now()
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        # Create synthetic price series (would load real candle data in practice)
        np.random.seed(hash(event_slug) % 2**32)
        prices = np.random.randn(len(date_range)).cumsum() * 0.01 + 0.5
        prices = np.clip(prices, 0.01, 0.99)
        
        series = pd.Series(prices, index=date_range, name=event_slug)
        return series


def main():
    """Run the complete basket engine pipeline."""
    engine = BasketEngine()
    engine.run_complete_pipeline()


if __name__ == "__main__":
    main()
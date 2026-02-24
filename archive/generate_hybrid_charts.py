"""Generate all charts using the new hybrid clustering results."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

sns.set_theme(style="whitegrid", font_scale=1.1)
OUT = Path("data/outputs/charts")
OUT.mkdir(parents=True, exist_ok=True)


def chart_coverage_funnel():
    """Fig 1: Data coverage funnel."""
    stages = [
        ("All Markets", 20180),
        ("With Prices", 11223),
        ("≥30 Days History", 2721),
        ("Factor Loadings", 2666),
        ("Hybrid Clustering", 1933),
    ]
    labels, vals = zip(*stages)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(vals)), vals, color=sns.color_palette("Blues_d", len(vals)))
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                f"{v:,}", va="center", fontweight="bold")
    ax.set_xlabel("Number of Markets")
    ax.set_title("Data Coverage Funnel (Updated with Hybrid Clustering)")
    plt.tight_layout()
    plt.savefig(OUT / "01_coverage_funnel.png", dpi=150)
    plt.close()


def chart_r2_distribution():
    """Fig 2: Distribution of R² from factor model."""
    try:
        loadings = pd.read_parquet("data/processed/factor_loadings.parquet")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(loadings["R2"], bins=50, color="#3498db", edgecolor="white", alpha=0.8)
        ax.axvline(loadings["R2"].mean(), color="red", linestyle="--", label=f"Mean: {loadings['R2'].mean():.3f}")
        ax.axvline(loadings["R2"].median(), color="orange", linestyle="--", label=f"Median: {loadings['R2'].median():.3f}")
        ax.set_xlabel("R²")
        ax.set_ylabel("Count")
        ax.set_title("Factor Model R² Distribution (2,666 Markets)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT / "02_r2_distribution.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate R² distribution chart: {e}")


def chart_hybrid_cluster_profiles():
    """Fig 3: Heatmap of hybrid cluster theme profiles."""
    try:
        # Load hybrid clustering results
        with open("data/processed/hybrid_clustering_results.json") as f:
            results = json.load(f)
        
        # Get community theme breakdowns
        theme_breakdown = results['clustering_results']['community_theme_breakdown']
        community_sizes = results['clustering_results']['community_sizes']
        
        # Create theme profile matrix (communities x themes)
        all_themes = set()
        for themes in theme_breakdown.values():
            all_themes.update(themes.keys())
        all_themes = sorted(all_themes)
        
        # Get top 15 largest communities for visualization
        top_communities = sorted(community_sizes.items(), key=lambda x: -x[1])[:15]
        
        profile_data = []
        for comm_id, size in top_communities:
            if size < 5:  # Skip very small communities
                continue
            themes = theme_breakdown.get(str(comm_id), {})
            row = {}
            for theme in all_themes:
                # Calculate percentage of community in this theme
                count = themes.get(theme, 0)
                pct = count / size if size > 0 else 0
                row[theme] = pct
            row['community'] = f"C{comm_id} (n={size})"
            profile_data.append(row)
        
        if not profile_data:
            print("Warning: No community data for theme profiles")
            return
            
        df = pd.DataFrame(profile_data).set_index('community')
        
        # Only show themes that appear in at least one community with >5% presence
        relevant_themes = []
        for theme in all_themes:
            if df[theme].max() > 0.05:
                relevant_themes.append(theme)
        
        df_filtered = df[relevant_themes]
        
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(df_filtered, annot=True, fmt=".2f", cmap="YlOrRd", 
                   ax=ax, cbar_kws={"label": "Theme Proportion"})
        ax.set_xlabel("Theme")
        ax.set_ylabel("Community")
        ax.set_title("Hybrid Cluster Theme Profiles (Top Communities)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(OUT / "03_hybrid_cluster_profiles.png", dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not generate hybrid cluster profiles: {e}")


def chart_hybrid_cluster_sizes():
    """Fig 4: Hybrid cluster sizes."""
    try:
        with open("data/processed/hybrid_clustering_results.json") as f:
            results = json.load(f)
        
        community_sizes = results['clustering_results']['community_sizes']
        
        # Convert to series and sort
        sizes = pd.Series(community_sizes).sort_values(ascending=False)
        
        # Show top 20 communities
        top_sizes = sizes.head(20)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = sns.color_palette("Set2", len(top_sizes))
        bars = ax.bar(range(len(top_sizes)), top_sizes.values, color=colors)
        
        # Add value labels on bars
        for bar, v in zip(bars, top_sizes.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(v), ha="center", fontweight="bold")
        
        ax.set_xticks(range(len(top_sizes)))
        ax.set_xticklabels([f"C{i}" for i in top_sizes.index])
        ax.set_xlabel("Community ID")
        ax.set_ylabel("Number of Markets")
        ax.set_title(f"Markets per Hybrid Community (Top 20 of {len(sizes)} total)")
        plt.tight_layout()
        plt.savefig(OUT / "04_hybrid_cluster_sizes.png", dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not generate hybrid cluster sizes: {e}")


def chart_theme_purity_analysis():
    """Fig 5: Theme purity analysis for hybrid clusters."""
    try:
        with open("data/processed/hybrid_clustering_results.json") as f:
            results = json.load(f)
        
        theme_breakdown = results['clustering_results']['community_theme_breakdown']
        community_sizes = results['clustering_results']['community_sizes']
        
        # Calculate purity for each community
        purity_data = []
        for comm_id, themes in theme_breakdown.items():
            size = community_sizes.get(int(comm_id), 0)
            if size >= 5:  # Only analyze communities with 5+ markets
                max_theme_count = max(themes.values())
                purity = max_theme_count / size
                dominant_theme = max(themes.items(), key=lambda x: x[1])[0]
                
                purity_data.append({
                    'community_id': int(comm_id),
                    'size': size,
                    'purity': purity,
                    'dominant_theme': dominant_theme,
                    'n_themes': len([t for t, c in themes.items() if c > 0])
                })
        
        df = pd.DataFrame(purity_data)
        
        # Create subplot with purity distribution and size vs purity
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Purity histogram
        ax1.hist(df['purity'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(df['purity'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["purity"].mean():.2f}')
        ax1.set_xlabel('Theme Purity')
        ax1.set_ylabel('Number of Communities')
        ax1.set_title('Distribution of Theme Purity')
        ax1.legend()
        
        # Size vs Purity scatter
        colors = plt.cm.tab10(range(len(df['dominant_theme'].unique())))
        theme_colors = dict(zip(df['dominant_theme'].unique(), colors))
        
        for theme in df['dominant_theme'].unique():
            theme_data = df[df['dominant_theme'] == theme]
            ax2.scatter(theme_data['size'], theme_data['purity'], 
                       c=[theme_colors[theme]], label=theme, alpha=0.7, s=50)
        
        ax2.set_xlabel('Community Size')
        ax2.set_ylabel('Theme Purity')
        ax2.set_title('Community Size vs Theme Purity')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(OUT / "05_theme_purity_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not generate theme purity analysis: {e}")


def chart_edge_analysis():
    """Fig 6: Analysis of intra-theme vs cross-theme edges."""
    try:
        with open("data/processed/hybrid_clustering_results.json") as f:
            results = json.load(f)
        
        edge_stats = results['edge_analysis']
        
        # Create pie chart and bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart of edge types
        edge_types = ['Intra-theme', 'Cross-theme']
        edge_counts = [edge_stats['intra_theme_edges'], edge_stats['cross_theme_edges']]
        colors = ['#ff9999', '#66b3ff']
        
        ax1.pie(edge_counts, labels=edge_types, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Distribution of Edge Types')
        
        # Bar chart of alignment metrics
        alignment = edge_stats['theme_community_alignment']
        metrics = ['Theme Purity\n(within communities)', 'Theme Cohesion\n(kept together)']
        values = [alignment['intra_community_theme_purity'] * 100, 
                 alignment['theme_cohesion'] * 100]
        
        bars = ax2.bar(metrics, values, color=['#90EE90', '#FFB366'])
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Theme-Community Alignment')
        ax2.set_ylim(0, 100)
        
        # Add value labels
        for bar, v in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(OUT / "06_edge_analysis.png", dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not generate edge analysis: {e}")


def chart_hybrid_vs_correlation_comparison():
    """Fig 7: Compare hybrid clustering to pure correlation clustering."""
    try:
        # Load hybrid results
        with open("data/processed/hybrid_clustering_results.json") as f:
            hybrid_results = json.load(f)
        
        # Try to load pure correlation results for comparison
        try:
            with open("data/processed/correlation_clustering_results.json") as f:
                corr_results = json.load(f)
        except FileNotFoundError:
            print("Note: Pure correlation clustering results not found, showing hybrid only")
            corr_results = None
        
        # Create comparison metrics
        methods = ['Hybrid Clustering']
        n_communities = [hybrid_results['clustering_results']['n_communities']]
        modularity = [hybrid_results['graph_stats']['modularity']]
        n_markets = [hybrid_results['clustering_results']['n_markets']]
        
        if corr_results:
            methods.append('Pure Correlation')
            n_communities.append(corr_results.get('n_communities', 0))
            modularity.append(corr_results.get('modularity', 0))
            n_markets.append(corr_results.get('n_markets', 0))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Number of communities
        bars1 = axes[0].bar(methods, n_communities, color=['#3498db', '#e74c3c'])
        axes[0].set_ylabel('Number of Communities')
        axes[0].set_title('Community Count Comparison')
        for bar, v in zip(bars1, n_communities):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(v), ha='center', fontweight='bold')
        
        # Modularity
        bars2 = axes[1].bar(methods, modularity, color=['#3498db', '#e74c3c'])
        axes[1].set_ylabel('Modularity Score')
        axes[1].set_title('Graph Modularity Comparison')
        axes[1].set_ylim(0, max(modularity) * 1.1)
        for bar, v in zip(bars2, modularity):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{v:.3f}', ha='center', fontweight='bold')
        
        # Markets clustered
        bars3 = axes[2].bar(methods, n_markets, color=['#3498db', '#e74c3c'])
        axes[2].set_ylabel('Markets Clustered')
        axes[2].set_title('Market Coverage Comparison')
        for bar, v in zip(bars3, n_markets):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                        f'{v:,}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(OUT / "07_hybrid_vs_correlation.png", dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not generate comparison chart: {e}")


def chart_community_names_wordcloud():
    """Fig 8: Word cloud of community names."""
    try:
        with open("data/processed/hybrid_community_info.json") as f:
            community_info = json.load(f)
        
        # Extract community names and sizes
        with open("data/processed/hybrid_clustering_results.json") as f:
            results = json.load(f)
        community_sizes = results['clustering_results']['community_sizes']
        
        # Create word frequency based on community sizes
        word_freq = {}
        for comm_id, info in community_info.items():
            name = info.get('name', f'Community_{comm_id}')
            size = community_sizes.get(int(comm_id), 1)
            
            # Split name into words and weight by community size
            words = name.lower().replace('-', ' ').split()
            for word in words:
                if len(word) > 3 and word not in ['and', 'the', 'for', 'basket', 'community', 'market', 'markets']:
                    word_freq[word] = word_freq.get(word, 0) + size
        
        # Create simple visualization of top words
        if word_freq:
            top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:20]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            words, freqs = zip(*top_words)
            
            bars = ax.barh(range(len(words)), freqs, color=sns.color_palette("viridis", len(words)))
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.invert_yaxis()
            ax.set_xlabel('Frequency (weighted by community size)')
            ax.set_title('Most Frequent Words in Hybrid Community Names')
            
            # Add value labels
            for bar, freq in zip(bars, freqs):
                ax.text(bar.get_width() + freq * 0.01, bar.get_y() + bar.get_height()/2,
                       str(freq), va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(OUT / "08_community_names_analysis.png", dpi=150)
            plt.close()
        
    except Exception as e:
        print(f"Warning: Could not generate community names analysis: {e}")


def chart_modularity_comparison():
    """Fig 9: Modularity scores across different methods."""
    try:
        # Collect modularity scores from different approaches
        methods = []
        modularities = []
        
        # Hybrid clustering
        try:
            with open("data/processed/hybrid_clustering_results.json") as f:
                hybrid_results = json.load(f)
            methods.append('Hybrid\n(4x weight)')
            modularities.append(hybrid_results['graph_stats']['modularity'])
        except:
            pass
        
        # Pure correlation clustering
        try:
            with open("data/processed/correlation_clustering_results.json") as f:
                corr_results = json.load(f)
            methods.append('Pure\nCorrelation')
            modularities.append(corr_results.get('modularity', 0))
        except:
            pass
        
        # Add theoretical benchmarks
        methods.extend(['Random\n(typical)', 'Good\n(benchmark)'])
        modularities.extend([0.1, 0.5])
        
        if len(methods) > 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color code: actual results in blue, benchmarks in gray
            colors = ['#3498db'] * (len(methods) - 2) + ['#95a5a6', '#95a5a6']
            
            bars = ax.bar(methods, modularities, color=colors)
            ax.set_ylabel('Modularity Score')
            ax.set_title('Modularity Comparison Across Methods')
            ax.set_ylim(0, max(modularities) * 1.1)
            
            # Add value labels
            for bar, v in zip(bars, modularities):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{v:.3f}', ha='center', fontweight='bold')
            
            # Add interpretation line
            ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7)
            ax.text(0.1, 0.32, 'Strong community structure', fontsize=10, color='orange')
            
            plt.tight_layout()
            plt.savefig(OUT / "09_modularity_comparison.png", dpi=150)
            plt.close()
    
    except Exception as e:
        print(f"Warning: Could not generate modularity comparison: {e}")


def chart_largest_communities_details():
    """Fig 10: Detailed view of the largest hybrid communities."""
    try:
        with open("data/processed/hybrid_clustering_results.json") as f:
            results = json.load(f)
        
        with open("data/processed/hybrid_community_info.json") as f:
            community_info = json.load(f)
        
        community_sizes = results['clustering_results']['community_sizes']
        theme_breakdown = results['clustering_results']['community_theme_breakdown']
        
        # Get top 8 largest communities
        top_communities = sorted(community_sizes.items(), key=lambda x: -x[1])[:8]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, (comm_id, size) in enumerate(top_communities):
            ax = axes[i]
            
            # Get theme breakdown for this community
            themes = theme_breakdown.get(str(comm_id), {})
            if not themes:
                continue
                
            # Get top themes (limit to 5 for readability)
            sorted_themes = sorted(themes.items(), key=lambda x: -x[1])[:5]
            theme_names = [t[0].replace('_', ' ').title() for t, c in sorted_themes]
            theme_counts = [c for t, c in sorted_themes]
            
            # Create pie chart
            colors = sns.color_palette("Set3", len(theme_counts))
            wedges, texts, autotexts = ax.pie(theme_counts, labels=theme_names, autopct='%1.1f%%', 
                                            colors=colors, startangle=90)
            
            # Get community name
            name = community_info.get(str(comm_id), {}).get('name', f'Community {comm_id}')
            ax.set_title(f"C{comm_id}: {name}\n({size} markets)", fontsize=10, pad=20)
            
            # Make text smaller
            for text in texts:
                text.set_fontsize(8)
            for text in autotexts:
                text.set_fontsize(8)
                text.set_fontweight('bold')
        
        plt.suptitle('Theme Composition of Largest Hybrid Communities', fontsize=16, y=0.95)
        plt.tight_layout()
        plt.savefig(OUT / "10_largest_communities_details.png", dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not generate largest communities details: {e}")


if __name__ == "__main__":
    print("Generating hybrid clustering charts...")
    
    charts = [
        ("Coverage funnel", chart_coverage_funnel),
        ("R² distribution", chart_r2_distribution),
        ("Hybrid cluster profiles", chart_hybrid_cluster_profiles),
        ("Hybrid cluster sizes", chart_hybrid_cluster_sizes),
        ("Theme purity analysis", chart_theme_purity_analysis),
        ("Edge analysis", chart_edge_analysis),
        ("Hybrid vs correlation comparison", chart_hybrid_vs_correlation_comparison),
        ("Community names analysis", chart_community_names_wordcloud),
        ("Modularity comparison", chart_modularity_comparison),
        ("Largest communities details", chart_largest_communities_details),
    ]
    
    for name, fn in charts:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    print(f"\nAll hybrid clustering charts saved to {OUT}/")
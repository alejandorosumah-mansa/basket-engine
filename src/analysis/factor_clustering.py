"""
Factor-Based Clustering

Cluster markets by their factor loading vectors (not raw returns).
Compare with LLM theme labels to find where they agree/disagree.
Use disagreements to refine basket definitions.

Process:
1. Load factor loadings from market_factors.py
2. Standardize loading vectors
3. Determine optimal k via silhouette/elbow
4. Run k-means and hierarchical clustering
5. Compare clusters with LLM themes
6. Output final basket definitions
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

BETA_COLS = [
    "beta_SPY", "beta_TNX", "beta_GLD", "beta_VIX",
    "beta_DX_Y_NYB", "beta_USO", "beta_BTC_USD", "beta_TLT", "beta_IRX"
]


def load_factor_loadings(path: str = "data/processed/factor_loadings.parquet") -> pd.DataFrame:
    """Load and clean factor loadings."""
    df = pd.read_parquet(path)
    # Drop markets with NaN loadings
    df = df.dropna(subset=BETA_COLS)
    return df


def find_optimal_k(X: np.ndarray, k_range: range = range(3, 16)) -> dict:
    """Evaluate clustering quality for different k values."""
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)))
        inertia = km.inertia_
        results[k] = {"silhouette": sil, "inertia": inertia}
        print(f"  k={k}: silhouette={sil:.4f}, inertia={inertia:.1f}")
    
    best_k = max(results, key=lambda k: results[k]["silhouette"])
    print(f"  Best k by silhouette: {best_k}")
    return results


def run_clustering(
    loadings: pd.DataFrame,
    n_clusters: int = 8,
    method: str = "kmeans",
) -> pd.DataFrame:
    """Cluster markets by factor loadings.
    
    Args:
        loadings: DataFrame with factor loading columns
        n_clusters: number of clusters (default: 8)
        method: 'kmeans' or 'hierarchical'
    
    Returns:
        loadings DataFrame with 'cluster' column added
    """
    beta_cols = [c for c in BETA_COLS if c in loadings.columns]
    X = loadings[beta_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Remove outliers (>3 std in any dimension)
    outlier_mask = (np.abs(X_std) < 3).all(axis=1)
    print(f"Removing {(~outlier_mask).sum()} outlier markets (>{3} std)")
    
    X_clean = X_std[outlier_mask]
    loadings_clean = loadings[outlier_mask].copy()
    
    print(f"Clustering {len(X_clean)} markets with k={n_clusters}, method={method}")
    
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_clean)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(X_clean)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    loadings_clean["cluster"] = labels
    
    # Assign outliers to nearest cluster
    if (~outlier_mask).any() and method == "kmeans":
        outlier_labels = model.predict(X_std[~outlier_mask])
        loadings_outlier = loadings[~outlier_mask].copy()
        loadings_outlier["cluster"] = outlier_labels
        loadings_clean = pd.concat([loadings_clean, loadings_outlier])
    
    sil = silhouette_score(X_clean, labels, sample_size=min(2000, len(X_clean)))
    print(f"Silhouette score: {sil:.4f}")
    
    # Print cluster sizes
    sizes = loadings_clean["cluster"].value_counts().sort_index()
    for c, n in sizes.items():
        print(f"  Cluster {c}: {n} markets")
    
    return loadings_clean


def get_cluster_profiles(clustered: pd.DataFrame) -> pd.DataFrame:
    """Compute mean factor loadings per cluster."""
    beta_cols = [c for c in BETA_COLS if c in clustered.columns]
    profiles = clustered.groupby("cluster")[beta_cols + ["R2", "idio_vol", "total_vol"]].agg(["mean", "count"])
    return profiles


def compare_with_themes(
    clustered: pd.DataFrame,
    markets: pd.DataFrame,
    llm_cache_path: str = "data/processed/llm_classifications_cache_v2.json",
) -> pd.DataFrame:
    """Compare factor clusters with LLM theme labels.
    
    Returns DataFrame showing cluster-theme cross-tabulation and disagreements.
    """
    # Get LLM themes from cache
    if Path(llm_cache_path).exists():
        with open(llm_cache_path) as f:
            llm_cache = json.load(f)
        
        # Build market_id → theme mapping
        theme_map = {}
        for key, val in llm_cache.items():
            if isinstance(val, dict) and "theme" in val:
                # Key might be the market title
                theme_map[key] = val["theme"]
    else:
        print("No LLM classification cache found")
        return pd.DataFrame()
    
    # Match markets to themes via title
    title_to_id = dict(zip(markets["title"], markets["market_id"]))
    market_themes = {}
    for title, theme in theme_map.items():
        if title in title_to_id:
            market_themes[title_to_id[title]] = theme
    
    clustered = clustered.copy()
    clustered["llm_theme"] = clustered.index.map(market_themes)
    
    # Cross-tabulation
    has_both = clustered.dropna(subset=["llm_theme"])
    if len(has_both) == 0:
        print("No markets with both cluster and theme labels")
        return clustered
    
    cross_tab = pd.crosstab(has_both["cluster"], has_both["llm_theme"])
    print(f"\nCluster-Theme cross-tabulation ({len(has_both)} markets):")
    print(cross_tab.to_string())
    
    # Agreement metric: for each cluster, what % belongs to dominant theme
    agreement = {}
    for cluster in cross_tab.index:
        row = cross_tab.loc[cluster]
        dominant_theme = row.idxmax()
        pct = row.max() / row.sum()
        agreement[cluster] = {
            "dominant_theme": dominant_theme,
            "agreement_pct": pct,
            "n_markets": row.sum(),
        }
    
    print("\nCluster agreement with LLM themes:")
    for cluster, info in sorted(agreement.items()):
        print(f"  Cluster {cluster}: {info['dominant_theme']} ({info['agreement_pct']:.1%} of {info['n_markets']} markets)")
    
    return clustered


def build_basket_definitions(
    clustered: pd.DataFrame,
    markets: pd.DataFrame,
) -> dict:
    """Create final basket definitions from factor clusters.
    
    Each cluster becomes a basket. Name based on dominant factor exposures.
    """
    beta_cols = [c for c in BETA_COLS if c in clustered.columns]
    
    baskets = {}
    for cluster_id in sorted(clustered["cluster"].unique()):
        cluster_markets = clustered[clustered["cluster"] == cluster_id]
        
        # Characterize by mean factor loadings
        mean_betas = cluster_markets[beta_cols].mean()
        
        # Name based on dominant exposures
        top_factors = mean_betas.abs().nlargest(3)
        factor_desc = []
        for factor_name in top_factors.index:
            clean_name = factor_name.replace("beta_", "")
            direction = "+" if mean_betas[factor_name] > 0 else "-"
            factor_desc.append(f"{direction}{clean_name}")
        
        basket_name = f"cluster_{cluster_id}"
        
        # Get LLM theme if available
        if "llm_theme" in cluster_markets.columns:
            themes = cluster_markets["llm_theme"].dropna().value_counts()
            dominant_theme = themes.index[0] if len(themes) > 0 else "unknown"
        else:
            dominant_theme = "unknown"
        
        baskets[basket_name] = {
            "cluster_id": int(cluster_id),
            "n_markets": len(cluster_markets),
            "dominant_theme": dominant_theme,
            "factor_signature": factor_desc,
            "mean_R2": float(cluster_markets["R2"].mean()),
            "mean_idio_vol": float(cluster_markets["idio_vol"].mean()),
            "market_ids": cluster_markets.index.tolist(),
            "mean_betas": {col.replace("beta_", ""): float(mean_betas[col]) for col in beta_cols},
        }
    
    return baskets


def run_factor_clustering(
    loadings_path: str = "data/processed/factor_loadings.parquet",
    markets_path: str = "data/processed/markets.parquet",
    output_baskets_path: str = "data/processed/basket_definitions.json",
    output_clusters_path: str = "data/processed/cluster_assignments.parquet",
) -> dict:
    """Full pipeline: load, cluster, compare, define baskets."""
    print("=== Factor-Based Clustering ===")
    
    loadings = load_factor_loadings(loadings_path)
    print(f"Loaded {len(loadings)} markets with factor loadings")
    
    markets = pd.read_parquet(markets_path)
    
    # Run clustering
    clustered = run_clustering(loadings, method="kmeans")
    
    # Cluster profiles
    print("\nCluster profiles:")
    profiles = get_cluster_profiles(clustered)
    for cluster_id in sorted(clustered["cluster"].unique()):
        n = (clustered["cluster"] == cluster_id).sum()
        beta_cols = [c for c in BETA_COLS if c in clustered.columns]
        means = clustered[clustered["cluster"] == cluster_id][beta_cols].mean()
        top = means.abs().nlargest(3)
        desc = ", ".join([f"{k.replace('beta_','')}={means[k]:+.4f}" for k in top.index])
        print(f"  Cluster {cluster_id} (n={n}): {desc}")
    
    # Compare with LLM themes
    clustered = compare_with_themes(clustered, markets)
    
    # Build basket definitions
    baskets = build_basket_definitions(clustered, markets)
    
    # Compute cross-basket correlations
    print("\nCross-cluster mean factor loading correlation:")
    beta_cols = [c for c in BETA_COLS if c in clustered.columns]
    cluster_means = clustered.groupby("cluster")[beta_cols].mean()
    corr = cluster_means.T.corr()
    print(corr.round(3).to_string())
    
    # Save
    Path(output_baskets_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_baskets_path, "w") as f:
        json.dump(baskets, f, indent=2, default=str)
    
    clustered.to_parquet(output_clusters_path)
    
    print(f"\nSaved basket definitions to {output_baskets_path}")
    print(f"Saved cluster assignments to {output_clusters_path}")
    print(f"Total baskets: {len(baskets)}")
    
    return baskets


if __name__ == "__main__":
    run_factor_clustering()

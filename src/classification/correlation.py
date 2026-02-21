"""Statistical clustering of prediction markets based on return correlations."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from pathlib import Path
from typing import Optional
import yaml
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_and_filter_returns(
    min_history_days: int = 30,
    min_volume: float = 10000,
) -> pd.DataFrame:
    """Load returns, filter to serious/investable markets with sufficient history.
    
    Returns a pivot table: index=date, columns=market_id, values=return.
    """
    settings = load_settings()
    
    # Load data
    returns_df = pd.read_parquet(DATA_DIR / "returns.parquet")
    prices_df = pd.read_parquet(DATA_DIR / "prices.parquet")
    classifications = pd.read_csv(DATA_DIR / "llm_classifications_full.csv")
    
    # Filter to serious markets (exclude sports_entertainment and uncategorized)
    serious = classifications[
        ~classifications["primary_theme"].isin(["sports_entertainment", "uncategorized"])
    ]
    serious_ids = set(serious["event_id"])
    
    # Filter returns to serious markets
    returns_df = returns_df[returns_df["market_id"].isin(serious_ids)]
    
    # Volume filter
    vol_per_market = prices_df.groupby("market_id")["volume"].sum()
    high_vol_ids = set(vol_per_market[vol_per_market > min_volume].index)
    returns_df = returns_df[returns_df["market_id"].isin(high_vol_ids)]
    
    # History filter
    days_per_market = returns_df.groupby("market_id").size()
    sufficient_ids = set(days_per_market[days_per_market >= min_history_days].index)
    returns_df = returns_df[returns_df["market_id"].isin(sufficient_ids)]
    
    if len(returns_df) == 0:
        logger.warning("No markets pass all filters!")
        return pd.DataFrame()
    
    # Pivot to matrix form
    returns_df["date"] = pd.to_datetime(returns_df["date"])
    matrix = returns_df.pivot_table(index="date", columns="market_id", values="return")
    
    logger.info(f"Returns matrix: {matrix.shape[1]} markets x {matrix.shape[0]} days")
    return matrix


def compute_correlation_matrix(returns_matrix: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """Compute pairwise correlation matrix."""
    if method == "spearman":
        corr = returns_matrix.corr(method="spearman")
    else:
        corr = returns_matrix.corr(method="pearson")
    return corr


def run_hierarchical_clustering(
    corr_matrix: pd.DataFrame,
    linkage_method: str = "ward",
    min_clusters: int = 2,
    max_clusters: int = 20,
) -> tuple[np.ndarray, int, list[float]]:
    """Run hierarchical clustering and find optimal cluster count.
    
    Returns: (linkage_matrix, optimal_k, silhouette_scores)
    """
    # Distance = 1 - abs(correlation)
    corr_filled = corr_matrix.fillna(0)
    dist = 1 - corr_filled.abs().values
    np.fill_diagonal(dist, 0)
    # Ensure symmetry, non-negative, finite
    dist = (dist + dist.T) / 2
    dist = np.clip(dist, 0, None)
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=0.0)
    
    # Convert to condensed form
    condensed = squareform(dist, checks=False)
    
    # Linkage
    Z = linkage(condensed, method=linkage_method)
    
    # Find optimal k via silhouette score
    n_markets = corr_matrix.shape[0]
    max_clusters = min(max_clusters, n_markets - 1)
    min_clusters = max(min_clusters, 2)
    
    if max_clusters < min_clusters:
        # Too few markets for meaningful clustering
        labels = np.zeros(n_markets, dtype=int)
        return Z, 1, []
    
    scores = []
    k_range = range(min_clusters, max_clusters + 1)
    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            scores.append(-1)
            continue
        score = silhouette_score(dist, labels, metric="precomputed")
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    optimal_k = list(k_range)[optimal_idx]
    
    logger.info(f"Optimal clusters: {optimal_k} (silhouette={scores[optimal_idx]:.3f})")
    return Z, optimal_k, list(zip(k_range, scores))


def assign_clusters(
    corr_matrix: pd.DataFrame,
    linkage_matrix: np.ndarray,
    n_clusters: int,
) -> pd.DataFrame:
    """Assign markets to clusters."""
    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    assignments = pd.DataFrame({
        "market_id": corr_matrix.columns,
        "cluster": labels,
    })
    return assignments


def visualize_clustering(
    corr_matrix: pd.DataFrame,
    linkage_matrix: np.ndarray,
    silhouette_scores: list[tuple[int, float]],
    output_dir: Optional[Path] = None,
):
    """Generate dendrogram, heatmap, and silhouette score plots."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dendrogram
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=list(corr_matrix.columns), leaf_rotation=90, ax=ax)
    ax.set_title("Market Clustering Dendrogram")
    ax.set_ylabel("Distance (1 - |correlation|)")
    plt.tight_layout()
    fig.savefig(output_dir / "dendrogram.png", dpi=150)
    plt.close(fig)
    
    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_title("Spearman Rank Correlation Matrix")
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    
    # 3. Silhouette scores
    if silhouette_scores:
        fig, ax = plt.subplots(figsize=(8, 5))
        ks, scores = zip(*silhouette_scores)
        ax.plot(ks, scores, "o-")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Optimal Cluster Count")
        best_idx = np.argmax(scores)
        ax.axvline(ks[best_idx], color="red", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(output_dir / "silhouette_scores.png", dpi=150)
        plt.close(fig)
    
    logger.info(f"Visualizations saved to {output_dir}")


def run_clustering_pipeline() -> pd.DataFrame:
    """Run the full clustering pipeline. Returns cluster assignments."""
    settings = load_settings()
    cls_settings = settings.get("classification", {})
    
    min_days = cls_settings.get("clustering_min_history_days", 30)
    method = cls_settings.get("clustering_distance_metric", "spearman")
    linkage_method = cls_settings.get("clustering_linkage", "ward")
    
    # Try with 30 days first; if too few markets, relax to 14
    matrix = load_and_filter_returns(min_history_days=min_days)
    
    if matrix.shape[1] < 5:
        logger.warning(f"Only {matrix.shape[1]} markets with {min_days}+ days. Relaxing to 14 days.")
        matrix = load_and_filter_returns(min_history_days=14, min_volume=5000)
    
    if matrix.shape[1] < 3:
        logger.warning(f"Only {matrix.shape[1]} markets available. Cannot cluster meaningfully.")
        # Return trivial single-cluster assignment
        if matrix.shape[1] > 0:
            assignments = pd.DataFrame({
                "market_id": matrix.columns,
                "cluster": 1,
            })
        else:
            assignments = pd.DataFrame(columns=["market_id", "cluster"])
        assignments.to_csv(DATA_DIR / "cluster_assignments.csv", index=False)
        return assignments
    
    # Compute correlation
    corr = compute_correlation_matrix(matrix, method=method)
    
    # Cluster
    max_k = min(20, matrix.shape[1] - 1)
    Z, optimal_k, sil_scores = run_hierarchical_clustering(
        corr, linkage_method=linkage_method, min_clusters=2, max_clusters=max_k
    )
    
    # Assign
    assignments = assign_clusters(corr, Z, optimal_k)
    
    # Visualize
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    visualize_clustering(corr, Z, sil_scores)
    
    # Save
    assignments.to_csv(DATA_DIR / "cluster_assignments.csv", index=False)
    logger.info(f"Cluster assignments saved: {len(assignments)} markets in {optimal_k} clusters")
    
    return assignments


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    assignments = run_clustering_pipeline()
    print(assignments.value_counts("cluster"))

"""Hybrid reconciliation: merge statistical clusters with LLM theme assignments."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"


def build_agreement_matrix(
    clusters: pd.DataFrame,
    classifications: pd.DataFrame,
) -> pd.DataFrame:
    """Build cluster x theme agreement matrix.
    
    Args:
        clusters: DataFrame with market_id, cluster columns
        classifications: DataFrame with event_id, primary_theme columns
    
    Returns:
        Pivot table of counts: rows=cluster, columns=theme
    """
    merged = clusters.merge(
        classifications[["event_id", "primary_theme"]],
        left_on="market_id",
        right_on="event_id",
        how="left",
    )
    merged = merged.dropna(subset=["primary_theme"])
    
    agreement = pd.crosstab(merged["cluster"], merged["primary_theme"])
    return agreement


def find_dominant_themes(agreement_matrix: pd.DataFrame) -> dict:
    """For each cluster, find the dominant LLM theme.
    
    Returns: {cluster_id: {"dominant_theme": str, "count": int, "total": int, "purity": float}}
    """
    result = {}
    for cluster_id in agreement_matrix.index:
        row = agreement_matrix.loc[cluster_id]
        dominant = row.idxmax()
        count = row.max()
        total = row.sum()
        purity = count / total if total > 0 else 0
        result[cluster_id] = {
            "dominant_theme": dominant,
            "count": int(count),
            "total": int(total),
            "purity": float(purity),
        }
    return result


def flag_disagreements(
    clusters: pd.DataFrame,
    classifications: pd.DataFrame,
    dominant_themes: dict,
    purity_threshold: float = 0.5,
) -> pd.DataFrame:
    """Flag markets where cluster assignment disagrees with LLM theme.
    
    Returns DataFrame of disagreements.
    """
    merged = clusters.merge(
        classifications[["event_id", "primary_theme"]],
        left_on="market_id",
        right_on="event_id",
        how="left",
    )
    
    # Map cluster to dominant theme
    merged["cluster_theme"] = merged["cluster"].map(
        lambda c: dominant_themes.get(c, {}).get("dominant_theme", "unknown")
    )
    merged["cluster_purity"] = merged["cluster"].map(
        lambda c: dominant_themes.get(c, {}).get("purity", 0)
    )
    
    # Flag disagreements
    disagree = merged[merged["primary_theme"] != merged["cluster_theme"]].copy()
    
    # Also flag low-purity clusters
    low_purity = merged[merged["cluster_purity"] < purity_threshold].copy()
    
    logger.info(f"Disagreements: {len(disagree)} markets")
    logger.info(f"Low-purity clusters: {low_purity['cluster'].nunique()} clusters")
    
    return disagree


def produce_final_classifications(
    classifications: pd.DataFrame,
    clusters: pd.DataFrame,
    dominant_themes: dict,
) -> pd.DataFrame:
    """Produce reconciled final classifications.
    
    Strategy: Trust LLM themes for markets that were classified. For clustered
    markets, add cluster info as secondary signal. If a market's LLM theme
    disagrees with its cluster's dominant theme and the cluster has high purity,
    flag it for review but keep the LLM assignment (LLM is primary authority).
    """
    # Start with all serious classifications
    serious = classifications[
        ~classifications["primary_theme"].isin(["sports_entertainment", "uncategorized"])
    ].copy()
    
    # Merge cluster info
    if not clusters.empty:
        serious = serious.merge(
            clusters,
            left_on="event_id",
            right_on="market_id",
            how="left",
        )
        
        # Add cluster's dominant theme
        serious["cluster_dominant_theme"] = serious["cluster"].map(
            lambda c: dominant_themes.get(c, {}).get("dominant_theme") if pd.notna(c) else None
        )
        serious["cluster_purity"] = serious["cluster"].map(
            lambda c: dominant_themes.get(c, {}).get("purity") if pd.notna(c) else None
        )
        
        # Reconciled theme: use LLM as primary, flag disagreements
        serious["reconciled_theme"] = serious["primary_theme"]
        serious["cluster_agrees"] = (
            serious["primary_theme"] == serious["cluster_dominant_theme"]
        ) | serious["cluster_dominant_theme"].isna()
    else:
        serious["cluster"] = None
        serious["cluster_dominant_theme"] = None
        serious["cluster_purity"] = None
        serious["reconciled_theme"] = serious["primary_theme"]
        serious["cluster_agrees"] = True
    
    return serious


def run_hybrid_pipeline() -> pd.DataFrame:
    """Run the full hybrid reconciliation pipeline."""
    classifications = pd.read_csv(DATA_DIR / "llm_classifications_full.csv")
    
    cluster_path = DATA_DIR / "cluster_assignments.csv"
    if cluster_path.exists():
        clusters = pd.read_csv(cluster_path)
    else:
        logger.warning("No cluster assignments found. Using LLM-only classifications.")
        clusters = pd.DataFrame(columns=["market_id", "cluster"])
    
    # Build agreement matrix
    if not clusters.empty:
        agreement = build_agreement_matrix(clusters, classifications)
        dominant = find_dominant_themes(agreement)
        
        # Save agreement matrix
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        agreement.to_csv(OUTPUT_DIR / "cluster_theme_agreement.csv")
        
        # Flag disagreements
        disagree = flag_disagreements(clusters, classifications, dominant)
        if not disagree.empty:
            disagree.to_csv(OUTPUT_DIR / "cluster_disagreements.csv", index=False)
        
        logger.info("Agreement matrix:")
        logger.info(f"\n{agreement}")
        logger.info(f"\nDominant themes per cluster:")
        for c, info in dominant.items():
            logger.info(f"  Cluster {c}: {info['dominant_theme']} "
                       f"({info['count']}/{info['total']}, purity={info['purity']:.2f})")
    else:
        dominant = {}
    
    # Produce final classifications
    final = produce_final_classifications(classifications, clusters, dominant)
    final.to_csv(DATA_DIR / "final_classifications.csv", index=False)
    
    logger.info(f"Final classifications: {len(final)} markets")
    logger.info(f"Theme distribution:\n{final['reconciled_theme'].value_counts()}")
    
    return final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    final = run_hybrid_pipeline()

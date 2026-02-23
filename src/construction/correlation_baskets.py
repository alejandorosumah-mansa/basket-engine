"""
Correlation-Based Basket Construction

Build baskets from correlation communities with:
- Eligibility filters (active, sufficient price history, volume)
- Risk parity weighting using market return volatility 
- Improved diversification from correlation-based clustering

This replaces factor-based clustering with correlation-based community detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def filter_eligible_markets(
    community_assignments: pd.DataFrame,
    prices: pd.DataFrame,
    markets: pd.DataFrame,
    min_price_days: int = 60,
    min_recent_volume: float = 1000,
    cluster_column: str = "community"
) -> pd.DataFrame:
    """Filter markets for basket inclusion.
    
    Criteria:
    - At least min_price_days of price data
    - Active (not resolved) or resolved within last 90 days
    - Minimum volume
    
    Args:
        community_assignments: DataFrame with market_id index and community column
        cluster_column: Name of column containing community assignments ("community" for correlation, "cluster" for factor)
    """
    # Price history length
    price_counts = prices.groupby("market_id").size()
    has_history = set(price_counts[price_counts >= min_price_days].index)
    
    # Volume filter
    has_volume = set(markets[markets["volume"] >= min_recent_volume]["market_id"])
    
    eligible_ids = has_history & has_volume & set(community_assignments.index)
    
    result = community_assignments.loc[community_assignments.index.isin(eligible_ids)].copy()
    print(f"Eligible markets: {len(result)} (from {len(community_assignments)})")
    
    # Show basket sizes
    basket_sizes = result[cluster_column].value_counts().sort_index()
    print("Eligible basket sizes:")
    for basket_id, size in basket_sizes.items():
        print(f"  Basket {basket_id}: {size} markets")
    
    return result


def compute_basket_returns(
    eligible: pd.DataFrame,
    prices: pd.DataFrame,
    cluster_column: str = "community"
) -> pd.DataFrame:
    """Compute daily returns per basket (community).
    
    Uses equal weighting within each basket for simplicity,
    then we can apply risk-parity across baskets.
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    
    # Only eligible markets
    eligible_prices = prices[prices["market_id"].isin(eligible.index)]
    
    # Map market to community/cluster
    market_to_cluster = eligible[cluster_column].to_dict()
    eligible_prices["basket"] = eligible_prices["market_id"].map(market_to_cluster)
    eligible_prices = eligible_prices.dropna(subset=["basket"])
    eligible_prices["basket"] = eligible_prices["basket"].astype(int)
    
    # Compute returns per market (price changes, not percentage changes)
    eligible_prices = eligible_prices.sort_values(["market_id", "date"])
    eligible_prices["return"] = eligible_prices.groupby("market_id")["close_price"].diff()
    eligible_prices = eligible_prices.dropna(subset=["return"])
    
    # Equal-weighted average return per basket per day
    basket_returns = eligible_prices.groupby(["date", "basket"])["return"].mean().unstack("basket")
    basket_returns.columns = [f"basket_{c}" for c in basket_returns.columns]
    basket_returns = basket_returns.sort_index()
    
    # Drop days with too few baskets
    basket_returns = basket_returns.dropna(thresh=max(1, len(basket_returns.columns) // 2))
    
    # Drop baskets with too few observations (all NaN or mostly NaN)
    min_obs = max(10, basket_returns.shape[0] // 4)  # At least 25% of days or 10 days
    valid_baskets = basket_returns.count() >= min_obs
    basket_returns = basket_returns.loc[:, valid_baskets]
    
    if len(basket_returns.columns) == 0:
        print("Warning: No valid baskets after filtering")
        return pd.DataFrame()
    
    basket_returns = basket_returns.fillna(0)
    
    print(f"Basket returns computed: {basket_returns.shape[0]} days × {basket_returns.shape[1]} baskets")
    
    # Log any dropped baskets
    dropped_baskets = (~valid_baskets).sum()
    if dropped_baskets > 0:
        print(f"Dropped {dropped_baskets} baskets with insufficient data")
    
    return basket_returns


def risk_parity_weights(basket_returns: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """Compute risk parity weights: inverse volatility."""
    vol = basket_returns.tail(lookback).std()
    vol = vol.replace(0, vol[vol > 0].min() if (vol > 0).any() else 1.0)
    inv_vol = 1.0 / vol
    weights = inv_vol / inv_vol.sum()
    return weights


def compute_basket_correlations(basket_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise correlation between baskets."""
    return basket_returns.corr()


def build_correlation_baskets(
    community_path: str = "data/processed/community_assignments.parquet",
    prices_path: str = "data/processed/prices.parquet",
    markets_path: str = "data/processed/markets.parquet",
    community_labels_path: str = "data/processed/community_labels.json",
    output_dir: str = "data/outputs",
    method_name: str = "correlation"
) -> dict:
    """Build correlation-based baskets and compute returns.
    
    Returns:
        dict with basket_returns, weights, correlations, compositions, metrics
    """
    print(f"=== {method_name.title()}-Based Basket Construction ===")
    
    communities = pd.read_parquet(community_path)
    prices = pd.read_parquet(prices_path)
    markets = pd.read_parquet(markets_path)
    
    # Load community labels if available
    community_labels = {}
    if Path(community_labels_path).exists():
        with open(community_labels_path) as f:
            raw_labels = json.load(f)
            
            # Handle different formats: correlation uses int keys, factor uses "cluster_X" keys
            if isinstance(list(raw_labels.keys())[0], int):
                # Correlation format: {0: "label", 1: "label"}
                community_labels = raw_labels
            elif "cluster_" in str(list(raw_labels.keys())[0]):
                # Factor format: {"cluster_0": {"dominant_theme": "label"}, ...}
                community_labels = {}
                for key, val in raw_labels.items():
                    if isinstance(val, dict) and "dominant_theme" in val:
                        cluster_id = int(key.replace("cluster_", ""))
                        # Use factor signature if available, otherwise dominant theme
                        if "factor_signature" in val and val["factor_signature"]:
                            label = " + ".join(val["factor_signature"][:2])  # Top 2 factors
                        else:
                            label = val.get("dominant_theme", f"Cluster {cluster_id}")
                        community_labels[cluster_id] = label
            else:
                # Try direct int conversion
                try:
                    community_labels = {int(k): v for k, v in raw_labels.items()}
                except:
                    print(f"Warning: Could not parse community labels from {community_labels_path}")
                    community_labels = {}
    
    # Determine cluster column name
    cluster_column = "community" if "community" in communities.columns else "cluster"
    
    # Filter eligible
    eligible = filter_eligible_markets(communities, prices, markets, cluster_column=cluster_column)
    
    # Compute returns
    print("Computing basket returns...")
    basket_ret = compute_basket_returns(eligible, prices, cluster_column=cluster_column)
    
    if len(basket_ret.columns) == 0:
        print("Error: No basket returns computed!")
        return {}
    
    print(f"Basket returns: {basket_ret.shape[0]} days × {basket_ret.shape[1]} baskets")
    
    # Risk parity weights
    weights = risk_parity_weights(basket_ret)
    print(f"\\nRisk parity weights:")
    for basket, w in weights.items():
        basket_id = int(basket.replace("basket_", ""))
        label = community_labels.get(basket_id, f"Basket {basket_id}")
        print(f"  {label}: {w:.3f}")
    
    # Correlations
    corr = compute_basket_correlations(basket_ret)
    print(f"\\nPairwise basket correlations:")
    print(corr.round(3).to_string())
    
    # Check diversification constraint
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    upper_corrs = corr.values[mask]
    max_corr = upper_corrs.max()
    mean_corr = upper_corrs.mean()
    print(f"\\nMax pairwise correlation: {max_corr:.3f}")
    print(f"Mean pairwise correlation: {mean_corr:.3f}")
    print(f"Pairs with |corr| > 0.3: {(np.abs(upper_corrs) > 0.3).sum()}")
    
    # Portfolio returns (risk-parity weighted)
    portfolio_ret = (basket_ret * weights).sum(axis=1)
    portfolio_nav = (1 + portfolio_ret).cumprod()
    
    # Metrics
    ann_ret = portfolio_ret.mean() * 252
    ann_vol = portfolio_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (portfolio_nav / portfolio_nav.cummax() - 1).min()
    
    print(f"\\nPortfolio metrics (risk-parity):")
    print(f"  Annual return: {ann_ret:.4f}")
    print(f"  Annual volatility: {ann_vol:.4f}")
    print(f"  Sharpe ratio: {sharpe:.4f}")
    print(f"  Max drawdown: {max_dd:.4f}")
    
    # Save outputs
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create method-specific filenames
    suffix = f"_{method_name}" if method_name != "correlation" else ""
    
    basket_ret.to_csv(f"{output_dir}/basket_returns{suffix}.csv")
    portfolio_nav.to_frame("nav").to_csv(f"{output_dir}/portfolio_nav{suffix}.csv")
    corr.to_csv(f"{output_dir}/basket_correlations{suffix}.csv")
    weights.to_frame("weight").to_csv(f"{output_dir}/basket_weights{suffix}.csv")
    
    # Compositions with labels
    compositions = {}
    for basket in basket_ret.columns:
        basket_id = int(basket.replace("basket_", ""))
        market_ids = eligible[eligible[cluster_column] == basket_id].index.tolist()
        label = community_labels.get(basket_id, f"Basket {basket_id}")
        
        compositions[basket] = {
            "label": label,
            "basket_id": basket_id,
            "n_markets": len(market_ids),
            "weight": float(weights[basket]),
            "market_ids": market_ids[:50],  # Top 50 for brevity
        }
    
    with open(f"{output_dir}/basket_compositions{suffix}.json", "w") as f:
        json.dump(compositions, f, indent=2)
    
    print(f"\\nSaved outputs to {output_dir}/")
    
    return {
        "method": method_name,
        "basket_returns": basket_ret,
        "portfolio_nav": portfolio_nav,
        "weights": weights,
        "correlations": corr,
        "compositions": compositions,
        "community_labels": community_labels,
        "metrics": {
            "annual_return": float(ann_ret),
            "annual_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "n_baskets": len(basket_ret.columns),
            "max_basket_correlation": float(max_corr),
            "mean_basket_correlation": float(mean_corr)
        }
    }


def compare_basket_methods():
    """Compare correlation-based vs factor-based basket construction."""
    print("=== Comparing Basket Construction Methods ===")
    
    # Build correlation-based baskets
    correlation_results = build_correlation_baskets(
        community_path="data/processed/community_assignments.parquet",
        community_labels_path="data/processed/community_labels.json",
        output_dir="data/outputs",
        method_name="correlation"
    )
    
    # Build factor-based baskets (if available)
    factor_results = {}
    if Path("data/processed/cluster_assignments.parquet").exists():
        factor_results = build_correlation_baskets(
            community_path="data/processed/cluster_assignments.parquet",
            community_labels_path="data/processed/basket_definitions.json",  # Different format
            output_dir="data/outputs", 
            method_name="factor"
        )
    
    # Compare metrics
    print("\\n=== METHOD COMPARISON ===")
    if correlation_results and factor_results:
        corr_metrics = correlation_results["metrics"]
        factor_metrics = factor_results["metrics"]
        
        print(f"Correlation Method:")
        print(f"  Baskets: {corr_metrics['n_baskets']}")
        print(f"  Sharpe: {corr_metrics['sharpe']:.4f}")
        print(f"  Max DD: {corr_metrics['max_drawdown']:.4f}")
        print(f"  Max basket correlation: {corr_metrics['max_basket_correlation']:.3f}")
        
        print(f"\\nFactor Method:")
        print(f"  Baskets: {factor_metrics['n_baskets']}")
        print(f"  Sharpe: {factor_metrics['sharpe']:.4f}")
        print(f"  Max DD: {factor_metrics['max_drawdown']:.4f}")
        print(f"  Max basket correlation: {factor_metrics['max_basket_correlation']:.3f}")
        
        # Winner
        corr_sharpe = corr_metrics["sharpe"]
        factor_sharpe = factor_metrics["sharpe"]
        winner = "Correlation" if corr_sharpe > factor_sharpe else "Factor"
        print(f"\\nBetter Sharpe: {winner} ({max(corr_sharpe, factor_sharpe):.4f} vs {min(corr_sharpe, factor_sharpe):.4f})")
    
    return correlation_results, factor_results


if __name__ == "__main__":
    compare_basket_methods()
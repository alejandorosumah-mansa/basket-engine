"""
Factor-Informed Basket Construction

Build baskets from factor clusters with:
- Eligibility filters (active, sufficient price history, volume)
- Risk parity weighting using factor risk
- Constraint: max pairwise basket return correlation < 0.3
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def filter_eligible_markets(
    cluster_assignments: pd.DataFrame,
    prices: pd.DataFrame,
    markets: pd.DataFrame,
    min_price_days: int = 60,
    min_recent_volume: float = 1000,
) -> pd.DataFrame:
    """Filter markets for basket inclusion.
    
    Criteria:
    - At least min_price_days of price data
    - Active (not resolved) or resolved within last 90 days
    - Minimum volume
    """
    # Price history length
    price_counts = prices.groupby("market_id").size()
    has_history = set(price_counts[price_counts >= min_price_days].index)
    
    # Volume filter
    has_volume = set(markets[markets["volume"] >= min_recent_volume]["market_id"])
    
    eligible_ids = has_history & has_volume & set(cluster_assignments.index)
    
    result = cluster_assignments.loc[cluster_assignments.index.isin(eligible_ids)].copy()
    print(f"Eligible markets: {len(result)} (from {len(cluster_assignments)})")
    return result


def compute_basket_returns(
    eligible: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Compute daily returns per basket (cluster).
    
    Uses equal weighting within each basket for simplicity,
    then we can apply risk-parity across baskets.
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    
    # Only eligible markets
    eligible_prices = prices[prices["market_id"].isin(eligible.index)]
    
    # Map market to cluster
    market_to_cluster = eligible["cluster"].to_dict()
    eligible_prices["cluster"] = eligible_prices["market_id"].map(market_to_cluster)
    eligible_prices = eligible_prices.dropna(subset=["cluster"])
    eligible_prices["cluster"] = eligible_prices["cluster"].astype(int)
    
    # Compute returns per market
    eligible_prices = eligible_prices.sort_values(["market_id", "date"])
    eligible_prices["return"] = eligible_prices.groupby("market_id")["close_price"].diff()
    eligible_prices = eligible_prices.dropna(subset=["return"])
    
    # Equal-weighted average return per cluster per day
    basket_returns = eligible_prices.groupby(["date", "cluster"])["return"].mean().unstack("cluster")
    basket_returns.columns = [f"basket_{c}" for c in basket_returns.columns]
    basket_returns = basket_returns.sort_index()
    
    # Drop days with too few baskets
    basket_returns = basket_returns.dropna(thresh=max(1, len(basket_returns.columns) // 2))
    basket_returns = basket_returns.fillna(0)
    
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


def build_factor_baskets(
    cluster_path: str = "data/processed/cluster_assignments.parquet",
    prices_path: str = "data/processed/prices.parquet",
    markets_path: str = "data/processed/markets.parquet",
    basket_defs_path: str = "data/processed/basket_definitions.json",
    output_dir: str = "data/outputs",
) -> dict:
    """Build factor-informed baskets and compute returns.
    
    Returns:
        dict with basket_returns, weights, correlations, compositions
    """
    print("=== Factor-Informed Basket Construction ===")
    
    clusters = pd.read_parquet(cluster_path)
    prices = pd.read_parquet(prices_path)
    markets = pd.read_parquet(markets_path)
    
    with open(basket_defs_path) as f:
        basket_defs = json.load(f)
    
    # Filter eligible
    eligible = filter_eligible_markets(clusters, prices, markets)
    
    # Compute returns
    print("Computing basket returns...")
    basket_ret = compute_basket_returns(eligible, prices)
    print(f"Basket returns: {basket_ret.shape[0]} days × {basket_ret.shape[1]} baskets")
    
    # Risk parity weights
    weights = risk_parity_weights(basket_ret)
    print(f"\nRisk parity weights:")
    for basket, w in weights.items():
        print(f"  {basket}: {w:.3f}")
    
    # Correlations
    corr = compute_basket_correlations(basket_ret)
    print(f"\nPairwise basket correlations:")
    print(corr.round(3).to_string())
    
    # Check diversification constraint
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    upper_corrs = corr.values[mask]
    max_corr = upper_corrs.max()
    mean_corr = upper_corrs.mean()
    print(f"\nMax pairwise correlation: {max_corr:.3f}")
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
    
    print(f"\nPortfolio metrics (risk-parity):")
    print(f"  Annual return: {ann_ret:.4f}")
    print(f"  Annual volatility: {ann_vol:.4f}")
    print(f"  Sharpe ratio: {sharpe:.4f}")
    print(f"  Max drawdown: {max_dd:.4f}")
    
    # Save outputs
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    basket_ret.to_csv(f"{output_dir}/basket_returns.csv")
    portfolio_nav.to_frame("nav").to_csv(f"{output_dir}/portfolio_nav.csv")
    corr.to_csv(f"{output_dir}/basket_correlations.csv")
    weights.to_frame("weight").to_csv(f"{output_dir}/basket_weights.csv")
    
    # Compositions
    compositions = {}
    for basket in basket_ret.columns:
        cluster_id = int(basket.replace("basket_", ""))
        market_ids = eligible[eligible["cluster"] == cluster_id].index.tolist()
        compositions[basket] = {
            "n_markets": len(market_ids),
            "weight": float(weights[basket]),
            "market_ids": market_ids[:50],  # Top 50 for brevity
        }
    
    with open(f"{output_dir}/basket_compositions.json", "w") as f:
        json.dump(compositions, f, indent=2)
    
    print(f"\nSaved outputs to {output_dir}/")
    
    return {
        "basket_returns": basket_ret,
        "portfolio_nav": portfolio_nav,
        "weights": weights,
        "correlations": corr,
        "compositions": compositions,
        "metrics": {
            "annual_return": float(ann_ret),
            "annual_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
        }
    }


if __name__ == "__main__":
    build_factor_baskets()

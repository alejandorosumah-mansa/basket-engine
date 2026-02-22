"""
Compute daily basket-level returns from individual market returns.

Each basket's daily return = equal-weighted average of constituent market returns.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_basket_returns() -> pd.DataFrame:
    """Compute daily returns per theme basket.
    
    Returns:
        DataFrame with DatetimeIndex and one column per theme (daily returns).
    """
    returns = pd.read_parquet("data/processed/returns.parquet")
    markets = pd.read_parquet("data/processed/markets_with_themes.parquet")
    
    # Merge theme info
    merged = returns.merge(markets[["market_id", "theme"]], on="market_id", how="inner")
    
    # Filter out themes with very few markets (< 3 distinct markets with returns)
    theme_counts = merged.groupby("theme")["market_id"].nunique()
    valid_themes = theme_counts[theme_counts >= 3].index
    merged = merged[merged["theme"].isin(valid_themes)]
    
    # Equal-weighted average return per theme per day
    basket_returns = merged.groupby(["date", "theme"])["return"].mean().unstack("theme")
    basket_returns = basket_returns.sort_index()
    
    # Fill NaN with 0 (days where a basket had no trading)
    basket_returns = basket_returns.fillna(0)
    
    print(f"Basket returns: {basket_returns.shape}")
    print(f"Themes: {basket_returns.columns.tolist()}")
    print(f"Date range: {basket_returns.index.min()} to {basket_returns.index.max()}")
    
    return basket_returns


def compute_basket_cumulative(basket_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative NAV from daily returns."""
    return (1 + basket_returns).cumprod()


if __name__ == "__main__":
    br = compute_basket_returns()
    print(br.describe())

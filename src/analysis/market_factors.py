"""
Market-Level Factor Decomposition

Regress EACH individual market's returns against external macro factors
to produce factor loadings per market. This is the foundation for
factor-based clustering and basket construction.

External factors:
    - SPY: equity market
    - TNX: 10Y yield (rates)
    - GLD: gold (safe haven)
    - VIX: volatility (fear)
    - DX_Y_NYB: dollar strength
    - USO: oil (energy/inflation)
    - BTC_USD: crypto sentiment
    - TLT: long bonds (duration)
    - IRX: short rates (front-end)

For each market with >= MIN_OBS days of returns:
    market_return_t = α + Σ βᵢ * factor_i_t + ε_t

Outputs:
    - factor_loadings.parquet: market_id × [alpha, beta_SPY, ..., R², adj_R², idio_vol]
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

MIN_OBS = 30  # Minimum observations for regression
FACTORS = ["SPY", "TNX", "GLD", "VIX", "DX_Y_NYB", "USO", "BTC_USD", "TLT", "IRX"]


def compute_market_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns for each market from price data.
    
    Args:
        prices: DataFrame with columns [market_id, date, close_price, volume]
    
    Returns:
        Pivoted DataFrame: date × market_id with daily returns
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["market_id", "date"])
    
    # Pivot to wide format
    pivot = prices.pivot_table(index="date", columns="market_id", values="close_price")
    
    # Daily returns (simple differences for prediction markets since prices are probabilities 0-1)
    returns = pivot.diff()
    
    return returns


def compute_factor_returns(benchmarks: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns for benchmark factors.
    
    For price-based factors (SPY, GLD, etc.): pct_change
    For level-based factors (VIX, TNX, IRX): simple difference
    """
    bench = benchmarks.copy()
    
    # Level-based factors: use differences
    level_factors = ["VIX", "TNX", "IRX"]
    pct_factors = [c for c in bench.columns if c not in level_factors]
    
    factor_ret = pd.DataFrame(index=bench.index)
    
    for col in pct_factors:
        if col in bench.columns:
            factor_ret[col] = bench[col].pct_change()
    
    for col in level_factors:
        if col in bench.columns:
            factor_ret[col] = bench[col].diff()
    
    return factor_ret


def run_market_factor_regressions(
    market_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    min_obs: int = MIN_OBS,
    factors: Optional[list] = None,
) -> pd.DataFrame:
    """Run factor regression for each market.
    
    Args:
        market_returns: date × market_id DataFrame
        factor_returns: date × factor DataFrame
        min_obs: minimum overlapping observations
        factors: list of factor columns to use (default: all available)
    
    Returns:
        DataFrame indexed by market_id with columns:
            alpha, beta_<factor>, tstat_<factor>, R2, adj_R2, n_obs, idio_vol
    """
    if factors is None:
        factors = [f for f in FACTORS if f in factor_returns.columns]
    
    results = []
    n_markets = market_returns.shape[1]
    
    for i, market_id in enumerate(market_returns.columns):
        if (i + 1) % 500 == 0:
            print(f"  Processing market {i+1}/{n_markets}")
        
        y = market_returns[market_id].dropna()
        if len(y) < min_obs:
            continue
        
        # Align with factors
        common_idx = y.index.intersection(factor_returns.dropna().index)
        if len(common_idx) < min_obs:
            continue
        
        y_aligned = y.loc[common_idx]
        X_aligned = factor_returns[factors].loc[common_idx]
        
        # Drop any remaining NaN rows
        mask = ~(y_aligned.isna() | X_aligned.isna().any(axis=1))
        y_clean = y_aligned[mask]
        X_clean = X_aligned[mask]
        
        if len(y_clean) < min_obs:
            continue
        
        # Standardize X for comparable betas
        X_std = (X_clean - X_clean.mean()) / X_clean.std()
        X_std = X_std.fillna(0)  # Handle zero-variance factors
        X_const = sm.add_constant(X_std)
        
        try:
            model = sm.OLS(y_clean, X_const).fit()
        except Exception:
            continue
        
        row = {"market_id": market_id}
        row["alpha"] = model.params.get("const", 0)
        row["alpha_tstat"] = model.tvalues.get("const", 0)
        
        for factor in factors:
            row[f"beta_{factor}"] = model.params.get(factor, 0)
            row[f"tstat_{factor}"] = model.tvalues.get(factor, 0)
        
        row["R2"] = model.rsquared
        row["adj_R2"] = model.rsquared_adj
        row["n_obs"] = int(model.nobs)
        row["idio_vol"] = np.std(model.resid) * np.sqrt(252)
        row["total_vol"] = np.std(y_clean) * np.sqrt(252)
        
        results.append(row)
    
    df = pd.DataFrame(results).set_index("market_id")
    print(f"Computed factor loadings for {len(df)} markets")
    return df


def run_factor_decomposition(
    prices_path: str = "data/processed/prices.parquet",
    benchmarks_path: str = "data/processed/benchmarks.parquet",
    output_path: str = "data/processed/factor_loadings.parquet",
) -> pd.DataFrame:
    """Full pipeline: load data, compute returns, run regressions, save.
    
    Returns:
        Factor loadings DataFrame
    """
    print("=== Market-Level Factor Decomposition ===")
    
    # Load data
    print("Loading prices...")
    prices = pd.read_parquet(prices_path)
    print(f"  {prices['market_id'].nunique()} markets, {len(prices)} price observations")
    
    print("Loading benchmarks...")
    benchmarks = pd.read_parquet(benchmarks_path)
    print(f"  {len(benchmarks)} days, {len(benchmarks.columns)} factors")
    
    # Compute returns
    print("Computing market returns...")
    market_ret = compute_market_returns(prices)
    print(f"  Returns matrix: {market_ret.shape}")
    
    print("Computing factor returns...")
    factor_ret = compute_factor_returns(benchmarks)
    
    # Align date index types
    market_ret.index = pd.to_datetime(market_ret.index)
    factor_ret.index = pd.to_datetime(factor_ret.index)
    
    # Run regressions
    print("Running factor regressions...")
    loadings = run_market_factor_regressions(market_ret, factor_ret)
    
    # Summary stats
    print(f"\n=== Summary ===")
    print(f"Markets with factor loadings: {len(loadings)}")
    print(f"Mean R²: {loadings['R2'].mean():.4f}")
    print(f"Median R²: {loadings['R2'].median():.4f}")
    print(f"Markets with R² > 0.1: {(loadings['R2'] > 0.1).sum()}")
    print(f"Mean idio vol: {loadings['idio_vol'].mean():.4f}")
    
    # Factor loading summary
    beta_cols = [c for c in loadings.columns if c.startswith("beta_")]
    print(f"\nFactor loading means:")
    for col in beta_cols:
        factor = col.replace("beta_", "")
        mean_beta = loadings[col].mean()
        sig_count = (loadings[f"tstat_{factor}"].abs() > 1.96).sum()
        print(f"  {factor:12s}: mean={mean_beta:+.4f}, significant={sig_count}/{len(loadings)}")
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    loadings.to_parquet(output_path)
    print(f"\nSaved to {output_path}")
    
    return loadings


if __name__ == "__main__":
    run_factor_decomposition()

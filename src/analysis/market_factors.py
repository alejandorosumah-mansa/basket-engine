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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

MIN_OBS = 30  # Minimum observations for regression

# All available factors (expanded from 9 to 41)
FACTORS = [
    # Original 10 factors
    "SPY", "QQQ", "GLD", "TLT", "TNX", "IRX", "VIX", "DX_Y_NYB", "BTC_USD", "USO",
    
    # US Rates (new)
    "SHY", "FVX", "TLH", "TYX",
    
    # Global Rates 
    "IGLT_L", "IBGL_L", "BNDX", "EMB", "BWX", "IGOV",
    
    # Global Equity Indices
    "FTSE", "GDAXI", "N225", "000001_SS", "FCHI", "STOXX50E", 
    "HSI", "BSESN", "BVSP", "KS11",
    
    # Country ETFs
    "EWC", "EWA", "EWW", "EWT", "EIDO", "TUR", "EZA", "KSA", "EWL", "EWS",
    
    # Additional Commodities
    "NG_F"
]


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
    
    For price-based factors (ETFs, indices, commodities): pct_change
    For level-based factors (yields, VIX): simple difference
    """
    bench = benchmarks.copy()
    
    # Level-based factors: use differences (yields, volatility)
    level_factors = ["VIX", "TNX", "IRX", "FVX", "TYX"]
    pct_factors = [c for c in bench.columns if c not in level_factors]
    
    factor_ret = pd.DataFrame(index=bench.index)
    
    # Price-based factors: percentage returns
    for col in pct_factors:
        if col in bench.columns:
            factor_ret[col] = bench[col].pct_change()
    
    # Level-based factors: simple differences
    for col in level_factors:
        if col in bench.columns:
            factor_ret[col] = bench[col].diff()
    
    return factor_ret


def run_market_factor_regressions(
    market_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    min_obs: int = MIN_OBS,
    factors: Optional[list] = None,
    use_ridge: bool = None,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """Run factor regression for each market.
    
    Args:
        market_returns: date × market_id DataFrame
        factor_returns: date × factor DataFrame
        min_obs: minimum overlapping observations
        factors: list of factor columns to use (default: all available)
        use_ridge: whether to use Ridge regression. If None, auto-detect based on factor count
        ridge_alpha: regularization strength for Ridge regression
    
    Returns:
        DataFrame indexed by market_id with columns:
            alpha, beta_<factor>, [tstat_<factor>], R2, adj_R2, n_obs, idio_vol
    """
    if factors is None:
        factors = [f for f in FACTORS if f in factor_returns.columns]
    
    # Auto-detect regularization need
    if use_ridge is None:
        use_ridge = len(factors) > 15  # Use Ridge for >15 factors to avoid overfitting
    
    print(f"Using {'Ridge' if use_ridge else 'OLS'} regression with {len(factors)} factors")
    if use_ridge:
        print(f"Ridge alpha: {ridge_alpha}")
    
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
        
        row = {"market_id": market_id}
        
        try:
            if use_ridge:
                # Ridge regression
                ridge_model = Ridge(alpha=ridge_alpha, fit_intercept=True)
                ridge_model.fit(X_std, y_clean)
                
                # Predict to calculate R²
                y_pred = ridge_model.predict(X_std)
                r2 = r2_score(y_clean, y_pred)
                
                # Calculate adjusted R²
                n, p = X_std.shape
                adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1) if n > p + 1 else r2
                
                # Store results
                row["alpha"] = ridge_model.intercept_
                for j, factor in enumerate(factors):
                    row[f"beta_{factor}"] = ridge_model.coef_[j]
                    # No t-stats for Ridge (regularized), set to NaN
                    row[f"tstat_{factor}"] = np.nan
                
                row["R2"] = r2
                row["adj_R2"] = adj_r2
                row["n_obs"] = len(y_clean)
                
                # Calculate residuals for idiosyncratic volatility
                residuals = y_clean - y_pred
                row["idio_vol"] = np.std(residuals) * np.sqrt(252)
                
            else:
                # Standard OLS
                X_const = sm.add_constant(X_std)
                model = sm.OLS(y_clean, X_const).fit()
                
                row["alpha"] = model.params.get("const", 0)
                row["alpha_tstat"] = model.tvalues.get("const", 0)
                
                for factor in factors:
                    row[f"beta_{factor}"] = model.params.get(factor, 0)
                    row[f"tstat_{factor}"] = model.tvalues.get(factor, 0)
                
                row["R2"] = model.rsquared
                row["adj_R2"] = model.rsquared_adj
                row["n_obs"] = int(model.nobs)
                row["idio_vol"] = np.std(model.resid) * np.sqrt(252)
            
        except Exception as e:
            print(f"Error processing market {market_id}: {e}")
            continue
        
        row["total_vol"] = np.std(y_clean) * np.sqrt(252)
        results.append(row)
    
    df = pd.DataFrame(results).set_index("market_id")
    print(f"Computed factor loadings for {len(df)} markets")
    return df


def run_factor_decomposition(
    prices_path: str = "data/processed/prices.parquet",
    benchmarks_path: str = "data/processed/benchmarks.parquet",
    output_path: str = "data/processed/factor_loadings.parquet",
    use_ridge: bool = None,
    ridge_alpha: float = 1.0,
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
    loadings = run_market_factor_regressions(market_ret, factor_ret, use_ridge=use_ridge, ridge_alpha=ridge_alpha)
    
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
        
        # Check if t-stats are available (not available for Ridge)
        if f"tstat_{factor}" in loadings.columns and not loadings[f"tstat_{factor}"].isna().all():
            sig_count = (loadings[f"tstat_{factor}"].abs() > 1.96).sum()
            print(f"  {factor:12s}: mean={mean_beta:+.4f}, significant={sig_count}/{len(loadings)}")
        else:
            print(f"  {factor:12s}: mean={mean_beta:+.4f} (Ridge - no t-stats)")
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    loadings.to_parquet(output_path)
    print(f"\nSaved to {output_path}")
    
    return loadings


if __name__ == "__main__":
    run_factor_decomposition()

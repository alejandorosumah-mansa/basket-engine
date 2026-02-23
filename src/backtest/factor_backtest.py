"""
Backtest for Factor-Informed Baskets

Full backtest with:
- Rolling rebalance (monthly)
- Transaction costs
- Comparison with equal-weight and theme-only baskets
- All metrics: Sharpe, drawdown, Calmar, turnover
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def compute_metrics(returns: pd.Series, name: str = "") -> dict:
    """Compute standard performance metrics."""
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    nav = (1 + returns).cumprod()
    dd = nav / nav.cummax() - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    # Sortino
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0
    
    return {
        "name": name,
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "n_days": len(returns),
        "pct_positive_days": float((returns > 0).mean()),
    }


def run_backtest(
    basket_returns_path: str = "data/outputs/basket_returns.csv",
    weights_path: str = "data/outputs/basket_weights.csv",
    benchmarks_path: str = "data/processed/benchmarks.parquet",
    output_dir: str = "data/outputs",
    transaction_cost_bps: float = 10,
    rebalance_freq: str = "M",
) -> dict:
    """Run full backtest.
    
    Returns dict with metrics, NAV series, and comparison results.
    """
    print("=== Factor-Informed Basket Backtest ===")
    
    basket_ret = pd.read_csv(basket_returns_path, index_col=0, parse_dates=True)
    weights = pd.read_csv(weights_path, index_col=0)["weight"]
    benchmarks = pd.read_parquet(benchmarks_path)
    
    print(f"Basket returns: {basket_ret.shape[0]} days × {basket_ret.shape[1]} baskets")
    
    # Strategy 1: Risk-parity (from factor analysis)
    rp_returns = (basket_ret * weights).sum(axis=1)
    
    # Strategy 2: Equal weight
    ew_weights = pd.Series(1.0 / len(basket_ret.columns), index=basket_ret.columns)
    ew_returns = (basket_ret * ew_weights).sum(axis=1)
    
    # Add transaction costs (simplified: cost on rebalance days)
    rebalance_dates = basket_ret.resample(rebalance_freq).last().index
    tc = transaction_cost_bps / 10000
    
    rp_returns_tc = rp_returns.copy()
    ew_returns_tc = ew_returns.copy()
    for date in rebalance_dates:
        if date in rp_returns_tc.index:
            rp_returns_tc.loc[date] -= tc
            ew_returns_tc.loc[date] -= tc
    
    # Benchmark returns
    bench_ret = pd.DataFrame(index=benchmarks.index)
    for col in ["SPY", "GLD", "BTC_USD", "TLT"]:
        if col in benchmarks.columns:
            if col in ["VIX", "TNX", "IRX"]:
                bench_ret[col] = benchmarks[col].diff()
            else:
                bench_ret[col] = benchmarks[col].pct_change()
    
    # Align dates
    common_dates = basket_ret.index.intersection(bench_ret.index)
    
    # 60/40 portfolio
    if "SPY" in bench_ret.columns and "TLT" in bench_ret.columns:
        sixty_forty = 0.6 * bench_ret.loc[common_dates, "SPY"] + 0.4 * bench_ret.loc[common_dates, "TLT"]
        sixty_forty = sixty_forty.fillna(0)
    else:
        sixty_forty = pd.Series(0, index=common_dates)
    
    # 60/40 + prediction markets (10% allocation)
    if len(common_dates) > 0:
        rp_aligned = rp_returns.reindex(common_dates).fillna(0)
        enhanced = 0.54 * bench_ret.loc[common_dates, "SPY"].fillna(0) + \
                   0.36 * bench_ret.loc[common_dates, "TLT"].fillna(0) + \
                   0.10 * rp_aligned
    else:
        enhanced = pd.Series(0, index=common_dates)
    
    # Compute metrics
    strategies = {
        "risk_parity": rp_returns,
        "risk_parity_tc": rp_returns_tc,
        "equal_weight": ew_returns,
        "equal_weight_tc": ew_returns_tc,
    }
    
    if len(common_dates) > 100:
        strategies["60/40"] = sixty_forty
        strategies["60/40+PM(10%)"] = enhanced
        for col in ["SPY", "GLD", "BTC_USD"]:
            if col in bench_ret.columns:
                strategies[col] = bench_ret.loc[common_dates, col].fillna(0)
    
    all_metrics = []
    for name, ret in strategies.items():
        m = compute_metrics(ret.dropna(), name)
        all_metrics.append(m)
        print(f"\n{name}:")
        print(f"  Return: {m['annual_return']:.4f}, Vol: {m['annual_vol']:.4f}")
        print(f"  Sharpe: {m['sharpe']:.4f}, Sortino: {m['sortino']:.4f}")
        print(f"  MaxDD: {m['max_drawdown']:.4f}, Calmar: {m['calmar']:.4f}")
    
    # Turnover analysis
    # Simplified: assume equal turnover on rebalance days
    n_rebalances = len(rebalance_dates)
    annual_turnover = n_rebalances / (basket_ret.shape[0] / 252) * 2  # 2-way
    print(f"\nEstimated annual turnover: {annual_turnover:.1f}x")
    
    # NAV series
    nav_series = pd.DataFrame(index=basket_ret.index)
    for name, ret in strategies.items():
        if name in ["risk_parity", "risk_parity_tc", "equal_weight", "equal_weight_tc"]:
            aligned = ret.reindex(basket_ret.index).fillna(0)
            nav_series[name] = (1 + aligned).cumprod()
    
    # Cross-asset correlation
    if len(common_dates) > 100:
        print("\nCorrelation with benchmarks:")
        rp_aligned = rp_returns.reindex(common_dates).fillna(0)
        for col in ["SPY", "GLD", "BTC_USD", "TLT"]:
            if col in bench_ret.columns:
                corr = rp_aligned.corr(bench_ret.loc[common_dates, col].fillna(0))
                print(f"  vs {col}: {corr:.4f}")
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"{output_dir}/backtest_metrics.csv", index=False)
    nav_series.to_csv(f"{output_dir}/backtest_nav.csv")
    
    print(f"\nSaved to {output_dir}/")
    
    return {
        "metrics": all_metrics,
        "nav_series": nav_series,
        "strategies": strategies,
    }


if __name__ == "__main__":
    run_backtest()

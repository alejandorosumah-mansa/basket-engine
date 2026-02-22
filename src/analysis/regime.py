"""
Economic Regime Analysis

Define market regimes from benchmark data and analyze basket performance in each.

Regimes:
    - Risk-on: SPY trending up, VIX low/declining
    - Risk-off: SPY trending down, VIX high/rising
    - Rate hiking: 10Y yield rising
    - Rate cutting: 10Y yield falling
    - Dollar strong: DXY rising
    - Dollar weak: DXY falling
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.benchmarks.fetch import load_benchmarks
from src.analysis.basket_returns import compute_basket_returns


def classify_regimes(benchmarks: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Classify each trading day into economic regimes.
    
    Uses rolling lookback window to determine regime state.
    
    Args:
        benchmarks: Daily benchmark prices
        lookback: Rolling window for trend determination (trading days)
    
    Returns:
        DataFrame with boolean columns for each regime state
    """
    regimes = pd.DataFrame(index=benchmarks.index)
    
    # SPY momentum for risk-on/risk-off
    if "SPY" in benchmarks.columns:
        spy_ret = benchmarks["SPY"].pct_change(lookback)
        vix = benchmarks.get("VIX")
        
        if vix is not None:
            vix_change = vix.diff(lookback)
            regimes["risk_on"] = (spy_ret > 0) & (vix_change < 0)
            regimes["risk_off"] = (spy_ret < 0) & (vix_change > 0)
        else:
            regimes["risk_on"] = spy_ret > 0
            regimes["risk_off"] = spy_ret < 0
    
    # Rate regime from 10Y yield
    if "TNX" in benchmarks.columns:
        tnx_change = benchmarks["TNX"].diff(lookback)
        regimes["rate_hiking"] = tnx_change > 0
        regimes["rate_cutting"] = tnx_change < 0
    
    # Dollar regime
    if "DX_Y_NYB" in benchmarks.columns:
        dxy_ret = benchmarks["DX_Y_NYB"].pct_change(lookback)
        regimes["dollar_strong"] = dxy_ret > 0
        regimes["dollar_weak"] = dxy_ret < 0
    
    return regimes.dropna()


def regime_performance(basket_ret: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Calculate average basket performance during each regime.
    
    Returns:
        DataFrame: baskets (rows) × regimes (columns), values = annualized mean return
    """
    common = basket_ret.index.intersection(regimes.index)
    basket_ret = basket_ret.loc[common]
    regimes = regimes.loc[common]
    
    results = {}
    for regime in regimes.columns:
        mask = regimes[regime].astype(bool)
        if mask.sum() < 10:
            continue
        regime_returns = basket_ret.loc[mask].mean() * 252  # annualize
        results[regime] = regime_returns
    
    return pd.DataFrame(results)


def regime_statistics(basket_ret: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Detailed regime statistics: mean, vol, Sharpe, hit rate per regime per basket."""
    common = basket_ret.index.intersection(regimes.index)
    basket_ret = basket_ret.loc[common]
    regimes = regimes.loc[common]
    
    records = []
    for regime in regimes.columns:
        mask = regimes[regime].astype(bool)
        if mask.sum() < 10:
            continue
        for basket in basket_ret.columns:
            r = basket_ret.loc[mask, basket]
            records.append({
                "regime": regime,
                "basket": basket,
                "n_days": int(mask.sum()),
                "mean_daily": round(r.mean(), 6),
                "annualized_return": round(r.mean() * 252, 4),
                "annualized_vol": round(r.std() * np.sqrt(252), 4),
                "sharpe": round((r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0, 2),
                "hit_rate": round((r > 0).mean(), 3),
            })
    
    return pd.DataFrame(records)


def plot_regime_performance(perf: pd.DataFrame, output_dir: Path):
    """Heatmap of basket returns by regime."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    display = perf.copy()
    display.index = [i.replace("_", " ").title() for i in display.index]
    display.columns = [c.replace("_", " ").title() for c in display.columns]
    
    import seaborn as sns
    sns.heatmap(display, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax)
    
    ax.set_title("Basket Performance by Economic Regime (Annualized Returns)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Economic Regime")
    ax.set_ylabel("Prediction Market Basket")
    
    plt.figtext(0.99, 0.01, "Generated: 2026-02-22", ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout()
    
    path = output_dir / "regime_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def run_regime_analysis(output_dir: Path = Path("data/outputs/charts")) -> dict:
    """Run full regime analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ECONOMIC REGIME ANALYSIS")
    print("=" * 60)
    
    benchmarks = load_benchmarks()
    basket_ret = compute_basket_returns()
    
    # Align dates
    common = basket_ret.index.intersection(benchmarks.index)
    basket_ret = basket_ret.loc[common]
    benchmarks = benchmarks.loc[common]
    
    regimes = classify_regimes(benchmarks)
    print(f"\nRegime classification: {len(regimes)} days")
    print(f"Regime prevalence:")
    for col in regimes.columns:
        pct = regimes[col].mean() * 100
        print(f"  {col:20s}: {pct:.1f}% of days")
    
    perf = regime_performance(basket_ret, regimes)
    print(f"\nPerformance matrix (annualized returns):")
    print(perf.round(4).to_string())
    
    stats = regime_statistics(basket_ret, regimes)
    
    # Key insights
    print("\n--- Key Regime Insights ---")
    if "risk_off" in perf.columns:
        best_riskoff = perf["risk_off"].idxmax()
        worst_riskoff = perf["risk_off"].idxmin()
        print(f"Best basket in risk-off: {best_riskoff} ({perf.loc[best_riskoff, 'risk_off']:.4f})")
        print(f"Worst basket in risk-off: {worst_riskoff} ({perf.loc[worst_riskoff, 'risk_off']:.4f})")
    
    plot_regime_performance(perf, output_dir)
    
    return {"regimes": regimes, "performance": perf, "statistics": stats}


if __name__ == "__main__":
    run_regime_analysis()

"""
Cross-Asset Correlation Analysis

For each prediction market basket, calculate rolling and static correlations
with traditional asset benchmarks (SPY, QQQ, GLD, TLT, VIX, BTC, etc.).

Key question: Which baskets track equities? Gold? Rates? Crypto?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.benchmarks.fetch import load_benchmarks, CLEAN_NAMES
from src.analysis.basket_returns import compute_basket_returns


def compute_benchmark_returns(benchmarks: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns for benchmarks.
    
    For yield indices (TNX, IRX), use absolute change (like prediction markets).
    For price indices, use log returns.
    """
    yield_cols = [c for c in benchmarks.columns if c in ("TNX", "IRX", "VIX")]
    price_cols = [c for c in benchmarks.columns if c not in yield_cols]
    
    bench_ret = pd.DataFrame(index=benchmarks.index)
    
    for c in price_cols:
        bench_ret[c] = np.log(benchmarks[c] / benchmarks[c].shift(1))
    
    for c in yield_cols:
        bench_ret[c] = benchmarks[c].diff()
    
    return bench_ret.iloc[1:]  # drop first NaN row


def align_data(basket_returns: pd.DataFrame, bench_returns: pd.DataFrame):
    """Align basket and benchmark returns to common dates."""
    # Ensure both have datetime index
    basket_returns.index = pd.to_datetime(basket_returns.index)
    bench_returns.index = pd.to_datetime(bench_returns.index)
    
    common_idx = basket_returns.index.intersection(bench_returns.index)
    return basket_returns.loc[common_idx], bench_returns.loc[common_idx]


def static_correlation(basket_ret: pd.DataFrame, bench_ret: pd.DataFrame) -> pd.DataFrame:
    """Full-sample Pearson correlation matrix: baskets × benchmarks."""
    combined = pd.concat([basket_ret, bench_ret], axis=1)
    corr = combined.corr()
    
    # Extract cross-block: baskets (rows) × benchmarks (cols)
    cross_corr = corr.loc[basket_ret.columns, bench_ret.columns]
    return cross_corr


def rolling_correlation(basket_ret: pd.DataFrame, bench_ret: pd.DataFrame,
                        windows: list = [30, 60, 90]) -> dict:
    """Rolling correlation between each basket and each benchmark.
    
    Returns:
        dict of {window: DataFrame} where each DataFrame has MultiIndex columns
        (basket, benchmark).
    """
    results = {}
    for w in windows:
        rolling_corrs = {}
        for basket in basket_ret.columns:
            for bench in bench_ret.columns:
                key = (basket, bench)
                rolling_corrs[key] = basket_ret[basket].rolling(w).corr(bench_ret[bench])
        results[w] = pd.DataFrame(rolling_corrs)
    return results


def find_notable_correlations(cross_corr: pd.DataFrame, threshold: float = 0.3) -> list:
    """Identify basket-benchmark pairs with absolute correlation above threshold."""
    notable = []
    for basket in cross_corr.index:
        for bench in cross_corr.columns:
            val = cross_corr.loc[basket, bench]
            if abs(val) >= threshold:
                bench_name = CLEAN_NAMES.get(bench, bench)
                notable.append({
                    "basket": basket,
                    "benchmark": bench_name,
                    "correlation": round(val, 3),
                    "direction": "positive" if val > 0 else "negative",
                })
    return sorted(notable, key=lambda x: abs(x["correlation"]), reverse=True)


def plot_correlation_heatmap(cross_corr: pd.DataFrame, output_dir: Path):
    """Generate baskets × benchmarks correlation heatmap."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Rename columns for readability
    display_corr = cross_corr.copy()
    display_corr.columns = [CLEAN_NAMES.get(c, c) for c in display_corr.columns]
    display_corr.index = [i.replace("_", " ").title() for i in display_corr.index]
    
    sns.heatmap(display_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Cross-Asset Correlation: Prediction Market Baskets × Traditional Benchmarks",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Traditional Asset Benchmarks")
    ax.set_ylabel("Prediction Market Baskets")
    
    plt.figtext(0.99, 0.01, "Generated: 2026-02-22", ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout()
    
    path = output_dir / "cross_asset_correlation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_rolling_correlation_spy(basket_ret: pd.DataFrame, bench_ret: pd.DataFrame,
                                  output_dir: Path):
    """Time series of 60-day rolling correlation with SPY for all baskets."""
    spy_col = "SPY"
    if spy_col not in bench_ret.columns:
        print("SPY not found in benchmarks, skipping rolling correlation plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for basket in basket_ret.columns:
        rolling = basket_ret[basket].rolling(60).corr(bench_ret[spy_col])
        label = basket.replace("_", " ").title()
        ax.plot(rolling.index, rolling.values, label=label, alpha=0.7)
    
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_title("60-Day Rolling Correlation with S&P 500 (SPY)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_ylim(-1, 1)
    
    plt.figtext(0.99, 0.01, "Generated: 2026-02-22", ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout()
    
    path = output_dir / "rolling_correlation_spy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def run_cross_asset_analysis(output_dir: Path = Path("data/outputs/charts")) -> dict:
    """Run full cross-asset analysis and return results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CROSS-ASSET CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Load data
    benchmarks = load_benchmarks()
    bench_ret = compute_benchmark_returns(benchmarks)
    basket_ret = compute_basket_returns()
    
    # Align
    basket_ret, bench_ret = align_data(basket_ret, bench_ret)
    print(f"Aligned data: {len(basket_ret)} common trading days")
    
    # Static correlation
    cross_corr = static_correlation(basket_ret, bench_ret)
    print("\nCross-Asset Correlation Matrix:")
    print(cross_corr.round(3).to_string())
    
    # Notable correlations
    notable = find_notable_correlations(cross_corr)
    print(f"\nNotable correlations (|r| >= 0.3): {len(notable)}")
    for n in notable[:10]:
        print(f"  {n['basket']} ↔ {n['benchmark']}: {n['correlation']} ({n['direction']})")
    
    # Rolling correlations
    rolling = rolling_correlation(basket_ret, bench_ret)
    
    # Charts
    plot_correlation_heatmap(cross_corr, output_dir)
    plot_rolling_correlation_spy(basket_ret, bench_ret, output_dir)
    
    return {
        "cross_corr": cross_corr,
        "notable": notable,
        "rolling": rolling,
        "n_common_days": len(basket_ret),
    }


if __name__ == "__main__":
    results = run_cross_asset_analysis()

"""
External (Macro) Factor Model

Regress each prediction market basket against real macro factors:
    - SPY returns (equity)
    - TNX changes (10Y yield)
    - GLD returns (gold)
    - VIX changes (volatility)
    - DX_Y_NYB returns (dollar)
    - USO returns (oil)
    - BTC_USD returns (crypto)

Key question: How much of basket variance is explained by traditional markets?
    Low R² → genuine diversification
    High R² → just a proxy for existing assets
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from src.benchmarks.fetch import load_benchmarks, CLEAN_NAMES
from src.analysis.cross_asset import compute_benchmark_returns, align_data
from src.analysis.basket_returns import compute_basket_returns


MACRO_FACTORS = ["SPY", "TNX", "GLD", "VIX", "DX_Y_NYB", "USO", "BTC_USD"]


def run_macro_regressions(basket_ret: pd.DataFrame, bench_ret: pd.DataFrame) -> pd.DataFrame:
    """Regress each basket against macro factors.
    
    Returns:
        DataFrame with regression results per basket per factor.
    """
    # Use only available macro factors
    available_factors = [f for f in MACRO_FACTORS if f in bench_ret.columns]
    print(f"Available macro factors: {available_factors}")
    
    results = []
    
    for basket in basket_ret.columns:
        y = basket_ret[basket].dropna()
        X = bench_ret[available_factors].loc[y.index].dropna()
        common = y.index.intersection(X.index)
        y, X = y.loc[common], X.loc[common]
        
        if len(y) < 30:
            continue
        
        X_const = sm.add_constant(X)
        try:
            model = sm.OLS(y, X_const).fit(cov_type="HC1")
        except Exception as e:
            print(f"  Regression failed for {basket}: {e}")
            continue
        
        for factor in available_factors:
            results.append({
                "basket": basket,
                "factor": factor,
                "factor_name": CLEAN_NAMES.get(factor, factor),
                "beta": round(model.params.get(factor, 0), 6),
                "tstat": round(model.tvalues.get(factor, 0), 2),
                "pvalue": round(model.pvalues.get(factor, 1), 4),
            })
        
        results.append({
            "basket": basket,
            "factor": "_summary",
            "factor_name": "Summary",
            "beta": round(model.params.get("const", 0), 8),
            "tstat": round(model.tvalues.get("const", 0), 2),
            "pvalue": round(model.pvalues.get("const", 1), 4),
            "r_squared": round(model.rsquared, 4),
            "adj_r_squared": round(model.rsquared_adj, 4),
            "n_obs": int(model.nobs),
            "f_stat": round(model.fvalue, 2),
            "f_pvalue": round(model.f_pvalue, 4),
        })
    
    return pd.DataFrame(results)


def plot_macro_r_squared(results_df: pd.DataFrame, output_dir: Path):
    """Bar chart: how much variance explained by macro factors per basket."""
    summaries = results_df[results_df["factor"] == "_summary"].set_index("basket")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    baskets = summaries.index
    r2 = summaries["r_squared"].values
    adj_r2 = summaries["adj_r_squared"].values
    
    x = np.arange(len(baskets))
    ax.bar(x - 0.15, r2, 0.3, label="R²", color="steelblue", alpha=0.8)
    ax.bar(x + 0.15, adj_r2, 0.3, label="Adjusted R²", color="coral", alpha=0.8)
    
    ax.set_title("Macro Factor Model: Variance Explained by Traditional Markets",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("R² (Proportion of Variance Explained)")
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", " ").title() for b in baskets], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, max(0.5, max(r2) * 1.2) if len(r2) > 0 else 0.5)
    
    # Add interpretation line
    ax.axhline(y=0.1, color="green", linestyle="--", alpha=0.5, label="Low correlation threshold")
    ax.text(len(baskets) - 0.5, 0.11, "Low R² = genuine diversifier", fontsize=8, 
            color="green", ha="right")
    
    plt.figtext(0.99, 0.01, "Generated: 2026-02-22", ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout()
    
    path = output_dir / "macro_r_squared.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def run_macro_factor_model(output_dir: Path = Path("data/outputs/charts")) -> dict:
    """Run the full macro factor analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EXTERNAL MACRO FACTOR MODEL")
    print("=" * 60)
    
    benchmarks = load_benchmarks()
    bench_ret = compute_benchmark_returns(benchmarks)
    basket_ret = compute_basket_returns()
    basket_ret, bench_ret = align_data(basket_ret, bench_ret)
    
    results_df = run_macro_regressions(basket_ret, bench_ret)
    
    # Print summary
    summaries = results_df[results_df["factor"] == "_summary"]
    print("\n--- Macro Factor Model Summary ---")
    print(f"{'Basket':30s} {'R²':>8} {'Adj R²':>8} {'F-stat':>8} {'N':>5}")
    print("-" * 65)
    for _, row in summaries.iterrows():
        print(f"  {row['basket']:28s} {row.get('r_squared', 0):>8.4f} "
              f"{row.get('adj_r_squared', 0):>8.4f} {row.get('f_stat', 0):>8.2f} "
              f"{row.get('n_obs', 0):>5.0f}")
    
    # Key insight
    avg_r2 = summaries["r_squared"].mean()
    print(f"\nAverage R²: {avg_r2:.4f}")
    if avg_r2 < 0.1:
        print("→ Prediction market baskets offer GENUINE DIVERSIFICATION")
        print("  (Very little variance explained by traditional macro factors)")
    elif avg_r2 < 0.3:
        print("→ Moderate correlation with traditional markets")
        print("  (Some diversification benefit, but partial overlap)")
    else:
        print("→ High correlation with traditional markets")
        print("  (Baskets are partially proxies for existing assets)")
    
    plot_macro_r_squared(results_df, output_dir)
    
    return {"results": results_df, "avg_r2": avg_r2}


if __name__ == "__main__":
    run_macro_factor_model()

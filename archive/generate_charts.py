"""Generate all charts for RESEARCH.md."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

sns.set_theme(style="whitegrid", font_scale=1.1)
OUT = Path("data/outputs/charts")
OUT.mkdir(parents=True, exist_ok=True)


def chart_coverage_funnel():
    """Fig 1: Data coverage funnel."""
    stages = [
        ("All Markets", 20180),
        ("With Prices", 11223),
        ("≥30 Days History", 2721),
        ("Factor Loadings", 2666),
        ("Basket Eligible", 2134),
    ]
    labels, vals = zip(*stages)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(vals)), vals, color=sns.color_palette("Blues_d", len(vals)))
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                f"{v:,}", va="center", fontweight="bold")
    ax.set_xlabel("Number of Markets")
    ax.set_title("Data Coverage Funnel")
    plt.tight_layout()
    plt.savefig(OUT / "01_coverage_funnel.png", dpi=150)
    plt.close()


def chart_r2_distribution():
    """Fig 2: Distribution of R² from factor model."""
    loadings = pd.read_parquet("data/processed/factor_loadings.parquet")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(loadings["R2"], bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(loadings["R2"].mean(), color="red", linestyle="--", label=f"Mean: {loadings['R2'].mean():.3f}")
    ax.axvline(loadings["R2"].median(), color="orange", linestyle="--", label=f"Median: {loadings['R2'].median():.3f}")
    ax.set_xlabel("R²")
    ax.set_ylabel("Count")
    ax.set_title("Factor Model R² Distribution (2,666 Markets)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "02_r2_distribution.png", dpi=150)
    plt.close()


def chart_cluster_profiles():
    """Fig 3: Heatmap of cluster factor profiles."""
    clusters = pd.read_parquet("data/processed/cluster_assignments.parquet")
    beta_cols = [c for c in clusters.columns if c.startswith("beta_")]
    profiles = clusters.groupby("cluster")[beta_cols].mean()
    profiles.columns = [c.replace("beta_", "") for c in profiles.columns]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(profiles.T, cmap="RdBu_r", center=0, annot=True, fmt=".4f",
                ax=ax, cbar_kws={"label": "Mean β"})
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Factor")
    ax.set_title("Mean Factor Loadings by Cluster")
    plt.tight_layout()
    plt.savefig(OUT / "03_cluster_profiles.png", dpi=150)
    plt.close()


def chart_cluster_sizes():
    """Fig 4: Cluster sizes."""
    clusters = pd.read_parquet("data/processed/cluster_assignments.parquet")
    sizes = clusters["cluster"].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Set2", len(sizes))
    bars = ax.bar(sizes.index, sizes.values, color=colors)
    for bar, v in zip(bars, sizes.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(v), ha="center", fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Markets")
    ax.set_title("Markets per Cluster")
    plt.tight_layout()
    plt.savefig(OUT / "04_cluster_sizes.png", dpi=150)
    plt.close()


def chart_basket_correlations():
    """Fig 5: Basket correlation heatmap."""
    corr = pd.read_csv("data/outputs/basket_correlations.csv", index_col=0)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, annot=True, fmt=".3f",
                ax=ax, vmin=-0.5, vmax=0.5, square=True)
    ax.set_title("Pairwise Basket Correlation")
    plt.tight_layout()
    plt.savefig(OUT / "05_basket_correlations.png", dpi=150)
    plt.close()


def chart_backtest_nav():
    """Fig 6: Backtest NAV curves."""
    nav = pd.read_csv("data/outputs/backtest_nav.csv", index_col=0, parse_dates=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in nav.columns:
        label = col.replace("_", " ").title()
        ax.plot(nav.index, nav[col], label=label, linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.set_title("Basket Strategy NAV (Backtest)")
    ax.legend()
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / "06_backtest_nav.png", dpi=150)
    plt.close()


def chart_benchmark_correlation():
    """Fig 7: Correlation with traditional assets."""
    benchmarks = {"SPY": -0.097, "GLD": -0.020, "BTC": 0.009, "TLT": -0.003}
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in benchmarks.values()]
    bars = ax.bar(benchmarks.keys(), benchmarks.values(), color=colors, edgecolor="white")
    for bar, v in zip(bars, benchmarks.values()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.002 if v >= 0 else -0.008),
                f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylabel("Correlation")
    ax.set_title("PM Basket Correlation with Traditional Assets")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(-0.15, 0.05)
    plt.tight_layout()
    plt.savefig(OUT / "07_benchmark_correlation.png", dpi=150)
    plt.close()


def chart_risk_parity_weights():
    """Fig 8: Risk parity weights."""
    weights = pd.read_csv("data/outputs/basket_weights.csv", index_col=0)["weight"]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette("Set2", len(weights))
    labels = [f"Basket {i}\n({w:.1%})" for i, w in enumerate(weights)]
    ax.pie(weights, labels=labels, colors=colors, autopct="", startangle=90)
    ax.set_title("Risk Parity Basket Weights")
    plt.tight_layout()
    plt.savefig(OUT / "08_risk_parity_weights.png", dpi=150)
    plt.close()


def chart_semantic_exposures():
    """Fig 9: Top semantic exposures for categorical events."""
    with open("data/processed/semantic_exposures.json") as f:
        exposures = json.load(f)
    
    # Select interesting events
    selected = {
        "fed-decision-in-october": "Fed Oct",
        "fed-interest-rates-november-2024": "Fed Nov '24",
        "us-x-venezuela-military-engagement-by": "US-Venezuela",
        "balance-of-power-2024-election": "Election Balance",
        "what-price-will-ethereum-hit-in-2025": "ETH 2025",
    }
    
    dims = ["rates", "dollar", "equity", "gold", "oil", "crypto", "volatility", "growth"]
    data = []
    for slug, label in selected.items():
        if slug in exposures:
            for dim in dims:
                data.append({"Event": label, "Factor": dim, "Exposure": exposures[slug]["net_exposure"].get(dim, 0)})
    
    df = pd.DataFrame(data)
    pivot = df.pivot(index="Event", columns="Factor", values="Exposure")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot[dims], cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax,
                cbar_kws={"label": "Net Exposure"})
    ax.set_title("Semantic Factor Exposures (Selected Categorical Events)")
    plt.tight_layout()
    plt.savefig(OUT / "09_semantic_exposures.png", dpi=150)
    plt.close()


def chart_metrics_comparison():
    """Fig 10: Strategy comparison bar chart."""
    metrics = pd.read_csv("data/outputs/backtest_metrics.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, metric, label in zip(axes,
            ["sharpe", "max_drawdown", "annual_vol"],
            ["Sharpe Ratio", "Max Drawdown", "Annual Volatility"]):
        vals = metrics.set_index("name")[metric]
        colors = ["#3498db" if "parity" in n or "weight" in n else "#95a5a6" for n in vals.index]
        vals.plot(kind="barh", ax=ax, color=colors)
        ax.set_title(label)
        ax.set_xlabel("")
    
    plt.tight_layout()
    plt.savefig(OUT / "10_metrics_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    print("Generating charts...")
    for name, fn in [
        ("Coverage funnel", chart_coverage_funnel),
        ("R² distribution", chart_r2_distribution),
        ("Cluster profiles", chart_cluster_profiles),
        ("Cluster sizes", chart_cluster_sizes),
        ("Basket correlations", chart_basket_correlations),
        ("Backtest NAV", chart_backtest_nav),
        ("Benchmark correlation", chart_benchmark_correlation),
        ("Risk parity weights", chart_risk_parity_weights),
        ("Semantic exposures", chart_semantic_exposures),
        ("Metrics comparison", chart_metrics_comparison),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    print(f"\nAll charts saved to {OUT}/")

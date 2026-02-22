"""
Barra-Style Internal Factor Model for Prediction Market Baskets

Factors (constructed from prediction market baskets themselves):
    1. Market Factor: average return across all prediction markets
    2. Risk Appetite Factor: spread between high-vol and low-vol baskets
    3. Political Uncertainty Factor: US elections + legal/regulatory
    4. Geopolitical Factor: middle_east + russia_ukraine + china_us
    5. Monetary Policy Factor: fed_monetary_policy + us_economic
    6. Crypto Sentiment Factor: crypto_digital

For each basket:
    basket_return = α + β₁(market) + β₂(risk_appetite) + β₃(political) 
                    + β₄(geopolitical) + β₅(monetary) + β₆(crypto) + ε

Reports: factor loadings, t-stats, R², p-values, alpha, idiosyncratic vol
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from src.analysis.basket_returns import compute_basket_returns


# Factor definitions: which themes contribute to each factor
FACTOR_DEFINITIONS = {
    "political": ["us_elections", "legal_regulatory"],
    "geopolitical": ["middle_east", "russia_ukraine", "china_us"],
    "monetary": ["fed_monetary_policy", "us_economic"],
    "crypto": ["crypto_digital"],
}


def construct_factors(basket_ret: pd.DataFrame) -> pd.DataFrame:
    """Construct prediction market factors from basket returns.
    
    Args:
        basket_ret: DataFrame of daily basket returns (themes as columns)
    
    Returns:
        DataFrame with factor time series
    """
    factors = pd.DataFrame(index=basket_ret.index)
    
    # 1. Market factor: equal-weighted average of all baskets
    factors["market"] = basket_ret.mean(axis=1)
    
    # 2. Risk appetite: spread between high-vol and low-vol baskets
    rolling_vol = basket_ret.rolling(60).std()
    median_vol = rolling_vol.median(axis=1)
    high_vol_mask = rolling_vol.gt(median_vol, axis=0)
    low_vol_mask = ~high_vol_mask
    
    high_vol_ret = basket_ret.where(high_vol_mask).mean(axis=1)
    low_vol_ret = basket_ret.where(low_vol_mask).mean(axis=1)
    factors["risk_appetite"] = high_vol_ret - low_vol_ret
    
    # 3-6. Thematic factors
    for factor_name, themes in FACTOR_DEFINITIONS.items():
        available = [t for t in themes if t in basket_ret.columns]
        if available:
            factors[factor_name] = basket_ret[available].mean(axis=1)
        else:
            factors[factor_name] = 0.0
    
    factors = factors.fillna(0)
    return factors


def run_factor_regressions(basket_ret: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Run OLS regression for each basket against the factor model.
    
    Returns:
        DataFrame with columns: basket, factor, beta, tstat, pvalue, r_squared, 
        alpha, alpha_tstat, idio_vol
    """
    results = []
    
    for basket in basket_ret.columns:
        y = basket_ret[basket].dropna()
        X = factors.loc[y.index].dropna()
        common = y.index.intersection(X.index)
        y = y.loc[common]
        X = X.loc[common]
        
        if len(y) < 30:
            continue
        
        X_const = sm.add_constant(X)
        try:
            model = sm.OLS(y, X_const).fit(cov_type="HC1")  # White robust SE
        except Exception as e:
            print(f"  Regression failed for {basket}: {e}")
            continue
        
        # Extract results
        alpha = model.params.get("const", 0)
        alpha_tstat = model.tvalues.get("const", 0)
        alpha_pval = model.pvalues.get("const", 1)
        r_squared = model.rsquared
        idio_vol = np.std(model.resid) * np.sqrt(252)  # annualized
        
        for factor_name in factors.columns:
            results.append({
                "basket": basket,
                "factor": factor_name,
                "beta": round(model.params.get(factor_name, 0), 4),
                "tstat": round(model.tvalues.get(factor_name, 0), 2),
                "pvalue": round(model.pvalues.get(factor_name, 1), 4),
            })
        
        # Add summary row
        results.append({
            "basket": basket,
            "factor": "_summary",
            "beta": round(alpha, 6),
            "tstat": round(alpha_tstat, 2),
            "pvalue": round(alpha_pval, 4),
            "r_squared": round(r_squared, 4),
            "idio_vol": round(idio_vol, 4),
        })
    
    return pd.DataFrame(results)


def plot_factor_loadings(results_df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart of factor betas per basket."""
    # Filter out summary rows and get betas
    betas = results_df[results_df["factor"] != "_summary"].pivot(
        index="basket", columns="factor", values="beta"
    )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    betas.plot(kind="bar", ax=ax, width=0.8)
    
    ax.set_title("Internal Factor Loadings (Barra-Style)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Prediction Market Basket")
    ax.set_ylabel("Factor Beta (Loading)")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.legend(title="Factor", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    labels = [l.get_text().replace("_", " ").title() for l in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    
    plt.figtext(0.99, 0.01, "Generated: 2026-02-22", ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout()
    
    path = output_dir / "factor_loadings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_alpha_decomposition(results_df: pd.DataFrame, basket_ret: pd.DataFrame,
                              factors: pd.DataFrame, output_dir: Path):
    """Alpha vs factor-explained returns per basket."""
    summaries = results_df[results_df["factor"] == "_summary"].set_index("basket")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    baskets = summaries.index
    alphas = summaries["beta"].values  # alpha stored in beta column for summary
    r2s = summaries.get("r_squared", pd.Series(0, index=baskets)).values
    
    # Annualize alpha (daily → annual, ~252 days)
    annual_alpha = alphas * 252
    
    x = np.arange(len(baskets))
    total_ret = basket_ret[baskets].mean() * 252  # annualized mean return
    factor_ret = total_ret.values - annual_alpha
    
    ax.bar(x, factor_ret, label="Factor-Explained Return", color="steelblue", alpha=0.8)
    ax.bar(x, annual_alpha, bottom=factor_ret, label="Alpha (Excess Return)", color="coral", alpha=0.8)
    
    ax.set_title("Return Decomposition: Alpha vs Factor-Explained", fontsize=13, fontweight="bold")
    ax.set_ylabel("Annualized Return")
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", " ").title() for b in baskets], rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    
    plt.figtext(0.99, 0.01, "Generated: 2026-02-22", ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout()
    
    path = output_dir / "alpha_decomposition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def run_factor_model(output_dir: Path = Path("data/outputs/charts")) -> dict:
    """Run the full internal factor model analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BARRA-STYLE INTERNAL FACTOR MODEL")
    print("=" * 60)
    
    basket_ret = compute_basket_returns()
    factors = construct_factors(basket_ret)
    
    print(f"\nFactors: {factors.columns.tolist()}")
    print(f"Factor correlations:\n{factors.corr().round(3)}")
    
    results_df = run_factor_regressions(basket_ret, factors)
    
    # Print summary
    summaries = results_df[results_df["factor"] == "_summary"]
    print("\n--- Factor Model Summary ---")
    for _, row in summaries.iterrows():
        print(f"  {row['basket']:30s}  R²={row.get('r_squared', 'N/A'):>8}  "
              f"α={row['beta']:>10.6f} (t={row['tstat']:>5.2f})  "
              f"Idio Vol={row.get('idio_vol', 'N/A')}")
    
    # Charts
    plot_factor_loadings(results_df, output_dir)
    plot_alpha_decomposition(results_df, basket_ret, factors, output_dir)
    
    return {
        "results": results_df,
        "factors": factors,
        "basket_returns": basket_ret,
    }


if __name__ == "__main__":
    run_factor_model()

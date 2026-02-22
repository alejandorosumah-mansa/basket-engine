"""
Deep Economic Analysis Module

Comprehensive institutional-grade analysis including:
- Multi-window rolling correlations & conditional correlation
- Full Barra factor model with diagnostic tests
- Extended macro factor model with Granger causality & PCA
- Extended regime analysis with transition matrices
- Portfolio construction (MVO, 60/40 improvement, risk budgeting)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from src.benchmarks.fetch import load_benchmarks, CLEAN_NAMES
from src.analysis.cross_asset import compute_benchmark_returns, align_data
from src.analysis.basket_returns import compute_basket_returns
from src.analysis.factor_model import construct_factors

DATE_STAMP = "Generated: 2026-02-22"
DPI = 150


def _sig_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


# ──────────────────────────────────────────────
# 1. CROSS-ASSET DEEP DIVE
# ──────────────────────────────────────────────

def multi_window_rolling_correlations(basket_ret, bench_ret, output_dir):
    """Rolling correlations at 7d, 30d, 60d, 90d windows."""
    windows = [7, 30, 60, 90]
    spy_col = "SPY"
    if spy_col not in bench_ret.columns:
        print("SPY not in benchmarks, skipping multi-window rolling corr")
        return {}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    results = {}

    for ax, w in zip(axes.flat, windows):
        for basket in basket_ret.columns:
            rc = basket_ret[basket].rolling(w).corr(bench_ret[spy_col])
            results[(basket, w)] = rc
            ax.plot(rc.index, rc.values, alpha=0.6, linewidth=0.8,
                    label=basket.replace("_", " ").title())
        ax.axhline(0, color="black", ls="--", alpha=0.3)
        ax.set_title(f"{w}-Day Rolling Correlation with SPY", fontsize=11)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Correlation")

    axes[0, 0].legend(fontsize=6, ncol=2, loc="upper left")
    plt.suptitle("Multi-Window Rolling Correlation: Baskets vs S&P 500",
                 fontsize=14, fontweight="bold")
    plt.figtext(0.99, 0.01, DATE_STAMP, ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    path = output_dir / "rolling_correlation_multiwindow.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return results


def conditional_correlation_analysis(basket_ret, bench_ret, output_dir):
    """Compare correlations during normal times vs market stress (VIX > 75th pctile)."""
    spy_col, vix_col = "SPY", "VIX"
    if spy_col not in bench_ret.columns or vix_col not in bench_ret.columns:
        print("Missing SPY/VIX for conditional correlation")
        return {}

    benchmarks = load_benchmarks()
    vix_level = benchmarks["VIX"].reindex(basket_ret.index).ffill()
    vix_75 = vix_level.quantile(0.75)
    vix_25 = vix_level.quantile(0.25)

    stress_mask = vix_level > vix_75
    calm_mask = vix_level < vix_25

    corr_stress = {}
    corr_calm = {}
    for basket in basket_ret.columns:
        if stress_mask.sum() > 10:
            corr_stress[basket] = basket_ret.loc[stress_mask, basket].corr(
                bench_ret.loc[stress_mask, spy_col])
        if calm_mask.sum() > 10:
            corr_calm[basket] = basket_ret.loc[calm_mask, basket].corr(
                bench_ret.loc[calm_mask, spy_col])

    fig, ax = plt.subplots(figsize=(12, 6))
    baskets = list(corr_calm.keys())
    x = np.arange(len(baskets))
    w = 0.35
    calm_vals = [corr_calm.get(b, 0) for b in baskets]
    stress_vals = [corr_stress.get(b, 0) for b in baskets]

    ax.bar(x - w/2, calm_vals, w, label=f"Calm (VIX < {vix_25:.0f})", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, stress_vals, w, label=f"Stress (VIX > {vix_75:.0f})", color="crimson", alpha=0.8)
    ax.set_title("Conditional Correlation with SPY: Calm vs Stress Periods",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Correlation with SPY")
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", " ").title() for b in baskets], rotation=45, ha="right")
    ax.legend()
    ax.axhline(0, color="black", ls="--", alpha=0.3)
    plt.figtext(0.99, 0.01, DATE_STAMP, ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout()
    path = output_dir / "conditional_correlation.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    return {"calm": corr_calm, "stress": corr_stress,
            "vix_75th": float(vix_75), "vix_25th": float(vix_25),
            "stress_days": int(stress_mask.sum()), "calm_days": int(calm_mask.sum())}


# ──────────────────────────────────────────────
# 2. BARRA FACTOR MODEL - DEEP
# ──────────────────────────────────────────────

def deep_factor_model(basket_ret, output_dir):
    """Full Barra factor model with diagnostic tests."""
    factors = construct_factors(basket_ret)
    results = {}

    for basket in basket_ret.columns:
        y = basket_ret[basket].dropna()
        X = factors.loc[y.index].dropna()
        common = y.index.intersection(X.index)
        y, X = y.loc[common], X.loc[common]
        if len(y) < 30:
            continue

        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        model_hc = sm.OLS(y, X_const).fit(cov_type="HC1")

        # Diagnostic tests
        dw = durbin_watson(model.resid)
        try:
            _, white_p, _, _ = het_white(model.resid, X_const)
        except Exception:
            white_p = np.nan
        try:
            bg_lm, bg_p, _, _ = acorr_breusch_godfrey(model, nlags=5)
        except Exception:
            bg_lm, bg_p = np.nan, np.nan

        # ADF on residuals
        adf_stat, adf_p, _, _, _, _ = adfuller(model.resid, maxlag=10, autolag="AIC")

        # Variance decomposition
        var_decomp = {}
        total_var = y.var()
        for fac in factors.columns:
            contrib = model.params.get(fac, 0) * X[fac]
            var_decomp[fac] = float(contrib.var() / total_var) if total_var > 0 else 0
        var_decomp["residual"] = float(model.resid.var() / total_var) if total_var > 0 else 0

        # Information ratio
        alpha_annual = model.params.get("const", 0) * 252
        tracking_error = model.resid.std() * np.sqrt(252)
        info_ratio = alpha_annual / tracking_error if tracking_error > 0 else 0

        results[basket] = {
            "model": model_hc,
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "f_stat": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "durbin_watson": dw,
            "white_test_p": white_p,
            "breusch_godfrey_p": bg_p,
            "adf_resid_p": adf_p,
            "var_decomposition": var_decomp,
            "info_ratio": info_ratio,
            "alpha_annual": alpha_annual,
            "tracking_error": tracking_error,
            "params": dict(model_hc.params),
            "tvalues": dict(model_hc.tvalues),
            "pvalues": dict(model_hc.pvalues),
            "n_obs": int(model.nobs),
        }

    # Plot variance decomposition
    _plot_variance_decomposition(results, output_dir)
    # Plot regression coefficients with CI
    _plot_regression_coefficients(results, factors.columns, output_dir)

    return results


def _plot_variance_decomposition(results, output_dir):
    """Stacked bar chart of variance decomposition."""
    baskets = list(results.keys())
    if not baskets:
        return
    all_factors = list(list(results.values())[0]["var_decomposition"].keys())

    data = {f: [results[b]["var_decomposition"].get(f, 0) for b in baskets] for f in all_factors}
    df = pd.DataFrame(data, index=baskets)

    fig, ax = plt.subplots(figsize=(14, 7))
    df.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Variance Decomposition: Factor Contributions to Basket Variance",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Proportion of Total Variance")
    ax.set_xticklabels([b.replace("_", " ").title() for b in baskets], rotation=45, ha="right")
    ax.legend(title="Factor", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.figtext(0.99, 0.01, DATE_STAMP, ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout()
    path = output_dir / "variance_decomposition.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _plot_regression_coefficients(results, factor_names, output_dir):
    """Regression coefficients with 95% CI for each basket."""
    baskets = list(results.keys())
    if not baskets:
        return

    n_baskets = len(baskets)
    fig, axes = plt.subplots(1, min(n_baskets, 6), figsize=(min(n_baskets, 6) * 3, 6),
                             sharey=True, squeeze=False)
    axes = axes.flat

    for i, basket in enumerate(baskets[:6]):
        ax = axes[i]
        m = results[basket]["model"]
        params = m.params.drop("const", errors="ignore")
        ci = m.conf_int().drop("const", errors="ignore")
        pvals = m.pvalues.drop("const", errors="ignore")

        y_pos = np.arange(len(params))
        colors = ["crimson" if p < 0.05 else "steelblue" for p in pvals]
        ax.barh(y_pos, params.values, color=colors, alpha=0.7)
        ax.errorbar(params.values, y_pos,
                    xerr=[params.values - ci.iloc[:, 0].values, ci.iloc[:, 1].values - params.values],
                    fmt="none", color="black", capsize=3)
        ax.axvline(0, color="black", ls="--", alpha=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace("_", " ").title() for f in params.index], fontsize=8)
        ax.set_title(basket.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("Beta")

    plt.suptitle("Factor Loadings with 95% Confidence Intervals (red = p<0.05)",
                 fontsize=12, fontweight="bold")
    plt.figtext(0.99, 0.01, DATE_STAMP, ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    path = output_dir / "regression_coefficients_ci.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────
# 3. MACRO FACTOR MODEL - DEEP
# ──────────────────────────────────────────────

def deep_macro_model(basket_ret, bench_ret, benchmarks, output_dir):
    """Extended macro model with additional factors, Granger, PCA."""
    # Construct extended factors
    macro_ret = bench_ret.copy()

    # Credit spread proxy: TLT return - GLD return (simplified)
    if "TLT" in macro_ret.columns and "GLD" in macro_ret.columns:
        macro_ret["CREDIT_SPREAD"] = macro_ret["TLT"] - macro_ret["GLD"]

    # Yield curve slope proxy: TNX level change (already in bench)
    if "TNX" in benchmarks.columns and "IRX" in benchmarks.columns:
        slope = benchmarks["TNX"] - benchmarks["IRX"]
        macro_ret["YIELD_CURVE_SLOPE"] = slope.diff().reindex(macro_ret.index)

    # Inflation proxy: GLD - TLT (TIPS-like)
    if "GLD" in macro_ret.columns and "TLT" in macro_ret.columns:
        macro_ret["INFLATION_PROXY"] = macro_ret["GLD"] - macro_ret["TLT"]

    macro_ret = macro_ret.fillna(0)

    # Align
    common = basket_ret.index.intersection(macro_ret.index)
    br = basket_ret.loc[common]
    mr = macro_ret.loc[common]

    # Factor selection
    factor_cols = [c for c in mr.columns if c in [
        "SPY", "TNX", "GLD", "VIX", "DX_Y_NYB", "USO", "BTC_USD",
        "CREDIT_SPREAD", "YIELD_CURVE_SLOPE", "INFLATION_PROXY"
    ]]
    X = mr[factor_cols]

    # Run regressions
    reg_results = {}
    for basket in br.columns:
        y = br[basket].dropna()
        Xb = X.loc[y.index].dropna()
        ci = y.index.intersection(Xb.index)
        y, Xb = y.loc[ci], Xb.loc[ci]
        if len(y) < 30:
            continue
        Xc = sm.add_constant(Xb)
        model = sm.OLS(y, Xc).fit(cov_type="HC1")
        reg_results[basket] = {
            "model": model,
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "f_stat": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "params": dict(model.params),
            "tvalues": dict(model.tvalues),
            "pvalues": dict(model.pvalues),
        }

    # Granger causality
    granger_results = _granger_causality(br, mr, factor_cols)

    # PCA
    pca_results = _run_pca(br, output_dir)

    # Out-of-sample R²
    oos_r2 = _out_of_sample_r2(br, X)

    return {
        "regressions": reg_results,
        "granger": granger_results,
        "pca": pca_results,
        "oos_r2": oos_r2,
        "factor_cols": factor_cols,
    }


def _granger_causality(basket_ret, macro_ret, factor_cols, maxlag=5):
    """Test if macro factors Granger-cause basket returns."""
    results = []
    for basket in basket_ret.columns:
        for fac in factor_cols:
            try:
                df = pd.DataFrame({"basket": basket_ret[basket], "factor": macro_ret[fac]}).dropna()
                if len(df) < maxlag * 3:
                    continue
                gc = grangercausalitytests(df[["basket", "factor"]], maxlag=maxlag, verbose=False)
                # Best lag by min p-value
                best_lag = min(gc.keys(), key=lambda k: gc[k][0]["ssr_ftest"][1])
                f_stat = gc[best_lag][0]["ssr_ftest"][0]
                p_val = gc[best_lag][0]["ssr_ftest"][1]
                results.append({
                    "basket": basket, "factor": fac,
                    "best_lag": best_lag, "f_stat": round(f_stat, 3),
                    "p_value": round(p_val, 4), "sig": _sig_stars(p_val)
                })
            except Exception:
                pass
    return pd.DataFrame(results) if results else pd.DataFrame()


def _run_pca(basket_ret, output_dir):
    """PCA on basket returns."""
    clean = basket_ret.dropna()
    if len(clean) < 30:
        return {}

    n_comp = min(len(clean.columns), 10)
    pca = PCA(n_components=n_comp)
    pca.fit(clean)

    explained = pca.explained_variance_ratio_
    cum_explained = np.cumsum(explained)
    loadings = pd.DataFrame(pca.components_.T, index=clean.columns,
                            columns=[f"PC{i+1}" for i in range(n_comp)])

    # Scree plot + loadings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(range(1, n_comp+1), explained, alpha=0.7, color="steelblue", label="Individual")
    ax1.plot(range(1, n_comp+1), cum_explained, "ro-", label="Cumulative")
    ax1.axhline(0.9, ls="--", color="gray", alpha=0.5, label="90% threshold")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained")
    ax1.set_title("PCA Scree Plot", fontsize=12, fontweight="bold")
    ax1.legend()

    # Loadings heatmap for top 5 PCs
    display_loadings = loadings.iloc[:, :5]
    display_loadings.index = [i.replace("_", " ").title() for i in display_loadings.index]
    sns.heatmap(display_loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax2)
    ax2.set_title("PCA Loadings (Top 5 Components)", fontsize=12, fontweight="bold")

    plt.figtext(0.99, 0.01, DATE_STAMP, ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout()
    path = output_dir / "pca_analysis.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # How many PCs for 90%?
    n_for_90 = int(np.searchsorted(cum_explained, 0.9) + 1)

    return {
        "explained_variance": explained.tolist(),
        "cumulative": cum_explained.tolist(),
        "loadings": loadings,
        "n_for_90pct": n_for_90,
    }


def _out_of_sample_r2(basket_ret, X):
    """Walk-forward OOS R² with 70/30 split."""
    results = {}
    split = int(len(basket_ret) * 0.7)
    for basket in basket_ret.columns:
        y = basket_ret[basket]
        common = y.index.intersection(X.index)
        y, Xb = y.loc[common], X.loc[common]
        if len(y) < 60:
            continue

        y_train, y_test = y.iloc[:split], y.iloc[split:]
        X_train, X_test = Xb.iloc[:split], Xb.iloc[split:]

        Xc_train = sm.add_constant(X_train)
        Xc_test = sm.add_constant(X_test)
        try:
            model = sm.OLS(y_train, Xc_train).fit()
            y_pred = model.predict(Xc_test)
            ss_res = ((y_test - y_pred) ** 2).sum()
            ss_tot = ((y_test - y_test.mean()) ** 2).sum()
            oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            results[basket] = round(float(oos_r2), 4)
        except Exception:
            pass
    return results


# ──────────────────────────────────────────────
# 4. REGIME ANALYSIS - DEEP
# ──────────────────────────────────────────────

def deep_regime_analysis(basket_ret, benchmarks, output_dir):
    """Extended regime analysis with VIX-based, dollar, oil, and election regimes."""
    common = basket_ret.index.intersection(benchmarks.index)
    basket_ret = basket_ret.loc[common]
    bm = benchmarks.loc[common]

    regimes = pd.DataFrame(index=common)

    # VIX regimes
    if "VIX" in bm.columns:
        regimes["high_vol"] = bm["VIX"] > 20
        regimes["low_vol"] = bm["VIX"] < 15

    # Dollar regimes
    if "DX_Y_NYB" in bm.columns:
        dxy_ma = bm["DX_Y_NYB"].rolling(20).mean()
        regimes["dollar_strong"] = bm["DX_Y_NYB"] > dxy_ma
        regimes["dollar_weak"] = bm["DX_Y_NYB"] < dxy_ma

    # Oil shock: USO daily move > 2 std
    if "USO" in bm.columns:
        uso_ret = bm["USO"].pct_change()
        uso_std = uso_ret.rolling(60).std()
        regimes["oil_shock"] = uso_ret.abs() > 2 * uso_std

    # Election season (Sep-Nov of even years)
    regimes["election_season"] = regimes.index.to_series().apply(
        lambda d: d.month >= 9 and d.month <= 11 and d.year % 2 == 0
    ).values

    # Risk on/off from SPY
    if "SPY" in bm.columns and "VIX" in bm.columns:
        spy_ret20 = bm["SPY"].pct_change(20)
        vix_chg20 = bm["VIX"].diff(20)
        regimes["risk_on"] = (spy_ret20 > 0) & (vix_chg20 < 0)
        regimes["risk_off"] = (spy_ret20 < 0) & (vix_chg20 > 0)

    regimes = regimes.dropna()

    # Performance per regime
    regime_perf = {}
    regime_stats = []
    for reg in regimes.columns:
        mask = regimes[reg].astype(bool)
        if mask.sum() < 5:
            continue
        for basket in basket_ret.columns:
            r = basket_ret.loc[mask, basket].dropna()
            if len(r) < 5:
                continue
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            max_dd = (r.cumsum() - r.cumsum().cummax()).min()
            regime_stats.append({
                "regime": reg, "basket": basket,
                "n_days": int(mask.sum()),
                "ann_return": round(ann_ret, 4),
                "ann_vol": round(ann_vol, 4),
                "sharpe": round(sharpe, 2),
                "max_drawdown": round(float(max_dd), 4),
                "hit_rate": round(float((r > 0).mean()), 3),
            })
        regime_perf[reg] = basket_ret.loc[mask].mean() * 252

    stats_df = pd.DataFrame(regime_stats)

    # Transition matrix for risk_on/risk_off
    transition = _compute_transition_matrix(regimes)

    # Drawdown chart with regime shading
    _plot_drawdown_with_regimes(basket_ret, regimes, output_dir)

    # Regime transition diagram
    _plot_transition_matrix(transition, output_dir)

    return {"stats": stats_df, "transition": transition, "regimes": regimes}


def _compute_transition_matrix(regimes):
    """Compute daily transition probabilities between risk states."""
    if "risk_on" not in regimes.columns or "risk_off" not in regimes.columns:
        return {}

    states = []
    for i, row in regimes.iterrows():
        if row.get("risk_on", False):
            states.append("risk_on")
        elif row.get("risk_off", False):
            states.append("risk_off")
        else:
            states.append("neutral")

    transitions = {"risk_on": {}, "risk_off": {}, "neutral": {}}
    for s1, s2 in zip(states[:-1], states[1:]):
        transitions[s1][s2] = transitions[s1].get(s2, 0) + 1

    # Normalize
    for s1 in transitions:
        total = sum(transitions[s1].values())
        if total > 0:
            transitions[s1] = {s2: round(c / total, 3) for s2, c in transitions[s1].items()}

    return transitions


def _plot_transition_matrix(transition, output_dir):
    """Plot regime transition matrix as heatmap."""
    if not transition:
        return
    states = ["risk_on", "neutral", "risk_off"]
    mat = np.zeros((3, 3))
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            mat[i, j] = transition.get(s1, {}).get(s2, 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    labels = ["Risk On", "Neutral", "Risk Off"]
    sns.heatmap(mat, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels,
                cmap="YlOrRd", ax=ax)
    ax.set_title("Regime Transition Probabilities (Daily)", fontsize=13, fontweight="bold")
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    plt.figtext(0.99, 0.01, DATE_STAMP, ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout()
    path = output_dir / "regime_transition.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _plot_drawdown_with_regimes(basket_ret, regimes, output_dir):
    """Drawdown chart per basket with regime shading."""
    cum = basket_ret.cumsum()
    dd = cum - cum.cummax()

    n = len(basket_ret.columns)
    fig, axes = plt.subplots(min(n, 6), 1, figsize=(14, min(n, 6) * 2.5), sharex=True)
    if min(n, 6) == 1:
        axes = [axes]

    for ax, basket in zip(axes, list(basket_ret.columns)[:6]):
        ax.fill_between(dd.index, dd[basket].values, 0, alpha=0.5, color="crimson")
        ax.set_ylabel(basket.replace("_", " ").title(), fontsize=8)

        # Shade risk-off
        if "risk_off" in regimes.columns:
            mask = regimes["risk_off"].reindex(dd.index, fill_value=False).astype(bool)
            ax.fill_between(dd.index, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=mask, alpha=0.15, color="gray", label="Risk-Off")

    plt.suptitle("Basket Drawdowns with Risk-Off Regime Shading",
                 fontsize=13, fontweight="bold")
    plt.figtext(0.99, 0.01, DATE_STAMP, ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    path = output_dir / "drawdown_regime_shading.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────
# 5. PORTFOLIO CONSTRUCTION
# ──────────────────────────────────────────────

def portfolio_construction_analysis(basket_ret, bench_ret, output_dir):
    """Mean-variance optimization, 60/40 improvement, risk budgeting."""
    # Annualize
    mu = basket_ret.mean() * 252
    sigma = basket_ret.cov() * 252
    n = len(mu)

    # Min variance portfolio (analytical)
    sigma_inv = np.linalg.pinv(sigma.values)
    ones = np.ones(n)
    w_min_var = sigma_inv @ ones / (ones @ sigma_inv @ ones)
    w_min_var = w_min_var / w_min_var.sum()  # normalize

    # Max Sharpe (tangency) portfolio: w* ∝ Σ⁻¹ μ
    w_tangency = sigma_inv @ mu.values
    w_tangency = w_tangency / w_tangency.sum()

    # Equal weight
    w_equal = np.ones(n) / n

    # Portfolio metrics helper
    def port_metrics(w, label):
        ret = w @ mu.values
        vol = np.sqrt(w @ sigma.values @ w)
        sharpe = ret / vol if vol > 0 else 0
        # Daily returns for drawdown
        daily_ret = (basket_ret.values @ w)
        cum = np.cumsum(daily_ret)
        max_dd = float(np.min(cum - np.maximum.accumulate(cum)))
        return {"portfolio": label, "return": round(ret, 4), "vol": round(vol, 4),
                "sharpe": round(sharpe, 2), "max_drawdown": round(max_dd, 4),
                "weights": {c: round(float(wi), 4) for c, wi in zip(mu.index, w)}}

    portfolios = [
        port_metrics(w_equal, "Equal Weight"),
        port_metrics(w_min_var, "Min Variance"),
        port_metrics(w_tangency, "Max Sharpe"),
    ]

    # 60/40 improvement analysis
    if "SPY" in bench_ret.columns and "TLT" in bench_ret.columns:
        common = basket_ret.index.intersection(bench_ret.index)
        spy_r = bench_ret.loc[common, "SPY"]
        tlt_r = bench_ret.loc[common, "TLT"]
        pm_r = basket_ret.loc[common].mean(axis=1)  # equal-weight PM basket

        # 60/40
        trad = 0.6 * spy_r + 0.4 * tlt_r
        # 55/35/10 (with PM)
        enhanced = 0.55 * spy_r + 0.35 * tlt_r + 0.10 * pm_r

        def _calc_sharpe_dd(r, label):
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            cum = r.cumsum()
            max_dd = float((cum - cum.cummax()).min())
            return {"portfolio": label, "return": round(ann_ret, 4),
                    "vol": round(ann_vol, 4), "sharpe": round(sharpe, 2),
                    "max_drawdown": round(max_dd, 4)}

        trad_m = _calc_sharpe_dd(trad, "60/40 Traditional")
        enhanced_m = _calc_sharpe_dd(enhanced, "55/35/10 (+PM)")
        pm_only_m = _calc_sharpe_dd(pm_r, "PM Equal-Weight")

        improvement = {
            "traditional_60_40": trad_m,
            "enhanced_with_pm": enhanced_m,
            "pm_only": pm_only_m,
            "sharpe_improvement": round(enhanced_m["sharpe"] - trad_m["sharpe"], 3),
            "dd_improvement": round(enhanced_m["max_drawdown"] - trad_m["max_drawdown"], 4),
        }

        # Plot
        _plot_portfolio_improvement(trad, enhanced, pm_r, trad_m, enhanced_m, output_dir)
        _plot_efficient_frontier(mu, sigma, w_min_var, w_tangency, output_dir)
    else:
        improvement = {}

    # Risk budgeting
    risk_budget = {}
    total_port_var = w_equal @ sigma.values @ w_equal
    for i, c in enumerate(mu.index):
        marginal = (sigma.values @ w_equal)[i]
        contrib = w_equal[i] * marginal
        risk_budget[c] = round(float(contrib / total_port_var) if total_port_var > 0 else 0, 4)

    return {
        "portfolios": portfolios,
        "improvement": improvement,
        "risk_budget": risk_budget,
    }


def _plot_portfolio_improvement(trad, enhanced, pm_r, trad_m, enhanced_m, output_dir):
    """Plot cumulative NAV: 60/40 vs 60/40+PM."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

    cum_trad = (1 + trad).cumprod()
    cum_enh = (1 + enhanced).cumprod()
    cum_pm = (1 + pm_r).cumprod()

    ax1.plot(cum_trad.index, cum_trad.values, label=f"60/40 (Sharpe={trad_m['sharpe']:.2f})",
             linewidth=1.5, color="steelblue")
    ax1.plot(cum_enh.index, cum_enh.values, label=f"55/35/10 +PM (Sharpe={enhanced_m['sharpe']:.2f})",
             linewidth=1.5, color="coral")
    ax1.set_title("Portfolio Improvement: Adding Prediction Markets to 60/40",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Cumulative NAV")
    ax1.legend()

    # Drawdown comparison
    dd_trad = cum_trad / cum_trad.cummax() - 1
    dd_enh = cum_enh / cum_enh.cummax() - 1
    ax2.fill_between(dd_trad.index, dd_trad.values, 0, alpha=0.5, color="steelblue", label="60/40")
    ax2.fill_between(dd_enh.index, dd_enh.values, 0, alpha=0.5, color="coral", label="55/35/10 +PM")
    ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=8)

    plt.figtext(0.99, 0.01, DATE_STAMP, ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout()
    path = output_dir / "portfolio_improvement.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _plot_efficient_frontier(mu, sigma, w_mv, w_tang, output_dir):
    """Plot efficient frontier with and without prediction markets."""
    n = len(mu)
    np.random.seed(42)
    n_ports = 5000

    rets, vols = [], []
    for _ in range(n_ports):
        w = np.random.dirichlet(np.ones(n))
        r = w @ mu.values
        v = np.sqrt(w @ sigma.values @ w)
        rets.append(r)
        vols.append(v)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(vols, rets, c=np.array(rets) / np.array(vols), cmap="viridis",
                    alpha=0.3, s=5)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    # Min var point
    mv_ret = w_mv @ mu.values
    mv_vol = np.sqrt(w_mv @ sigma.values @ w_mv)
    ax.scatter(mv_vol, mv_ret, marker="*", s=200, c="red", label="Min Variance", zorder=5)

    # Tangency
    t_ret = w_tang @ mu.values
    t_vol = np.sqrt(w_tang @ sigma.values @ w_tang)
    ax.scatter(t_vol, t_ret, marker="*", s=200, c="gold", label="Max Sharpe", zorder=5)

    # Equal weight
    eq_ret = mu.mean()
    eq_vol = np.sqrt(np.ones(n)/n @ sigma.values @ np.ones(n)/n)
    ax.scatter(eq_vol, eq_ret, marker="D", s=100, c="green", label="Equal Weight", zorder=5)

    ax.set_title("Efficient Frontier: Prediction Market Baskets", fontsize=13, fontweight="bold")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.legend()
    plt.figtext(0.99, 0.01, DATE_STAMP, ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout()
    path = output_dir / "efficient_frontier.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────
# MAIN RUNNER
# ──────────────────────────────────────────────

def run_deep_analysis(output_dir=None):
    """Run all deep economic analyses."""
    if output_dir is None:
        output_dir = Path("data/outputs/charts")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("DEEP ECONOMIC ANALYSIS - INSTITUTIONAL GRADE")
    print("=" * 70)

    # Load data
    benchmarks = load_benchmarks()
    bench_ret = compute_benchmark_returns(benchmarks)
    basket_ret = compute_basket_returns()
    basket_ret, bench_ret = align_data(basket_ret, bench_ret)
    print(f"\nData: {len(basket_ret)} common days, {len(basket_ret.columns)} baskets, "
          f"{len(bench_ret.columns)} benchmarks")

    # 1. Cross-asset deep dive
    print("\n" + "=" * 60)
    print("1. CROSS-ASSET DEEP DIVE")
    print("=" * 60)
    multi_window_rolling_correlations(basket_ret, bench_ret, output_dir)
    cond_corr = conditional_correlation_analysis(basket_ret, bench_ret, output_dir)
    if cond_corr:
        print(f"\nConditional Correlation Analysis:")
        print(f"  Calm periods ({cond_corr['calm_days']} days, VIX < {cond_corr['vix_25th']:.1f}):")
        for b, v in cond_corr["calm"].items():
            print(f"    {b}: {v:.3f}")
        print(f"  Stress periods ({cond_corr['stress_days']} days, VIX > {cond_corr['vix_75th']:.1f}):")
        for b, v in cond_corr["stress"].items():
            print(f"    {b}: {v:.3f}")

    # 2. Barra Factor Model - Deep
    print("\n" + "=" * 60)
    print("2. BARRA FACTOR MODEL - DEEP DIAGNOSTICS")
    print("=" * 60)
    factor_results = deep_factor_model(basket_ret, output_dir)
    print(f"\n{'Basket':25s} {'R²':>6} {'Adj R²':>7} {'DW':>6} {'White p':>8} {'BG p':>8} {'IR':>6}")
    print("-" * 75)
    for basket, res in factor_results.items():
        print(f"  {basket:23s} {res['r_squared']:>6.4f} {res['adj_r_squared']:>7.4f} "
              f"{res['durbin_watson']:>6.2f} {res['white_test_p']:>8.4f} "
              f"{res['breusch_godfrey_p']:>8.4f} {res['info_ratio']:>6.2f}")

    # 3. Macro Factor Model - Deep
    print("\n" + "=" * 60)
    print("3. MACRO FACTOR MODEL - DEEP")
    print("=" * 60)
    macro_results = deep_macro_model(basket_ret, bench_ret, benchmarks, output_dir)

    if not macro_results["granger"].empty:
        sig_granger = macro_results["granger"][macro_results["granger"]["p_value"] < 0.05]
        print(f"\nSignificant Granger causality results ({len(sig_granger)} pairs):")
        for _, row in sig_granger.head(15).iterrows():
            print(f"  {row['factor']:>15s} → {row['basket']:25s} "
                  f"F={row['f_stat']:>7.3f} p={row['p_value']:.4f}{row['sig']}")

    if macro_results["pca"]:
        pca = macro_results["pca"]
        print(f"\nPCA: {pca['n_for_90pct']} components explain 90% of variance")
        for i, ev in enumerate(pca["explained_variance"][:5]):
            print(f"  PC{i+1}: {ev:.1%}")

    if macro_results["oos_r2"]:
        print(f"\nOut-of-Sample R²:")
        for b, r2 in macro_results["oos_r2"].items():
            print(f"  {b:25s}: {r2:.4f}")

    # 4. Regime Analysis - Deep
    print("\n" + "=" * 60)
    print("4. REGIME ANALYSIS - DEEP")
    print("=" * 60)
    regime_results = deep_regime_analysis(basket_ret, benchmarks, output_dir)
    if not regime_results["stats"].empty:
        print("\nRegime Performance Summary:")
        pivot = regime_results["stats"].pivot_table(
            index="basket", columns="regime", values="sharpe", aggfunc="first"
        )
        print(pivot.round(2).to_string())

    if regime_results["transition"]:
        print("\nTransition Matrix:")
        for s1, trans in regime_results["transition"].items():
            print(f"  {s1}: {trans}")

    # 5. Portfolio Construction
    print("\n" + "=" * 60)
    print("5. PORTFOLIO CONSTRUCTION")
    print("=" * 60)
    port_results = portfolio_construction_analysis(basket_ret, bench_ret, output_dir)

    print("\nOptimal Portfolios:")
    for p in port_results["portfolios"]:
        print(f"  {p['portfolio']:20s}: Return={p['return']:>7.4f}  Vol={p['vol']:>7.4f}  "
              f"Sharpe={p['sharpe']:>5.2f}  MaxDD={p['max_drawdown']:>8.4f}")

    if port_results["improvement"]:
        imp = port_results["improvement"]
        print(f"\n60/40 vs 60/40 + Prediction Markets:")
        print(f"  Traditional 60/40:  Sharpe={imp['traditional_60_40']['sharpe']:.2f}  "
              f"MaxDD={imp['traditional_60_40']['max_drawdown']:.4f}")
        print(f"  Enhanced 55/35/10:  Sharpe={imp['enhanced_with_pm']['sharpe']:.2f}  "
              f"MaxDD={imp['enhanced_with_pm']['max_drawdown']:.4f}")
        print(f"  Sharpe improvement: {imp['sharpe_improvement']:+.3f}")

    print(f"\nRisk Budget (Equal Weight):")
    for b, rb in port_results["risk_budget"].items():
        print(f"  {b:25s}: {rb:.1%}")

    print("\n" + "=" * 70)
    print("DEEP ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        "conditional_correlation": cond_corr,
        "factor_model": factor_results,
        "macro_model": macro_results,
        "regime": regime_results,
        "portfolio": port_results,
    }


if __name__ == "__main__":
    run_deep_analysis()

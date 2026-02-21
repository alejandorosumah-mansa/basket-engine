"""Backtest runner: simulate basket performance over historical period."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from functools import partial
from typing import Optional
import yaml
import json
import logging

from src.construction.weighting import compute_weights
from src.construction.rebalance import RebalanceEngine, get_rebalance_dates

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def compute_sharpe(nav_series: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Annualized Sharpe ratio."""
    returns = nav_series.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    rf_daily = risk_free_rate / 252
    excess = returns.mean() - rf_daily
    return float(excess / returns.std() * np.sqrt(252))


def compute_max_drawdown(nav_series: pd.Series) -> float:
    """Maximum drawdown (negative number)."""
    running_max = nav_series.cummax()
    drawdowns = (nav_series - running_max) / running_max
    return float(drawdowns.min())


def compute_total_return(nav_series: pd.Series) -> float:
    """Total return over period."""
    return float(nav_series.iloc[-1] / nav_series.iloc[0] - 1)


def run_backtest(
    method: str,
    returns_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    classifications_df: pd.DataFrame,
    start_date: str = "2025-08-01",
    end_date: str = "2026-02-01",
    basket_settings: Optional[dict] = None,
    weighting_settings: Optional[dict] = None,
) -> dict:
    """Run a single backtest for a given weighting method.
    
    Returns dict with nav_series, metrics, rebalance_log.
    """
    settings = load_settings()
    bs = basket_settings or settings["basket"]
    ws = weighting_settings or settings["weighting"]
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Get all serious market IDs
    serious = classifications_df[
        ~classifications_df["primary_theme"].isin(["sports_entertainment", "uncategorized"])
    ]
    serious_ids = set(serious["event_id"])
    
    # Filter data to date range
    returns_df = returns_df.copy()
    returns_df["date"] = pd.to_datetime(returns_df["date"])
    returns_df = returns_df[(returns_df["date"] >= start) & (returns_df["date"] <= end)]
    returns_df = returns_df[returns_df["market_id"].isin(serious_ids)]
    
    prices_df = prices_df.copy()
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    
    # Get all trading dates
    all_dates = sorted(returns_df["date"].unique())
    if len(all_dates) == 0:
        logger.warning(f"No trading data for {method} between {start_date} and {end_date}")
        return {"nav_series": pd.DataFrame(), "metrics": {}, "rebalance_log": []}
    
    # Rebalance dates
    rebal_dates = get_rebalance_dates(start, end)
    
    # Volume and liquidity lookups
    vol_by_market = prices_df.groupby("market_id")["volume"].sum().to_dict()
    
    # Create weighting function
    def weight_fn(market_ids, returns_df=None, volumes=None, liquidity=None):
        return compute_weights(
            method=method,
            market_ids=market_ids,
            returns_df=returns_df,
            volumes=volumes or vol_by_market,
            liquidity=liquidity or vol_by_market,  # Fixed: use liquidity parameter correctly
            vol_window=ws.get("volatility_window_days", 30),
            max_weight=bs["max_single_weight"],
            min_weight=bs["min_single_weight"],
            min_markets=bs["min_markets"],
            max_markets=bs["max_markets"],
            liquidity_cap=ws.get("liquidity_cap_enabled", True),
        )
    
    engine = RebalanceEngine(
        basket_id=f"backtest_{method}",
        weighting_fn=weight_fn,
        initial_nav=100.0,
        min_markets=bs["min_markets"],
    )
    
    # Run day by day
    for date in all_dates:
        date_dt = pd.Timestamp(date).to_pydatetime()
        
        # Check if rebalance day
        is_rebal = any(
            abs((date_dt - rd).days) <= 1 for rd in rebal_dates
            if rd <= date_dt
        ) and (not engine.rebalance_events or 
               (date_dt - datetime.fromisoformat(engine.rebalance_events[-1].date)).days >= 25)
        
        # Initial rebalance or scheduled
        if not engine.current_weights or is_rebal:
            # Find eligible markets: those with returns data up to this date
            avail = returns_df[returns_df["date"] <= date]
            days_per = avail.groupby("market_id").size()
            eligible_ids = list(days_per[days_per >= 5].index)  # Relaxed for backtest
            
            if len(eligible_ids) >= bs["min_markets"]:
                # Get returns up to this date for vol calculation
                hist_returns = returns_df[returns_df["date"] <= date]
                # For backtest, we distinguish between volume (for volume_weighted) 
                # and liquidity (for risk parity cap) - use volume as proxy but could be different
                engine.rebalance(
                    date=date_dt,
                    eligible_market_ids=eligible_ids,
                    returns_df=hist_returns,
                    volumes=vol_by_market,
                    liquidity=vol_by_market,  # In a real system, this would be separate liquidity data
                )
        
        # Get today's returns
        day_returns = returns_df[returns_df["date"] == date]
        ret_dict = dict(zip(day_returns["market_id"], day_returns["return"]))
        
        engine.step(date_dt, ret_dict)
    
    # Compute metrics
    nav_df = engine.get_nav_series()
    metrics = {}
    if len(nav_df) > 1:
        metrics = {
            "method": method,
            "total_return": compute_total_return(nav_df["nav"]),
            "sharpe_ratio": compute_sharpe(nav_df["nav"]),
            "max_drawdown": compute_max_drawdown(nav_df["nav"]),
            "n_rebalances": len(engine.rebalance_events),
            "avg_turnover": np.mean([e.turnover for e in engine.rebalance_events]) if engine.rebalance_events else 0,
            "final_nav": nav_df["nav"].iloc[-1],
            "n_days": len(nav_df),
        }
    
    return {
        "nav_series": nav_df,
        "metrics": metrics,
        "rebalance_log": engine.get_rebalance_log(),
    }


def run_all_backtests() -> dict:
    """Run backtests for all three weighting methods and save results."""
    settings = load_settings()
    
    # Load data
    returns_df = pd.read_parquet(DATA_DIR / "returns.parquet")
    prices_df = pd.read_parquet(DATA_DIR / "prices.parquet")
    classifications_df = pd.read_csv(DATA_DIR / "llm_classifications_full.csv")
    
    start = settings["backtest"]["start_date"]
    end = settings["backtest"]["end_date"]
    
    methods = ["risk_parity_liquidity_cap", "equal", "volume_weighted"]
    results = {}
    
    for method in methods:
        logger.info(f"Running backtest: {method}")
        results[method] = run_backtest(
            method=method,
            returns_df=returns_df,
            prices_df=prices_df,
            classifications_df=classifications_df,
            start_date=start,
            end_date=end,
        )
    
    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # NAV series
    all_nav = []
    for method, res in results.items():
        if not res["nav_series"].empty:
            nav = res["nav_series"].copy()
            nav["method"] = method
            all_nav.append(nav)
    
    if all_nav:
        nav_combined = pd.concat(all_nav, ignore_index=True)
        nav_combined.to_csv(OUTPUT_DIR / "backtest_nav_series.csv", index=False)
    
    # Metrics summary
    metrics_rows = [res["metrics"] for res in results.values() if res["metrics"]]
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(OUTPUT_DIR / "backtest_metrics.csv", index=False)
        logger.info(f"\nBacktest Results:\n{metrics_df.to_string()}")
    
    # Rebalance logs
    for method, res in results.items():
        if res["rebalance_log"]:
            with open(OUTPUT_DIR / f"rebalance_log_{method}.json", "w") as f:
                json.dump(res["rebalance_log"], f, indent=2, default=str)
    
    # Visualization: NAV comparison
    if all_nav:
        fig, ax = plt.subplots(figsize=(12, 6))
        for method, res in results.items():
            if not res["nav_series"].empty:
                nav = res["nav_series"]
                ax.plot(nav["date"], nav["nav"], label=method)
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV")
        ax.set_title("Basket NAV Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "backtest_nav_comparison.png", dpi=150)
        plt.close(fig)
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_all_backtests()
    
    for method, res in results.items():
        if res["metrics"]:
            print(f"\n{method}:")
            for k, v in res["metrics"].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

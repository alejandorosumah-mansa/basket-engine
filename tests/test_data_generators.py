"""
Realistic fake market data generators for testing.

These generators produce market data with known, controllable properties
so that tests can make deterministic assertions about processing outcomes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import uuid


def generate_market_id(platform: str = "polymarket") -> str:
    """Generate a unique market ID."""
    return f"{platform}_{uuid.uuid4().hex[:12]}"


def generate_price_series(
    n_days: int = 60,
    start_price: float = 0.50,
    volatility: float = 0.03,
    start_date: Optional[datetime] = None,
    trend: float = 0.0,
    resolution: Optional[float] = None,
    resolution_day: Optional[int] = None,
    gaps: Optional[list[int]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a synthetic daily price series for a prediction market.

    Args:
        n_days: Number of trading days.
        start_price: Initial price in [0, 1].
        volatility: Daily standard deviation of returns.
        start_date: First date of the series.
        trend: Daily drift (positive = trending toward YES).
        resolution: If set, the terminal price (0.0 or 1.0).
        resolution_day: Day index when market resolves (defaults to last day).
        gaps: List of day indices to remove (simulates missing data).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: date, price, volume.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    start_date = start_date or datetime(2025, 8, 1)

    prices = [start_price]
    for i in range(1, n_days):
        ret = trend + volatility * rng.standard_normal()
        new_price = prices[-1] + ret
        new_price = np.clip(new_price, 0.01, 0.99)
        prices.append(new_price)

    if resolution is not None:
        day = resolution_day if resolution_day is not None else n_days - 1
        prices[day] = resolution
        # Truncate after resolution
        prices = prices[: day + 1]

    dates = [start_date + timedelta(days=i) for i in range(len(prices))]
    volumes = (rng.exponential(5000, size=len(prices)) + 100).astype(int)

    df = pd.DataFrame({"date": dates, "price": prices, "volume": volumes})

    if gaps:
        df = df[~df.index.isin(gaps)].reset_index(drop=True)

    return df


def generate_market(
    market_id: Optional[str] = None,
    platform: str = "polymarket",
    title: str = "Will event X happen?",
    description: str = "This market resolves YES if event X happens before the end date.",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    resolution: Optional[str] = None,
    resolution_value: Optional[float] = None,
    total_volume: float = 50000.0,
    daily_volume_7d: float = 2000.0,
    liquidity: float = 5000.0,
    tags: Optional[list[str]] = None,
    **extra_fields,
) -> dict:
    """Generate a single market metadata record.

    Args:
        market_id: Unique ID (auto-generated if None).
        platform: 'polymarket' or 'kalshi'.
        title: Market question title.
        description: Longer description.
        start_date: Market creation date.
        end_date: Market expiration date.
        resolution: 'YES', 'NO', or None (active).
        resolution_value: Terminal price (1.0 for YES, 0.0 for NO).
        total_volume: Lifetime trading volume in $.
        daily_volume_7d: 7-day average daily volume.
        liquidity: Current open interest / liquidity.
        tags: Platform tags/categories.
        **extra_fields: Any additional fields to include.

    Returns:
        dict with market metadata.
    """
    start_date = start_date or datetime(2025, 6, 1)
    end_date = end_date or datetime(2026, 6, 1)

    market = {
        "market_id": market_id or generate_market_id(platform),
        "platform": platform,
        "title": title,
        "description": description,
        "start_date": start_date,
        "end_date": end_date,
        "resolution": resolution,
        "resolution_value": resolution_value,
        "total_volume": total_volume,
        "daily_volume_7d": daily_volume_7d,
        "liquidity": liquidity,
        "tags": tags or [],
        "last_price": 0.55,
    }
    market.update(extra_fields)
    return market


def generate_market_batch(
    n: int = 20,
    platforms: Optional[list[str]] = None,
    seed: int = 42,
    include_resolved: bool = True,
    include_illiquid: bool = True,
    include_expired: bool = False,
) -> list[dict]:
    """Generate a batch of diverse markets for testing.

    Produces markets with a mix of properties: active/resolved, high/low volume,
    different platforms, varying liquidity.

    Args:
        n: Number of markets to generate.
        platforms: List of platforms to sample from.
        seed: Random seed.
        include_resolved: Include some resolved markets.
        include_illiquid: Include some low-liquidity markets.
        include_expired: Include some already-expired markets.

    Returns:
        List of market dicts.
    """
    rng = np.random.default_rng(seed)
    platforms = platforms or ["polymarket", "kalshi"]

    titles = [
        "Will Bitcoin exceed $100K by end of 2026?",
        "Will the Fed cut rates before July 2026?",
        "US unemployment above 5% by Q3 2026?",
        "Will GPT-5 be released before July 2026?",
        "Will Democrats win the Senate in 2026?",
        "Will there be a ceasefire in Ukraine before June?",
        "Will US strike Iran before April?",
        "WTI crude oil above $90 in 2026?",
        "Will SpaceX land on Mars before 2030?",
        "Will WHO declare a new pandemic in 2026?",
        "Will Trump approval rating exceed 50%?",
        "Will the EU impose new Russia sanctions?",
        "Will China invade Taiwan before 2027?",
        "CPI year-over-year above 3% in March?",
        "Will SCOTUS overturn major precedent?",
        "Global temperature anomaly above 1.5C?",
        "Will Tether depeg in 2026?",
        "Will Congress pass immigration reform?",
        "Will a Category 5 hurricane hit the US?",
        "Will nuclear fusion achieve net energy gain?",
        "Will AI replace 10% of coding jobs by 2027?",
        "Will France hold snap elections?",
        "Will lithium prices double in 2026?",
        "Will a bird flu vaccine be approved?",
        "Will OPEC cut production again?",
        "Will US ban TikTok?",
        "Will quantum supremacy be demonstrated?",
        "Will there be a US recession in 2026?",
        "Will Bitcoin ETF inflows exceed $50B?",
        "Will NATO deploy troops to Ukraine?",
    ]

    markets = []
    for i in range(n):
        platform = platforms[i % len(platforms)]
        title = titles[i % len(titles)]

        is_resolved = include_resolved and i % 5 == 0 and i > 0
        is_illiquid = include_illiquid and i % 7 == 0 and i > 0
        is_expired = include_expired and i % 10 == 0 and i > 0

        volume = float(rng.exponential(50000) + 1000)
        if is_illiquid:
            volume = float(rng.uniform(100, 500))

        start = datetime(2025, 6, 1) + timedelta(days=int(rng.integers(0, 120)))
        end = start + timedelta(days=int(rng.integers(30, 365)))
        if is_expired:
            end = datetime(2025, 12, 1)

        resolution = None
        resolution_value = None
        if is_resolved:
            resolution = rng.choice(["YES", "NO"])
            resolution_value = 1.0 if resolution == "YES" else 0.0

        m = generate_market(
            platform=platform,
            title=f"{title} (#{i})",
            description=f"Test market #{i}: {title}",
            start_date=start,
            end_date=end,
            resolution=resolution,
            resolution_value=resolution_value,
            total_volume=volume,
            daily_volume_7d=volume / 30,
            liquidity=float(rng.exponential(3000) + 500) if not is_illiquid else float(rng.uniform(50, 200)),
        )
        markets.append(m)

    return markets


def generate_returns_matrix(
    n_markets: int = 10,
    n_days: int = 60,
    n_correlated_groups: int = 3,
    within_group_corr: float = 0.6,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a returns matrix with known correlation structure.

    Creates groups of correlated markets so clustering tests can verify
    that the correct structure is recovered.

    Args:
        n_markets: Total number of markets.
        n_days: Number of days.
        n_correlated_groups: Number of correlated groups.
        within_group_corr: Correlation within each group.
        seed: Random seed.

    Returns:
        DataFrame indexed by date, columns are market IDs.
    """
    rng = np.random.default_rng(seed)

    markets_per_group = n_markets // n_correlated_groups
    remainder = n_markets % n_correlated_groups

    all_returns = []
    market_ids = []

    for g in range(n_correlated_groups):
        count = markets_per_group + (1 if g < remainder else 0)
        # Generate correlated returns: common factor + idiosyncratic
        common = rng.standard_normal(n_days) * 0.02
        for j in range(count):
            idio = rng.standard_normal(n_days) * 0.02
            r = within_group_corr * common + (1 - within_group_corr) * idio
            all_returns.append(r)
            market_ids.append(f"market_g{g}_{j}")

    dates = pd.date_range("2025-08-01", periods=n_days, freq="D")
    df = pd.DataFrame(np.array(all_returns).T, index=dates, columns=market_ids)
    return df


def generate_basket_weights(
    n_markets: int = 10,
    method: str = "equal",
    seed: int = 42,
) -> dict[str, float]:
    """Generate basket weights for testing.

    Args:
        n_markets: Number of markets in basket.
        method: 'equal', 'random', or 'concentrated'.
        seed: Random seed.

    Returns:
        Dict mapping market_id to weight.
    """
    rng = np.random.default_rng(seed)
    market_ids = [f"market_{i}" for i in range(n_markets)]

    if method == "equal":
        w = 1.0 / n_markets
        weights = {m: w for m in market_ids}
    elif method == "random":
        raw = rng.dirichlet(np.ones(n_markets))
        weights = {m: float(r) for m, r in zip(market_ids, raw)}
    elif method == "concentrated":
        # One market gets 50%, rest split equally
        weights = {m: 0.5 / (n_markets - 1) for m in market_ids}
        weights[market_ids[0]] = 0.5
    else:
        raise ValueError(f"Unknown method: {method}")

    return weights


def generate_nav_series(
    n_days: int = 120,
    start_nav: float = 100.0,
    daily_return_mean: float = 0.001,
    daily_return_std: float = 0.02,
    drawdown_at: Optional[int] = None,
    drawdown_pct: float = 0.15,
    rebalance_days: Optional[list[int]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic NAV series for backtest testing.

    Args:
        n_days: Number of days.
        start_nav: Starting NAV value.
        daily_return_mean: Average daily return.
        daily_return_std: Daily return volatility.
        drawdown_at: Day index to inject a drawdown.
        drawdown_pct: Size of injected drawdown.
        rebalance_days: Day indices where rebalances occur.
        seed: Random seed.

    Returns:
        DataFrame with columns: date, nav, daily_return.
    """
    rng = np.random.default_rng(seed)

    returns = rng.normal(daily_return_mean, daily_return_std, n_days)
    if drawdown_at is not None and drawdown_at < n_days:
        returns[drawdown_at] = -drawdown_pct

    nav = [start_nav]
    for r in returns[1:]:
        nav.append(nav[-1] * (1 + r))

    dates = pd.date_range("2025-08-01", periods=n_days, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "nav": nav,
        "daily_return": returns,
    })

    if rebalance_days:
        df["is_rebalance"] = df.index.isin(rebalance_days)
    else:
        df["is_rebalance"] = False

    return df


def generate_classification_result(
    market_id: str = "test_market_1",
    primary_theme: str = "us_elections",
    secondary_theme: Optional[str] = None,
    confidence: float = 0.85,
    reasoning: str = "Market clearly about US elections.",
) -> dict:
    """Generate a single classification result."""
    return {
        "market_id": market_id,
        "primary_theme": primary_theme,
        "secondary_theme": secondary_theme,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def generate_rebalance_event(
    date: Optional[datetime] = None,
    basket_id: str = "us_elections",
    n_additions: int = 2,
    n_removals: int = 1,
    n_weight_changes: int = 5,
    turnover: float = 0.23,
    market_count: int = 12,
) -> dict:
    """Generate a synthetic rebalance event for testing."""
    date = date or datetime(2026, 1, 1)

    additions = [
        {"market_id": f"new_market_{i}", "weight": 0.08, "reason": "newly eligible"}
        for i in range(n_additions)
    ]
    removals = [
        {"market_id": f"removed_market_{i}", "reason": "resolved YES", "final_return": 1.33}
        for i in range(n_removals)
    ]
    weight_changes = [
        {"market_id": f"market_{i}", "old_weight": 0.12, "new_weight": 0.09}
        for i in range(n_weight_changes)
    ]

    return {
        "date": date.isoformat(),
        "basket_id": basket_id,
        "additions": additions,
        "removals": removals,
        "weight_changes": weight_changes,
        "turnover": turnover,
        "market_count": market_count,
    }

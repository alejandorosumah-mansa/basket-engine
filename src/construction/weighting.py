"""Basket weighting schemes: risk parity, equal weight, volume-weighted."""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def compute_rolling_volatility(
    returns_series: pd.Series,
    window: int = 30,
) -> float:
    """Compute rolling volatility (std of returns) over trailing window."""
    if len(returns_series) < 2:
        return np.nan
    recent = returns_series.tail(window)
    vol = recent.std()
    return vol if vol > 0 else 1e-6  # Floor to avoid division by zero


def risk_parity_weights(
    volatilities: dict[str, float],
) -> dict[str, float]:
    """Compute risk parity weights: w_i = (1/vol_i) / sum(1/vol_j).
    
    Args:
        volatilities: {market_id: volatility}
    
    Returns:
        {market_id: weight}
    """
    inv_vols = {m: 1.0 / v for m, v in volatilities.items() if v > 0 and np.isfinite(v)}
    if not inv_vols:
        # Fallback to equal weight
        n = len(volatilities)
        return {m: 1.0 / n for m in volatilities}
    
    total = sum(inv_vols.values())
    return {m: iv / total for m, iv in inv_vols.items()}


def equal_weights(market_ids: list[str]) -> dict[str, float]:
    """Equal weight across all markets."""
    n = len(market_ids)
    if n == 0:
        return {}
    w = 1.0 / n
    return {m: w for m in market_ids}


def volume_weighted(volumes: dict[str, float]) -> dict[str, float]:
    """Volume-weighted: w_i = volume_i / sum(volume_j)."""
    total = sum(volumes.values())
    if total <= 0:
        return equal_weights(list(volumes.keys()))
    return {m: v / total for m, v in volumes.items()}


def apply_liquidity_cap(
    weights: dict[str, float],
    liquidity: dict[str, float],
    cap_multiplier: float = 2.0,
) -> dict[str, float]:
    """Cap weights by liquidity share with more reasonable limits.
    
    Instead of capping at liquidity_share, cap at cap_multiplier * liquidity_share
    to allow some deviation from pure liquidity weighting.
    
    Args:
        weights: Original weights to cap
        liquidity: Liquidity values 
        cap_multiplier: Allow weights up to this multiple of liquidity share
    """
    total_liq = sum(liquidity.values())
    if total_liq <= 0:
        return weights
    
    liq_shares = {m: liquidity.get(m, 0) / total_liq for m in weights}
    
    # Cap at cap_multiplier * liquidity share (more reasonable)
    capped = dict(weights)
    any_capped = False
    
    for m in list(capped.keys()):
        max_allowed = liq_shares.get(m, 0) * cap_multiplier
        if capped[m] > max_allowed:
            capped[m] = max_allowed
            any_capped = True
    
    # Re-normalize if any weights were capped
    if any_capped:
        total = sum(capped.values())
        if total > 0:
            capped = {m: w / total for m, w in capped.items()}
    
    return capped


def enforce_constraints(
    weights: dict[str, float],
    max_weight: float = 0.20,
    min_weight: float = 0.02,
    min_markets: int = 5,
    max_markets: int = 30,
) -> dict[str, float]:
    """Enforce basket constraints: max/min weight, basket size.
    
    Process:
    1. Remove markets below min_weight (dust)
    2. Cap markets above max_weight
    3. Re-normalize
    4. Trim to max_markets (keep highest-weighted)
    5. Ensure min_markets (relax min_weight if needed)
    """
    if not weights:
        return {}
    
    # Sort by weight descending
    sorted_markets = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    # Trim to max_markets
    if len(sorted_markets) > max_markets:
        sorted_markets = sorted_markets[:max_markets]
    
    # Remove dust positions (but keep at least min_markets)
    filtered = [(m, w) for m, w in sorted_markets if w >= min_weight]
    if len(filtered) < min_markets and len(sorted_markets) >= min_markets:
        # Keep top min_markets even if below min_weight
        filtered = sorted_markets[:min_markets]
    elif len(filtered) < min_markets:
        filtered = sorted_markets  # Keep whatever we have
    
    # Re-normalize
    total = sum(w for _, w in filtered)
    if total <= 0:
        return {}
    result = {m: w / total for m, w in filtered}
    
    # Cap max weight (iterative)
    for _ in range(20):
        any_capped = False
        for m in list(result.keys()):
            if result[m] > max_weight:
                result[m] = max_weight
                any_capped = True
        total = sum(result.values())
        if total > 0:
            result = {m: w / total for m, w in result.items()}
        if not any_capped:
            break
    
    return result


def compute_weights(
    method: str,
    market_ids: list[str],
    returns_df: Optional[pd.DataFrame] = None,
    volumes: Optional[dict[str, float]] = None,
    liquidity: Optional[dict[str, float]] = None,
    vol_window: int = 30,
    max_weight: float = 0.20,
    min_weight: float = 0.02,
    min_markets: int = 5,
    max_markets: int = 30,
    liquidity_cap: bool = True,
) -> dict[str, float]:
    """Compute basket weights using specified method with all constraints.
    
    Args:
        method: 'risk_parity_liquidity_cap', 'equal', 'volume_weighted'
        market_ids: List of eligible market IDs
        returns_df: Returns DataFrame (market_id, date, return) for vol calc
        volumes: {market_id: total_volume} for volume weighting
        liquidity: {market_id: liquidity} for liquidity cap
        vol_window: Rolling window for volatility
        max_weight, min_weight: Weight bounds
        min_markets, max_markets: Basket size bounds
        liquidity_cap: Whether to apply liquidity cap
    
    Returns:
        {market_id: weight} summing to 1.0
    """
    if len(market_ids) == 0:
        return {}
    
    if method == "equal":
        weights = equal_weights(market_ids)
    
    elif method == "volume_weighted":
        if volumes is None:
            weights = equal_weights(market_ids)
        else:
            vols_filtered = {m: volumes.get(m, 0) for m in market_ids}
            weights = volume_weighted(vols_filtered)
    
    elif method in ("risk_parity", "risk_parity_liquidity_cap"):
        if returns_df is None or returns_df.empty:
            logger.warning("No returns data for risk parity, falling back to equal weight")
            weights = equal_weights(market_ids)
        else:
            volatilities = {}
            for mid in market_ids:
                mret = returns_df[returns_df["market_id"] == mid]["return"]
                vol = compute_rolling_volatility(mret, window=vol_window)
                volatilities[mid] = vol if np.isfinite(vol) else 1e-6
            weights = risk_parity_weights(volatilities)
            
            # Apply liquidity cap
            if liquidity_cap and liquidity and method == "risk_parity_liquidity_cap":
                weights = apply_liquidity_cap(weights, liquidity, cap_multiplier=2.0)
    else:
        logger.warning(f"Unknown method '{method}', using equal weight")
        weights = equal_weights(market_ids)
    
    # Enforce constraints
    weights = enforce_constraints(
        weights, max_weight=max_weight, min_weight=min_weight,
        min_markets=min_markets, max_markets=max_markets,
    )
    
    return weights

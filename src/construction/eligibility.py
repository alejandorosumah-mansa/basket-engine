"""Market eligibility filters for basket construction."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import yaml
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def check_eligibility(
    market: dict,
    price_history_days: int,
    current_price: float,
    reference_date: Optional[datetime] = None,
    settings: Optional[dict] = None,
) -> tuple[bool, list[str]]:
    """Check if a single market passes all eligibility filters.
    
    Args:
        market: Market metadata dict with keys: market_id, resolution, end_date,
                total_volume, daily_volume_7d, liquidity
        price_history_days: Number of days of price history available
        current_price: Current market price
        reference_date: Date to check against (default: now)
        settings: Eligibility settings dict
    
    Returns:
        (is_eligible, list_of_failure_reasons)
    """
    if settings is None:
        settings = load_settings()["eligibility"]
    
    reference_date = reference_date or datetime.now()
    reasons = []
    
    # Must be active (not resolved)  
    if market.get("resolution") is not None:
        reasons.append("resolved")
    
    # Must be open/active status (exclude closed/cancelled markets)
    status = market.get("status", "unknown").lower()
    if status not in ["open", "active", "live"]:
        reasons.append(f"status_not_active:{status}")
    
    # Days to expiry
    end_date = market.get("end_date")
    if end_date:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        days_to_exp = (end_date - reference_date).days
        if days_to_exp < settings["min_days_to_expiration"]:
            reasons.append(f"expiry_too_soon:{days_to_exp}d")
    
    # Volume
    if market.get("total_volume", 0) < settings["min_total_volume"]:
        reasons.append(f"low_total_volume:{market.get('total_volume', 0)}")
    
    # 7d avg daily volume
    if market.get("daily_volume_7d", 0) < settings["min_7d_avg_daily_volume"]:
        reasons.append(f"low_daily_volume:{market.get('daily_volume_7d', 0)}")
    
    # Liquidity
    if market.get("liquidity", 0) < settings["min_liquidity"]:
        reasons.append(f"low_liquidity:{market.get('liquidity', 0)}")
    
    # Price history
    if price_history_days < settings["min_price_history_days"]:
        reasons.append(f"insufficient_history:{price_history_days}d")
    
    # Price range
    if current_price < settings["price_range_min"]:
        reasons.append(f"price_too_low:{current_price}")
    if current_price > settings["price_range_max"]:
        reasons.append(f"price_too_high:{current_price}")
    
    return len(reasons) == 0, reasons


def filter_eligible_markets(
    markets: list[dict],
    prices_df: pd.DataFrame,
    reference_date: Optional[datetime] = None,
    settings: Optional[dict] = None,
) -> tuple[list[dict], pd.DataFrame]:
    """Filter a list of markets to only eligible ones.
    
    Args:
        markets: List of market metadata dicts
        prices_df: Price history with market_id, date, close_price, volume columns
        reference_date: Reference date for eligibility checks
        settings: Eligibility settings
    
    Returns:
        (eligible_markets, eligibility_report_df)
    """
    if settings is None:
        settings = load_settings()["eligibility"]
    
    reference_date = reference_date or datetime.now()
    
    eligible = []
    report_rows = []
    
    for market in markets:
        mid = market["market_id"]
        
        # Get price history for this market
        mprices = prices_df[prices_df["market_id"] == mid]
        history_days = len(mprices)
        current_price = mprices["close_price"].iloc[-1] if len(mprices) > 0 else 0.5
        
        # Compute 7d avg daily volume if not in market dict
        if "daily_volume_7d" not in market and len(mprices) > 0:
            recent = mprices.tail(7)
            market["daily_volume_7d"] = recent["volume"].mean()
        
        is_eligible, reasons = check_eligibility(
            market, history_days, current_price, reference_date, settings
        )
        
        report_rows.append({
            "market_id": mid,
            "eligible": is_eligible,
            "reasons": "; ".join(reasons) if reasons else "pass",
            "history_days": history_days,
            "current_price": current_price,
            "total_volume": market.get("total_volume", 0),
        })
        
        if is_eligible:
            eligible.append(market)
    
    report = pd.DataFrame(report_rows)
    logger.info(f"Eligibility: {len(eligible)}/{len(markets)} markets pass")
    
    return eligible, report

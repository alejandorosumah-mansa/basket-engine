"""Monthly rebalance engine with chain-link pricing and resolution handling."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class RebalanceEvent:
    """Record of a single rebalance."""
    date: str
    basket_id: str
    old_weights: dict = field(default_factory=dict)
    new_weights: dict = field(default_factory=dict)
    additions: list = field(default_factory=list)
    removals: list = field(default_factory=list)
    weight_changes: list = field(default_factory=list)
    turnover: float = 0.0
    market_count: int = 0
    resolutions: list = field(default_factory=list)
    nav_before: float = 0.0
    nav_after: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_turnover(old_weights: dict, new_weights: dict) -> float:
    """Turnover = sum(|new_w - old_w|) / 2."""
    all_markets = set(old_weights) | set(new_weights)
    return sum(abs(new_weights.get(m, 0) - old_weights.get(m, 0)) for m in all_markets) / 2


def handle_resolution(
    weights: dict,
    resolved_market: str,
    last_price: float,
    resolution_value: float,
) -> tuple[dict, float]:
    """Handle a market resolution: compute terminal return and redistribute weight.
    
    Returns: (new_weights, terminal_return_contribution)
    """
    weight = weights.get(resolved_market, 0)
    if weight == 0:
        return weights, 0.0
    
    # Terminal return
    if last_price > 0:
        terminal_return = (resolution_value - last_price) / last_price
    else:
        terminal_return = 0.0
    
    contribution = weight * terminal_return
    
    # Remove and redistribute
    remaining = {m: w for m, w in weights.items() if m != resolved_market}
    total = sum(remaining.values())
    if total > 0:
        remaining = {m: w / total for m, w in remaining.items()}
    
    return remaining, contribution


def chain_link_nav(
    nav_prev: float,
    weights: dict,
    daily_returns: dict,
) -> float:
    """Chain-link pricing: NAV(t) = NAV(t-1) * (1 + sum(w_i * r_i)).
    
    Args:
        nav_prev: Previous NAV value
        weights: Current weights {market_id: weight}
        daily_returns: Today's returns {market_id: return}
    
    Returns:
        New NAV value
    """
    portfolio_return = sum(
        weights.get(m, 0) * daily_returns.get(m, 0)
        for m in set(weights) | set(daily_returns)
    )
    return nav_prev * (1 + portfolio_return)


def get_rebalance_dates(
    start_date: datetime,
    end_date: datetime,
    frequency: str = "monthly",
) -> list[datetime]:
    """Generate rebalance dates."""
    dates = []
    if frequency == "monthly":
        current = start_date.replace(day=1)
        while current <= end_date:
            if current >= start_date:
                dates.append(current)
            # Move to first of next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
    return dates


class RebalanceEngine:
    """Monthly rebalance engine for basket management."""
    
    def __init__(
        self,
        basket_id: str,
        weighting_fn,
        eligibility_fn=None,
        initial_nav: float = 100.0,
        min_markets: int = 5,
    ):
        self.basket_id = basket_id
        self.weighting_fn = weighting_fn
        self.eligibility_fn = eligibility_fn
        self.initial_nav = initial_nav
        self.min_markets = min_markets
        
        self.current_weights: dict = {}
        self.nav_history: list = []
        self.rebalance_events: list = []
        self.current_nav = initial_nav
    
    def rebalance(
        self,
        date: datetime,
        eligible_market_ids: list[str],
        returns_df: Optional[pd.DataFrame] = None,
        volumes: Optional[dict] = None,
        liquidity: Optional[dict] = None,
    ) -> RebalanceEvent:
        """Execute a rebalance."""
        old_weights = dict(self.current_weights)
        
        # Compute new weights
        new_weights = self.weighting_fn(
            market_ids=eligible_market_ids,
            returns_df=returns_df,
            volumes=volumes,
            liquidity=liquidity,
        )
        
        # Compute turnover
        turnover = compute_turnover(old_weights, new_weights)
        
        # Identify additions and removals
        old_set = set(old_weights.keys())
        new_set = set(new_weights.keys())
        additions = [{"market_id": m, "weight": new_weights[m]} for m in new_set - old_set]
        removals = [{"market_id": m, "old_weight": old_weights[m]} for m in old_set - new_set]
        
        # Weight changes for continuing markets
        weight_changes = []
        for m in old_set & new_set:
            if abs(old_weights[m] - new_weights[m]) > 1e-6:
                weight_changes.append({
                    "market_id": m,
                    "old_weight": old_weights[m],
                    "new_weight": new_weights[m],
                })
        
        event = RebalanceEvent(
            date=date.isoformat(),
            basket_id=self.basket_id,
            old_weights=old_weights,
            new_weights=new_weights,
            additions=additions,
            removals=removals,
            weight_changes=weight_changes,
            turnover=turnover,
            market_count=len(new_weights),
            nav_before=self.current_nav,
            nav_after=self.current_nav,  # NAV doesn't jump at rebalance
        )
        
        self.current_weights = new_weights
        self.rebalance_events.append(event)
        
        logger.info(f"Rebalance {date.date()}: {len(new_weights)} markets, "
                    f"turnover={turnover:.2%}, additions={len(additions)}, removals={len(removals)}")
        
        return event
    
    def step(self, date: datetime, daily_returns: dict):
        """Process one day: update NAV using chain-link pricing."""
        if not self.current_weights:
            self.nav_history.append({
                "date": date,
                "nav": self.current_nav,
                "portfolio_return": 0.0,
            })
            return
        
        portfolio_return = sum(
            self.current_weights.get(m, 0) * daily_returns.get(m, 0)
            for m in self.current_weights
        )
        
        self.current_nav = self.current_nav * (1 + portfolio_return)
        
        self.nav_history.append({
            "date": date,
            "nav": self.current_nav,
            "portfolio_return": portfolio_return,
        })
    
    def get_nav_series(self) -> pd.DataFrame:
        """Return NAV history as DataFrame."""
        return pd.DataFrame(self.nav_history)
    
    def get_rebalance_log(self) -> list[dict]:
        """Return all rebalance events as dicts."""
        return [e.to_dict() for e in self.rebalance_events]

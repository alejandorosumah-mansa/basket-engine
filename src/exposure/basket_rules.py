"""Basket construction rules accounting for side/exposure.

Enforces:
1. No opposing exposures in same basket (long YES + long NO of same event = useless)
2. Categorical awareness: one outcome per event
3. One side per event per basket
4. Detect hidden correlations from opposing events
"""

import logging
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


def check_exposure_conflicts(
    market_ids: list[str],
    exposure_map: dict,
) -> list[dict]:
    """Check for exposure conflicts within a set of markets.

    Detects:
    - Same event, opposing directions (cancel each other)
    - Same event, multiple categorical outcomes (fake diversification)

    Returns:
        List of conflict dicts with details.
    """
    conflicts = []

    # Group by event
    event_markets = defaultdict(list)
    for mid in market_ids:
        info = exposure_map.get(mid)
        if info and info.event_id:
            event_markets[info.event_id].append(info)

    for event_id, infos in event_markets.items():
        if len(infos) < 2:
            continue

        # Check for opposing directions within same event
        directions = set(i.exposure_direction for i in infos)
        if len(directions) > 1:
            conflicts.append({
                "type": "opposing_exposure",
                "event_id": event_id,
                "markets": [i.market_id for i in infos],
                "directions": {i.market_id: i.exposure_direction for i in infos},
                "severity": "high",
                "description": f"Event '{event_id}' has both long and short exposures — they cancel out.",
            })

        # Check for multiple categorical outcomes from same event
        token_sides = [i.token_side for i in infos]
        non_binary_sides = [s for s in token_sides if s not in ("YES", "NO")]
        if len(non_binary_sides) > 1:
            conflicts.append({
                "type": "categorical_overlap",
                "event_id": event_id,
                "markets": [i.market_id for i in infos],
                "outcomes": non_binary_sides,
                "severity": "medium",
                "description": (
                    f"Event '{event_id}' has multiple categorical outcomes in basket: "
                    f"{non_binary_sides}. These are anti-correlated by construction."
                ),
            })

    return conflicts


def filter_opposing_exposures(
    market_ids: list[str],
    exposure_map: dict,
    liquidity: Optional[dict] = None,
) -> list[str]:
    """Filter market list to enforce one side per event.

    For each event with multiple markets, keeps the most liquid one.
    Removes categorical duplicates (keeps most liquid outcome per event).

    Args:
        market_ids: Candidate market IDs
        exposure_map: {market_id: ExposureInfo}
        liquidity: {market_id: liquidity_value} for tie-breaking

    Returns:
        Filtered list of market IDs with no exposure conflicts.
    """
    liquidity = liquidity or {}

    # Group by event
    event_markets = defaultdict(list)
    no_event = []

    for mid in market_ids:
        info = exposure_map.get(mid)
        if info and info.event_id:
            event_markets[info.event_id].append(mid)
        else:
            no_event.append(mid)

    filtered = list(no_event)

    for event_id, mids in event_markets.items():
        if len(mids) == 1:
            filtered.append(mids[0])
            continue

        # Pick the most liquid market for this event
        best = max(mids, key=lambda m: liquidity.get(m, 0))
        filtered.append(best)

        removed = [m for m in mids if m != best]
        if removed:
            logger.debug(
                f"Event '{event_id}': kept {best}, removed {removed} (exposure dedup)"
            )

    logger.info(
        f"Exposure filter: {len(market_ids)} → {len(filtered)} markets "
        f"({len(market_ids) - len(filtered)} removed)"
    )
    return filtered

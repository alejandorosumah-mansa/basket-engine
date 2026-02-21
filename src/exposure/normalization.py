"""Exposure normalization: map all positions to a consistent directional framework."""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExposureInfo:
    """Exposure information for a single market/CUSIP."""
    market_id: str
    event_id: Optional[str]
    token_side: str           # YES / NO / outcome name
    phrasing_polarity: str    # positive / negative / neutral
    exposure_direction: str   # long / short
    normalized_direction: float  # 1.0 or -1.0
    event_positive_label: Optional[str] = None  # human-readable "positive direction"

    @property
    def is_long(self) -> bool:
        return self.normalized_direction > 0

    @property
    def is_short(self) -> bool:
        return self.normalized_direction < 0


def normalize_exposures(markets_df) -> dict[str, ExposureInfo]:
    """Build an exposure map from a markets DataFrame that has side columns.

    Expects columns: market_id, event_slug (or event_id), token_side,
    phrasing_polarity, exposure_direction, normalized_direction.

    Returns:
        {market_id: ExposureInfo}
    """
    exposure_map = {}

    event_col = "event_slug" if "event_slug" in markets_df.columns else "event_id"
    if event_col not in markets_df.columns:
        event_col = None

    for _, row in markets_df.iterrows():
        mid = row["market_id"]
        exposure_map[mid] = ExposureInfo(
            market_id=mid,
            event_id=row.get(event_col) if event_col else None,
            token_side=row.get("token_side", "YES"),
            phrasing_polarity=row.get("phrasing_polarity", "positive"),
            exposure_direction=row.get("exposure_direction", "long"),
            normalized_direction=row.get("normalized_direction", 1.0),
        )

    n_long = sum(1 for e in exposure_map.values() if e.is_long)
    n_short = sum(1 for e in exposure_map.values() if e.is_short)
    logger.info(f"Exposure map: {n_long} long, {n_short} short, {len(exposure_map)} total")

    return exposure_map


def adjust_return_for_exposure(
    raw_return: float,
    normalized_direction: float,
) -> float:
    """Adjust a raw return by the exposure direction.

    For short-exposure positions, a price increase is actually a loss in
    the normalized framework (the "bad" outcome became more likely).

    Args:
        raw_return: Raw price change (e.g., 0.05 means price went up 5 cents)
        normalized_direction: 1.0 (long) or -1.0 (short)

    Returns:
        Direction-adjusted return.
    """
    return raw_return * normalized_direction

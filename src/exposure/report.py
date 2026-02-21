"""Portfolio-level exposure report for baskets."""

import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


def generate_exposure_report(
    weights: dict[str, float],
    exposure_map: dict,
    classifications: Optional[dict] = None,
) -> dict:
    """Generate a portfolio-level exposure report for a basket.

    Args:
        weights: {market_id: weight}
        exposure_map: {market_id: ExposureInfo}
        classifications: {market_id: {"primary_theme": str}} optional theme data

    Returns:
        Report dict with:
        - net_direction: overall net long/short
        - theme_exposures: net directional exposure per theme
        - conflicts: any hidden cancellations
        - contradictions: positions that offset each other
        - summary: human-readable summary lines
    """
    from .basket_rules import check_exposure_conflicts

    # Net directional exposure
    net_exposure = 0.0
    long_weight = 0.0
    short_weight = 0.0

    theme_long = defaultdict(float)
    theme_short = defaultdict(float)

    for mid, weight in weights.items():
        info = exposure_map.get(mid)
        if not info:
            continue

        direction = info.normalized_direction
        if direction > 0:
            long_weight += weight
        else:
            short_weight += weight
        net_exposure += weight * direction

        # Theme breakdown
        theme = "unknown"
        if classifications and mid in classifications:
            theme = classifications[mid].get("primary_theme", "unknown")
        elif info.event_id and classifications:
            # Try event_id lookup
            theme = classifications.get(info.event_id, {}).get("primary_theme", "unknown")

        if direction > 0:
            theme_long[theme] += weight
        else:
            theme_short[theme] += weight

    # Theme net exposures
    all_themes = set(theme_long.keys()) | set(theme_short.keys())
    theme_exposures = {}
    for theme in sorted(all_themes):
        net = theme_long.get(theme, 0) - theme_short.get(theme, 0)
        theme_exposures[theme] = {
            "long_weight": round(theme_long.get(theme, 0), 4),
            "short_weight": round(theme_short.get(theme, 0), 4),
            "net_exposure": round(net, 4),
            "direction": "LONG" if net > 0 else "SHORT" if net < 0 else "NEUTRAL",
        }

    # Check conflicts
    conflicts = check_exposure_conflicts(list(weights.keys()), exposure_map)

    # Build summary
    summary = []
    summary.append(f"Net portfolio exposure: {net_exposure:+.2%} "
                   f"(long={long_weight:.1%}, short={short_weight:.1%})")

    for theme, info in theme_exposures.items():
        if abs(info["net_exposure"]) > 0.01:
            summary.append(f"  {theme}: {info['direction']} {abs(info['net_exposure']):.1%}")

    if conflicts:
        summary.append(f"⚠️  {len(conflicts)} exposure conflict(s) detected")
        for c in conflicts:
            summary.append(f"  - {c['type']}: {c['description']}")

    return {
        "net_exposure": round(net_exposure, 4),
        "long_weight": round(long_weight, 4),
        "short_weight": round(short_weight, 4),
        "net_direction": "LONG" if net_exposure > 0.01 else "SHORT" if net_exposure < -0.01 else "NEUTRAL",
        "theme_exposures": theme_exposures,
        "conflicts": conflicts,
        "summary": summary,
    }

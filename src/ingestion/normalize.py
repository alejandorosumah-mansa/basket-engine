"""Normalize Polymarket + Kalshi data into unified parquet format."""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import Cache

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def normalize_polymarket_markets(cache: Cache) -> pd.DataFrame:
    """Convert cached Polymarket markets to normalized DataFrame."""
    rows = []
    for status in ["open", "closed"]:
        markets = cache.get(f"markets_{status}") or []
        for m in markets:
            cid = m.get("condition_id")
            if not cid:
                continue
            rows.append({
                "market_id": f"poly_{m.get('market_slug', cid[:16])}",
                "platform": "polymarket",
                "title": m.get("title", ""),
                "description": (m.get("description") or "")[:1000],
                "tags": ",".join(m.get("tags") or []),
                "start_date": _ts_to_date(m.get("start_time")),
                "end_date": _ts_to_date(m.get("end_time")),
                "volume": m.get("volume_total", 0),
                "status": m.get("status", "unknown"),
                "condition_id": cid,
                "ticker": None,
                "event_slug": m.get("event_slug"),
                "winning_side": m.get("winning_side"),
            })
    return pd.DataFrame(rows)


def normalize_kalshi_markets(cache: Cache) -> pd.DataFrame:
    """Convert cached Kalshi markets to normalized DataFrame."""
    rows = []
    for status in ["open", "closed"]:
        markets = cache.get(f"markets_{status}") or []
        for m in markets:
            ticker = m.get("market_ticker")
            if not ticker:
                continue
            rows.append({
                "market_id": f"kalshi_{ticker}",
                "platform": "kalshi",
                "title": m.get("title", ""),
                "description": "",
                "tags": "",
                "start_date": _ts_to_date(m.get("start_time")),
                "end_date": _ts_to_date(m.get("end_time")),
                "volume": m.get("volume", 0),
                "status": m.get("status", "unknown"),
                "condition_id": None,
                "ticker": ticker,
                "event_slug": m.get("event_ticker"),
                "winning_side": str(m.get("result")) if m.get("result") is not None else None,
            })
    return pd.DataFrame(rows)


def build_polymarket_prices(cache: Cache, markets_df: pd.DataFrame) -> pd.DataFrame:
    """Build daily prices from Polymarket candlesticks."""
    rows = []
    poly_markets = markets_df[markets_df["platform"] == "polymarket"]

    for _, market in poly_markets.iterrows():
        cid = market["condition_id"]
        if not cid:
            continue
        candles_raw = cache.get(f"candles_{cid}") or []

        # Handle nested format: [[candle_dicts], token_info]
        candles = candles_raw
        if candles and isinstance(candles[0], list):
            candles = candles[0]

        for c in candles:
            if not isinstance(c, dict):
                continue
            ts = c.get("end_period_ts")
            price_data = c.get("price", {})
            close_str = price_data.get("close_dollars")
            if ts and close_str:
                close = float(close_str)
                if 0 <= close <= 1:
                    rows.append({
                        "market_id": market["market_id"],
                        "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                        "close_price": close,
                        "volume": c.get("volume", 0),
                    })

    return pd.DataFrame(rows)


def build_kalshi_prices(cache: Cache, markets_df: pd.DataFrame) -> pd.DataFrame:
    """Build daily prices from Kalshi trades (aggregate to daily close = last trade price)."""
    rows = []
    kalshi_markets = markets_df[markets_df["platform"] == "kalshi"]

    for _, market in kalshi_markets.iterrows():
        ticker = market["ticker"]
        if not ticker:
            continue
        trades = cache.get(f"trades_{ticker}") or []
        if not trades:
            continue

        # Group trades by day, take last trade price as close
        daily = {}
        for t in trades:
            ts = t.get("created_time")
            price = t.get("yes_price_dollars")
            if ts and price is not None:
                date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                count = t.get("count", 1)
                if date not in daily or ts > daily[date]["ts"]:
                    daily[date] = {"ts": ts, "price": float(price), "volume": 0}
                daily[date]["volume"] += count

        for date, info in daily.items():
            if 0 <= info["price"] <= 1:
                rows.append({
                    "market_id": market["market_id"],
                    "date": date,
                    "close_price": info["price"],
                    "volume": info["volume"],
                })

    return pd.DataFrame(rows)


def detect_cross_platform_duplicates(markets_df: pd.DataFrame) -> pd.DataFrame:
    """Flag potential cross-platform duplicates using title similarity."""
    poly = markets_df[markets_df["platform"] == "polymarket"][["market_id", "title"]].copy()
    kalshi = markets_df[markets_df["platform"] == "kalshi"][["market_id", "title"]].copy()

    if poly.empty or kalshi.empty:
        return pd.DataFrame(columns=["poly_id", "kalshi_id", "poly_title", "kalshi_title"])

    # Simple normalized title matching
    def normalize_title(t):
        return re.sub(r'[^a-z0-9 ]', '', t.lower().strip())

    poly["norm"] = poly["title"].apply(normalize_title)
    kalshi["norm"] = kalshi["title"].apply(normalize_title)

    # Find exact matches on normalized titles
    merged = poly.merge(kalshi, on="norm", suffixes=("_poly", "_kalshi"))
    dupes = merged[["market_id_poly", "market_id_kalshi", "title_poly", "title_kalshi"]].rename(
        columns={"market_id_poly": "poly_id", "market_id_kalshi": "kalshi_id",
                  "title_poly": "poly_title", "title_kalshi": "kalshi_title"}
    )
    if not dupes.empty:
        logger.info(f"Found {len(dupes)} potential cross-platform duplicates")
    return dupes


def flag_price_gaps(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Flag gaps in daily price series (>2 day gaps)."""
    if prices_df.empty:
        return pd.DataFrame(columns=["market_id", "gap_start", "gap_end", "gap_days"])

    gaps = []
    for mid, group in prices_df.groupby("market_id"):
        dates = pd.to_datetime(group["date"]).sort_values()
        if len(dates) < 2:
            continue
        diffs = dates.diff().dt.days
        for i, d in enumerate(diffs):
            if d and d > 2:
                gaps.append({
                    "market_id": mid,
                    "gap_start": str(dates.iloc[i - 1].date()),
                    "gap_end": str(dates.iloc[i].date()),
                    "gap_days": int(d),
                })
    return pd.DataFrame(gaps)


def run_normalization():
    """Build all parquet files from cached data."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    poly_cache = Cache("polymarket")
    kalshi_cache = Cache("kalshi")

    # Markets
    logger.info("Normalizing markets...")
    poly_markets = normalize_polymarket_markets(poly_cache)
    kalshi_markets = normalize_kalshi_markets(kalshi_cache)
    markets = pd.concat([poly_markets, kalshi_markets], ignore_index=True)
    markets = markets.drop_duplicates(subset=["market_id"])
    markets.to_parquet(OUTPUT_DIR / "markets.parquet", index=False)
    logger.info(f"Saved {len(markets)} markets to markets.parquet "
                f"(poly={len(poly_markets)}, kalshi={len(kalshi_markets)})")

    # Prices
    logger.info("Building price series...")
    poly_prices = build_polymarket_prices(poly_cache, markets)
    kalshi_prices = build_kalshi_prices(kalshi_cache, markets)
    prices = pd.concat([poly_prices, kalshi_prices], ignore_index=True)
    prices = prices.drop_duplicates(subset=["market_id", "date"])
    prices = prices.sort_values(["market_id", "date"])
    prices.to_parquet(OUTPUT_DIR / "prices.parquet", index=False)
    logger.info(f"Saved {len(prices)} price observations to prices.parquet")

    # Returns
    logger.info("Computing returns...")
    if not prices.empty:
        prices_sorted = prices.sort_values(["market_id", "date"])
        prices_sorted["prev_price"] = prices_sorted.groupby("market_id")["close_price"].shift(1)
        prices_sorted["return"] = prices_sorted["close_price"] - prices_sorted["prev_price"]
        returns = prices_sorted.dropna(subset=["return"])[["market_id", "date", "return"]].copy()
        returns.to_parquet(OUTPUT_DIR / "returns.parquet", index=False)
        logger.info(f"Saved {len(returns)} return observations to returns.parquet")
    else:
        pd.DataFrame(columns=["market_id", "date", "return"]).to_parquet(
            OUTPUT_DIR / "returns.parquet", index=False)

    # Data quality
    logger.info("Running data quality checks...")
    dupes = detect_cross_platform_duplicates(markets)
    if not dupes.empty:
        dupes.to_parquet(OUTPUT_DIR / "cross_platform_duplicates.parquet", index=False)

    gaps = flag_price_gaps(prices)
    if not gaps.empty:
        gaps.to_parquet(OUTPUT_DIR / "price_gaps.parquet", index=False)
        logger.info(f"Flagged {len(gaps)} price gaps across {gaps['market_id'].nunique()} markets")

    # Summary
    logger.info("=== SUMMARY ===")
    logger.info(f"Markets: {len(markets)} (poly={len(poly_markets)}, kalshi={len(kalshi_markets)})")
    logger.info(f"Price observations: {len(prices)}")
    logger.info(f"Markets with prices: {prices['market_id'].nunique() if not prices.empty else 0}")
    logger.info(f"Cross-platform duplicates: {len(dupes)}")
    logger.info(f"Price gaps: {len(gaps)}")


def _ts_to_date(ts):
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return None


if __name__ == "__main__":
    run_normalization()

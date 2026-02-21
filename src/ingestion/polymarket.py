"""Polymarket data ingestion via Dome API (markets) + CLOB API (prices)."""

import logging
import time
from datetime import datetime, timezone, timedelta

import requests

from .api_client import api_get
from .cache import Cache

logger = logging.getLogger(__name__)

CLOB_BASE = "https://clob.polymarket.com"


def fetch_all_markets(cache: Cache, force: bool = False) -> list:
    """Fetch all Polymarket markets (open + closed) with pagination."""
    all_markets = []

    for status in ["open", "closed"]:
        cache_key = f"markets_{status}"
        if not force and not cache.needs_update(cache_key, max_age_hours=12):
            cached = cache.get(cache_key)
            if cached:
                logger.info(f"Using cached {status} markets: {len(cached)} markets")
                all_markets.extend(cached)
                continue

        logger.info(f"Fetching {status} Polymarket markets...")
        markets = []
        offset = 0
        limit = 100

        while True:
            try:
                data = api_get("/polymarket/markets", params={
                    "status": status, "limit": limit, "offset": offset
                })
            except Exception as e:
                if "400" in str(e):
                    logger.info(f"  {status}: hit API offset limit at {offset}, stopping")
                    break
                raise
            batch = data.get("markets", [])
            if not batch:
                break
            markets.extend(batch)
            logger.info(f"  {status}: fetched {len(markets)} markets so far...")
            offset += limit

            if len(batch) < limit:
                break

        cache.put(cache_key, markets)
        cache.set_last_fetch(cache_key)
        logger.info(f"Fetched {len(markets)} {status} Polymarket markets")
        all_markets.extend(markets)

    # Deduplicate by condition_id
    seen = set()
    unique = []
    for m in all_markets:
        cid = m.get("condition_id")
        if cid and cid not in seen:
            seen.add(cid)
            unique.append(m)
    logger.info(f"Total unique Polymarket markets: {len(unique)}")
    return unique


def _get_token_id(market: dict) -> str:
    """Extract YES token ID from market data."""
    side_a = market.get("side_a", {})
    if side_a and side_a.get("id"):
        return side_a["id"]
    return None


def _market_time_range(market: dict) -> tuple:
    """Get the active time range for a market.
    
    Returns (start_ts, end_ts) where end_ts is:
    - completed_time or close_time for resolved/closed markets
    - now for active markets
    """
    start_ts = market.get("start_time")
    status = market.get("status", "")
    
    if status == "closed":
        # Use completed_time, close_time, or end_time (whichever is available)
        end_ts = market.get("completed_time") or market.get("close_time") or market.get("end_time")
    else:
        end_ts = int(time.time())
    
    return start_ts, end_ts


def fetch_price_history(market: dict, cache: Cache, force: bool = False) -> list:
    """Fetch price history from Polymarket CLOB API using token ID.
    
    Uses the /prices-history endpoint with the YES token ID.
    Works for both active and resolved markets.
    """
    cid = market.get("condition_id", "")
    cache_key = f"candles_{cid}"

    if not force:
        cached = cache.get(cache_key)
        if cached is not None and not cache.needs_update(cache_key, max_age_hours=12):
            return cached

    token_id = _get_token_id(market)
    if not token_id:
        logger.warning(f"No token ID for {market.get('title', cid)[:40]}")
        cache.put(cache_key, [])
        cache.set_last_fetch(cache_key)
        return []

    try:
        url = f"{CLOB_BASE}/prices-history"
        resp = requests.get(url, params={
            "market": token_id,
            "interval": "all",
            "fidelity": 1440,  # daily
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        history = data.get("history", [])

        # Convert CLOB format {t, p} to our candle format for compatibility
        candles = []
        for point in history:
            ts = point.get("t")
            price = point.get("p")
            if ts is not None and price is not None:
                candles.append({
                    "end_period_ts": ts,
                    "price": {"close_dollars": str(price)},
                    "volume": 0,
                })

        cache.put(cache_key, candles)
        cache.set_last_fetch(cache_key)
        return candles

    except Exception as e:
        logger.warning(f"CLOB price history failed for {cid[:20]}: {e}")
        cache.put(cache_key, [])
        cache.set_last_fetch(cache_key)
        return []


def run_polymarket_ingestion(force: bool = False):
    """Main entry point for Polymarket data collection."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    cache = Cache("polymarket")

    markets = fetch_all_markets(cache, force=force)
    
    # Fetch prices for all markets with meaningful volume
    MIN_VOL = 100000  # $100K - lowered to maximize coverage
    vol_markets = [m for m in markets if (m.get("volume_total") or 0) >= MIN_VOL]
    logger.info(f"Fetching price history for {len(vol_markets)} markets with volume >= ${MIN_VOL:,} "
                f"(skipping {len(markets) - len(vol_markets)} low-volume)...")

    success = 0
    failed = 0
    for i, market in enumerate(vol_markets):
        cid = market.get("condition_id")
        if not cid:
            continue

        candles = fetch_price_history(market, cache, force=force)
        if candles:
            success += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i + 1}/{len(vol_markets)} (success={success}, failed={failed})")

        # CLOB API rate limit - be gentle
        time.sleep(0.2)

    logger.info(f"Polymarket ingestion complete: {success} with data, {failed} failed")
    return markets


if __name__ == "__main__":
    run_polymarket_ingestion()

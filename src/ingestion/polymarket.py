"""Polymarket data ingestion via Dome API."""

import logging
import time
from datetime import datetime, timezone, timedelta

from .api_client import api_get
from .cache import Cache

logger = logging.getLogger(__name__)


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
            data = api_get("/polymarket/markets", params={
                "status": status, "limit": limit, "offset": offset
            })
            batch = data.get("markets", [])
            if not batch:
                break
            markets.extend(batch)
            logger.info(f"  {status}: fetched {len(markets)} markets so far...")
            offset += limit

            # Check if no more - Polymarket doesn't return pagination field
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


def fetch_candlesticks(condition_id: str, cache: Cache,
                       months_back: int = 6, force: bool = False) -> list:
    """Fetch daily candlesticks for a market going back N months."""
    cache_key = f"candles_{condition_id}"

    if not force:
        cached = cache.get(cache_key)
        last_fetch = cache.get_last_fetch(cache_key)
        if cached is not None and last_fetch and not cache.needs_update(cache_key, max_age_hours=12):
            return cached

    now = int(time.time())
    start = int((datetime.now(timezone.utc) - timedelta(days=30 * months_back)).timestamp())

    try:
        data = api_get(f"/polymarket/candlesticks/{condition_id}", params={
            "start_time": start,
            "end_time": now,
            "interval": 1440  # daily
        })
    except Exception as e:
        logger.warning(f"Failed to fetch candles for {condition_id[:20]}...: {e}")
        return []

    candles_raw = data.get("candlesticks", [])

    # Response is [candles_list, token_info] - extract candles
    candles = []
    if candles_raw and isinstance(candles_raw[0], list):
        candles = candles_raw[0]
    elif candles_raw and isinstance(candles_raw[0], dict):
        candles = candles_raw

    cache.put(cache_key, candles)
    cache.set_last_fetch(cache_key)
    return candles


def run_polymarket_ingestion(force: bool = False):
    """Main entry point for Polymarket data collection."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    cache = Cache("polymarket")

    markets = fetch_all_markets(cache, force=force)
    # Only fetch candles for markets with meaningful volume (>$10K)
    vol_markets = [m for m in markets if (m.get("volume_total") or 0) >= 10000]
    logger.info(f"Fetching candlesticks for {len(vol_markets)} markets with volume >= $10K "
                f"(skipping {len(markets) - len(vol_markets)} low-volume)...")

    success = 0
    failed = 0
    for i, market in enumerate(vol_markets):
        cid = market.get("condition_id")
        if not cid:
            continue

        candles = fetch_candlesticks(cid, cache, force=force)
        if candles:
            success += 1
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(vol_markets)} (success={success}, failed={failed})")

    logger.info(f"Polymarket ingestion complete: {success} with data, {failed} failed")
    return markets


if __name__ == "__main__":
    run_polymarket_ingestion()

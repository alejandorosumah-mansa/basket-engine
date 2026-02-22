"""Kalshi data ingestion via Dome API."""

import logging
import time
from datetime import datetime, timezone, timedelta

from .api_client import api_get
from .cache import Cache

logger = logging.getLogger(__name__)

# Kalshi has 477K+ markets. We filter to those with meaningful volume.
MIN_VOLUME = 50000  # minimum total volume (contracts) - lowered from 500K to capture more markets


def fetch_all_markets(cache: Cache, force: bool = False) -> list:
    """Fetch all Kalshi markets (open + closed) with pagination.
    
    Filters to markets with volume >= MIN_VOLUME to avoid pulling 477K garbage markets.
    """
    all_markets = []

    for status in ["open", "closed"]:
        cache_key = f"markets_{status}"
        if not force and not cache.needs_update(cache_key, max_age_hours=12):
            cached = cache.get(cache_key)
            if cached:
                logger.info(f"Using cached {status} Kalshi markets: {len(cached)} markets")
                all_markets.extend(cached)
                continue

        logger.info(f"Fetching {status} Kalshi markets...")
        markets = []
        offset = 0
        limit = 100
        empty_pages = 0

        while True:
            try:
                data = api_get("/kalshi/markets", params={
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

            # Filter by volume
            filtered = [m for m in batch if (m.get("volume") or 0) >= MIN_VOLUME]
            markets.extend(filtered)

            pagination = data.get("pagination", {})
            offset += limit

            # Track if we're getting low-volume markets (they're sorted by volume desc)
            if not filtered:
                empty_pages += 1
                if empty_pages >= 3:
                    logger.info(f"  {status}: 3 consecutive pages with no qualifying markets, stopping")
                    break
            else:
                empty_pages = 0

            if not pagination.get("has_more", len(batch) == limit):
                break

            if len(markets) % 500 == 0 and len(markets) > 0:
                logger.info(f"  {status}: fetched {len(markets)} qualifying markets so far...")

        cache.put(cache_key, markets)
        cache.set_last_fetch(cache_key)
        logger.info(f"Fetched {len(markets)} {status} Kalshi markets (volume >= {MIN_VOLUME})")
        all_markets.extend(markets)

    # Deduplicate by market_ticker
    seen = set()
    unique = []
    for m in all_markets:
        ticker = m.get("market_ticker")
        if ticker and ticker not in seen:
            seen.add(ticker)
            unique.append(m)
    logger.info(f"Total unique Kalshi markets: {len(unique)}")
    return unique


def fetch_trades(ticker: str, cache: Cache,
                 months_back: int = 6, force: bool = False) -> list:
    """Fetch trade history for a Kalshi market."""
    cache_key = f"trades_{ticker}"

    if not force:
        cached = cache.get(cache_key)
        if cached is not None and not cache.needs_update(cache_key, max_age_hours=12):
            return cached

    now = int(time.time())
    start = int((datetime.now(timezone.utc) - timedelta(days=30 * months_back)).timestamp())

    all_trades = []
    offset = 0
    limit = 100

    try:
        while True:
            data = api_get("/kalshi/trades", params={
                "ticker": ticker,
                "start_time": start,
                "end_time": now,
                "limit": limit,
                "offset": offset
            })
            batch = data.get("trades", [])
            if not batch:
                break
            all_trades.extend(batch)
            
            pagination = data.get("pagination", {})
            if not pagination.get("has_more", len(batch) == limit):
                break
            offset += limit

            # Cap at 200 trades per market to keep it fast
            if len(all_trades) >= 200:
                break
    except Exception as e:
        logger.warning(f"Failed to fetch trades for {ticker}: {e}")

    cache.put(cache_key, all_trades)
    cache.set_last_fetch(cache_key)
    return all_trades


def run_kalshi_ingestion(force: bool = False):
    """Main entry point for Kalshi data collection."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    cache = Cache("kalshi")

    markets = fetch_all_markets(cache, force=force)
    logger.info(f"Fetching trades for {len(markets)} markets...")

    success = 0
    failed = 0
    for i, market in enumerate(markets):
        ticker = market.get("market_ticker")
        if not ticker:
            continue

        trades = fetch_trades(ticker, cache, force=force)
        if trades:
            success += 1
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(markets)} (success={success}, failed={failed})")

    logger.info(f"Kalshi ingestion complete: {success} with trades, {failed} empty")
    return markets


if __name__ == "__main__":
    run_kalshi_ingestion()

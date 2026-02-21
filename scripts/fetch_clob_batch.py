#!/usr/bin/env python3
"""Fetch CLOB prices in small batches to avoid memory issues."""

import json
import logging
import time
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

CLOB_BASE = "https://clob.polymarket.com"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "polymarket"

to_fetch = json.loads((RAW_DIR / "_to_fetch.json").read_text())
batch_start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 500

batch = to_fetch[batch_start:batch_start + batch_size]
logger.info(f"Batch {batch_start}-{batch_start + len(batch)} of {len(to_fetch)}")

session = requests.Session()
success = 0
empty = 0
failed = 0

for i, entry in enumerate(batch):
    cid = entry["c"]
    token_id = entry["t"]
    try:
        resp = session.get(f"{CLOB_BASE}/prices-history", params={
            "market": token_id, "interval": "all", "fidelity": 1440
        }, timeout=30)
        resp.raise_for_status()
        history = resp.json().get("history", [])
        candles = [{"end_period_ts": p["t"], "price": {"close_dollars": str(p["p"])}, "volume": 0}
                   for p in history if "t" in p and "p" in p]
        (RAW_DIR / f"candles_{cid}.json").write_text(json.dumps(candles))
        if candles:
            success += 1
        else:
            empty += 1
    except Exception as e:
        failed += 1
        (RAW_DIR / f"candles_{cid}.json").write_text("[]")

    if (i + 1) % 100 == 0:
        logger.info(f"  {i+1}/{len(batch)} ok={success} empty={empty} fail={failed}")
    time.sleep(0.12)

logger.info(f"Batch done: {success} ok, {empty} empty, {failed} fail")

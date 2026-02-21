#!/usr/bin/env python3
"""Fetch one market's price history from CLOB and save."""
import json, sys, requests
cid, token_id, outfile = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    r = requests.get("https://clob.polymarket.com/prices-history",
                      params={"market": token_id, "interval": "all", "fidelity": 1440}, timeout=30)
    r.raise_for_status()
    h = r.json().get("history", [])
    candles = [{"end_period_ts": p["t"], "price": {"close_dollars": str(p["p"])}, "volume": 0}
               for p in h if "t" in p and "p" in p]
    open(outfile, "w").write(json.dumps(candles))
except:
    open(outfile, "w").write("[]")

#!/usr/bin/env python3
"""Convert clob_raw_*.json files to candles_*.json format."""
import json, os, sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "polymarket")
converted = 0
skipped = 0
errors = 0

for fname in os.listdir(RAW_DIR):
    if not fname.startswith("clob_raw_"):
        continue
    cid = fname[9:-5]
    candle_file = os.path.join(RAW_DIR, f"candles_{cid}.json")
    
    # Skip if candle file already has data
    if os.path.exists(candle_file) and os.path.getsize(candle_file) > 5:
        skipped += 1
        continue
    
    raw_path = os.path.join(RAW_DIR, fname)
    try:
        with open(raw_path) as f:
            data = json.load(f)
        history = data.get("history", [])
        candles = [
            {"end_period_ts": p["t"], "price": {"close_dollars": str(p["p"])}, "volume": 0}
            for p in history if "t" in p and "p" in p
        ]
        with open(candle_file, "w") as f:
            json.dump(candles, f)
        converted += 1
    except Exception as e:
        # Write empty candle
        with open(candle_file, "w") as f:
            f.write("[]")
        errors += 1

print(f"Converted: {converted}, Skipped: {skipped}, Errors: {errors}")

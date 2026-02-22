"""
Generate classification output CSVs for data/outputs/.

Outputs:
- event_classifications.csv: every event with theme, confidence, exposure direction
- ticker_event_mapping.csv: CUSIP/Ticker → Event mapping
- theme_summary.csv: count per theme, avg confidence, avg volume
- exposure_summary.csv: long/short breakdown per theme
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json


def main():
    out = Path("data/outputs")
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    fc = pd.read_csv("data/processed/final_classifications.csv")
    with open("data/processed/exposure_classifications.json") as f:
        exposures = json.load(f)

    # 1. event_classifications.csv
    exp_df = pd.DataFrame([
        {"event_id": k, "exposure_direction": v["exposure_direction"],
         "exposure_description": v["exposure_description"],
         "exposure_confidence": v["confidence"]}
        for k, v in exposures.items()
    ])

    ec = fc[["event_id", "platform", "event_title", "reconciled_theme",
             "primary_theme", "secondary_theme", "confidence", "volume"]].copy()
    ec = ec.rename(columns={"reconciled_theme": "theme"})
    ec = ec.merge(exp_df, on="event_id", how="left")
    ec.to_csv(out / "event_classifications.csv", index=False)
    print(f"event_classifications.csv: {len(ec)} rows")

    # 2. ticker_event_mapping.csv
    has_ticker = fc[fc["ticker"].notna() & (fc["ticker"] != "")][
        ["event_id", "platform", "ticker", "condition_id", "event_title", "reconciled_theme"]
    ].copy()
    has_ticker = has_ticker.rename(columns={"reconciled_theme": "theme"})
    has_ticker.to_csv(out / "ticker_event_mapping.csv", index=False)
    print(f"ticker_event_mapping.csv: {len(has_ticker)} rows")

    # 3. theme_summary.csv
    ts = fc.groupby("reconciled_theme").agg(
        event_count=("event_id", "count"),
        avg_confidence=("confidence", "mean"),
        avg_volume=("volume", "mean"),
        total_volume=("volume", "sum"),
        min_confidence=("confidence", "min"),
        max_confidence=("confidence", "max"),
    ).round(4).sort_values("event_count", ascending=False)
    ts.index.name = "theme"
    ts.to_csv(out / "theme_summary.csv")
    print(f"theme_summary.csv: {len(ts)} themes")

    # 4. exposure_summary.csv
    # Merge exposures with themes
    merged = ec[["event_id", "theme", "exposure_direction", "volume"]].dropna(subset=["exposure_direction"])
    es = merged.groupby(["theme", "exposure_direction"]).agg(
        event_count=("event_id", "count"),
        avg_volume=("volume", "mean"),
        total_volume=("volume", "sum"),
    ).round(2)
    es.to_csv(out / "exposure_summary.csv")
    print(f"exposure_summary.csv: {len(es)} rows")

    print("\nAll classification outputs generated in data/outputs/")


if __name__ == "__main__":
    main()

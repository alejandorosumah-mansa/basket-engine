#!/usr/bin/env python3
"""Classify all 20K markets from markets.parquet using gpt-4o-mini with concurrency."""

import json
import os
import sys
import time
import hashlib
import random
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from classification.taxonomy import load_taxonomy, format_taxonomy_for_prompt, list_themes

load_dotenv(Path.home() / ".openclaw" / ".env")

DATA_DIR = ROOT / "data" / "processed"
CACHE_PATH = DATA_DIR / "llm_classifications_cache_v2.json"
OUTPUT_CSV = DATA_DIR / "llm_classifications_full.csv"
REPORT_PATH = DATA_DIR / "classification_report_full.md"

MAX_WORKERS = 20  # concurrent API calls


def market_hash(row):
    key = f"{row.get('platform', '')}|{row.get('market_id', '')}|{row.get('title', '')}"
    return hashlib.md5(key.encode()).hexdigest()


def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def build_system_prompt(themes):
    taxonomy_text = format_taxonomy_for_prompt(themes)
    theme_ids = list_themes(themes)

    few_shot = [
        ("Will Trump impose new tariffs on China?", "china_us", 0.92),
        ("Will the Fed cut rates before July?", "fed_monetary_policy", 0.98),
        ("Will the Lakers win the NBA championship?", "sports_entertainment", 0.99),
        ("Bitcoin price above $100K?", "crypto_digital", 0.95),
        ("US CPI above 3.5% YoY?", "us_economic", 0.97),
    ]
    few_shot_text = "\n".join(f'"{t}" → {{"primary_theme":"{th}","confidence":{c}}}' for t,th,c in few_shot)

    return f"""You are a prediction market classifier. Assign each market to exactly one primary investment theme.

TAXONOMY:
{taxonomy_text}

## sports_entertainment: Sports & Entertainment
Description: Sports outcomes, player stats, awards shows, celebrity events, TV ratings, movie box office, music charts, reality TV, esports.
Examples: "Will the Lakers win the NBA Finals?", "Super Bowl winner?", "Oscar Best Picture?"

VALID THEME IDS: {json.dumps(theme_ids + ['sports_entertainment'])}

RULES:
1. Assign exactly 1 primary_theme. Optionally 1 secondary_theme.
2. Confidence 0.0-1.0. If no theme fits, use "uncategorized".
3. Sports/entertainment/pop culture → sports_entertainment.
4. BTC/crypto price predictions → crypto_digital.
5. Focus on CORE SUBJECT.

EXAMPLES:
{few_shot_text}

Return ONLY valid JSON: {{"primary_theme":"id","secondary_theme":null,"confidence":0.9,"reasoning":"one sentence"}}"""


def classify_one(client, system_prompt, row_data, valid_themes):
    """Classify a single market. row_data is a dict."""
    h, platform, title, desc, tags = row_data
    user_msg = f'Platform: {platform}\nTitle: {title}\nDescription: {desc[:300]}\nTags: {tags}\nReturn JSON only.'
    
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(resp.choices[0].message.content)
            if result.get("primary_theme") not in valid_themes:
                result["primary_theme"] = "uncategorized"
            return h, {
                "primary_theme": result.get("primary_theme", "uncategorized"),
                "secondary_theme": result.get("secondary_theme"),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", "")
            }, None
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                time.sleep(min(60, 2 ** (attempt + 2)))
            elif attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return h, {
                    "primary_theme": "uncategorized",
                    "secondary_theme": None,
                    "confidence": 0.0,
                    "reasoning": f"Error: {e}"
                }, str(e)
    return h, {"primary_theme": "uncategorized", "secondary_theme": None, "confidence": 0.0, "reasoning": "max retries"}, "max retries"


def main():
    print("Loading markets from parquet...", flush=True)
    df = pd.read_parquet(DATA_DIR / "markets.parquet")
    print(f"Loaded {len(df)} markets", flush=True)

    themes = load_taxonomy()
    theme_ids = list_themes(themes)
    valid_themes = set(theme_ids + ["sports_entertainment", "uncategorized"])
    system_prompt = build_system_prompt(themes)

    cache = load_cache()
    print(f"Cache has {len(cache)} entries", flush=True)

    # Compute hashes
    hashes = []
    for _, row in df.iterrows():
        hashes.append(market_hash(row))
    df["_hash"] = hashes

    # Find uncached
    to_classify = []
    for _, row in df.iterrows():
        h = row["_hash"]
        if h not in cache:
            to_classify.append((
                h,
                str(row.get("platform", "")),
                str(row.get("title", "")),
                str(row.get("description", ""))[:300],
                str(row.get("tags", ""))
            ))

    print(f"Need to classify: {len(to_classify)} ({len(df) - len(to_classify)} cached)", flush=True)

    if to_classify:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        start_time = time.time()
        done = 0
        errors = 0

        print(f"Classifying with {MAX_WORKERS} workers...", flush=True)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            batch_size = 500
            
            for batch_start in range(0, len(to_classify), batch_size):
                batch = to_classify[batch_start:batch_start + batch_size]
                futures_batch = {
                    executor.submit(classify_one, client, system_prompt, rd, valid_themes): rd[0]
                    for rd in batch
                }
                
                for future in as_completed(futures_batch):
                    h, result, err = future.result()
                    cache[h] = result
                    done += 1
                    if err:
                        errors += 1
                    
                    if done % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = done / elapsed
                        eta = (len(to_classify) - done) / rate if rate > 0 else 0
                        print(f"  [{done}/{len(to_classify)}] {rate:.1f}/s ETA:{eta/60:.1f}min errs:{errors}", flush=True)
                
                # Save after each batch
                save_cache(cache)
                print(f"  Saved cache ({len(cache)} entries)", flush=True)

        elapsed = time.time() - start_time
        print(f"Done: {done} calls in {elapsed/60:.1f}min ({errors} errors)", flush=True)

    # Build output
    print("Building results...", flush=True)
    results = []
    for _, row in df.iterrows():
        h = row["_hash"]
        results.append(cache.get(h, {"primary_theme": "uncategorized", "secondary_theme": None, "confidence": 0.0, "reasoning": "missing"}))

    result_df = pd.DataFrame(results)
    result_df.index = df.index
    out = pd.concat([df.drop(columns=["_hash"]), result_df], axis=1)

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(out)} to {OUTPUT_CSV}", flush=True)

    report = generate_report(out)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report saved to {REPORT_PATH}", flush=True)
    print("\n" + report, flush=True)


def generate_report(df):
    lines = [
        "# Full Classification Report",
        f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"\n## Overview",
        f"- Total markets: {len(df):,}",
        f"- Classified: {(df['primary_theme'] != 'uncategorized').sum():,}",
        f"- Uncategorized: {(df['primary_theme'] == 'uncategorized').sum():,}",
        f"- Mean confidence: {df['confidence'].mean():.3f}",
        f"- Median confidence: {df['confidence'].median():.3f}",
        f"- With secondary theme: {df['secondary_theme'].notna().sum():,}",
    ]

    lines.append("\n## Theme Distribution")
    lines.append("| Theme | Count | % | Avg Conf | Flag |")
    lines.append("|-------|-------|---|----------|------|")
    counts = df["primary_theme"].value_counts()
    for theme, count in counts.items():
        pct = count / len(df) * 100
        avg_conf = df[df["primary_theme"] == theme]["confidence"].mean()
        flag = "⚠️ <5" if count < 5 else ("⚠️ >5000" if count > 5000 else "")
        lines.append(f"| {theme} | {count:,} | {pct:.1f}% | {avg_conf:.2f} | {flag} |")

    # Category breakdown
    sports = (df["primary_theme"] == "sports_entertainment").sum()
    crypto = (df["primary_theme"] == "crypto_digital").sum()
    serious_themes = ["us_elections", "fed_monetary_policy", "us_economic", "ai_technology",
                      "china_us", "russia_ukraine", "middle_east", "europe_politics",
                      "legal_regulatory", "climate_environment", "pandemic_health",
                      "energy_commodities", "space_frontier"]
    serious = df["primary_theme"].isin(serious_themes).sum()
    uncat = (df["primary_theme"] == "uncategorized").sum()
    lines.append(f"\n## Category Breakdown")
    lines.append(f"- Sports/Entertainment: {sports:,} ({sports/len(df)*100:.1f}%)")
    lines.append(f"- Crypto/Digital: {crypto:,} ({crypto/len(df)*100:.1f}%)")
    lines.append(f"- Serious/Political/Economic: {serious:,} ({serious/len(df)*100:.1f}%)")
    lines.append(f"- Uncategorized: {uncat:,} ({uncat/len(df)*100:.1f}%)")

    # Confidence distribution
    lines.append(f"\n## Confidence Distribution")
    for t in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        n = (df["confidence"] >= t).sum()
        lines.append(f"- >= {t}: {n:,} ({n/len(df)*100:.1f}%)")

    # Flagged
    lines.append(f"\n## Flagged Themes")
    for theme, count in counts.items():
        if count < 5:
            lines.append(f"- ⚠️ {theme}: only {count} markets")
        if count > 5000:
            lines.append(f"- ⚠️ {theme}: {count:,} markets (>5000)")

    # Random sample
    lines.append(f"\n## Random Sample (20)")
    sample = df.sample(min(20, len(df)), random_state=42)
    for _, row in sample.iterrows():
        title = str(row.get("title", ""))[:80]
        lines.append(f"- [{row['confidence']:.2f}] **{row['primary_theme']}** | {title}")
        if row.get("reasoning"):
            lines.append(f"  > {str(row['reasoning'])[:120]}")

    # Low confidence
    lines.append(f"\n## Low Confidence (< 0.5)")
    low = df[df["confidence"] < 0.5]
    lines.append(f"Count: {len(low):,}")
    for _, row in low.head(10).iterrows():
        lines.append(f"- [{row['confidence']:.2f}] {str(row.get('title',''))[:80]} → {row['primary_theme']}")

    # Platform
    lines.append(f"\n## By Platform")
    for plat in df["platform"].unique():
        sub = df[df["platform"] == plat]
        top = sub["primary_theme"].value_counts().head(3)
        top_str = ", ".join(f"{t}({c})" for t,c in top.items())
        lines.append(f"- {plat}: {len(sub):,} markets. Top: {top_str}")

    return "\n".join(lines)


if __name__ == "__main__":
    main()

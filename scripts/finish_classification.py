"""Finish classification using v2 cache and markets.parquet."""
import json
import sys
import os
import random
import hashlib
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path.home() / ".openclaw" / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from classification.taxonomy import load_taxonomy, format_taxonomy_for_prompt, list_themes
from classification.llm_classifier import _build_system_prompt, classify_market

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
CACHE_PATH = DATA_DIR / "llm_classifications_cache_v2.json"

def market_hash(row):
    key = f"{row.get('platform', '')}|{row.get('market_id', '')}|{row.get('title', '')}"
    return hashlib.md5(key.encode()).hexdigest()

def main():
    df = pd.read_parquet(DATA_DIR / "markets.parquet")
    print(f"Loaded {len(df)} markets")
    
    # Rename columns so classifier prompts work
    df = df.rename(columns={"title": "event_title", "market_id": "event_id"})
    
    # Load cache
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    print(f"Cache has {len(cache)} entries")
    
    # Compute hashes using original column names
    df_orig = pd.read_parquet(DATA_DIR / "markets.parquet")
    hash_keys = df_orig["platform"].fillna("") + "|" + df_orig["market_id"].fillna("") + "|" + df_orig["title"].fillna("")
    hashes = [hashlib.md5(k.encode()).hexdigest() for k in hash_keys]
    
    uncached_indices = [i for i, h in enumerate(hashes) if h not in cache]
    print(f"Uncached: {len(uncached_indices)} markets")
    
    if uncached_indices:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        themes = load_taxonomy()
        theme_ids = list_themes(themes)
        system_prompt = _build_system_prompt(themes)
        
        api_count = 0
        for idx in uncached_indices:
            row = df.iloc[idx]
            h = hashes[idx]
            result = classify_market(client, system_prompt, row, theme_ids)
            result["market_hash"] = h
            cache[h] = result
            api_count += 1
            time.sleep(0.02)
            
            if api_count % 50 == 0:
                with open(CACHE_PATH, "w") as f:
                    json.dump(cache, f)
                print(f"  Progress: {api_count}/{len(uncached_indices)}")
        
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)
        print(f"Classified {api_count} new markets")
    
    # Build results
    results = []
    for h in hashes:
        results.append(cache[h])
    
    result_df = pd.DataFrame(results)
    result_df.index = df.index
    out = pd.concat([df, result_df], axis=1)
    
    # Save CSV
    csv_path = DATA_DIR / "llm_classifications_full.csv"
    out.to_csv(csv_path, index=False)
    print(f"Saved {len(out)} to {csv_path}")
    
    # Generate report
    report = generate_full_report(out)
    report_path = DATA_DIR / "classification_report_full.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")


def generate_full_report(df):
    lines = ["# Full LLM Classification Report\n"]
    lines.append(f"**Total markets:** {len(df)}")
    lines.append(f"**Classified (non-uncategorized):** {(df['primary_theme'] != 'uncategorized').sum()}")
    lines.append(f"**Uncategorized:** {(df['primary_theme'] == 'uncategorized').sum()}")
    lines.append(f"**Mean confidence:** {df['confidence'].mean():.3f}")
    lines.append(f"**Median confidence:** {df['confidence'].median():.3f}")
    lines.append(f"**Markets with secondary theme:** {df['secondary_theme'].notna().sum()}")
    
    # Theme counts
    lines.append("\n## Theme Counts\n")
    lines.append("| Theme | Count | % | Avg Confidence |")
    lines.append("|-------|-------|---|----------------|")
    counts = df["primary_theme"].value_counts()
    for theme, count in counts.items():
        pct = count / len(df) * 100
        avg_conf = df[df["primary_theme"] == theme]["confidence"].mean()
        lines.append(f"| {theme} | {count} | {pct:.1f}% | {avg_conf:.2f} |")
    
    # Confidence distribution
    lines.append("\n## Confidence Distribution\n")
    bins = [(0, 0.4, "Very Low"), (0.4, 0.6, "Low"), (0.6, 0.8, "Medium"), (0.8, 0.9, "High"), (0.9, 1.01, "Very High")]
    for lo, hi, label in bins:
        n = ((df["confidence"] >= lo) & (df["confidence"] < hi)).sum()
        lines.append(f"- **{label}** ({lo:.1f}-{hi:.1f}): {n} ({n/len(df)*100:.1f}%)")
    
    # Noise analysis
    lines.append("\n## Noise vs Serious Markets\n")
    noise_themes = {"sports", "entertainment", "meme_culture", "uncategorized"}
    # Check for sports-like themes
    all_themes = df["primary_theme"].unique()
    sports_themes = [t for t in all_themes if any(kw in t.lower() for kw in ["sport", "nba", "nfl", "mlb", "soccer", "football", "entertainment", "meme", "pop_culture", "celebrity"])]
    noise_themes_actual = noise_themes | set(sports_themes)
    
    noise = df[df["primary_theme"].isin(noise_themes_actual)]
    serious = df[~df["primary_theme"].isin(noise_themes_actual)]
    lines.append(f"- **Noise markets** (sports/entertainment/meme/uncategorized): {len(noise)} ({len(noise)/len(df)*100:.1f}%)")
    lines.append(f"- **Serious/investable markets**: {len(serious)} ({len(serious)/len(df)*100:.1f}%)")
    
    if noise_themes_actual:
        lines.append(f"- Noise theme IDs: {sorted(noise_themes_actual)}")
    
    # BTC up/down check
    btc_noise = df[df["event_title"].str.contains("BTC|Bitcoin", case=False, na=False) & 
                   df["event_title"].str.contains("above|below|price|hit", case=False, na=False)]
    lines.append(f"- **BTC price-level markets**: {len(btc_noise)}")
    
    # Random sample for human review
    lines.append("\n## Random Sample for Human Review (20 markets)\n")
    sample = df.sample(n=min(20, len(df)), random_state=42)
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        title = str(row.get("event_title", "N/A"))[:100]
        lines.append(f"### {i}. {title}")
        lines.append(f"- **Platform:** {row.get('platform', 'N/A')}")
        lines.append(f"- **Primary:** {row['primary_theme']} | **Secondary:** {row.get('secondary_theme', 'None')}")
        lines.append(f"- **Confidence:** {row['confidence']:.2f}")
        reasoning = str(row.get('reasoning', 'N/A'))[:200]
        lines.append(f"- **Reasoning:** {reasoning}")
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()

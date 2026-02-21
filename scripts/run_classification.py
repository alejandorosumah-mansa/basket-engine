"""Run classification with concurrent API calls for speed."""

import json
import os
import sys
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path.home() / ".openclaw" / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.classification.taxonomy import load_taxonomy, format_taxonomy_for_prompt, list_themes

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
CACHE_PATH = DATA_DIR / "llm_classifications_cache.json"

CSV_PATH = Path.home() / ".openclaw/workspace/basket-trader/scripts/seed_data/markets_with_baskets_final_20260119.csv"

# Build prompts
themes = load_taxonomy()
theme_ids = list_themes(themes)
taxonomy_text = format_taxonomy_for_prompt(themes)

FEW_SHOT = [
    {"title": "Will Trump impose new tariffs on China in Q1 2026?", "tags": "Politics, Trump, Trade",
     "output": {"primary_theme": "china_us", "secondary_theme": "legal_regulatory", "confidence": 0.92,
                "reasoning": "Tariffs on China are about bilateral trade relations, not US elections."}},
    {"title": "Will the Fed cut rates before July 2026?", "tags": "Economics, Fed Rates",
     "output": {"primary_theme": "fed_monetary_policy", "secondary_theme": None, "confidence": 0.98,
                "reasoning": "Direct Fed rate action."}},
    {"title": "Will Trump fire the FBI director?", "tags": "Politics, Trump",
     "output": {"primary_theme": "legal_regulatory", "secondary_theme": "us_elections", "confidence": 0.85,
                "reasoning": "Executive/regulatory action, not an election event."}},
    {"title": "US CPI above 3.5% YoY in March?", "tags": "Economics",
     "output": {"primary_theme": "us_economic", "secondary_theme": None, "confidence": 0.97,
                "reasoning": "CPI is a measurable economic indicator."}},
]

few_shot_text = "\n".join([
    f"Market: \"{ex['title']}\" | Tags: {ex['tags']}\nOutput: {json.dumps(ex['output'])}"
    for ex in FEW_SHOT
])

SYSTEM_PROMPT = f"""You are a prediction market classifier. Assign each market to exactly one primary investment theme.

TAXONOMY:
{taxonomy_text}

VALID THEME IDS: {json.dumps(theme_ids)}

RULES:
1. Assign exactly 1 primary_theme from valid IDs above.
2. Optionally assign 1 secondary_theme if market genuinely spans two themes.
3. Rate confidence 0.0-1.0.
4. If no theme fits, use "uncategorized".
5. Pay attention to anti-examples. "Trump tariffs on China" = china_us, NOT us_elections.
6. Focus on CORE SUBJECT, not who is involved.

FEW-SHOT EXAMPLES:
{few_shot_text}

Return ONLY valid JSON: {{"primary_theme": "id", "secondary_theme": "id_or_null", "confidence": 0.85, "reasoning": "one sentence"}}"""


def market_hash(row):
    key = f"{row.get('platform', '')}|{row.get('event_id', '')}|{row.get('event_title', '')}"
    return hashlib.md5(key.encode()).hexdigest()


def classify_one(client, row_data):
    """Classify a single market."""
    idx, row = row_data
    title = str(row.get("event_title", ""))
    desc = str(row.get("description", ""))[:500]
    tags = str(row.get("tags", ""))
    platform = str(row.get("platform", ""))
    
    user_msg = f"Classify:\nPlatform: {platform}\nTitle: {title}\nDescription: {desc}\nTags: {tags}\n\nReturn JSON only."
    
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini", temperature=0,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_msg}],
                response_format={"type": "json_object"}
            )
            result = json.loads(resp.choices[0].message.content)
            if result.get("primary_theme") not in theme_ids and result.get("primary_theme") != "uncategorized":
                result["primary_theme"] = "uncategorized"
            if result.get("secondary_theme") and result["secondary_theme"] not in theme_ids:
                result["secondary_theme"] = None
            return idx, {
                "primary_theme": result.get("primary_theme", "uncategorized"),
                "secondary_theme": result.get("secondary_theme"),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", "")
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return idx, {"primary_theme": "uncategorized", "secondary_theme": None,
                        "confidence": 0.0, "reasoning": f"Error: {e}"}


def main():
    print(f"Loading markets from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} markets")
    
    # Load cache
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    print(f"Cache has {len(cache)} entries")
    
    # Split into cached vs needs-API
    to_classify = []
    results = {}
    for idx, row in df.iterrows():
        mh = market_hash(row)
        if mh in cache:
            results[idx] = cache[mh]
        else:
            to_classify.append((idx, row, mh))
    
    print(f"Cached: {len(results)}, Need API: {len(to_classify)}")
    
    if to_classify:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        # Use 10 concurrent threads
        done = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for idx, row, mh in to_classify:
                f = executor.submit(classify_one, client, (idx, row))
                futures[f] = mh
            
            for future in as_completed(futures):
                mh = futures[future]
                idx, result = future.result()
                results[idx] = result
                cache[mh] = result
                done += 1
                if done % 50 == 0:
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    with open(CACHE_PATH, "w") as f:
                        json.dump(cache, f)
                    print(f"  Progress: {done}/{len(to_classify)} API calls done")
        
        # Final cache save
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)
    
    # Build result DataFrame
    result_rows = []
    for idx in range(len(df)):
        result_rows.append(results[idx])
    
    result_df = pd.DataFrame(result_rows)
    out = pd.concat([df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)
    
    # Save
    out_path = DATA_DIR / "llm_classifications.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} classifications to {out_path}")
    
    # Report
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(f"Total: {len(out)}")
    print(f"Classified: {(out['primary_theme'] != 'uncategorized').sum()}")
    print(f"Uncategorized: {(out['primary_theme'] == 'uncategorized').sum()}")
    print(f"Mean confidence: {out['confidence'].mean():.3f}")
    print(f"Median confidence: {out['confidence'].median():.3f}")
    print(f"\nTheme counts:")
    counts = out["primary_theme"].value_counts()
    for theme, count in counts.items():
        avg_conf = out[out["primary_theme"] == theme]["confidence"].mean()
        print(f"  {theme:30s} {count:4d}  (avg conf: {avg_conf:.2f})")
    
    # Low confidence
    low = out[out["confidence"] < 0.6].sort_values("confidence")
    print(f"\nLow confidence (<0.6): {len(low)} markets")
    for _, row in low.head(10).iterrows():
        print(f"  [{row['confidence']:.2f}] {str(row.get('event_title', ''))[:70]} → {row['primary_theme']}")
    
    # Save report
    report_lines = [
        "# LLM Classification Report\n",
        f"Total markets: {len(out)}",
        f"Classified: {(out['primary_theme'] != 'uncategorized').sum()}",
        f"Uncategorized: {(out['primary_theme'] == 'uncategorized').sum()}",
        f"Mean confidence: {out['confidence'].mean():.3f}",
        f"Median confidence: {out['confidence'].median():.3f}",
        f"\n## Theme Counts",
    ]
    for theme, count in counts.items():
        avg_conf = out[out["primary_theme"] == theme]["confidence"].mean()
        report_lines.append(f"- {theme}: {count} (avg confidence: {avg_conf:.2f})")
    
    report_path = DATA_DIR / "classification_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()

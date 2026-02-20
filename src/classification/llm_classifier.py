"""LLM-based market classification pipeline using OpenAI."""

import json
import os
import time
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from .taxonomy import load_taxonomy, format_taxonomy_for_prompt, list_themes

# Load env
load_dotenv(Path.home() / ".openclaw" / ".env")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
CACHE_PATH = DATA_DIR / "llm_classifications_cache.json"

FEW_SHOT_EXAMPLES = [
    {
        "title": "Will Trump impose new tariffs on China in Q1 2026?",
        "tags": "Politics, Trump, Trade",
        "output": {
            "primary_theme": "china_us",
            "secondary_theme": "legal_regulatory",
            "confidence": 0.92,
            "reasoning": "Tariffs on China are fundamentally about bilateral trade relations, not US elections despite involving Trump."
        }
    },
    {
        "title": "Will the Fed cut rates before July 2026?",
        "tags": "Economics, Fed Rates",
        "output": {
            "primary_theme": "fed_monetary_policy",
            "secondary_theme": None,
            "confidence": 0.98,
            "reasoning": "Direct Fed rate action, clearly monetary policy."
        }
    },
    {
        "title": "Will Trump fire the FBI director?",
        "tags": "Politics, Trump",
        "output": {
            "primary_theme": "legal_regulatory",
            "secondary_theme": "us_elections",
            "confidence": 0.85,
            "reasoning": "Firing an agency head is executive/regulatory action, not an election event. Secondary link to partisan politics."
        }
    },
    {
        "title": "US CPI above 3.5% year-over-year in March?",
        "tags": "Economics",
        "output": {
            "primary_theme": "us_economic",
            "secondary_theme": None,
            "confidence": 0.97,
            "reasoning": "CPI is a measurable economic indicator, not a Fed action."
        }
    },
    {
        "title": "Will NVIDIA stock hit $200?",
        "tags": "Stocks, AI",
        "output": {
            "primary_theme": "ai_technology",
            "secondary_theme": None,
            "confidence": 0.55,
            "reasoning": "NVIDIA stock is primarily a stock market bet, but closely tied to AI demand. Low confidence as it's borderline."
        }
    }
]


def _build_system_prompt(themes: dict) -> str:
    taxonomy_text = format_taxonomy_for_prompt(themes)
    theme_ids = list_themes(themes)

    few_shot_text = "\n".join([
        f"Market: \"{ex['title']}\" | Tags: {ex['tags']}\nOutput: {json.dumps(ex['output'])}"
        for ex in FEW_SHOT_EXAMPLES
    ])

    return f"""You are a prediction market classifier. Your job is to assign each market to exactly one primary investment theme from the taxonomy below.

TAXONOMY:
{taxonomy_text}

VALID THEME IDS: {json.dumps(theme_ids)}

CLASSIFICATION RULES:
1. Assign exactly 1 primary_theme from the valid theme IDs above.
2. Optionally assign 1 secondary_theme if the market genuinely spans two themes.
3. Rate confidence from 0.0 to 1.0.
4. If no theme fits well, use "uncategorized" as primary_theme.
5. Pay careful attention to anti-examples. "Trump tariffs on China" = china_us, NOT us_elections.
6. Focus on the CORE SUBJECT of the market, not who is involved. Trump doing something about China = china_us. Trump running for office = us_elections.
7. Stock prices of companies should be classified by their sector (e.g., AI company stock → ai_technology).

FEW-SHOT EXAMPLES:
{few_shot_text}

OUTPUT FORMAT: Return ONLY valid JSON with these exact keys:
{{"primary_theme": "theme_id", "secondary_theme": "theme_id_or_null", "confidence": 0.85, "reasoning": "one sentence"}}"""


def _build_user_prompt(row: pd.Series) -> str:
    title = str(row.get("event_title", ""))
    description = str(row.get("description", ""))[:500]
    tags = str(row.get("tags", ""))
    platform = str(row.get("platform", ""))
    category = str(row.get("category", ""))
    
    return f"""Classify this prediction market:

Platform: {platform}
Title: {title}
Description: {description}
Category: {category}
Tags: {tags}

Return JSON only."""


def _market_hash(row: pd.Series) -> str:
    """Create a stable hash for cache key."""
    key = f"{row.get('platform', '')}|{row.get('event_id', '')}|{row.get('event_title', '')}"
    return hashlib.md5(key.encode()).hexdigest()


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def classify_market(client: OpenAI, system_prompt: str, row: pd.Series, 
                    theme_ids: list[str]) -> dict:
    """Classify a single market. Returns dict with classification fields."""
    user_prompt = _build_user_prompt(row)
    
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            text = resp.choices[0].message.content
            result = json.loads(text)
            
            # Validate
            if result.get("primary_theme") not in theme_ids and result.get("primary_theme") != "uncategorized":
                result["primary_theme"] = "uncategorized"
                result["confidence"] = max(0, result.get("confidence", 0) - 0.2)
            
            if result.get("secondary_theme") and result["secondary_theme"] not in theme_ids:
                result["secondary_theme"] = None
            
            return {
                "primary_theme": result.get("primary_theme", "uncategorized"),
                "secondary_theme": result.get("secondary_theme"),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", "")
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return {
                "primary_theme": "uncategorized",
                "secondary_theme": None,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }


def classify_all(df: pd.DataFrame, batch_size: int = 50, 
                 rate_limit_delay: float = 0.1) -> pd.DataFrame:
    """Classify all markets in the dataframe. Uses cache to skip already-classified."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    themes = load_taxonomy()
    theme_ids = list_themes(themes)
    system_prompt = _build_system_prompt(themes)
    cache = _load_cache()
    
    results = []
    cached_count = 0
    api_count = 0
    
    for idx, row in df.iterrows():
        mhash = _market_hash(row)
        
        if mhash in cache:
            results.append(cache[mhash])
            cached_count += 1
            continue
        
        result = classify_market(client, system_prompt, row, theme_ids)
        result["market_hash"] = mhash
        cache[mhash] = result
        results.append(result)
        api_count += 1
        
        # Rate limiting
        time.sleep(rate_limit_delay)
        
        # Save cache periodically
        if api_count % batch_size == 0:
            _save_cache(cache)
            print(f"  Progress: {cached_count + api_count}/{len(df)} "
                  f"(cached: {cached_count}, API: {api_count})")
    
    _save_cache(cache)
    print(f"Done: {cached_count} cached, {api_count} API calls")
    
    result_df = pd.DataFrame(results)
    # Align index with input df
    result_df.index = df.index
    
    # Merge with original
    out = pd.concat([df, result_df], axis=1)
    return out


def save_classifications(df: pd.DataFrame, path: Optional[Path] = None):
    """Save classification results to CSV."""
    path = path or DATA_DIR / "llm_classifications.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} classifications to {path}")


def generate_report(df: pd.DataFrame) -> str:
    """Generate a summary report of classifications."""
    lines = ["# LLM Classification Report\n"]
    
    lines.append(f"Total markets: {len(df)}")
    lines.append(f"Classified: {(df['primary_theme'] != 'uncategorized').sum()}")
    lines.append(f"Uncategorized: {(df['primary_theme'] == 'uncategorized').sum()}")
    lines.append(f"Mean confidence: {df['confidence'].mean():.3f}")
    lines.append(f"Median confidence: {df['confidence'].median():.3f}")
    lines.append(f"Markets with secondary theme: {df['secondary_theme'].notna().sum()}")
    
    lines.append("\n## Theme Counts")
    counts = df["primary_theme"].value_counts()
    for theme, count in counts.items():
        avg_conf = df[df["primary_theme"] == theme]["confidence"].mean()
        lines.append(f"  {theme}: {count} markets (avg confidence: {avg_conf:.2f})")
    
    lines.append("\n## Low Confidence Markets (< 0.6)")
    low_conf = df[df["confidence"] < 0.6].sort_values("confidence")
    for _, row in low_conf.head(20).iterrows():
        lines.append(f"  [{row['confidence']:.2f}] {row.get('event_title', 'N/A')[:80]} → {row['primary_theme']}")
        if "reasoning" in row and row["reasoning"]:
            lines.append(f"         Reason: {row['reasoning'][:100]}")
    
    if len(low_conf) > 20:
        lines.append(f"  ... and {len(low_conf) - 20} more")
    
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else (
        Path.home() / ".openclaw/workspace/basket-trader/scripts/seed_data/markets_with_baskets_final_20260119.csv"
    )
    
    print(f"Loading markets from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} markets")
    
    print("Classifying...")
    classified = classify_all(df)
    
    save_classifications(classified)
    
    report = generate_report(classified)
    print("\n" + report)
    
    report_path = DATA_DIR / "classification_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

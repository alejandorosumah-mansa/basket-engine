"""
Semantic Exposure Layer for Categorical Markets

For categorical events (multiple outcomes per event), map each outcome
to economic factor exposures using LLM, then probability-weight to get
net event exposure.

Example:
    Event: "How many Fed rate cuts in 2025?"
    Outcomes: 0 cuts (hawkish), 1-2 cuts (neutral), 3+ cuts (dovish)
    Each outcome maps to factor exposures:
        0 cuts → {rates: +1, dollar: +1, gold: -1, equity: -0.5}
        3+ cuts → {rates: -1, dollar: -1, gold: +1, equity: +0.5}
    Probability-weight: net event exposure

Output: semantic_exposures.json
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import time

FACTOR_DIMENSIONS = [
    "rates",      # Higher rates (+) / lower rates (-)
    "dollar",     # Stronger dollar (+) / weaker dollar (-)  
    "equity",     # Risk-on / bullish equities (+) / bearish (-)
    "gold",       # Safe haven demand (+) / less demand (-)
    "oil",        # Higher oil (+) / lower oil (-)
    "crypto",     # Pro-crypto (+) / anti-crypto (-)
    "volatility", # Higher vol (+) / lower vol (-)
    "growth",     # Pro-growth (+) / anti-growth (-)
]

# Cache for LLM responses
CACHE_PATH = "data/processed/semantic_exposure_cache.json"


def load_cache() -> dict:
    if Path(CACHE_PATH).exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def get_event_exposure_prompt(event_slug: str, outcomes: List[dict]) -> str:
    """Build prompt for LLM to map categorical outcomes to factor exposures."""
    outcome_text = "\n".join([
        f"  - \"{o['title']}\" (current probability: {o['probability']:.1%})"
        for o in outcomes
    ])
    
    return f"""Analyze this prediction market event and map each outcome to economic factor exposures.

Event: {event_slug}
Outcomes:
{outcome_text}

For each outcome, rate its impact on these factors from -2 (strong negative) to +2 (strong positive), or 0 if no clear impact:
- rates: impact on interest rates (+ = higher rates, - = lower)
- dollar: impact on USD strength (+ = stronger, - = weaker)
- equity: impact on stock market (+ = bullish, - = bearish)
- gold: impact on gold/safe havens (+ = higher, - = lower)
- oil: impact on energy prices (+ = higher, - = lower)
- crypto: impact on crypto sentiment (+ = bullish, - = bearish)
- volatility: impact on market volatility (+ = higher vol, - = lower)
- growth: impact on economic growth expectations (+ = positive, - = negative)

If the event is NOT related to economics, policy, or financial markets (e.g., sports, entertainment, weather), respond with ONLY: "NOT_ECONOMIC"

Otherwise, respond in this exact JSON format:
{{
  "outcomes": [
    {{
      "title": "<outcome title>",
      "exposures": {{"rates": 0, "dollar": 0, "equity": 0, "gold": 0, "oil": 0, "crypto": 0, "volatility": 0, "growth": 0}}
    }}
  ],
  "rationale": "<brief explanation>"
}}"""


def call_llm(prompt: str, cache: dict) -> Optional[str]:
    """Call OpenAI API with caching."""
    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    
    if cache_key in cache:
        return cache[cache_key]
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
        )
        result = response.choices[0].message.content.strip()
        cache[cache_key] = result
        return result
    except Exception as e:
        print(f"  LLM error: {e}")
        return None


def parse_llm_response(response: str) -> Optional[Dict]:
    """Parse LLM response into structured exposure data."""
    if not response or "NOT_ECONOMIC" in response:
        return None
    
    try:
        # Extract JSON from response (may have markdown code blocks)
        text = response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        data = json.loads(text.strip())
        return data
    except (json.JSONDecodeError, IndexError, KeyError):
        return None


def compute_net_exposure(outcomes_data: Dict, market_probabilities: Dict[str, float]) -> Dict[str, float]:
    """Probability-weight outcome exposures to get net event exposure."""
    net = {dim: 0.0 for dim in FACTOR_DIMENSIONS}
    total_prob = 0.0
    
    for outcome in outcomes_data.get("outcomes", []):
        title = outcome.get("title", "")
        prob = market_probabilities.get(title, 0.0)
        exposures = outcome.get("exposures", {})
        
        for dim in FACTOR_DIMENSIONS:
            net[dim] += prob * exposures.get(dim, 0)
        total_prob += prob
    
    # Normalize if probabilities don't sum to 1
    if total_prob > 0 and abs(total_prob - 1.0) > 0.01:
        for dim in FACTOR_DIMENSIONS:
            net[dim] /= total_prob
    
    return net


def identify_categorical_events(markets: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Find categorical events with multiple outcomes and economic relevance."""
    # Events with multiple tickers
    event_counts = markets.groupby("event_slug").size()
    categorical_slugs = event_counts[event_counts > 1].index
    
    cat_markets = markets[markets.event_slug.isin(categorical_slugs)].copy()
    
    # Filter for active/recent markets with prices
    markets_with_prices = set(prices.market_id.unique())
    cat_markets = cat_markets[cat_markets.market_id.isin(markets_with_prices)]
    
    # Get latest price as proxy for probability
    latest_prices = prices.sort_values("date").groupby("market_id").last()["close_price"]
    cat_markets["probability"] = cat_markets["market_id"].map(latest_prices).fillna(0.5)
    
    return cat_markets


def run_semantic_exposure(
    markets_path: str = "data/processed/markets.parquet",
    prices_path: str = "data/processed/prices.parquet",
    output_path: str = "data/processed/semantic_exposures.json",
    max_events: int = 200,
) -> Dict:
    """Run semantic exposure analysis on categorical events.
    
    Returns:
        Dict of event_slug → {net_exposure, outcomes, rationale}
    """
    print("=== Semantic Exposure Layer ===")
    
    markets = pd.read_parquet(markets_path)
    prices = pd.read_parquet(prices_path)
    
    cat_markets = identify_categorical_events(markets, prices)
    print(f"Categorical events with prices: {cat_markets.event_slug.nunique()}")
    
    # Focus on economic/policy events by filtering tags
    economic_tags = [
        "politics", "economy", "fed", "trump", "policy", "regulation",
        "government", "tariff", "inflation", "recession", "bitcoin",
        "crypto", "china", "russia", "iran", "oil", "treasury",
        "trade", "sanctions", "monetary", "fiscal", "geopolitical",
    ]
    
    def is_economic(row):
        tags = str(row.get("tags", "")).lower()
        title = str(row.get("title", "")).lower()
        combined = tags + " " + title
        return any(t in combined for t in economic_tags)
    
    econ_markets = cat_markets[cat_markets.apply(is_economic, axis=1)]
    econ_events = econ_markets.event_slug.unique()
    print(f"Economic categorical events: {len(econ_events)}")
    
    # Limit to top events by volume
    event_volume = econ_markets.groupby("event_slug")["volume"].sum().nlargest(max_events)
    target_events = event_volume.index.tolist()
    print(f"Processing top {len(target_events)} events by volume")
    
    cache = load_cache()
    results = {}
    economic_count = 0
    
    for i, slug in enumerate(target_events):
        if (i + 1) % 20 == 0:
            print(f"  Processing event {i+1}/{len(target_events)}")
            save_cache(cache)  # Periodic save
        
        event_markets = econ_markets[econ_markets.event_slug == slug]
        outcomes = [
            {
                "title": row["title"],
                "probability": row["probability"],
                "market_id": row["market_id"],
            }
            for _, row in event_markets.iterrows()
        ]
        
        # Build and call LLM
        prompt = get_event_exposure_prompt(slug, outcomes)
        response = call_llm(prompt, cache)
        
        if response is None:
            continue
        
        parsed = parse_llm_response(response)
        if parsed is None:
            continue
        
        # Compute probability-weighted net exposure
        prob_map = {o["title"]: o["probability"] for o in outcomes}
        net = compute_net_exposure(parsed, prob_map)
        
        results[slug] = {
            "net_exposure": net,
            "n_outcomes": len(outcomes),
            "outcomes": parsed.get("outcomes", []),
            "rationale": parsed.get("rationale", ""),
            "market_ids": [o["market_id"] for o in outcomes],
        }
        economic_count += 1
    
    save_cache(cache)
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Events with semantic exposure: {economic_count}")
    
    if results:
        # Aggregate exposure stats
        exposures = pd.DataFrame([r["net_exposure"] for r in results.values()])
        print(f"\nNet exposure statistics:")
        print(exposures.describe().round(3))
    
    print(f"\nSaved to {output_path}")
    return results


if __name__ == "__main__":
    run_semantic_exposure()

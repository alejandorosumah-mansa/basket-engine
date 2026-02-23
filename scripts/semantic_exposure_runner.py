#!/usr/bin/env python3
"""Run semantic exposure analysis in batches to avoid OOM."""
import os, json, hashlib, sys
import pandas as pd
from pathlib import Path

# Step 1: Prepare event list and save to disk
print("Preparing events...", flush=True)
markets = pd.read_parquet('data/processed/markets.parquet')
prices = pd.read_parquet('data/processed/prices.parquet')

event_counts = markets.groupby('event_slug').size()
cat_slugs = event_counts[event_counts > 1].index
cat = markets[markets.event_slug.isin(cat_slugs)]
cat = cat[cat.market_id.isin(prices.market_id.unique())]
latest_prices = prices.sort_values('date').groupby('market_id').last()['close_price']
cat = cat.copy()
cat['probability'] = cat['market_id'].map(latest_prices).fillna(0.5)

econ_tags = ['politics','economy','fed','trump','policy','regulation','government',
             'tariff','inflation','recession','bitcoin','crypto','china','russia',
             'iran','oil','treasury','trade','sanctions','monetary','fiscal','geopolitical']

def is_econ(row):
    combined = (str(row.get('tags','')).lower() + ' ' + str(row.get('title','')).lower())
    return any(t in combined for t in econ_tags)

econ = cat[cat.apply(is_econ, axis=1)]
event_vol = econ.groupby('event_slug')['volume'].sum().nlargest(50)

# Save event data to temp file
events_data = {}
for slug in event_vol.index:
    event_markets = econ[econ.event_slug == slug]
    outcomes = [{'title': r['title'], 'probability': float(r['probability']), 
                 'market_id': r['market_id']} 
                for _, r in event_markets.head(20).iterrows()]
    events_data[slug] = outcomes

with open('data/processed/_temp_events.json', 'w') as f:
    json.dump(events_data, f)

print(f"Prepared {len(events_data)} events", flush=True)

# Free all memory
del markets, prices, cat, econ, event_vol, latest_prices, event_counts
import gc; gc.collect()

# Step 2: Process events with LLM
import openai
client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

cache_path = 'data/processed/semantic_exposure_cache.json'
cache = json.load(open(cache_path)) if Path(cache_path).exists() else {}

FACTOR_DIMS = ['rates','dollar','equity','gold','oil','crypto','volatility','growth']
results = {}

for i, (slug, outcomes) in enumerate(events_data.items()):
    outcome_text = "\n".join([f'  - "{o["title"]}" (prob: {o["probability"]:.1%})' for o in outcomes])
    prompt = f"""Analyze this prediction market event and map each outcome to economic factor exposures.

Event: {slug}
Outcomes:
{outcome_text}

For each outcome, rate impact on these factors from -2 to +2 (0 if no impact): rates, dollar, equity, gold, oil, crypto, volatility, growth

If NOT economic/policy related, respond: "NOT_ECONOMIC"

Otherwise respond in JSON:
{{"outcomes": [{{"title": "...", "exposures": {{"rates": 0, "dollar": 0, "equity": 0, "gold": 0, "oil": 0, "crypto": 0, "volatility": 0, "growth": 0}}}}], "rationale": "..."}}"""

    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    
    if cache_key not in cache:
        try:
            resp = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role':'user','content':prompt}],
                temperature=0.1, max_tokens=1000
            )
            cache[cache_key] = resp.choices[0].message.content.strip()
            print(f'  [{i+1}/{len(events_data)}] API: {slug}', flush=True)
        except Exception as e:
            print(f'  [{i+1}] Error: {e}', flush=True)
            continue
    else:
        print(f'  [{i+1}/{len(events_data)}] Cached: {slug}', flush=True)
    
    # Save cache every 10 items
    if (i+1) % 10 == 0:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    
    response = cache[cache_key]
    if 'NOT_ECONOMIC' in response:
        continue
    
    try:
        text = response
        if '```json' in text: text = text.split('```json')[1].split('```')[0]
        elif '```' in text: text = text.split('```')[1].split('```')[0]
        parsed = json.loads(text.strip())
    except:
        continue
    
    net = {d: 0.0 for d in FACTOR_DIMS}
    total_prob = 0.0
    prob_map = {o['title']: o['probability'] for o in outcomes}
    for out in parsed.get('outcomes', []):
        prob = prob_map.get(out.get('title',''), 0.0)
        for d in FACTOR_DIMS:
            net[d] += prob * out.get('exposures',{}).get(d, 0)
        total_prob += prob
    if total_prob > 0 and abs(total_prob - 1.0) > 0.01:
        for d in FACTOR_DIMS: net[d] /= total_prob
    
    results[slug] = {
        'net_exposure': net,
        'n_outcomes': len(outcomes),
        'market_ids': [o['market_id'] for o in outcomes],
        'rationale': parsed.get('rationale','')
    }

# Final save
with open(cache_path, 'w') as f:
    json.dump(cache, f, indent=2)
with open('data/processed/semantic_exposures.json', 'w') as f:
    json.dump(results, f, indent=2)

# Cleanup temp
Path('data/processed/_temp_events.json').unlink(missing_ok=True)

print(f'\nDone: {len(results)} events with semantic exposures', flush=True)

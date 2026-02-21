#!/usr/bin/env python3
"""Analyze structural grouping potential using event_slug and patterns."""

import pandas as pd
import re
from collections import defaultdict

print('=== STEP 1: STRUCTURAL GROUPING ANALYSIS ===')

markets = pd.read_parquet('data/processed/markets.parquet')
print(f'Total markets: {len(markets):,}')
print(f'Unique event_slugs: {markets["event_slug"].nunique():,}')

# Analyze event_slug grouping structure
slug_analysis = markets.groupby('event_slug').agg({
    'market_id': 'count',
    'title': lambda x: list(x)[:3],  # Sample titles
    'volume': 'mean'
}).rename(columns={'market_id': 'market_count'})

slug_analysis = slug_analysis.sort_values('market_count', ascending=False)

print('\n=== TOP EVENT SLUGS BY MARKET COUNT ===')
for i, (slug, row) in enumerate(slug_analysis.head(10).iterrows()):
    print(f'{i+1}. {slug}: {row["market_count"]} markets')
    print(f'   Sample titles:')
    for title in row["title"][:2]:
        print(f'     - {title}')
    print(f'   Avg volume: ${row["volume"]:,.0f}')
    print()

# Analyze patterns for events with NO event_slug 
no_slug = markets[markets['event_slug'].isnull()]
print(f'Markets WITHOUT event_slug: {len(no_slug):,} ({len(no_slug)/len(markets)*100:.1f}%)')

# Look for patterns that can be grouped
print('\n=== DETECTING GROUPING PATTERNS IN NO-SLUG MARKETS ===')

# Iran strike patterns
iran_markets = no_slug[no_slug['title'].str.contains('Iran', case=False, na=False)]
print(f'Iran-related markets: {len(iran_markets)}')
for title in iran_markets['title'].head(5):
    print(f'  - {title}')

# Bitcoin patterns  
btc_markets = no_slug[no_slug['title'].str.contains('Bitcoin|BTC', case=False, na=False)]
print(f'\nBitcoin-related markets: {len(btc_markets)}')
for title in btc_markets['title'].head(5):
    print(f'  - {title}')

# Fed/interest rate patterns
fed_markets = no_slug[no_slug['title'].str.contains('Fed|interest rate|Federal Reserve', case=False, na=False)]
print(f'\nFed/interest rate markets: {len(fed_markets)}')
for title in fed_markets['title'].head(5):
    print(f'  - {title}')

# Democratic nomination patterns
dem_markets = no_slug[no_slug['title'].str.contains('Democratic.*nomination|Democratic.*president', case=False, na=False)]
print(f'\nDemocratic nomination markets: {len(dem_markets)}')
for title in dem_markets['title'].head(5):
    print(f'  - {title}')

print(f'\n=== COMPRESSION POTENTIAL ===')
total_events_estimated = markets["event_slug"].nunique()
print(f'Raw markets: {len(markets):,}')
print(f'Event_slug groups: {total_events_estimated:,}')
print(f'Current compression from event_slug alone: {len(markets) / total_events_estimated:.1f}x')

# Estimate additional compression from pattern-based grouping
pattern_groups = 0
pattern_groups += len(iran_markets) // 10 if len(iran_markets) > 0 else 0  # Estimate ~10 Iran markets per event
pattern_groups += len(btc_markets) // 15 if len(btc_markets) > 0 else 0   # Estimate ~15 BTC markets per event  
pattern_groups += len(fed_markets) // 8 if len(fed_markets) > 0 else 0    # Estimate ~8 Fed markets per event
pattern_groups += len(dem_markets) // 20 if len(dem_markets) > 0 else 0   # Estimate ~20 candidate markets per event

estimated_total_events = total_events_estimated + pattern_groups
print(f'Estimated total unique events after pattern grouping: {estimated_total_events:,}')
print(f'Estimated final compression: {len(markets) / estimated_total_events:.1f}x')

# Show specific examples of what needs grouping
print(f'\n=== EXAMPLES OF MARKETS THAT SHOULD BE GROUPED ===')

# Find Iran strikes by date - these should be one event with time variants
iran_by_date = []
for title in iran_markets['title'].head(20):
    if 'by' in title.lower():
        iran_by_date.append(title)

if iran_by_date:
    print(f'\nIran Strike Time Variants (should be ONE event):')
    for title in iran_by_date[:8]:
        print(f'  - {title}')

# Find Democratic nomination candidates - these should be one event with categorical outcomes  
dem_candidates = []
for title in dem_markets['title'].head(20):
    dem_candidates.append(title)

if dem_candidates:
    print(f'\nDemocratic Nomination Categorical Outcomes (should be ONE event):')
    for title in dem_candidates[:5]:
        print(f'  - {title}')

print(f'\n=== RECOMMENDED APPROACH ===')
print('1. Use event_slug where available (groups 99%+ of markets)')
print('2. For null event_slug, apply regex pattern grouping')
print('3. Create ticker-level classification on grouped events')
print('4. Deduplicate: pick one representative market per ticker')
print('5. Theme classify the deduped ticker universe, not individual markets')
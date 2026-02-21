#!/usr/bin/env python3
"""Implement the correct taxonomy hierarchy: CUSIP → Ticker → Event → Theme"""

import pandas as pd
import numpy as np
import hashlib
import re
from collections import defaultdict

def generate_cusip(market_id: str, title: str, platform: str) -> str:
    """Generate unique CUSIP for individual market instance."""
    content = f"{platform}:{market_id}:{title}"
    hash_obj = hashlib.sha256(content.encode('utf-8'))
    cusip = hash_obj.hexdigest()[:12].upper()
    return cusip

def extract_ticker_from_title(title: str, event_slug: str) -> str:
    """Extract ticker (outcome) from individual market title."""
    title_lower = title.lower()
    
    # For player/candidate-specific markets, extract the person's name
    if 'win' in title_lower:
        # Pattern: "Will [PERSON] win the [AWARD/ELECTION]?"
        match = re.search(r'will\s+([^?]+?)\s+win\s+(?:the\s+)?(.+?)\?', title_lower)
        if match:
            person = match.group(1).strip()
            context = match.group(2).strip()
            # Clean up the person name (remove common prefixes)
            person = re.sub(r'^(?:president |mr\.?\s+|ms\.?\s+|dr\.?\s+)', '', person)
            return person.upper().replace(' ', '_')
    
    # For team/country specific markets
    team_patterns = [
        r'(\w+)\s+team\s+total',  # "Ravens Team Total: O/U 24.5"
        r'(\w+)\s+vs\.?\s+\w+',   # "Ravens vs Browns"  
        r'will\s+(\w+)\s+record', # "Will China record a medal"
    ]
    
    for pattern in team_patterns:
        match = re.search(pattern, title_lower)
        if match:
            return match.group(1).upper()
    
    # For over/under markets, extract the line/number
    ou_match = re.search(r'o/u\s+(\d+(?:\.\d+)?)', title_lower)
    if ou_match:
        return f"OU_{ou_match.group(1).replace('.', '_')}"
    
    # For spread markets
    spread_match = re.search(r'spread.*?\(([+-]?\d+(?:\.\d+)?)\)', title_lower)
    if spread_match:
        return f"SPREAD_{spread_match.group(1).replace('.', '_').replace('-', 'NEG').replace('+', 'POS')}"
    
    # Default: use a hash of the title to create unique tickers
    title_hash = hashlib.md5(title.encode()).hexdigest()[:8]
    return f"TICKER_{title_hash.upper()}"

def identify_event_type(event_markets: pd.DataFrame) -> str:
    """Determine if an event is binary or categorical based on its markets."""
    titles = event_markets['title'].tolist()
    
    # If all titles are very similar, likely binary with time variants
    if len(titles) <= 3:
        return 'binary'
    
    # Check for player names, candidates, teams - indicates categorical
    name_patterns = [
        r'will\s+\w+\s+\w+\s+win',  # "Will John Doe win"
        r'will\s+\w+\s+record',     # "Will China record" 
        r'^\w+\s+team\s+total',     # "Ravens Team Total"
        r'^\w+\s+vs\.?\s+\w+'       # "Ravens vs Browns"
    ]
    
    name_count = 0
    for title in titles:
        for pattern in name_patterns:
            if re.search(pattern, title.lower()):
                name_count += 1
                break
    
    # If most titles match name patterns, it's categorical
    if name_count / len(titles) > 0.7:
        return 'categorical'
    
    # Otherwise assume binary
    return 'binary'

def classify_event_theme(event_slug: str, sample_titles: list) -> str:
    """Classify EVENT into theme based on event_slug and sample titles."""
    text = f'{event_slug} {" ".join(sample_titles[:3])}'.lower()
    
    # Classification rules at EVENT level
    if any(word in text for word in ['election', 'president', 'nomination', 'democrat', 'republican', 'trump', 'biden']):
        return 'us_elections'
    elif any(word in text for word in ['nfl', 'nba', 'nhl', 'mlb', 'olympics', 'sports', 'game', 'team', 'trophy', 'championship']):
        return 'sports_entertainment'  
    elif any(word in text for word in ['fed', 'federal reserve', 'interest rate', 'monetary']):
        return 'fed_monetary_policy'
    elif any(word in text for word in ['iran', 'israel', 'gaza', 'hamas', 'middle east', 'lebanon']):
        return 'middle_east'
    elif any(word in text for word in ['ukraine', 'russia', 'putin', 'ceasefire']):
        return 'russia_ukraine'
    elif any(word in text for word in ['china', 'chinese', 'taiwan']):
        return 'china_us'
    elif any(word in text for word in ['crypto', 'bitcoin', 'ethereum', 'blockchain']):
        return 'crypto_digital'
    elif any(word in text for word in ['ai', 'artificial intelligence', 'technology', 'microsoft', 'google']):
        return 'ai_technology'
    elif any(word in text for word in ['gdp', 'unemployment', 'inflation', 'economy']):
        return 'us_economic'
    elif any(word in text for word in ['climate', 'environment', 'carbon']):
        return 'climate_environment'
    else:
        return 'uncategorized'

def main():
    print('=== IMPLEMENTING CORRECT TAXONOMY HIERARCHY ===')
    print('CUSIP → Ticker → Event → Theme')
    
    # Load raw markets
    markets = pd.read_parquet('data/processed/markets.parquet')
    print(f'Total markets (CUSIPs): {len(markets):,}')
    print(f'Unique events: {markets["event_slug"].nunique():,}')
    
    # Build the taxonomy hierarchy
    taxonomy_data = []
    event_data = []
    
    for event_slug, event_markets in markets.groupby('event_slug'):
        print(f'Processing event: {event_slug} ({len(event_markets)} markets)')
        
        # Identify event type (binary vs categorical)
        event_type = identify_event_type(event_markets)
        
        # Extract tickers for this event
        tickers_in_event = {}
        
        for _, market in event_markets.iterrows():
            # CUSIP level (individual market)
            cusip = generate_cusip(market['market_id'], market['title'], market['platform'])
            
            # Ticker level (outcome)  
            ticker = extract_ticker_from_title(market['title'], event_slug)
            
            # Store the ticker->CUSIP mapping
            if ticker not in tickers_in_event:
                tickers_in_event[ticker] = []
            tickers_in_event[ticker].append({
                'cusip': cusip,
                'market_id': market['market_id'],
                'title': market['title'],
                'volume': market['volume'],
                'platform': market['platform']
            })
            
            # Add to taxonomy data
            taxonomy_data.append({
                'cusip': cusip,
                'market_id': market['market_id'],  
                'ticker': ticker,
                'event_slug': event_slug,
                'event_type': event_type,
                'title': market['title'],
                'volume': market['volume'],
                'platform': market['platform']
            })
        
        # Event level data
        sample_titles = event_markets['title'].tolist()[:5]
        theme = classify_event_theme(event_slug, sample_titles)
        total_volume = event_markets['volume'].sum()
        
        event_data.append({
            'event_slug': event_slug,
            'event_type': event_type, 
            'theme': theme,
            'num_tickers': len(tickers_in_event),
            'num_cusips': len(event_markets),
            'total_volume': total_volume,
            'sample_titles': sample_titles
        })
    
    # Create DataFrames
    taxonomy_df = pd.DataFrame(taxonomy_data)
    events_df = pd.DataFrame(event_data)
    
    print(f'\\n=== HIERARCHY SUMMARY ===')
    print(f'CUSIPs (individual markets): {len(taxonomy_df):,}')
    print(f'Tickers (unique outcomes): {taxonomy_df["ticker"].nunique():,}')  
    print(f'Events (parent questions): {len(events_df):,}')
    print(f'Themes: {events_df["theme"].nunique():,}')
    
    # Theme classification results
    print(f'\\n=== THEME CLASSIFICATION (EVENT LEVEL) ===')
    theme_counts = events_df['theme'].value_counts()
    total_events = len(events_df)
    
    for theme, count in theme_counts.items():
        pct = (count / total_events) * 100
        avg_volume = events_df[events_df['theme'] == theme]['total_volume'].mean()
        print(f'  {theme:20s}: {count:3d} events ({pct:5.1f}%) (avg vol: ${avg_volume:,.0f})')
    
    # Event type analysis  
    print(f'\\n=== EVENT TYPE ANALYSIS ===')
    event_type_counts = events_df['event_type'].value_counts()
    for event_type, count in event_type_counts.items():
        pct = (count / total_events) * 100
        avg_tickers = events_df[events_df['event_type'] == event_type]['num_tickers'].mean()
        print(f'  {event_type:15s}: {count:3d} events ({pct:5.1f}%) (avg {avg_tickers:.1f} tickers/event)')
    
    # Show examples of categorical events (multiple tickers)
    print(f'\\n=== EXAMPLES OF CATEGORICAL EVENTS ===')
    categorical_events = events_df[events_df['event_type'] == 'categorical'].nlargest(5, 'num_tickers')
    for _, event in categorical_events.iterrows():
        event_markets = taxonomy_df[taxonomy_df['event_slug'] == event['event_slug']]
        unique_tickers = event_markets['ticker'].unique()
        print(f'\\nEvent: {event["event_slug"]}')
        print(f'  Theme: {event["theme"]}')
        print(f'  Tickers: {len(unique_tickers)} ({", ".join(unique_tickers[:5])}{"..." if len(unique_tickers) > 5 else ""})')
        print(f'  Total CUSIPs: {event["num_cusips"]}')
    
    # Propagate theme from event level to all CUSIPs
    print(f'\\n=== PROPAGATING THEMES TO ALL LEVELS ===')
    event_theme_map = events_df.set_index('event_slug')['theme'].to_dict()
    taxonomy_df['theme'] = taxonomy_df['event_slug'].map(event_theme_map)
    
    # For basket construction: create EVENT-level representatives
    print(f'\\n=== CREATING EVENT-LEVEL REPRESENTATIVES ===')
    
    # For each event, pick the most liquid CUSIP as the representative
    event_representatives = []
    
    for event_slug, event_group in taxonomy_df.groupby('event_slug'):
        # Pick the CUSIP with highest volume as event representative
        representative = event_group.loc[event_group['volume'].idxmax()]
        event_info = events_df[events_df['event_slug'] == event_slug].iloc[0]
        
        event_representatives.append({
            'event_slug': event_slug,
            'theme': representative['theme'],
            'event_type': event_info['event_type'],
            'representative_cusip': representative['cusip'],
            'representative_market_id': representative['market_id'],
            'representative_title': representative['title'],
            'num_tickers': event_info['num_tickers'],
            'num_cusips': event_info['num_cusips'], 
            'total_volume': event_info['total_volume'],
            'representative_volume': representative['volume']
        })
    
    representatives_df = pd.DataFrame(event_representatives)
    
    print(f'Event representatives: {len(representatives_df):,}')
    print(f'Ready for basket construction: one position per EVENT')
    
    # Filter to events with price data for basket construction
    try:
        prices = pd.read_parquet('data/processed/prices.parquet')
        markets_with_prices = set(prices['market_id'])
        
        tradeable_events = representatives_df[
            representatives_df['representative_market_id'].isin(markets_with_prices)
        ]
        
        print(f'\\n=== TRADEABLE EVENT UNIVERSE ===')
        print(f'Events with price data: {len(tradeable_events):,}')
        
        # Apply volume filter at event level
        min_volume = 10000
        eligible_events = tradeable_events[tradeable_events['total_volume'] >= min_volume]
        print(f'Events with total volume >= ${min_volume:,}: {len(eligible_events):,}')
        
        # Theme breakdown for eligible events
        eligible_themes = eligible_events['theme'].value_counts()
        print(f'\\nEligible events by theme:')
        for theme, count in eligible_themes.items():
            pct = (count / len(eligible_events)) * 100
            print(f'  {theme:20s}: {count:3d} events ({pct:5.1f}%)')
        
        # Save eligible events for basket construction  
        eligible_events.to_parquet('data/processed/eligible_events_correct_taxonomy.parquet')
        print(f'\\nSaved eligible events: data/processed/eligible_events_correct_taxonomy.parquet')
        
    except Exception as e:
        print(f'Could not process price data: {e}')
    
    # Save complete taxonomy
    print(f'\\n=== SAVING COMPLETE TAXONOMY ===')
    
    taxonomy_df.to_parquet('data/processed/complete_taxonomy.parquet')
    events_df.to_parquet('data/processed/events_taxonomy.parquet') 
    representatives_df.to_parquet('data/processed/event_representatives.parquet')
    
    print('Saved complete taxonomy to: data/processed/complete_taxonomy.parquet')
    print('Saved events data to: data/processed/events_taxonomy.parquet')
    print('Saved event representatives to: data/processed/event_representatives.parquet')
    
    print(f'\\n=== TAXONOMY IMPLEMENTATION COMPLETE ===')
    print('✅ Correct hierarchy: CUSIP → Ticker → Event → Theme')
    print('✅ Theme classification at EVENT level only')
    print('✅ Binary vs categorical event identification') 
    print('✅ One representative per event for basket construction')
    print('✅ Ready for proper basket construction!')

if __name__ == '__main__':
    main()
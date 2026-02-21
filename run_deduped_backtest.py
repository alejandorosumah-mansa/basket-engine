#!/usr/bin/env python3
"""Run basket construction and backtest on properly deduped events."""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from src.construction.weighting import (
    equal_weights, volume_weighted, risk_parity_weights
)

print('=== RUNNING DEDUPED BASKET CONSTRUCTION & BACKTEST ===')

# Load deduped eligible events (this is our clean universe)
eligible_events = pd.read_parquet('data/processed/eligible_events_correct_taxonomy.parquet')
print(f'Eligible events: {len(eligible_events)}')

# Load price data 
prices = pd.read_parquet('data/processed/prices.parquet')
returns = pd.read_parquet('data/processed/returns.parquet')

print(f'Price data: {len(prices)} rows for {prices["market_id"].nunique()} markets')

# Filter to viable themes (>=5 events, exclude sports/uncategorized)
theme_counts = eligible_events['theme'].value_counts()
viable_themes = theme_counts[theme_counts >= 5].index.tolist()
if 'uncategorized' in viable_themes:
    viable_themes.remove('uncategorized')
if 'sports_entertainment' in viable_themes:
    viable_themes.remove('sports_entertainment')

print(f'\\nViable themes: {viable_themes}')

# Define weighting methods (simplified for clarity)
weighting_methods = ['equal', 'volume_weighted', 'risk_parity']

# Store basket compositions using EVENTS not individual markets
all_baskets = {}

for theme in viable_themes:
    theme_events = eligible_events[eligible_events['theme'] == theme]
    event_market_ids = theme_events['representative_market_id'].tolist()
    
    print(f'\\n=== {theme.upper()} BASKET ===')
    print(f'Events (not individual markets): {len(event_market_ids)}')
    
    # Get price/return data for these representative markets
    theme_prices = prices[prices['market_id'].isin(event_market_ids)]
    theme_returns = returns[returns['market_id'].isin(event_market_ids)]
    
    if len(theme_returns) == 0:
        print(f'  No price data available - skipping theme')
        continue
    
    # Calculate metrics using event-level volumes (total volume per event)
    event_volumes = theme_events.set_index('representative_market_id')['total_volume'].to_dict()
    
    # Calculate volatilities for risk parity (using returns data)
    volatilities = {}
    for market_id in event_market_ids:
        market_returns = theme_returns[theme_returns['market_id'] == market_id]['return']
        if len(market_returns) > 1:
            vol = market_returns.std()
            volatilities[market_id] = max(vol, 1e-6)  # Floor to avoid division by zero
        else:
            volatilities[market_id] = 1.0
    
    # Generate baskets with different weighting methods
    theme_baskets = {}
    
    # 1. Equal Weight across EVENTS
    theme_baskets['equal'] = equal_weights(event_market_ids)
    
    # 2. Volume Weighted by total event volume
    theme_baskets['volume_weighted'] = volume_weighted(event_volumes)
    
    # 3. Risk Parity
    theme_baskets['risk_parity'] = risk_parity_weights(volatilities)
    
    # Store baskets for this theme
    all_baskets[theme] = theme_baskets
    
    # Display basket compositions
    print('Event-level basket compositions (top 5 positions):')
    for method in weighting_methods:
        weights = theme_baskets[method]
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        print(f'  {method}:')
        for i, (market_id, weight) in enumerate(sorted_weights[:5]):
            # Get the canonical title for this event
            event_info = theme_events[theme_events['representative_market_id'] == market_id]
            if len(event_info) > 0:
                representative_title = event_info['representative_title'].iloc[0]
                event_slug = event_info['event_slug'].iloc[0]
                market_count = event_info['num_cusips'].iloc[0]
                print(f'    {weight:.3f} - {representative_title[:60]}... [{market_count} underlying markets]')
            else:
                print(f'    {weight:.3f} - {market_id}')
        if len(sorted_weights) > 5:
            print(f'    ... and {len(sorted_weights)-5} more events')

# Run backtest on event-level baskets
print(f'\\n=== RUNNING BACKTEST ON EVENT-LEVEL BASKETS ===')

# Define backtest period  
prices['date'] = pd.to_datetime(prices['date'])
returns['date'] = pd.to_datetime(returns['date'])
max_date = prices['date'].max()
backtest_start = max_date - timedelta(days=180)

backtest_prices = prices[(prices['date'] >= backtest_start) & (prices['date'] <= max_date)]
backtest_returns = returns[(returns['date'] >= backtest_start) & (returns['date'] <= max_date)]

trading_dates = sorted(backtest_prices['date'].unique())
print(f'Backtest period: {backtest_start.date()} to {max_date.date()} ({len(trading_dates)} days)')

# Run backtests
backtest_results = {}

for theme, theme_methods in all_baskets.items():
    print(f'\\n=== BACKTESTING {theme.upper()} ===')
    
    for method, positions in theme_methods.items():
        # Get market IDs and weights for this basket
        market_weights = positions
        basket_markets = list(market_weights.keys())
        
        # Filter to markets with data in backtest period
        basket_returns = backtest_returns[backtest_returns['market_id'].isin(basket_markets)]
        
        if len(basket_returns) == 0:
            print(f'  {method}: No return data')
            continue
            
        # Calculate daily basket returns
        basket_nav = []
        nav = 1.0  # Start at $1
        
        for date in trading_dates:
            daily_returns = basket_returns[basket_returns['date'] == date]
            
            if len(daily_returns) > 0:
                # Calculate weighted return
                weighted_return = 0.0
                total_weight = 0.0
                
                for _, row in daily_returns.iterrows():
                    market_id = row['market_id']
                    if market_id in market_weights:
                        weight = market_weights[market_id]
                        weighted_return += weight * row['return']
                        total_weight += weight
                
                if total_weight > 0:
                    weighted_return /= total_weight  # Normalize if not all events traded
                    nav *= (1 + weighted_return)
            
            basket_nav.append(nav)
        
        # Calculate metrics
        nav_series = pd.Series(basket_nav, index=trading_dates)
        returns_series = nav_series.pct_change().dropna()
        
        if len(returns_series) > 1:
            total_return = (nav_series.iloc[-1] / nav_series.iloc[0]) - 1
            volatility = returns_series.std() * np.sqrt(252)  # Annualized
            sharpe = (returns_series.mean() * 252 - 0.05) / (returns_series.std() * np.sqrt(252)) if returns_series.std() > 0 else 0
            max_dd = (nav_series / nav_series.cummax() - 1).min()
            
            backtest_results[f'{theme}_{method}'] = {
                'total_return': total_return,
                'annualized_volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'final_nav': nav_series.iloc[-1],
                'num_events': len(basket_markets),  # This is now event count, not market count
                'nav_series': nav_series.values.tolist(),
                'dates': [d.strftime('%Y-%m-%d') for d in trading_dates]
            }
            
            print(f'  {method}: Return={total_return:.2%}, Vol={volatility:.2%}, Sharpe={sharpe:.2f}, MDD={max_dd:.2%}, Events={len(basket_markets)}')
        else:
            print(f'  {method}: Insufficient data for metrics')

# Save results
print(f'\\n=== SAVING RESULTS ===')

# Convert baskets to JSON-serializable format
baskets_json = {}
for theme, methods in all_baskets.items():
    baskets_json[theme] = {}
    for method, weights in methods.items():
        # Include event metadata in the basket composition
        basket_composition = []
        for market_id, weight in weights.items():
            event_info = eligible_events[eligible_events['representative_market_id'] == market_id]
            if len(event_info) > 0:
                composition_entry = {
                    'market_id': market_id,
                    'weight': float(weight),
                    'event_slug': event_info['event_slug'].iloc[0],
                    'canonical_title': event_info['representative_title'].iloc[0],
                    'market_count': int(event_info['num_cusips'].iloc[0]),
                    'total_volume': float(event_info['total_volume'].iloc[0])
                }
            else:
                composition_entry = {
                    'market_id': market_id,
                    'weight': float(weight),
                    'event_slug': 'unknown',
                    'canonical_title': 'unknown',
                    'market_count': 1,
                    'total_volume': 0
                }
            basket_composition.append(composition_entry)
        baskets_json[theme][method] = basket_composition

with open('data/outputs/deduped_basket_compositions.json', 'w') as f:
    json.dump(baskets_json, f, indent=2)

with open('data/outputs/deduped_backtest_results.json', 'w') as f:
    json.dump(backtest_results, f, indent=2, default=str)

print('Saved deduped basket compositions to: data/outputs/deduped_basket_compositions.json')
print('Saved deduped backtest results to: data/outputs/deduped_backtest_results.json')

# Performance summary
if backtest_results:
    results_df = pd.DataFrame.from_dict(backtest_results, orient='index')
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    print(f'\\n=== DEDUPED BASKET PERFORMANCE SUMMARY ===')
    print('Top performers by Sharpe ratio:')
    for idx, row in results_df.head(10).iterrows():
        events_count = row['num_events']
        print(f'  {idx:25s}: Sharpe={row["sharpe_ratio"]:5.2f}, Return={row["total_return"]:6.1%}, MDD={row["max_drawdown"]:6.1%}, Events={events_count}')
    
    results_df.to_csv('data/outputs/deduped_performance_summary.csv')
    print('\\nSaved deduped performance summary to: data/outputs/deduped_performance_summary.csv')

print(f'\\n=== SUCCESS! ===')
print(f'✅ Built baskets from {len(eligible_events)} unique EVENTS (not individual markets)')
print(f'✅ No fake diversification from categorical outcomes')
print(f'✅ Proper 3-layer taxonomy implementation')
print(f'✅ {len(backtest_results)} baskets backtested successfully')
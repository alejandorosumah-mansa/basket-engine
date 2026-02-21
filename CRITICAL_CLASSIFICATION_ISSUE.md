# CRITICAL: Wrong Classification Order Causing High Uncategorized Rate

## Problem
The theme classification is happening at the wrong level, causing 49% uncategorized rate. We're classifying individual market fragments instead of coherent underlying events.

## Root Cause
Current pipeline does classification FIRST, then deduplication. Should be:
1. **Structural grouping** (ticker/CUSIP) - group related markets
2. **Event-level deduplication** - pick one representative per event  
3. **Theme classification** - classify the deduped events

## Current Bad Approach
❌ Classifying each individual market:
- "Will Pete Buttigieg win the 2028 Democratic nomination?" → confusing fragment
- "Will Kamala Harris win the 2028 Democratic nomination?" → confusing fragment  
- "Will Gavin Newsom win the 2028 Democratic nomination?" → confusing fragment
- "Will Bitcoin be above $68,000 on February 21?" → confusing fragment
- "Will Bitcoin be above $70,000 on February 21?" → confusing fragment

Result: LLM sees 500 fragmented betting markets, can't understand context, marks as uncategorized.

## Correct Approach
✅ First group into events, then classify the parent event:
1. **Group**: All "2028 Democratic nomination" markets → one ticker "DEM_2028_PRIMARY"
2. **Dedupe**: Pick most liquid market as representative
3. **Classify**: "2028 Democratic Primary" → us_elections (clear and obvious)

1. **Group**: All "Bitcoin price on Feb 21" markets → one ticker "BTC_PRICE_FEB21"  
2. **Dedupe**: Pick most liquid price level as representative
3. **Classify**: "Bitcoin Price February 2026" → crypto_digital (clear and obvious)

## Data Evidence
- We have 4,128 unique event_slugs for 20,366 markets (5x compression)
- event_slug "nhl-2025-26-hart-memorial-trophy" contains 118 individual player markets
- event_slug "winter-games-2026-countries-to-win-a-gold-medal" contains 80 country markets
- These should become ONE classified event each, not 118 or 80 separate classifications

## Impact
- High uncategorized rate (49%) due to context confusion
- Wasted LLM API calls on redundant classifications  
- Fake diversification in baskets (multiple outcomes of same event)
- Poor thematic coherence

## Fix Required
1. Implement structural grouping using event_slug + regex patterns
2. Create deduped event universe (one market per underlying event)
3. Run theme classification ONLY on deduped events with proper context
4. Propagate theme classification back to all markets in each group

## Priority: CRITICAL
This fundamentally breaks the entire basket construction logic and explains the poor results.
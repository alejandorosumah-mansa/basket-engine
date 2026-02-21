# CORRECTED: Taxonomy Hierarchy Implementation

## The Correct Bottom-Up Hierarchy

### CUSIP (Most Granular - Individual Market Instance)
- **Definition**: A specific market instance with specific time/date conditions
- **Examples**: 
  - "US strikes Iran by February 28th" 
  - "Bitcoin Up or Down Feb 22 5:40AM-5:45AM"
  - "Harris wins nomination by March 2028"
- **Implementation**: Each individual market gets a unique CUSIP

### Ticker (Outcome Level)
- **Definition**: The underlying outcome, stripped of time/date specifics
- **Binary Example**: "US strikes Iran" (one ticker per event)
- **Categorical Example**: 
  - "Harris wins nomination" (ticker 1)
  - "Clinton wins nomination" (ticker 2) 
  - "Buttigieg wins nomination" (ticker 3)
- **Implementation**: Group CUSIPs by outcome type

### Event (The Question)
- **Definition**: The parent question or event being predicted
- **Binary Events**: 1 ticker per event (ticker = event)
  - Event: "US strikes Iran" = Ticker: "US strikes Iran" 
- **Categorical Events**: Multiple tickers per event
  - Event: "Who wins Democratic nomination?" has tickers: {Harris, Clinton, Buttigieg, ...}
- **Implementation**: event_slug represents this level

### Theme (Thematic Basket)
- **Definition**: Classification of the EVENT for basket construction
- **Examples**:
  - Event "Who wins Democratic nomination?" → Theme: us_elections
  - Event "US strikes Iran" → Theme: middle_east
- **Implementation**: Theme classification happens at EVENT level

## Key Rules for Implementation

1. **Binary Events**: 1 ticker per event (ticker = event)
2. **Categorical Events**: Multiple tickers per event  
3. **Theme Classification**: Happens at EVENT level, not ticker or CUSIP
4. **Basket Construction**: One exposure per EVENT (not per ticker)
5. **CUSIPs**: Are just time variants of a ticker

## Current Implementation Errors

❌ **Wrong**: Using event_slug as ticker level
✅ **Correct**: event_slug represents EVENT level

❌ **Wrong**: Classifying individual markets  
✅ **Correct**: Classify EVENTs, propagate to all child tickers/CUSIPs

❌ **Wrong**: One position per market or ticker
✅ **Correct**: One position per EVENT

## Required Fixes

1. **Data Model**: CUSIP → Ticker → Event → Theme hierarchy
2. **Categorization**: Identify binary vs categorical events within event_slug
3. **Classification**: Theme classification at event level only
4. **Deduplication**: One exposure per event for baskets
5. **Documentation**: Update all docs to reflect correct hierarchy
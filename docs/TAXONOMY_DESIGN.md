# Prediction Market Taxonomy Design
*CORRECTED: A 4-Layer Bottom-Up Model for Intelligent Basket Construction*

## The Correct Hierarchy (Bottom-Up)

```
CUSIP → Ticker → Event → Theme
```

### CUSIP (Layer 1: Most Granular)
**Definition**: A specific market instance with a specific time/date

**Examples**:
- "US strikes Iran by February 28th"
- "Bitcoin Up or Down Feb 22 5:40AM-5:45AM" 
- "Harris wins nomination by March 2028"
- "Ravens Team Total O/U 24.5 points"

**Key property**: These are individual tradeable markets with specific temporal conditions

### Ticker (Layer 2: Outcome Level)
**Definition**: The underlying outcome, stripped of time

**Examples**:
- **Binary event ticker**: "US strikes Iran" (one ticker per event)
- **Categorical event tickers**: "Harris wins nomination", "Clinton wins nomination", "Sanders wins nomination" (multiple tickers per event)
- **Continuous event tickers**: "Ravens Team Total O/U 24.5", "Ravens Team Total O/U 26.5" (multiple tickers per event)

**Key insight**: Binary events have ticker = event. Categorical events have multiple tickers per event.

### Event (Layer 3: The Question)
**Definition**: The parent question/event that contains tickers

**Examples**:
- **Binary event**: "US strikes Iran" → binary, so event = ticker
- **Categorical event**: "Who wins Democratic nomination?" → categorical, so event has MULTIPLE tickers (Harris, Clinton, Sanders...)
- **Categorical event**: "Ravens Team Total Points" → continuous/categorical, with multiple O/U tickers

**Key property**: This is the structural grouping level. One EVENT can generate many CUSIPs through time variants.

### Theme (Layer 4: Thematic Basket)
**Definition**: Classification applied at the EVENT level for basket construction

**Examples**:
- "Who wins Democratic nomination?" → us_elections
- "US strikes Iran" → middle_east  
- "Ravens Team Total Points" → sports_entertainment

**Key insight**: Classification happens at EVENT level, NOT at ticker or CUSIP level.

---

## Key Rules

### Binary vs Categorical Events

**Binary Events**:
- 1 ticker per event (ticker = event)
- Multiple CUSIPs through time variants
- Example: "US strikes Iran" event → "US strikes Iran" ticker → ["by Feb 28", "by March 31", "by June 30"] CUSIPs

**Categorical Events**:
- Multiple tickers per event  
- Each ticker can have multiple CUSIPs through time variants
- Example: "Democratic nomination" event → ["Harris wins", "Clinton wins", "Sanders wins"] tickers → Each ticker has ["by March 2028", "by June 2028"] CUSIPs

### Basket Construction Rules

1. **Theme classification happens at EVENT level**
2. **Basket construction: one exposure per EVENT**
3. **CUSIPs are just time variants of a ticker**
4. **For categorical events, pick ONE ticker per event for the basket**

---

## Detection Methods

### CUSIP Detection (Layer 1)
This is just our raw market data. Each market is a CUSIP.

### Ticker Detection (Layer 2)
**Method**: Strip time/date information from CUSIP titles

**Regex patterns for time stripping**:
```regex
# Remove date patterns
\b(?:by|before|after|until)\s+\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b
\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b

# Remove time patterns  
\d{1,2}:\d{2}(?:AM|PM)?[-\s]*\d{1,2}:\d{2}(?:AM|PM)?
```

**Examples**:
- CUSIP: "US strikes Iran by February 28th" → Ticker: "US strikes Iran"
- CUSIP: "Harris wins nomination by March 2028" → Ticker: "Harris wins nomination"

### Event Detection (Layer 3)
**Method**: Group tickers that represent outcomes of the same underlying question

**For Binary Events**: ticker = event (1:1 mapping)

**For Categorical Events**: Use pattern matching and semantic similarity
- Look for shared patterns: "[PERSON] wins [SAME_POSITION]"
- Group tickers with similar base text but different outcome actors
- Examples:
  - Tickers ["Harris wins nomination", "Clinton wins nomination"] → Event: "Democratic nomination"
  - Tickers ["McDavid wins Hart", "MacKinnon wins Hart"] → Event: "NHL Hart Trophy"

**Leverage existing structure**: Polymarket's `event_slug` often represents our EVENT level
- `event_slug: "nhl-2025-26-hart-memorial-trophy"` contains 118 different player tickers
- `event_slug: "2028-democratic-presidential-nomination"` contains different candidate tickers

### Theme Classification (Layer 4)
**Method**: Classify EVENTS (not individual tickers or CUSIPs) into thematic baskets

**LLM Prompt Structure**:
```
Event: "Democratic nomination"
Context: This event contains tickers like "Harris wins nomination", "Clinton wins nomination", "Sanders wins nomination"
Classify this EVENT into one theme: [us_elections, middle_east, crypto, etc.]
```

---

## Implementation Logic

### Step 1: CUSIP → Ticker Mapping
```python
def extract_ticker_from_cusip(cusip_title):
    """Strip time/date information to get ticker"""
    # Remove date patterns
    ticker = re.sub(date_patterns, '', cusip_title)
    # Remove time patterns
    ticker = re.sub(time_patterns, '', ticker)
    return ticker.strip()
```

### Step 2: Ticker → Event Grouping
```python
def group_tickers_into_events(tickers):
    """Group related tickers into events"""
    events = {}
    
    for ticker in tickers:
        if is_binary_event(ticker):
            # Binary: ticker = event
            event_name = ticker
            events[event_name] = [ticker]
        else:
            # Categorical: find the parent event
            event_name = extract_base_event(ticker)
            if event_name not in events:
                events[event_name] = []
            events[event_name].append(ticker)
    
    return events

def is_binary_event(ticker):
    """Detect if ticker represents binary vs categorical event"""
    # Binary patterns: "US strikes Iran", "Fed raises rates"
    # Categorical patterns: "Harris wins", "McDavid wins"
    binary_patterns = [
        r"^\w+\s+strikes\s+\w+$",
        r"^\w+\s+(?:raises|cuts)\s+rates$"
    ]
    return any(re.match(pattern, ticker) for pattern in binary_patterns)
```

### Step 3: Event → Theme Classification
```python
def classify_event_theme(event_name, event_tickers):
    """Classify an event into thematic basket"""
    context = f"Event: {event_name}\nTickers: {event_tickers}"
    
    prompt = f"""
    Classify this EVENT into one of these themes:
    - us_elections
    - crypto  
    - middle_east
    - sports_entertainment
    - economic_policy
    - [other themes...]
    
    Event context: {context}
    
    Theme:"""
    
    return llm_classify(prompt)
```

### Step 4: Basket Construction (One Exposure Per Event)
```python
def construct_theme_basket(theme_events):
    """Build basket with one position per event"""
    basket_positions = []
    
    for event_name, event_tickers in theme_events.items():
        # Pick the most liquid ticker for this event
        best_ticker = select_most_liquid_ticker(event_tickers)
        
        # Pick the most liquid CUSIP for this ticker
        ticker_cusips = get_cusips_for_ticker(best_ticker)
        best_cusip = select_most_liquid_cusip(ticker_cusips)
        
        basket_positions.append(best_cusip)
    
    return basket_positions
```

---

## Data Model

### Core Tables

**markets (CUSIPs)**
```sql
CREATE TABLE markets (
    market_id VARCHAR PRIMARY KEY,  -- This is our CUSIP
    title VARCHAR,
    description TEXT,
    volume DECIMAL,
    status VARCHAR,
    platform VARCHAR
);
```

**tickers**  
```sql
CREATE TABLE tickers (
    ticker_id VARCHAR PRIMARY KEY,   -- Ticker name
    ticker_text VARCHAR,             -- "Harris wins nomination"  
    event_id VARCHAR,                -- References events table
    is_binary BOOLEAN,               -- TRUE if binary event
    market_count INTEGER             -- Number of CUSIPs with this ticker
);
```

**events**
```sql  
CREATE TABLE events (
    event_id VARCHAR PRIMARY KEY,    -- Event identifier
    event_name VARCHAR,              -- "Democratic nomination"
    theme VARCHAR,                   -- "us_elections" 
    is_categorical BOOLEAN,          -- TRUE if multiple tickers per event
    ticker_count INTEGER             -- Number of tickers in this event
);
```

**cusip_ticker_mapping**
```sql
CREATE TABLE cusip_ticker_mapping (
    market_id VARCHAR,               -- CUSIP (references markets)
    ticker_id VARCHAR,               -- Ticker (references tickers)  
    extraction_method VARCHAR,       -- "regex", "manual", "llm"
    confidence DECIMAL(3,2)
);
```

---

## Real Data Examples

### Binary Event: "US strikes Iran"
- **Event**: "US strikes Iran" 
- **Ticker**: "US strikes Iran" (same as event)
- **CUSIPs**: 
  - "US strikes Iran by February 28th"
  - "US strikes Iran by March 31st" 
  - "US strikes Iran by June 30th"
  - [126 other time variants]
- **Theme**: middle_east
- **Basket exposure**: Pick ONE CUSIP (most liquid)

### Categorical Event: "NHL Hart Trophy"
- **Event**: "NHL Hart Trophy"
- **Tickers**: 
  - "McDavid wins Hart Trophy"
  - "MacKinnon wins Hart Trophy"  
  - "Keller wins Hart Trophy"
  - [115 other players]
- **CUSIPs**: Each ticker could have time variants, but most are season-end
- **Theme**: sports_entertainment  
- **Basket exposure**: Pick ONE ticker (most liquid player), then ONE CUSIP for that ticker

### Categorical Event: "Democratic Nomination"  
- **Event**: "Democratic nomination"
- **Tickers**:
  - "Harris wins nomination"
  - "Clinton wins nomination"
  - "Sanders wins nomination" 
  - [77 other candidates]
- **CUSIPs**: Each ticker has time variants like "by March 2028", "by June 2028"
- **Theme**: us_elections
- **Basket exposure**: Pick ONE ticker (most liquid candidate), then ONE CUSIP for that ticker

---

## Critical Insights

### Why This Hierarchy Matters

1. **Proper diversification**: One exposure per EVENT prevents fake diversification from including both "Harris wins" and "Clinton wins" in the same basket

2. **Correct classification**: Classifying at EVENT level (not CUSIP level) gives the LLM proper context. "Democratic nomination" is clearly us_elections, while individual CUSIPs like "Harris wins by March 2028" might be ambiguous.

3. **Structural deduplication**: Events provide natural grouping that eliminates time-variant fragmentation and categorical correlation

4. **Scalable framework**: Clear rules for binary vs categorical events, with systematic selection of representatives

### Previous Approach Was Backwards

The old approach tried to classify individual markets (CUSIPs) first, then group them. This created:
- High "uncategorized" rates (49%) because individual CUSIPs lack context
- Fake diversification from including multiple related positions  
- Temporal fragmentation from treating time variants as separate themes

The correct approach groups first (CUSIP → Ticker → Event), then classifies the EVENT, ensuring one exposure per underlying question.
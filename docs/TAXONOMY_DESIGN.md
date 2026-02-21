# Prediction Market Taxonomy Design
*A 3-Layer Model for Intelligent Basket Construction*

## Problem Statement

Naive market-level analysis is fundamentally broken for prediction market basket construction. Current approaches treat each market as an independent bet, leading to three critical errors:

1. **Fake Diversification**: Including multiple outcomes from the same categorical event (e.g., both "Harris wins 2028 nomination" and "Clinton wins 2028 nomination") creates illusory diversification when these outcomes are negatively correlated by construction.

2. **Temporal Fragmentation**: Treating "US strikes Iran by March 31" and "US strikes Iran by June 30" as separate themes dilutes exposure to the underlying geopolitical risk.

3. **Sports Betting Noise**: Thousands of NFL over/under lines and player props obscure genuine thematic signals with micro-betting noise.

**Example from real data**: Our dataset contains 129 "US strikes Iran" markets with different dates, 80 Democratic nomination markets for different candidates, and 67 "Ravens Team Total O/U" markets with different point spreads. Traditional basket construction would either ignore these patterns (losing thematic coherence) or include too many correlated positions (destroying diversification).

Correct basket construction requires understanding three distinct layers of market organization before any portfolio optimization can begin.

---

## The 3-Layer Taxonomy Model

### Layer 1: Underlying Events ("Ticker")
**Definition**: The fundamental question or event that drives market outcomes, regardless of specific variants.

**Examples from real data**:
- **Ticker: US_IRAN_CONFLICT** → 129 markets: "US strikes Iran by Feb 28", "US strikes Iran by March 31", "Israel strikes Iran by June 30"
- **Ticker: FED_RATE_DECISION** → 87 markets: Fed rate changes after March meeting, April meeting, June meeting, etc.
- **Ticker: DEM_2028_PRIMARY** → 80 markets: Harris wins, Clinton wins, Sanders wins, Yang wins, etc.
- **Ticker: BTC_PRICE_TRAJECTORY** → 392 markets: Bitcoin above $68K, Bitcoin reaches $150K, Bitcoin dips to $55K

**Key insight**: Multiple markets can represent the SAME underlying event/question. This is the fundamental unit for thematic exposure.

### Layer 2: Outcome Variants (Categorical)
**Definition**: Within one underlying event, different mutually exclusive outcomes that sum to approximately 100% probability.

**Real examples**:

**Democratic Primary (Layer 1) → Categorical Outcomes (Layer 2)**:
- Harris wins nomination (market 1)
- Clinton wins nomination (market 2)  
- Sanders wins nomination (market 3)
- Yang wins nomination (market 4)
- [78 other candidates...]

**NHL Hart Trophy (Layer 1) → Categorical Outcomes (Layer 2)**:
- McDavid wins Hart Trophy (market 1)
- MacKinnon wins Hart Trophy (market 2)
- Keller wins Hart Trophy (market 3)
- [115 other players...]

**Winter Olympics Gold Medals (Layer 1) → Categorical Outcomes (Layer 2)**:
- China wins a gold medal (market 1)
- Kazakhstan wins a gold medal (market 2)
- Japan wins a gold medal (market 3)
- [77 other countries...]

**Critical property**: These outcomes are **negatively correlated by construction**. If Harris wins the nomination, Clinton cannot. Including multiple categorical outcomes in the same basket creates fake diversification and destroys portfolio theory assumptions.

### Layer 3: Maturity/Time Variants ("CUSIP")
**Definition**: The same underlying question at different time horizons or with different temporal conditions.

**Real examples**:

**Iran Conflict Time Variants**:
- US strikes Iran by February 28, 2026
- US strikes Iran by March 31, 2026  
- US strikes Iran by June 30, 2026
- US strikes Iran by December 31, 2026

**Fed Meeting Time Variants**:
- Fed cuts rates after March 2026 meeting
- Fed cuts rates after April 2026 meeting
- Fed cuts rates after June 2026 meeting

**Bitcoin Price Time Variants**:
- Bitcoin above $68,000 on February 21
- Bitcoin above $70,000 on February 21
- Bitcoin above $74,000 on February 21
- Bitcoin reaches $150,000 in February
- Bitcoin reaches $150,000 in March

**Pattern recognition**: The "by [date]" pattern appears in 1,576+ markets. These are like bond maturities - same underlying risk, different duration exposure.

### Layer Interactions

**Layer 1 + 2**: An event has categorical outcomes
- Event: "2028 Democratic Primary" → Outcomes: {Harris, Clinton, Sanders, Yang, ...}

**Layer 1 + 3**: An event has time variants  
- Event: "US-Iran Military Conflict" → Time variants: {by Feb 28, by March 31, by June 30, ...}

**Layer 1 + 2 + 3**: Complex structure
- Event: "Fed Rate Policy" → Outcomes: {cut 25bps, cut 50bps, no change, hike 25bps} → Time variants: {March meeting, April meeting, June meeting}

---

## Detection Methods by Layer

### Layer 1: Event Detection

**What can be solved with regex/structural parsing?**
- Time pattern removal: Strip "by February 28", "after March meeting", "in 2026"
- Categorical pattern removal: Strip "Harris wins", "McDavid wins", specific candidate/player names
- Numerical variant removal: Strip "O/U 24.5", "above $68,000", specific price levels

**Current regex patterns working well**:
```regex
# Dates and times
\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b
\b(?:by|through|before|after|until)\s+\w+\b

# Over/under patterns  
O/U \d+\.?\d*

# Meeting patterns
after the \w+ \d{4} meeting
```

**What needs fuzzy matching?**
- Similar phrasings: "Trump wins 2028 election" vs "Trump to win 2028 presidential race"
- Abbreviations: "BTC" vs "Bitcoin", "Fed" vs "Federal Reserve"
- Minor variations: "US strikes Iran" vs "United States strikes Iran"

**What needs LLM reasoning?**
- Semantic equivalence: "Will there be a ceasefire?" vs "Will the war end?"
- Complex categorization: Is "Trump fires FBI director" legal/regulatory or partisan politics?
- Cross-platform normalization: Polymarket vs Kalshi phrasing differences

**What can use existing platform structure?**
- **Polymarket event_slug**: Already groups related markets (67 unique event_slugs for 20,191 markets)
- **Example**: event_slug "nhl-2025-26-hart-memorial-trophy" contains 118 different player markets
- **Limitation**: Only covers ~0.3% of unique events, but high-quality when available

### Layer 2: Categorical Outcome Detection

**Regex patterns that work**:
- Player/candidate name extraction: "Will [PERSON] win [AWARD]"
- Country/team extraction: "Will [COUNTRY] record a [medal type]"
- Over/under level extraction: "[TEAM] Total: O/U [NUMBER]"

**Fuzzy matching scenarios**:
- Name variations: "McDavid" vs "Connor McDavid"
- Team abbreviations: "Ravens" vs "Baltimore Ravens"

**LLM reasoning needed for**:
- Determining if outcomes are truly mutually exclusive
- Distinguishing categorical vs continuous variants (NFL point spreads are technically continuous but practically categorical)
- Complex outcome relationships

**Real pattern from data**: 274 markets starting with "Will the price of Bitcoin be above..." with different price levels. These are continuous variants that could be treated as categorical outcomes.

### Layer 3: Time/Maturity Detection

**Highly effective regex patterns**:
```regex
# Direct date references
by \w+ \d{1,2}(?:st|nd|rd|th)?, \d{4}
before \w+ \d{4}

# Meeting references  
after the \w+ \d{4} meeting

# Quarterly patterns
in Q[1-4] \d{4}
```

**Success metrics from real data**:
- 355 markets contain "by March" 
- 392 markets contain "before 2027"
- 1,576 markets contain "February"
- 87 Fed meeting time variants detected

**Minimal LLM reasoning needed**: Time extraction is largely deterministic.

---

## Edge Cases from Real Data

### Sports Betting Complexity
**Challenge**: NFL markets create massive categorical explosions
- Ravens Team Total has 67 different O/U lines (8.5, 16.5, 22.5, 26.5, 27.5, 34.5, 35.5, 37.5...)
- These are technically different outcomes but represent micro-variations on "Ravens scoring performance"

**Current approach**: Group by base event ("Ravens Team Total") but unclear if these should be:
1. Treated as one position (take most liquid line)
2. Treated as categorical outcomes (pick one line per basket)
3. Treated as separate events (different risk/reward profiles)

### Cryptocurrency Price Targets
**Challenge**: Continuous vs categorical treatment
- "Bitcoin above $68,000 on Feb 21"
- "Bitcoin above $70,000 on Feb 21"  
- "Bitcoin above $74,000 on Feb 21"

These are about the same underlying (BTC price on a specific date) but represent different strike prices. Are they categorical outcomes or different events?

### Temporal Granularity Explosion
**Challenge**: Iran strike markets with excessive time granularity
- 129 total markets with dates ranging from specific days to quarters
- "US strikes Iran by February 28" vs "by March 31" (monthly)
- "US strikes Iran on February 19" vs "on February 20" (daily)
- "US strikes Iran during week of Feb 15-21" (weekly)

**Pattern**: Same underlying event fragmented across multiple time granularities simultaneously.

### Cross-Platform Inconsistency
**Challenge**: Polymarket vs Kalshi express similar events differently
- Polymarket: "Will Trump nominate Kevin Warsh as the next Fed chair?"
- Kalshi: Hypothetical equivalent might be "FEDCHAIR-WARSH-26"

**Opportunity**: Cross-platform arbitrage detection, but complicates event grouping.

### Semantic Boundary Cases
**Challenge**: Events that could belong to multiple Layer 1 categories
- "Will Trump impose tariffs on China?"
  - Could be: US_POLITICS (Trump policy)
  - Could be: CHINA_US_TRADE (bilateral trade)
  - Could be: US_ECONOMIC (tariff impact)

**Real example from classifications**: LLM classified 77 markets as "uncategorized", suggesting genuine boundary cases exist.

---

## Implications for Basket Construction

### Handling Categorical Outcomes (Layer 2)

**Option 1: Pick One Representative**
- Choose the most liquid market from each categorical set
- Pro: Clean diversification, no correlation issues
- Con: May miss nuanced views (e.g., Clinton vs Harris have different risk profiles)

**Option 2: Aggregate Probabilities** 
- Create synthetic "Any Democrat except Biden" positions by combining probabilities
- Pro: Captures broader thematic exposure
- Con: Complex to implement, harder to trade

**Option 3: Weighted Composite**
- Weight by historical performance, liquidity, or polling data
- Pro: Sophisticated exposure management
- Con: Requires external data and complex rebalancing

**Recommended approach**: Start with Option 1 (pick most liquid) for MVP, evolve to Option 3.

### Handling Time Variants (Layer 3)

**Option 1: Pick Nearest Maturity**
- For current date Feb 21, pick "by February 28" over "by March 31"
- Pro: Most immediate risk exposure
- Con: May miss longer-term thematics

**Option 2: Pick Longest Maturity**
- Choose "by December 31" over "by February 28" 
- Pro: Avoids excessive turnover from short-dated expirations
- Con: Less sensitive to near-term events

**Option 3: Pick Most Liquid**
- Volume and open interest determine selection
- Pro: Most tradeable, reflects market consensus  
- Con: May not align with investment horizon

**Recommended approach**: Option 3 (most liquid) with minimum 30-day expiration filter.

### The "One Exposure Per Underlying" Rule

**Definition**: Each basket should contain at most one position representing each Layer 1 underlying event.

**Implementation**:
1. Group all markets by Layer 1 ticker
2. Within each ticker group, apply Layer 2 deduplication (pick one categorical outcome)
3. Within each ticker group, apply Layer 3 deduplication (pick one time variant)
4. Result: Each ticker appears exactly once in each basket

**Example**: From 129 "US strikes Iran" markets, select exactly one for the Middle East basket:
- Layer 2 deduplication: Choose "US strikes" over "Israel strikes" (most liquid actor)
- Layer 3 deduplication: Choose "by June 30" (sufficient time horizon, high liquidity)
- Final selection: "US strikes Iran by June 30, 2026"

**Edge case**: What if one ticker spans multiple themes? 
- "Trump fires FBI director" could be LEGAL_REGULATORY or US_ELECTIONS
- **Solution**: Assign to primary theme only, accept that some cross-theme exposure is inevitable

---

## Proposed Data Model

### Core Tables

**markets_raw** (existing)
- market_id, platform, title, description, volume, dates, status

**taxonomy_classification**
```sql
market_id VARCHAR PRIMARY KEY
ticker VARCHAR NOT NULL           -- Layer 1: underlying event
theme VARCHAR NOT NULL            -- Final theme assignment  
outcome_variant VARCHAR           -- Layer 2: categorical outcome identifier
time_variant VARCHAR              -- Layer 3: time/maturity identifier
classification_method VARCHAR     -- "llm", "statistical", "hybrid"
confidence DECIMAL(3,2)           -- 0.0 to 1.0
```

**ticker_definitions**
```sql
ticker VARCHAR PRIMARY KEY
canonical_title VARCHAR NOT NULL  -- "US strikes Iran"
canonical_description TEXT        -- Detailed event description
theme VARCHAR NOT NULL            -- Primary theme assignment
market_count INTEGER              -- Number of markets with this ticker
total_volume DECIMAL              -- Sum of volume across all markets
```

**categorical_groups**  
```sql
ticker VARCHAR
outcome_variant VARCHAR
outcome_description VARCHAR       -- "Harris wins", "McDavid wins"
market_count INTEGER
is_mutually_exclusive BOOLEAN     -- TRUE for political races, FALSE for commodities
```

**time_variants**
```sql
ticker VARCHAR  
time_variant VARCHAR
expiration_date DATE              -- When the condition is evaluated
time_horizon_days INTEGER         -- Days from creation to expiration
market_count INTEGER
```

### Derived Views

**basket_eligible_markets**
```sql
-- Markets that pass all eligibility filters
SELECT m.*, t.ticker, t.theme
FROM markets_raw m 
JOIN taxonomy_classification t ON m.market_id = t.market_id
WHERE m.status = 'open'
  AND m.days_to_expiration > 30
  AND m.volume > 10000
  AND m.liquidity > 1000
```

**basket_canonical_positions**  
```sql
-- One market per ticker (after Layer 2 + 3 deduplication)
SELECT DISTINCT ON (ticker) 
  ticker, theme, market_id, volume, liquidity
FROM basket_eligible_markets
ORDER BY ticker, volume DESC  -- Pick highest volume market per ticker
```

---

## Implementation Approach

### Phase 1: Layer 1 Event Detection (Week 1)
1. **Regex-based ticker extraction** (extend existing ticker_cusip.py)
   - Implement time pattern removal
   - Implement categorical pattern removal  
   - Test on real data sample (1,000 markets)

2. **Leverage Polymarket event_slug structure**
   - Parse existing event_slug to ticker mapping
   - Validate against manual classification sample

3. **Basic fuzzy matching** 
   - Group similar titles (edit distance < 3)
   - Manual review of fuzzy match suggestions

**Success criteria**: 80%+ of markets get reasonable ticker assignments

### Phase 2: Layer 2 Categorical Detection (Week 2)  
1. **Pattern-based outcome extraction**
   - Extract person names from political markets
   - Extract team names from sports markets
   - Extract country names from Olympics markets

2. **Mutual exclusivity detection**
   - Flag outcome groups where probabilities should sum to ~1.0
   - Identify which outcomes are truly mutually exclusive vs independent

3. **Deduplication rules**
   - Most liquid market per categorical group
   - Handle edge cases (continuous outcomes like O/U lines)

**Success criteria**: Categorical groups have <0.3 average correlation within theme

### Phase 3: Layer 3 Time Variant Detection (Week 3)
1. **Date extraction pipeline**  
   - Parse "by [date]" patterns
   - Parse "after [meeting]" patterns
   - Calculate days to expiration

2. **Time horizon optimization**
   - Analyze turnover impact of different time selection rules
   - Backtest different maturity selection strategies

**Success criteria**: <50% monthly turnover from time variant selection

### Phase 4: LLM Integration (Week 4)
1. **Boundary case resolution**
   - Use LLM for complex ticker assignments
   - Use LLM for cross-theme event classification
   - Build confidence scoring

2. **Classification validation**
   - Human audit of 200 random classifications
   - Cross-validation across different LLM prompts

**Success criteria**: >90% human agreement on LLM classifications

### Phase 5: Integration Testing (Week 5)
1. **Full pipeline testing**
   - Process all 20K+ markets through 3-layer taxonomy
   - Generate final basket compositions
   - Validate diversity and investability

2. **Backtest integration**
   - Ensure chain-linking works with new taxonomy
   - Performance attribution by layer
   - Compare to naive equal-weight baseline

**Success criteria**: Baskets pass all diversification and tradability constraints

### Recommended Tool Selection

**Layer 1 (Event Detection)**:
- Primary: Regex + string processing (deterministic, fast)
- Secondary: Fuzzy matching for boundary cases  
- Tertiary: LLM for complex semantic cases
- Leverage: Polymarket event_slug where available

**Layer 2 (Categorical Detection)**:
- Primary: Pattern matching + named entity recognition
- Secondary: Statistical correlation analysis (negative correlation detection)
- Tertiary: LLM for mutual exclusivity determination

**Layer 3 (Time Detection)**: 
- Primary: Regex date parsing (highly effective)
- Secondary: NLP date extraction libraries
- Minimal LLM usage needed

---

## Open Questions for Alejandro

### Strategic Direction
1. **Basket count target**: Should we let the data determine number of themes (currently ~13) or do you have a target number (e.g., exactly 10 baskets for simplicity)?

2. **Sports betting treatment**: The data shows 4,419 over/under markets creating massive categorical explosions. Should we:
   - Exclude sports betting entirely from thematic baskets?
   - Create separate sports betting baskets (NFL, NHL, NBA)?
   - Treat as micro-themes within entertainment category?

3. **Cross-platform arbitrage**: We have equivalent events on Polymarket + Kalshi. Should we:
   - Include both platforms in same basket (diversification)?
   - Pick the more liquid platform per event?
   - Create cross-platform arbitrage detection?

### Technical Implementation
4. **Minimum viable product scope**: For initial launch, should we focus on:
   - Clean non-sports events only (higher quality, easier classification)?
   - Full universe including sports (more comprehensive)?
   - Specific themes only (politics, crypto, geopolitics)?

5. **Categorical outcome handling**: For mutually exclusive outcomes like Democratic nomination:
   - Pick single most liquid candidate per basket?
   - Create "Any Democratic nominee" synthetic positions?
   - Multiple separate baskets (Harris basket, Clinton basket)?

6. **Time variant selection**: For events with many time horizons:
   - Always pick longest maturity available (less turnover)?
   - Always pick most liquid maturity (better execution)?
   - Pick maturity closest to target investment horizon?

### Business Integration  
7. **Integration timeline**: Should this replace the current basket system:
   - Immediately upon completion (rip off band-aid)?
   - Gradual migration (run both systems parallel)?
   - A/B test with subset of users?

8. **Performance benchmarking**: What's the success criteria?
   - Beat equal-weight all-markets benchmark by X%?
   - Achieve target Sharpe ratio of Y?  
   - Deliver uncorrelated returns across baskets (<0.3 correlation)?

9. **Data freshness requirements**: How often should taxonomy be rebuilt?
   - Weekly (catch new events quickly)?
   - Monthly (align with rebalancing)?
   - Only when significant new event types emerge?

---

## Conclusion

The 3-layer taxonomy model provides a rigorous intellectual foundation for prediction market basket construction. By properly identifying underlying events (Layer 1), handling categorical outcomes (Layer 2), and managing time variants (Layer 3), we can construct baskets that offer genuine thematic diversification rather than illusory market-level diversification.

The real data analysis reveals clear patterns that validate this approach: 129 Iran strike markets should become 1 position, 80 Democratic nomination markets should become 1 representative position, and 4,419 sports over/under markets need systematic categorical treatment.

Implementation should prioritize deterministic methods (regex, pattern matching) over complex methods (LLM reasoning) where possible, while using existing platform structure (Polymarket event_slug) as a foundation. The result will be investable, diversified baskets that actually reflect thematic investment exposure rather than prediction market microstructure noise.

This taxonomy is the missing piece that transforms prediction markets from a collection of individual bets into a structured asset class suitable for institutional portfolio construction.
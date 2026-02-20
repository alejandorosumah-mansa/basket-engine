# Flagship Basket Rebalancing Engine - Technical Plan

## Problem Statement

Current basket construction is broken:
- 17 hand-labeled risk factors, completely wrong categorizations
- No rebalancing (baskets rot as markets expire)
- No historical backfill capability
- Equal weighting with no regard for volume, liquidity, or correlation
- Can't go back in time because there's no rebalancing methodology to replay

## Goal

Build a standalone system that:
1. Correctly categorizes prediction markets into investable themes
2. Constructs baskets with quantitative weighting
3. Rebalances monthly with full audit trail
4. Can replay historical rebalances to generate backtested performance
5. Validates itself through testable, measurable criteria

---

## Architecture Overview

**New repo: `basket-engine`** (standalone, no dependency on basket-trader)

```
basket-engine/
├── README.md
├── requirements.txt
├── config/
│   ├── taxonomy.yaml          # Theme definitions + rules
│   └── settings.yaml          # API keys, thresholds, params
├── data/
│   ├── raw/                   # Raw API pulls (cached)
│   │   ├── polymarket/        # Market metadata + price history
│   │   └── kalshi/            # Market metadata + price history
│   ├── processed/             # Cleaned, normalized data
│   │   ├── markets.parquet    # All markets with metadata
│   │   ├── prices.parquet     # Daily prices for all markets
│   │   └── returns.parquet    # Daily returns matrix
│   └── outputs/               # Final basket compositions
│       ├── baskets/           # One file per rebalance date
│       └── performance/       # Backtested NAV series
├── src/
│   ├── ingestion/             # Data collection
│   │   ├── polymarket.py      # Polymarket API client (Dome + native)
│   │   ├── kalshi.py          # Kalshi API client (Dome + native)
│   │   └── cache.py           # Local caching layer
│   ├── classification/        # Market categorization
│   │   ├── llm_classifier.py  # LLM-based theme assignment
│   │   ├── correlation.py     # Statistical clustering
│   │   ├── hybrid.py          # Combined approach
│   │   └── taxonomy.py        # Taxonomy loader + validation
│   ├── construction/          # Basket building
│   │   ├── eligibility.py     # Market eligibility filters
│   │   ├── weighting.py       # Weight calculation methods
│   │   └── rebalance.py       # Rebalance engine
│   ├── backtest/              # Historical replay
│   │   ├── engine.py          # Backtest orchestrator
│   │   ├── chain_link.py      # Price chain-linking
│   │   └── metrics.py         # Performance/risk metrics
│   └── validation/            # Testing + verification
│       ├── classification_audit.py
│       ├── backtest_sanity.py
│       └── stability.py
├── notebooks/                 # Jupyter for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_correlation_analysis.ipynb
│   ├── 03_clustering_experiments.ipynb
│   ├── 04_classification_comparison.ipynb
│   └── 05_backtest_results.ipynb
└── tests/
    ├── test_ingestion.py
    ├── test_classification.py
    ├── test_construction.py
    └── test_backtest.py
```

---

## Phase 1: Data Ingestion

### Objective
Pull all historical market data into a local normalized format.

### Data Sources

**Polymarket (via Dome API + native CLOB API):**
- `GET /polymarket/candlesticks/{condition_id}` - OHLC candles (1m, 1h, 1d intervals)
  - 1d interval: max range 1 year per request
  - Need condition_id for each market
- `GET /polymarket/market-price/{token_id}?at_time=X` - Point-in-time historical price
- Native CLOB: `GET /prices-history?market={token_id}&interval=all&fidelity=1440` - Daily price history
- `GET /polymarket/markets` - Market metadata (title, description, category, volume, liquidity, outcomes, dates)

**Kalshi (via Dome API + native API):**
- `GET /kalshi/market-price/{ticker}?at_time=X` - Point-in-time historical price
- `GET /kalshi/trades?ticker=X&start_time=X&end_time=X` - Trade history
- Native: `GET /trade-api/v2/markets` - Market metadata

**What we need per market:**
| Field | Source | Purpose |
|-------|--------|---------|
| market_id | API | Unique identifier |
| platform | API | polymarket / kalshi |
| title | API | For LLM classification |
| description | API | For LLM classification |
| category/tags | API | Supplementary classification signal |
| start_date | API | When market was created |
| end_date | API | When market expires/resolved |
| resolution | API | YES/NO/null if active |
| daily_close_price | Candlesticks | For returns matrix |
| daily_volume | Candlesticks/trades | For weighting |
| daily_liquidity | Orderbook snapshots | For eligibility |

### Caching Strategy
- Store raw API responses in `data/raw/` as JSON
- Process into parquet files in `data/processed/`
- Incremental updates: only fetch new data since last pull
- Rate limit: Dome free tier = 1 QPS, so full historical pull will take time. Budget for this.

### Data Quality Checks
- [ ] Markets with 0 volume for >7 consecutive days → flag as illiquid
- [ ] Price outside [0, 1] → data error
- [ ] Gaps in daily price series → interpolate or flag
- [ ] Duplicate market detection (same event on Poly + Kalshi)

### Deliverable
- `markets.parquet`: all markets with metadata (expect 1,000-2,000+ markets)
- `prices.parquet`: daily close prices, indexed by (market_id, date)
- `returns.parquet`: daily returns, indexed by (market_id, date)
- Data going back at least 6 months, ideally 12+

---

## Phase 2: Classification

### Approach: Hybrid (Statistical + LLM)

Two independent classification systems that get compared and merged.

### 2A: Statistical Clustering

**Input:** Returns matrix (markets x days)

**Method:**
1. Filter to markets with at least 30 days of price history and >$10K total volume
2. Compute pairwise Spearman rank correlation (not Pearson, because returns aren't normal)
3. Run hierarchical clustering (Ward linkage) on distance matrix (1 - correlation)
4. Use silhouette score + dendrogram to determine optimal number of clusters (expect 8-15)
5. For each cluster, examine constituent markets and assign interpretive label

**Why Spearman over Pearson:**
Prediction market returns are bounded [0,1] and often have heavy tails (big jumps on news). Spearman captures monotonic relationships without assuming linearity.

**Why hierarchical over k-means:**
- Produces a dendrogram (visual tree of relationships)
- No need to pre-specify k
- Can cut at different levels for different granularity (coarse: 8 baskets, fine: 20 sub-themes)

**Alternative to explore:** Conditional correlation
- Standard correlation misses tail dependencies
- Compute correlation only during periods when markets move >5% in a day
- "When shit hits the fan, which markets move together?"
- This captures systemic risk linkages better

**Output:** cluster_assignments.csv (market_id, cluster_id, cluster_label)

### 2B: LLM Classification

**Input:** Market title + description + platform tags

**Taxonomy (defined in taxonomy.yaml):**
```yaml
themes:
  us_elections:
    name: "US Elections & Partisan Politics"
    description: "Presidential, congressional, gubernatorial elections. Party control. Approval ratings. Electoral mechanics."
    examples:
      - "Will Democrats win the Senate in 2026?"
      - "Trump approval rating above 45%?"
    anti_examples:
      - "Will the Fed cut rates?" # This is monetary policy, not elections
      - "Will Trump impose tariffs on China?" # This is trade/geopolitics

  fed_monetary_policy:
    name: "Federal Reserve & Monetary Policy"
    description: "Fed rate decisions, QE/QT, Fed chair nominations, inflation targets, Treasury yields."
    examples:
      - "Fed funds rate above 4.5% in March?"
      - "Will Trump nominate a new Fed Chair?"
    anti_examples:
      - "US GDP growth above 2%?" # This is economic indicators

  us_economic:
    name: "US Economic Indicators"
    description: "GDP, unemployment, CPI, housing, consumer confidence. Measurable economic data."
    ...

  ai_technology:
    name: "AI & Automation"
    description: "AI capabilities, AI company milestones, automation replacing jobs, AI regulation."
    ...

  china_us:
    name: "China-US Relations"
    description: "Trade war, tariffs, Taiwan, tech restrictions, diplomatic tensions."
    ...

  russia_ukraine:
    name: "Russia-Ukraine Conflict"
    description: "War progression, ceasefire negotiations, sanctions, territorial control."
    ...

  middle_east:
    name: "Middle East & Iran"
    description: "Iran nuclear, Israel-Palestine, Yemen/Houthi, oil supply disruption from conflict."
    ...

  energy_commodities:
    name: "Energy & Commodities"
    description: "Oil prices, OPEC, natural gas, commodity supply/demand. NOT climate policy."
    ...

  crypto_digital:
    name: "Crypto & Digital Assets"
    description: "Bitcoin/ETH prices, ETF approvals, DeFi, stablecoin regulation, exchange events."
    ...

  climate_environment:
    name: "Climate & Environment"
    description: "Natural disasters, climate policy, emissions targets, weather events."
    ...

  europe_politics:
    name: "European Politics"
    description: "EU governance, Brexit effects, European elections, NATO policy."
    ...

  pandemic_health:
    name: "Pandemic & Biotech"
    description: "Disease outbreaks, vaccine milestones, health policy, pharma breakthroughs."
    ...

  legal_regulatory:
    name: "US Legal & Regulatory"
    description: "Supreme Court decisions, DOJ actions, antitrust, regulatory frameworks."
    ...

  space_frontier:
    name: "Space & Frontier Technology"
    description: "SpaceX milestones, space exploration, quantum computing, nuclear fusion."
    ...
```

**LLM Prompt Structure:**
```
You are classifying prediction markets into investment themes.

TAXONOMY:
[full taxonomy with descriptions, examples, anti-examples]

MARKET:
- Platform: {platform}
- Title: {title}
- Description: {first 500 chars of description}
- Tags: {tags if any}
- End date: {end_date}

INSTRUCTIONS:
1. Assign exactly 1 primary theme and optionally 1 secondary theme
2. Rate confidence 0.0-1.0
3. If no theme fits well (confidence < 0.5), assign "uncategorized"
4. A market can only have a secondary theme if it genuinely spans two themes

OUTPUT (JSON):
{
  "primary_theme": "theme_id",
  "secondary_theme": "theme_id or null",
  "confidence": 0.85,
  "reasoning": "one sentence why"
}
```

**Batch processing:**
- Run on all markets (1,063+)
- Use GPT-4o-mini for cost efficiency (classification is not a hard reasoning task)
- Cache results, only re-classify new markets
- Cost estimate: ~1,063 markets * ~500 tokens/market = ~530K tokens ≈ $0.08

**Output:** llm_classifications.csv (market_id, primary_theme, secondary_theme, confidence, reasoning)

### 2C: Hybrid Reconciliation

Compare statistical clusters vs LLM themes:

1. **Agreement matrix:** For each (cluster, theme) pair, count how many markets overlap
2. **Expected behavior:** High overlap = good. Statistical cluster 3 should map mostly to "US Elections" or similar
3. **Disagreement analysis:**
   - Markets where stats say cluster A but LLM says theme B → investigate manually
   - If stats are right: add as anti-example to LLM prompt
   - If LLM is right: the market's price correlation was coincidental, not causal
4. **Final assignment rules:**
   - If both agree → high confidence, use that theme
   - If stats have strong cluster but LLM disagrees → trust stats (prices don't lie)
   - If LLM assigns but market lacks price history for clustering → trust LLM
   - If neither has confidence → "uncategorized" (don't force it)

### Validation Metrics

**Classification quality:**
- [ ] Inter-annotator agreement: have LLM classify same markets with 3 different prompt phrasings. Agreement > 90%?
- [ ] Cluster purity: within each statistical cluster, what % of markets share the same LLM theme? Target > 75%
- [ ] Manual audit: randomly sample 50 markets, manually verify classification. Accuracy > 90%?
- [ ] Edge case audit: specifically check markets that could go either way (e.g., "Trump tariffs on China" - is this US Politics or China-US?)

**Stability:**
- [ ] Re-run clustering with slightly different time windows (60 days vs 90 days). Do clusters change dramatically? (They shouldn't)
- [ ] Add/remove 10% of markets randomly. Do clusters remain stable? (Robustness check)

---

## Phase 3: Basket Construction

### Eligibility Filters

A market must pass ALL of these to enter a basket:

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Status | Active (not resolved) | Can't trade resolved markets |
| Time to expiration | > 14 days | Avoid settlement risk |
| Total volume | > $10,000 | Minimum trading activity |
| 7-day avg daily volume | > $500 | Recent activity, not just old volume |
| Liquidity (open interest) | > $1,000 | Must be tradeable |
| Price history | > 14 days | Need data for weighting |
| Price range | 0.05 < price < 0.95 | Exclude near-certain markets (no info content) |

These thresholds are tunable. Start conservative, loosen if baskets are too sparse.

### Weighting Methodology

**Primary: Risk Parity with Liquidity Cap**

For each market i in basket B:

1. Compute 30-day rolling volatility: `vol_i = std(daily_returns_i, window=30)`
2. Raw risk parity weight: `w_i_raw = (1/vol_i) / sum(1/vol_j for all j in B)`
3. Liquidity cap: `w_i_cap = min(w_i_raw, liquidity_i / sum(liquidity_j))`
4. Re-normalize: `w_i_final = w_i_cap / sum(w_j_cap for all j in B)`

**Why risk parity:**
- Prevents high-vol markets from dominating basket returns
- A market swinging 20% daily shouldn't have the same weight as one moving 2%
- This is standard institutional methodology (Bridgewater, AQR)

**Why liquidity cap:**
- No point weighting 30% to a market with $500 liquidity
- Ensures baskets are actually tradeable
- Cap = your proportional share of available liquidity

**Alternative weighting schemes to test in backtest:**
- Equal weight (1/n) - baseline
- Volume-weighted
- Max diversification (optimize for diversification ratio)
- Minimum variance (optimize for lowest portfolio vol)

### Basket Constraints

| Constraint | Value | Rationale |
|-----------|-------|-----------|
| Min markets per basket | 5 | Diversification |
| Max markets per basket | 30 | Concentration, tradability |
| Max single market weight | 20% | No single market dominates |
| Min single market weight | 2% | Don't include dust positions |

If a theme has < 5 eligible markets, it doesn't become a basket (merge into a broader theme or flag for review).

### Resolution Handling

When a market resolves mid-month:
1. Record final price (0 or 1) and compute terminal return
2. That return accrues to the basket on resolution date
3. Redistribute weight: `new_w_j = old_w_j / (1 - old_w_resolved)` for all remaining
4. Log the event in rebalance trail
5. If basket drops below 5 markets between rebalances → emergency rebalance

---

## Phase 4: Rebalance Engine

### Monthly Rebalance (1st of each month)

**Inputs:**
- Current basket composition
- Updated market data (prices, volume, liquidity)
- New markets since last rebalance
- Resolved markets since last rebalance

**Process:**
1. **Re-classify new markets** (LLM, or stats if enough history)
2. **Apply eligibility filters** to full universe
3. **For each theme/basket:**
   a. Remove resolved/expired/ineligible markets
   b. Add newly eligible markets that match the theme
   c. Recalculate weights (risk parity + liquidity cap)
   d. Apply constraints (min/max weight, min/max count)
4. **Record RebalanceEvent:**
   ```json
   {
     "date": "2026-02-01",
     "basket_id": "us_elections",
     "additions": [{"market_id": "...", "weight": 0.08, "reason": "newly eligible"}],
     "removals": [{"market_id": "...", "reason": "resolved YES", "final_return": 1.33}],
     "weight_changes": [{"market_id": "...", "old_weight": 0.12, "new_weight": 0.09}],
     "turnover": 0.23,
     "market_count": 12
   }
   ```
5. **Chain-link basket price:**
   ```
   On rebalance day:
   - Close out old weights at closing prices
   - Open new weights at same closing prices  
   - NAV is continuous (no jump)
   - Divisor adjustment (like S&P 500 methodology)
   ```

### Turnover Analysis

Track monthly turnover per basket:
- `turnover = sum(|new_weight_i - old_weight_i|) / 2`
- Target: < 30% per month (reasonable for prediction markets given resolutions)
- If turnover is consistently > 50%, weighting methodology is unstable

---

## Phase 5: Backtest

### Historical Replay

**Process:**
1. Pick backtest start date (e.g., 6 months ago)
2. For each month M from start to present:
   a. Get all markets that existed at month M (from cached data)
   b. Apply eligibility filters using data available at month M (no lookahead)
   c. Classify markets using only info available at month M
   d. Construct baskets with weights computed from data up to month M
   e. Calculate returns for month M using actual price changes
   f. Handle resolutions that occurred during month M
   g. Chain-link to get cumulative NAV
3. Output: daily NAV series per basket

### NO Lookahead Bias Rules
- Classification: only use market title/description available at time M (not future resolution)
- Weighting: volatility computed from trailing 30 days before M, not future
- Eligibility: volume/liquidity from trailing period, not future
- Resolution: only process resolutions after they happen

### Performance Metrics

For each basket, compute:
| Metric | What it measures |
|--------|-----------------|
| Cumulative return | Total growth |
| Annualized return | Standardized growth rate |
| Volatility (annualized) | Risk |
| Sharpe ratio | Risk-adjusted return (assume rf = 5%) |
| Max drawdown | Worst peak-to-trough |
| Calmar ratio | Return / max drawdown |
| Avg monthly turnover | Trading cost proxy |
| Hit rate | % of resolved markets that were profitable |
| Avg markets per basket | Diversification |
| Correlation between baskets | Are baskets actually independent? |

### Comparison Tests

Run backtest for MULTIPLE methodologies side by side:

| Method | Classification | Weighting |
|--------|---------------|-----------|
| Baseline (current) | Manual CSV labels | Equal weight |
| LLM-only | LLM themes | Equal weight |
| LLM + risk parity | LLM themes | Risk parity + liquidity cap |
| Stats-only | Statistical clusters | Risk parity + liquidity cap |
| Hybrid | Reconciled hybrid | Risk parity + liquidity cap |
| Hybrid + equal | Reconciled hybrid | Equal weight |

This tells us exactly how much value each component adds.

### Sanity Checks
- [ ] Basket returns should be uncorrelated across themes (< 0.3 average pairwise correlation)
- [ ] No basket should have > 80% drawdown (would indicate broken construction)
- [ ] Resolved markets should contribute their terminal value correctly
- [ ] NAV should be continuous at rebalance dates (no gaps or jumps)
- [ ] Basket with all markets resolving NO should show negative performance
- [ ] Compare basket returns to equal-weight "all markets" benchmark

---

## Phase 6: Validation Framework

### Automated Test Suite

**test_ingestion.py:**
- [ ] All markets have required fields (title, platform, dates)
- [ ] Price data is within [0, 1]
- [ ] No duplicate market entries
- [ ] Date ranges are valid (start < end)
- [ ] Resolved markets have resolution value

**test_classification.py:**
- [ ] Every market gets a primary theme
- [ ] Confidence scores are in [0, 1]
- [ ] No theme has < 5 eligible markets (or gets merged)
- [ ] LLM classifications are deterministic (same input → same output, temp=0)
- [ ] Statistical clusters pass silhouette score threshold (> 0.3)

**test_construction.py:**
- [ ] All basket weights sum to 1.0
- [ ] No single market exceeds max weight (20%)
- [ ] No market below min weight (2%)
- [ ] All markets in basket pass eligibility filters
- [ ] Basket has between 5-30 markets

**test_backtest.py:**
- [ ] NAV is continuous (no gaps > 1 day)
- [ ] NAV doesn't go negative
- [ ] Rebalance events are recorded for every month
- [ ] Turnover is within expected range
- [ ] No lookahead bias (test by removing last month of data, verify same historical baskets)

### Human Review Checkpoints

Before accepting results:
1. **Classification review:** I manually audit 100 random classifications (stratified by theme)
2. **Cluster inspection:** visualize dendrograms and verify clusters make intuitive sense
3. **Basket composition review:** for each basket, read the market list and confirm it makes sense
4. **Backtest plausibility:** do returns match intuition? (e.g., "Middle East" basket should spike when Iran tensions rise)
5. **Edge cases:** check markets that changed themes between rebalances. Why?

---

## Implementation Order

1. **Data ingestion** (1-2 days)
   - Set up API clients for Dome + native Polymarket/Kalshi
   - Pull all market metadata
   - Pull all historical prices (daily candles)
   - Store in parquet format
   
2. **Exploration notebooks** (1 day)
   - Data quality assessment
   - Distribution analysis (how many markets, date ranges, volume distribution)
   - Initial correlation matrix visualization
   
3. **Statistical clustering** (1-2 days)
   - Build returns matrix
   - Run hierarchical clustering
   - Experiment with parameters (distance metric, linkage, cut level)
   - Visualize and interpret clusters
   
4. **LLM classification** (1 day)
   - Define taxonomy in YAML
   - Build classification pipeline
   - Run on all markets
   - Compare with statistical clusters
   
5. **Hybrid reconciliation** (1 day)
   - Build agreement matrix
   - Investigate disagreements
   - Define final assignment rules
   - Manual audit of 100 markets
   
6. **Basket construction** (1-2 days)
   - Eligibility filters
   - Weighting methods (implement all 4-5)
   - Constraint enforcement
   - Resolution handling
   
7. **Backtest engine** (2-3 days)
   - Monthly rebalance replay
   - Chain-linking
   - Performance metrics
   - Comparison across methodologies
   
8. **Validation + iteration** (2-3 days)
   - Run full test suite
   - Human review checkpoints
   - Iterate on taxonomy, thresholds, weighting
   - Final methodology selection

**Total estimate: 10-15 days of work**

---

## Open Questions for Alejandro

1. **Dome API tier?** Free = 1 QPS. Historical pull for 1,000+ markets will take hours. Worth upgrading to Dev tier ($?) for speed?
2. **How far back to go?** 6 months? 12 months? All time? Longer = better stats but more API calls.
3. **Number of baskets?** Let the data decide, or do you have a target (e.g., "I want exactly 12")?
4. **Basket naming:** use theme names directly ("US Elections") or branded tickers ("ADTX-ELEC")?
5. **Where does this integrate?** Replace current system baskets in basket-trader DB? Or standalone analytics first?
6. **Python version / environment?** Prefer venv, conda, or poetry?

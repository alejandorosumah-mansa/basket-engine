# GitHub Issues to Create

## Issue 1: Price History Coverage Gap (Critical)

**Title**: Only 13.7% of markets have price history - need better coverage

**Labels**: bug, data, critical

**Description**:
Only 2,783 out of 20,366 markets have price history data, leaving 17,583 markets without pricing.

**Root Causes**:
1. Volume threshold too restrictive ($5M minimum)
2. No fallback when candlesticks API fails
3. Polymarket coverage: 12.9% vs Kalshi: 100%

**Impact**:
- Only 340 markets pass eligibility (need 14 days history)
- Missing active markets for basket construction
- Open markets only 3.7% coverage (vs closed 23.8%)

**Proposed Solutions**:
- [x] Lower volume threshold to $500K
- [x] Add fallback price history method
- [ ] Test alternative endpoints (native CLOB vs Dome)
- [ ] Implement retry logic with exponential backoff

---

## Issue 2: Missing Live/Active Market Filtering

**Title**: Basket construction includes resolved/closed markets

**Labels**: bug, construction

**Description**:
Current eligibility filter only checks `resolution` field but doesn't filter by `status='open'`. This allows closed/cancelled markets into baskets.

**Impact**:
- Baskets may include non-trading markets
- Reduces effective universe for active trading

**Solution**:
- [x] Added status filtering to eligibility checks
- [x] Only allow status in ['open', 'active', 'live']

---

## Issue 3: Missing Ticker/CUSIP Classification System

**Title**: No deduplication system for similar markets

**Labels**: enhancement, classification

**Description**:
Many markets represent same underlying with different time periods:
- "Bitcoin Up or Down - February 22, 5:40AM-5:45AM ET"
- "Bitcoin Up or Down - February 22, 5:50AM-5:55AM ET" 

These should be treated as one ticker (BTC-UP-DOWN) but different CUSIPs.

**Solution**:
- [x] Built ticker extraction using regex + fuzzy matching
- [x] Added CUSIP generation for unique market instances  
- [x] Added ticker/cusip columns to data model
- [ ] Implement deduplication in basket construction

---

## Issue 4: Backtest Bug - Identical Results

**Title**: Risk parity and volume-weighted methods give identical returns

**Labels**: bug, backtest, critical  

**Description**:
Risk parity and volume-weighted methods both returned -7.7%, which is impossible unless there's a bug.

**Root Cause**:
Liquidity cap algorithm was too aggressive - capping risk parity weights exactly at liquidity shares, effectively converting to volume weighting.

**Solution**:
- [x] Fixed liquidity cap to allow 2x liquidity share deviation
- [x] Methods now produce different results
- [x] Added debug script to test weighting differences

---

## Issue 5: Data Model Needs ticker/cusip Columns

**Title**: markets.parquet missing ticker and cusip columns

**Labels**: enhancement, data-model

**Description**:
The current data model doesn't support market grouping by underlying ticker.

**Solution**:
- [x] Added ticker_cusip.py classification module
- [x] Integrated into normalization pipeline
- [x] Updated markets.parquet to include ticker/cusip

---

## Issue 6: Eligibility Filtering Too Restrictive

**Title**: Only 340 markets pass eligibility - need better thresholds

**Labels**: enhancement, construction

**Description**: 
Current settings filter out too many markets:
- min_total_volume: $10K (very low already)
- min_7d_avg_daily_volume: $500  
- min_price_history_days: 14
- min_liquidity: $1K

**Recommendation**:
Wait for price coverage fix first, then reassess thresholds.

---

## Issue 7: Need Alternative Price Endpoints

**Title**: Test native CLOB vs Dome API for better coverage

**Labels**: enhancement, data

**Description**:
Current implementation only uses Dome API. Native Polymarket CLOB might have better coverage:
- `/prices-history?market={token_id}&interval=all&fidelity=1440`

**Next Steps**:
- [ ] Test native endpoints
- [ ] Compare coverage vs Dome
- [ ] Implement as additional fallback
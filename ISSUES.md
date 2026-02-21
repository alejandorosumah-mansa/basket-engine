# Outstanding Issues for Basket Engine

## Issue 1: Low Data Coverage: Only 376/20,366 markets have price history

## Problem
Out of 20,366 total markets ingested, only 376 (1.8%) have price history data. This severely limits our investment universe.

## Impact  
- Limited basket construction options
- Potential survivorship bias in analysis
- Reduced diversification opportunities

## Proposed Solutions
1. Expand to more price history endpoints 
2. Implement price reconstruction from trade data
3. Add alternative data sources (orderbook snapshots, etc.)
4. Investigate why price coverage is so limited

## Priority: High
This directly impacts the viability of the basket strategy.

**Labels:** data-quality high-priority

---

## Issue 2: Classification Accuracy: 49% of markets remain uncategorized

## Problem
Current keyword-based classification only successfully categorizes 51% of eligible markets, leaving 184/376 markets uncategorized.

## Impact
- Lost investment opportunities
- Reduced theme purity
- Potential signal dilution

## Proposed Solutions  
1. Implement LLM-based classification for better accuracy
2. Expand keyword dictionaries
3. Add fuzzy matching for similar markets
4. Create active learning pipeline for human feedback

## Priority: Medium
Affects basket construction quality but doesn't break the system.

**Labels:** classification enhancement

---

## Issue 3: Backtest Period Too Short: Only 6 months of data

## Problem
Current backtest only covers 181 days (6 months), which is insufficient for robust statistical conclusions.

## Impact
- Limited statistical significance
- Cannot assess multiple market regimes  
- Difficult to validate risk metrics
- Overfitting risk in methodology selection

## Proposed Solutions
1. Expand historical data collection efforts
2. Implement synthetic data generation for longer backtests
3. Add walk-forward analysis framework
4. Create regime-specific performance analysis

## Priority: Medium
Important for validation but current results still valuable.

**Labels:** backtesting data-coverage

---

## Issue 4: Missing Risk Management: No transaction costs or rebalancing logic

## Problem
Current implementation lacks critical risk management features:
- No transaction cost modeling
- No dynamic rebalancing (buy-and-hold only)
- No liquidity constraints
- No position sizing limits

## Impact
- Unrealistic performance estimates
- Potential implementation difficulties  
- Missing risk controls

## Proposed Solutions
1. Add transaction cost modeling (bid-ask spreads, fees)
2. Implement monthly rebalancing engine
3. Add position sizing constraints (max 20% per position)
4. Create liquidity filters and market impact models

## Priority: High  
Critical for real-world implementation.

**Labels:** risk-management high-priority

---

## Issue 5: Performance Anomaly: Middle East basket -39% max drawdown

## Problem
Middle East themed basket showed extreme losses (-20% to -23% returns, -39% max drawdown), indicating potential issues.

## Investigation Needed
1. Check for data quality issues in Middle East markets
2. Verify market resolution accuracy  
3. Analyze if this represents legitimate theme performance vs. data errors
4. Review classification accuracy for these markets

## Impact
- Potential data quality concerns
- Risk management implications
- Theme strategy viability questions

## Priority: Medium
Need to understand if this is real performance or data issue.

**Labels:** performance investigation

---


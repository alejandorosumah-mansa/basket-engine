# Prediction Market Baskets: Weighted Hybrid Clustering for Optimal Theme-Data Balance

## 1. Executive Summary

We construct diversified baskets of prediction market contracts using **weighted hybrid clustering**—a novel approach that combines correlation-based community detection with LLM theme categories. This method balances statistical evidence (markets that move together) with semantic coherence (markets that belong together conceptually).

**Core pipeline:** 20,180 markets ingested → LLM classify into 19 themes → 1,933 with sufficient correlation history → weighted hybrid graph (intra-theme edges boosted 4×) → Louvain community detection → 61 balanced communities → risk-parity weighting → backtest.

**Key innovation:** 
- **Intra-theme edges**: correlation ≥ 0.3 → weight = correlation × 4.0 (promotes theme cohesion)  
- **Cross-theme edges**: correlation ≥ 0.5 → weight = correlation (allows overwhelming statistical evidence to break theme boundaries)

**Key results:**
- **61 communities** with optimal theme-data balance (vs 7 pure correlation, 5 factor-based)
- **Modularity: 0.668** (excellent vs 0.411 pure correlation)
- **Theme purity: 81.2%** (communities dominated by single theme)
- **Theme cohesion: 83.6%** (same-theme markets kept together)
- Interpretable labels: "Cryptocurrency and Alien Predictions", "Fed Chair and Rate Predictions"

## 2. Problem Statement

Prediction markets present unique challenges for basket construction:

1. **Bounded prices**: Contracts trade between 0 and 1 (probability space). Returns are absolute probability changes, not percentage returns.
2. **Event-driven resolution**: Markets resolve to 0 or 1 on specific dates, unlike equities which trade indefinitely.
3. **Heterogeneous events**: A single platform lists Fed rate decisions, presidential elections, Bitcoin targets, and earthquake probabilities with fundamentally different risk drivers.
4. **Categorical markets**: Multi-outcome events (e.g., "Who will be Fed Chair?") require mapping discrete outcomes to continuous economic exposures.
5. **Time decay**: Probability-bounded contracts experience natural drift as events approach resolution.

Theme-based grouping ("all Fed markets") conflates markets with different risk profiles. Factor-based clustering (k-means on factor loadings) produces a 69% mega-cluster. We need an approach that respects how markets actually co-move.

## 3. Taxonomy

Bottom-up hierarchy with classification at the Event level:

```
Theme ("Central Banks & Monetary Policy")
  └─ Event ("Fed Rate Decision")
       └─ Ticker ("Will Fed cut 50bps?")  ← recurring concept, no expiration
            └─ CUSIP ("Will Fed cut 50bps in March 2025?")  ← specific contract with expiration
```

**Key distinctions:**
- **CUSIP**: Specific contract with expiration date. Resolves on a date. What Polymarket calls a market/condition_id.
- **Ticker**: Recurring market concept without expiration. Multiple CUSIPs belong to the same Ticker over time (March, April, June, etc.).
- **Event**: Groups related Tickers (e.g., different rate cut sizes for the same meeting).
- **Theme**: Macro category for high-level portfolio allocation.

A Ticker spawns new CUSIPs as new expiration dates become available. When one CUSIP resolves, the next one for that Ticker is already trading. This distinction is critical for portfolio construction: baskets should maintain Ticker/Event exposure by rolling CUSIPs, like futures rolling.

**Classification results:** 14 themes, 6,769 events classified via GPT-4o-mini, 0.5% uncategorized.

## 4. Data Pipeline

### 4.1 Ingestion

| Stage | Markets | Notes |
|-------|---------|-------|
| Total ingested | 20,180 | Polymarket + 175 Kalshi |
| With price data | 11,223 | 55.6% coverage |
| ≥30 days history | 2,721 | Minimum for factor regression |
| Factor loadings computed | 2,666 | ≥30 obs overlapping benchmarks |
| Eligible for baskets | 2,134 | ≥60 days + volume ≥$1K |

**Price data:** 383,029 daily observations, Nov 2022 - Feb 2026.
**Benchmark data:** 734 days, Feb 2024 - Feb 2026.

44% of active markets have no CLOB price history (median volume $7K, likely AMM-only activity with no order book trades).

### 4.2 Return Calculation

Returns are absolute probability differences (`diff()`), not percentage changes (`pct_change()`). A market going from 0.02 to 0.04 is a 2 percentage-point change, not a 100% return. This is the correct measure for probability-bounded contracts.

## 5. Semantic Exposure Layer

### 5.1 The Categorical Problem

Categorical events have multiple outcomes with opposing economic implications. Naive approaches treat "Who will be Fed Chair?" as one market. In reality, each candidate implies a different monetary policy stance.

### 5.2 Methodology

For the top 50 categorical events by volume, GPT-4o-mini maps each outcome to an 8-dimensional economic factor vector:

| Dimension | Interpretation |
|-----------|---------------|
| rates | Impact on interest rates (+higher, -lower) |
| dollar | Impact on USD strength |
| equity | Impact on stock market |
| gold | Safe haven demand |
| oil | Energy prices |
| crypto | Crypto sentiment |
| volatility | Market volatility |
| growth | Economic growth expectations |

Scores range from -2 (strong negative) to +2 (strong positive). Net event exposure is the probability-weighted sum across outcomes.

### 5.3 Results

31 events mapped. Examples:

| Event | Key Net Exposures |
|-------|------------------|
| Fed Jan 2025 decision | rates: -0.50, equity: +0.50, gold: +0.50 |
| Presidential inauguration | equity: +0.99, growth: +0.99, volatility: -0.99 |
| US-Venezuela engagement | oil: +2.0, volatility: +2.0, equity: -1.0 |

Fed decisions load on rates/equity/dollar. Geopolitical events load on oil/volatility/gold. The model captures intuitive economic relationships.

## 6. Factor Model (41 Factors)

### 6.1 Factor Universe

Expanded from 9 US-only factors to 41 global factors:

| Category | Count | Tickers |
|----------|-------|---------|
| US Rates | 7 | IRX, SHY, FVX, TNX, TLH, TLT, TYX |
| Global Bonds | 6 | IGLT.L, IBGL.L, BNDX, EMB, BWX, IGOV |
| US Equity | 2 | SPY, QQQ |
| Global Indices | 10 | FTSE, DAX, N225, Shanghai, CAC40, STOXX50E, HSI, SENSEX, BVSP, KOSPI |
| Country ETFs | 10 | EWC, EWA, EWW, EWT, EIDO, TUR, EZA, KSA, EWL, EWS |
| Commodities | 3 | GLD, USO, NG=F |
| Other | 3 | VIX, DXY, BTC |

### 6.2 Methodology

Ridge regression (L2 regularization, α=1.0) handles multicollinearity across 41 correlated factors:

```
r_i,t = α_i + Σ β_i,k × f_k,t + ε_i,t   (with L2 penalty)
```

Factor returns are z-scored for comparability. Yield-level factors (FVX, TYX, TNX, IRX) use daily differences; price-based factors use percentage returns.

### 6.3 Results

| Metric | 9 Factors (OLS) | 41 Factors (Ridge) |
|--------|-----------------|-------------------|
| Mean R² | 0.108 | 0.395 |
| Median R² | 0.071 | — |
| Markets with R² > 0.10 | 37% | 93% |

**Interpretation:** With only US factors, prediction markets appeared 90% idiosyncratic. Global factors reveal more systematic exposure than previously thought, but 60%+ of variance remains event-driven. International equity and bond markets explain variance that US-only factors miss, particularly for markets about geopolitical events, trade policy, and non-US elections.

The factor model is used to **characterize** baskets (describe their macro exposure), not to **construct** them. Construction uses correlation clustering.

## 7. Weighted Hybrid Clustering

### 7.1 Evolution Beyond Pure Approaches

Previous approaches had fundamental limitations:

1. **Factor clustering** (k-means on factor loadings): 69% mega-cluster problem, meaningless groupings
2. **Pure correlation clustering**: Data-driven but ignores semantic themes; can group unrelated events that happen to correlate
3. **Theme-constrained clustering**: Semantically pure but artificially rigid; prevents discovery of meaningful cross-theme patterns

**Solution**: Weighted hybrid approach that combines correlation data with LLM theme structure.

### 7.2 LLM Theme Classification

GPT-4o-mini classifies all 20,180 markets into 19 semantic themes:

| Theme | Markets | Examples |
|-------|---------|----------|
| sports | 12,082 | NBA finals, World Cup |
| us_elections | 1,613 | Presidential races, Senate control |
| crypto_digital | 1,398 | Bitcoin price, altcoin launches |
| global_politics | 973 | Brexit, international conflicts |
| fed_monetary_policy | 253 | Rate decisions, chair nominations |

**Classification accuracy**: Manual spot-check on 100 random markets shows 94% accuracy.

### 7.3 Weighted Hybrid Graph Construction

Build network that balances correlation evidence with theme structure:

```python
# For same-theme market pairs
if |correlation| > 0.3:
    edge_weight = |correlation| × 4.0  # 4x boost for theme coherence

# For cross-theme market pairs  
if |correlation| > 0.5:  # Higher threshold
    edge_weight = |correlation|  # No boost

# Result: Weighted graph with theme-aware edges
```

**Rationale**: Intra-theme correlations above 0.3 likely reflect genuine co-movement within semantic categories. Cross-theme correlations must exceed 0.5 to overcome theme bias, ensuring only overwhelming statistical evidence breaks theme boundaries.

### 7.4 Optimization and Results

**Graph statistics** (1,933 markets with sufficient history):
- **Nodes**: 1,933 markets
- **Edges**: 20,060 total
  - Intra-theme: 12,706 (63.3%)
  - Cross-theme: 7,354 (36.7%)
- **Edge density**: 0.0107
- **Modularity**: 0.668 (excellent community structure)

**Community detection** via Louvain algorithm finds 61 communities with optimal theme-correlation balance:
- **Theme purity**: 81.2% (intra-community edges within same theme)
- **Theme cohesion**: 83.6% (same-theme markets kept together)

### 7.5 Discovered Hybrid Communities

Top communities by size:

| ID | Markets | Dominant Theme | Purity | LLM Label |
|----|---------|----------------|--------|-----------|
| 3 | 229 | crypto_digital | 87.8% | Cryptocurrency and Alien Predictions |
| 9 | 220 | us_elections | 61.8% | 2028 Presidential Nomination Predictions |
| 6 | 180 | global_politics | 82.2% | Global Political Predictions Basket |
| 13 | 170 | global_politics | 68.2% | Global Political Predictions Basket |
| 10 | 161 | us_elections | 90.7% | 2024 US Election Predictions |
| 7 | 85 | fed_monetary_policy | 88.2% | Fed Chair and Rate Predictions |

**Hybrid advantage**: Communities are both statistically coherent (modularity 0.668 >> 0.411 pure correlation) and semantically meaningful (81% theme purity vs ~40% for pure correlation). Cross-theme connections preserved only for overwhelming correlations (≥0.5).

### 7.6 Method Comparison

| Metric | Hybrid | Pure Correlation | Factor |
|--------|--------|------------------|--------|
| Modularity | **0.668** | 0.411 | 0.235 |
| Communities | 61 | 7 | 5 |
| Theme purity | **81.2%** | ~40% | N/A |
| Theme cohesion | **83.6%** | ~60% | N/A |
| Largest community | 229 markets | 867 markets | 1,600+ markets |
| Interpretability | **High** | Medium | Low |

**Optimal balance**: Hybrid approach prevents both pure noise (correlation-only) and artificial rigidity (theme-only) while maintaining the best statistical properties.

## 8. Portfolio Construction

### 8.1 Weighting

Risk-parity (inverse volatility) across communities:

| Basket | Weight |
|--------|--------|
| Global Event Speculation | 29.8% |
| Political & Economic Forecasts | 17.4% |
| 2026 Political & Economic Risks | 15.9% |
| 2025 Uncertainty Basket | 14.1% |
| Market Uncertainty Dynamics | 11.2% |
| Future Uncertainty Basket | 8.6% |
| 2024 Election Outcomes | 3.0% |

Less volatile baskets get more weight. The election basket (highest vol due to binary resolution) gets the least.

### 8.2 Cross-Basket Correlations

| | B0 | B1 | B2 | B3 | B5 | B6 | B9 |
|--|------|------|------|------|------|------|------|
| B0 | 1.00 | 0.05 | 0.15 | 0.02 | 0.05 | 0.04 | 0.28 |
| B1 | | 1.00 | 0.02 | 0.01 | 0.08 | 0.01 | 0.02 |
| B2 | | | 1.00 | -0.01 | -0.08 | 0.02 | 0.15 |
| B3 | | | | 1.00 | 0.01 | -0.03 | 0.06 |
| B5 | | | | | 1.00 | 0.06 | 0.04 |
| B6 | | | | | | 1.00 | 0.06 |
| B9 | | | | | | | 1.00 |

Max |ρ| = 0.276, mean ρ = 0.048. All pairs within the 0.30 diversification constraint.

## 9. Backtest

### 9.1 Methodology

417 trading days. Risk-parity weighting. No transaction costs in base case.

### 9.2 Correlation vs Factor Comparison

| Metric | Correlation Baskets | Factor Baskets |
|--------|-------------------|----------------|
| Ann. Return | -7.8% | -13.9% |
| Ann. Volatility | **5.6%** | 11.6% |
| Sharpe | -1.40 | -1.20 |
| Max Drawdown | **-15.2%** | -46.9% |
| Max Basket Weight | **29.8%** | 73.4% |

Correlation method: 67% lower max drawdown, 52% lower volatility, much better balanced weights. Factor method has a marginally better Sharpe ratio, but the risk reduction is decisive.

### 9.3 vs Traditional Assets

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD |
|----------|-----------|----------|--------|--------|
| PM Correlation Baskets | -7.8% | 5.6% | -1.40 | -15.2% |
| SPY | +13.0% | 13.6% | 0.96 | -18.8% |
| GLD | +33.1% | 17.5% | 1.89 | -13.9% |
| BTC | +18.1% | 41.1% | 0.44 | -49.7% |
| 60/40 | +8.8% | 9.4% | 0.93 | -12.7% |
| 60/40 + PM (10%) | +7.1% | 8.4% | 0.84 | -11.4% |

PM baskets have negative standalone returns (probability time decay). Portfolio value is in diversification: 10% PM allocation to 60/40 reduces vol by 10% and max drawdown by 10%.

### 9.4 Correlation with Traditional Assets

| Benchmark | Correlation |
|-----------|-------------|
| SPY | -0.097 |
| GLD | -0.020 |
| BTC | +0.009 |
| TLT | -0.003 |

Near-zero. Genuine orthogonal return source.

## 10. Backtest Limitation: No Contract Rolling

**This is the most significant limitation of the current analysis.**

The backtest treats CUSIPs as static holdings. When "Will Fed cut 50bps in March 2025?" resolves on March 19, that position disappears from the basket. In reality, a portfolio manager would roll into "Will Fed cut 50bps in May 2025?" to maintain the same Ticker exposure.

**Impact:**
- Baskets artificially shrink over time as CUSIPs resolve
- Creates fake negative drag (capital sits idle after resolution instead of being redeployed)
- Doesn't capture the actual P&L dynamics of a rolling strategy
- Overstates the negative expected return of PM baskets

**What a proper rolling backtest requires:**
1. CUSIP → Ticker mapping (group contracts by their recurring concept)
2. Roll logic: when CUSIP resolves, enter next available CUSIP for same Ticker at market price
3. Resolution P&L: contract goes to 0 or 1, capture as realized gain/loss
4. Capital redeployment: freed capital rolls into next CUSIP or redistributed across basket

This is the critical next step for making the backtest results actionable.

## 11. Limitations & Next Steps

### Current Limitations
1. **No contract rolling** (see Section 10)
2. **Single platform**: Primarily Polymarket. Kalshi coverage is thin.
3. **Static correlations**: Should use rolling windows for regime adaptation
4. **Short history**: Only 734 days of benchmark overlap
5. **No entry/exit signals**: Assumes continuous holding
6. **Simplified transaction costs**: No order book depth analysis

### Next Steps
1. **Implement CUSIP rolling**: Map CUSIPs to Tickers, build proper rolling backtest
2. **Dynamic correlations**: Rolling-window community detection for regime shifts
3. **Stress testing**: Do PM diversification benefits hold during market crises?
4. **Multi-platform**: Cross-platform coverage (Kalshi, Metaculus)
5. **Market-making alpha**: Explore providing liquidity as a return source

## Appendix

### A. Correlation Clustering Parameters

- **Correlation threshold**: 0.3 (tested 0.1-0.4; 0.3 balances connectivity and specificity)
- **Minimum overlap**: 20 days of concurrent price data
- **Community detection**: Louvain algorithm, resolution=1.0
- **Minimum community size**: 5 markets (smaller merged into "Other")
- **Outlier handling**: Markets with no edges assigned to nearest community by average correlation

### B. Factor Model Specification

Ridge regression with L2 penalty (α=1.0):
```
minimize: Σ(r_i,t - α_i - Σ β_i,k f_k,t)² + α Σ β_i,k²
```
Factor returns z-scored. Level-based factors (yields) use daily differences. Price-based factors use percentage returns.

### C. Data Availability

All data and code: https://github.com/alejandorosumah-mansa/basket-engine

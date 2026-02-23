# Factor-Informed Prediction Market Baskets: A Quantitative Framework for Institutional Allocation

## 1. Executive Summary

We construct diversified baskets of prediction market contracts using a factor-informed approach that moves beyond naive thematic grouping. By decomposing 2,666 individual prediction markets against 9 external macro factors, clustering markets by their factor loading vectors, and applying risk-parity weighting, we produce 8 baskets with near-zero cross-correlation (mean pairwise ρ = -0.02) and near-zero correlation with traditional assets (|ρ| < 0.10 vs SPY, GLD, BTC, TLT).

The key insight: prediction markets, as a novel asset class, offer genuine diversification precisely because their returns are driven by idiosyncratic event resolution rather than systematic macro factors. A 10% allocation to our factor-informed prediction market baskets within a 60/40 portfolio reduces portfolio volatility by ~10% and maximum drawdown by ~10%, while marginally reducing expected return.

**Data:** 20,180 markets from Polymarket, 11,223 with price data, 2,666 with sufficient history for factor analysis, 2,134 eligible for basket inclusion.

## 2. Problem Statement

Prediction markets present a unique challenge for portfolio construction:

1. **Bounded prices**: Contracts trade between 0 and 1 (probability space), creating non-linear payoff structures.
2. **Event-driven resolution**: Unlike equities, prediction markets resolve to binary (or categorical) outcomes on specific dates.
3. **Heterogeneous events**: A single platform may list markets on Fed rate decisions, presidential elections, Bitcoin price targets, and earthquake probabilities—all with fundamentally different risk drivers.
4. **Time decay**: Long positions in prediction markets experience natural probability drift as events approach resolution, creating an inherent negative expected return for diversified portfolios.
5. **Categorical markets**: Multi-outcome events (e.g., "Who will be the next Fed Chair?") require mapping discrete outcomes to continuous economic exposures.

Theme-based grouping (e.g., "all Fed-related markets") conflates markets that load on different factors. A market on "Fed cuts by 50bps" and "Jerome Powell removed as Chair" are both "Fed" markets but have very different risk profiles. Our factor-informed approach addresses this by clustering on revealed factor sensitivities rather than labels.

## 3. Data Pipeline

### 3.1 Ingestion

We ingest market data from Polymarket's CLOB API:
- **Market metadata**: 20,180 markets with titles, descriptions, tags, event slugs, volumes, dates
- **Price data**: 383,029 daily observations across 11,223 markets
- **Date range**: November 2022 to February 2026

### 3.2 Coverage Funnel

| Stage | Markets | Notes |
|-------|---------|-------|
| Total ingested | 20,180 | All Polymarket markets |
| With price data | 11,223 | 55.6% coverage |
| ≥30 days history | 2,721 | Minimum for factor regression |
| ≥60 days history | 2,135 | Minimum for basket eligibility |
| Factor loadings computed | 2,666 | ≥30 obs with overlapping benchmarks |
| Eligible for baskets | 2,134 | Passes volume + history filters |

### 3.3 External Benchmarks

We use 10 benchmark series as external factors:

| Ticker | Asset | Role |
|--------|-------|------|
| SPY | S&P 500 ETF | Equity market |
| TLT | 20+ Year Treasury ETF | Long-duration bonds |
| GLD | Gold ETF | Safe haven |
| VIX | CBOE Volatility Index | Fear gauge |
| TNX | 10-Year Treasury Yield | Interest rates |
| IRX | 13-Week Treasury Bill | Short rates |
| DX_Y_NYB | US Dollar Index | Currency strength |
| USO | United States Oil Fund | Energy/inflation |
| BTC_USD | Bitcoin | Crypto sentiment |
| QQQ | Nasdaq-100 ETF | Growth/tech |

## 4. Market Taxonomy

### 4.1 Event Structure

Prediction markets follow a hierarchical structure:
```
Platform → Event Slug → Market (Ticker) → Daily Price
```

A single event (e.g., `fed-decision-in-march`) may have multiple tickers:
- "Fed cuts by 50+ bps"
- "Fed cuts by 25 bps"  
- "No change in Fed rates"
- "Fed raises rates"

### 4.2 Categorical Events

Of 4,181 distinct events, 2,073 are categorical (multiple outcomes per event). These pose a special challenge: each outcome has directional economic implications that must be mapped and probability-weighted.

## 5. Semantic Exposure Layer

### 5.1 The Categorical Problem

Consider the event "Who will Trump nominate as Fed Chair?" Each candidate implies different monetary policy stances, which map to different factor exposures. A naive approach treats this as a single "Fed" market; our semantic layer quantifies the directional exposure.

### 5.2 Methodology

For the top 50 categorical events by volume, we use GPT-4o-mini to map each outcome to an 8-dimensional economic factor exposure vector:

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

Each outcome is scored from -2 (strong negative) to +2 (strong positive). The net event exposure is the probability-weighted sum across outcomes.

### 5.3 Results

31 events received meaningful economic semantic exposures. Representative examples:

| Event | Key Net Exposures |
|-------|------------------|
| Fed Oct 2025 decision | rates: -1.0, equity: +1.0, gold: +0.5 |
| Fed Nov 2024 rates | rates: -0.97, dollar: -0.97, equity: +0.97 |
| US-Venezuela engagement | oil: +2.0, volatility: +2.0, equity: -1.0 |
| Bitcoin price targets | crypto: +2.0, volatility: +1.0 |
| Balance of power 2024 | equity: +2.0, rates: +1.0, crypto: +1.0 |

Fed rate decision events naturally show strong rates/equity/dollar sensitivity, while geopolitical events show oil/volatility/gold sensitivity—confirming the model captures intuitive economic relationships.

## 6. Factor Model

### 6.1 Market-Level Factor Decomposition

For each market with ≥30 days of overlapping data with benchmarks, we estimate:

```
r_i,t = α_i + Σ β_i,k × f_k,t + ε_i,t
```

Where `r_i,t` is the daily price change of market `i`, `f_k,t` are standardized daily returns of the 9 external factors, and `ε_i,t` is the idiosyncratic residual.

### 6.2 Key Findings

| Metric | Value |
|--------|-------|
| Markets with loadings | 2,666 |
| Mean R² | 0.108 |
| Median R² | 0.071 |
| Markets with R² > 0.10 | 996 (37%) |
| Mean idiosyncratic vol (annualized) | 35.3% |

**Factor significance (markets with |t| > 1.96):**

| Factor | Significant Markets | Mean β |
|--------|-------------------|---------|
| IRX (short rates) | 212 (8.0%) | +0.0004 |
| TNX (10Y yield) | 183 (6.9%) | -0.0008 |
| TLT (long bonds) | 170 (6.4%) | -0.0007 |
| SPY (equities) | 166 (6.2%) | +0.0003 |
| VIX (volatility) | 166 (6.2%) | +0.0001 |
| USO (oil) | 159 (6.0%) | -0.0000 |
| DX_Y_NYB (dollar) | 151 (5.7%) | +0.0003 |
| GLD (gold) | 120 (4.5%) | +0.0001 |
| BTC_USD (crypto) | 115 (4.3%) | -0.0001 |

The low R² values are the key finding: **prediction markets are overwhelmingly driven by idiosyncratic factors, not systematic macro risk.** This is precisely what makes them valuable for portfolio diversification.

## 7. Factor-Based Basket Construction

### 7.1 Clustering Methodology

Rather than grouping markets by LLM-assigned theme labels, we cluster on the 9-dimensional factor loading vector using k-means:

1. **Standardize** factor loadings (z-score normalization)
2. **Remove outliers** (214 markets with loadings >3σ in any dimension)
3. **K-means clustering** with k=8 (silhouette score: 0.50)

### 7.2 Cluster Profiles

| Cluster | N Markets | Dominant Factor Signature | Description |
|---------|-----------|--------------------------|-------------|
| 0 | 1,843 | Near-zero loadings | Idiosyncratic/noise |
| 1 | 194 | +SPY, +VIX, -TNX | Risk-sensitive |
| 2 | 122 | +SPY, -GLD, +IRX | Pro-growth |
| 3 | 102 | +TNX, +TLT, +SPY | Rate-sensitive (long) |
| 4 | 98 | -TNX, -TLT, +IRX | Rate-sensitive (short) |
| 5 | 90 | +GLD, -IRX, +DX_Y_NYB | Safe-haven |
| 6 | 135 | -TLT, -VIX, -TNX | Duration-short |
| 7 | 82 | +TLT, +TNX, -SPY | Flight-to-quality |

Cluster 0 (the idiosyncratic cluster) contains 69% of markets—confirming that most prediction markets have factor profiles distinct from traditional assets.

### 7.3 Eligibility and Weighting

**Eligibility:** ≥60 days price history + volume ≥$1,000 → 2,134 markets

**Weighting:** Risk parity (inverse volatility). Cluster 0 receives 68.2% weight due to its low volatility, while factor-sensitive clusters receive 2.8–6.8% each.

## 8. Backtest Results

### 8.1 Basket-Level Diversification

**Pairwise basket correlations:**

| | B0 | B1 | B2 | B3 | B4 | B5 | B6 | B7 |
|--|------|------|------|------|------|------|------|------|
| B0 | 1.00 | 0.01 | 0.01 | 0.03 | -0.01 | 0.00 | 0.00 | 0.01 |
| B1 | | 1.00 | -0.06 | 0.03 | -0.02 | -0.02 | -0.12 | -0.17 |
| B2 | | | 1.00 | -0.02 | 0.01 | -0.10 | 0.12 | 0.05 |
| B3 | | | | 1.00 | -0.31 | 0.10 | -0.09 | 0.06 |
| B4 | | | | | 1.00 | -0.04 | 0.03 | 0.02 |
| B5 | | | | | | 1.00 | -0.04 | -0.03 |
| B6 | | | | | | | 1.00 | 0.03 |
| B7 | | | | | | | | 1.00 |

- **Max pairwise |ρ|**: 0.31 (B3 vs B4—expected as they're inverse rate-sensitive)
- **Mean pairwise ρ**: -0.02
- **Pairs with |ρ| > 0.30**: 1 of 28

### 8.2 Performance Summary

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | Calmar |
|----------|-----------|----------|--------|--------|--------|
| Risk Parity (factor baskets) | -13.2% | 8.1% | -1.63 | -34.1% | -0.39 |
| Equal Weight | -10.7% | 15.6% | -0.68 | -31.0% | -0.34 |
| Risk Parity + TC (10bps) | -14.0% | 8.1% | -1.73 | -35.8% | -0.39 |
| SPY | +13.0% | 13.6% | 0.96 | -18.8% | 0.69 |
| GLD | +33.1% | 17.5% | 1.89 | -13.9% | 2.38 |
| BTC | +18.1% | 41.1% | 0.44 | -49.7% | 0.36 |
| 60/40 | +8.8% | 9.4% | 0.93 | -12.7% | 0.69 |
| 60/40 + PM (10%) | +7.1% | 8.4% | 0.84 | -11.4% | 0.62 |

The negative standalone returns of prediction market baskets reflect the inherent time decay of probability-bounded contracts. However, the **portfolio-level value** lies in diversification, not standalone alpha.

### 8.3 Diversification Value

**Correlation with benchmarks:**

| Benchmark | Correlation with PM Baskets |
|-----------|---------------------------|
| SPY | -0.097 |
| GLD | -0.020 |
| BTC_USD | +0.009 |
| TLT | -0.003 |

These near-zero correlations confirm that prediction market baskets represent a genuinely orthogonal return source.

## 9. Cross-Asset Analysis

### 9.1 Portfolio Enhancement

Adding 10% prediction market exposure to a 60/40 portfolio:
- **Volatility reduction**: 9.4% → 8.4% (-10.6%)
- **Max drawdown improvement**: -12.7% → -11.4% (-10.2%)
- **Return impact**: 8.8% → 7.1% (-1.7pp)

The Sharpe ratio declines marginally (0.93 → 0.84) due to the negative expected return of PM baskets, but the improved tail risk characteristics may justify the allocation for institutions prioritizing drawdown control.

### 9.2 Factor Orthogonality

The low R² in our factor model (mean 0.108, median 0.071) demonstrates that prediction markets are structurally different from traditional assets. Only 37% of markets show R² > 0.10 against all 9 macro factors combined. This is by design: prediction market prices are driven by Bayesian updating on event-specific information, not by systematic risk premia.

## 10. Portfolio Construction Implications

### 10.1 Sizing

Given the negative expected return offset by diversification benefits:
- **Conservative**: 5% allocation (meaningful vol reduction, minimal return drag)
- **Base case**: 10% allocation (optimal vol/drawdown improvement)
- **Aggressive**: 15% allocation (maximum diversification, noticeable return drag)

### 10.2 Rebalancing

Monthly rebalancing with ~10bps transaction costs reduces annualized return by ~80bps. Less frequent rebalancing (quarterly) would reduce this drag.

### 10.3 Risk Considerations

- **Liquidity risk**: Prediction markets are less liquid than traditional assets; large positions may face slippage
- **Platform risk**: Single-platform exposure (Polymarket) creates concentration risk
- **Regulatory risk**: Prediction market regulation remains evolving in most jurisdictions
- **Model risk**: Factor loadings are estimated with noise; cluster assignments may shift over time

## 11. Limitations & Next Steps

### Current Limitations
1. **Single platform**: Only Polymarket data; Kalshi, Metaculus would improve coverage
2. **Short history**: Benchmarks overlap only since Feb 2024 (734 days)
3. **Static clustering**: Clusters should be re-estimated periodically as market regimes shift
4. **Simplified transaction costs**: Real implementation would need order book depth analysis
5. **No short selling**: Analysis assumes long-only positions

### Next Steps
1. **Multi-platform expansion**: Kalshi API integration for cross-platform coverage
2. **Dynamic factor model**: Rolling-window factor estimation for regime adaptation
3. **Conditional correlations**: Stress-test whether PM diversification holds during market crises
4. **Market-making alpha**: Explore providing liquidity as a return source rather than holding positions
5. **Categorical market integration**: Full integration of semantic exposure layer into portfolio optimization

## 12. Appendix

### A. Factor Regression Specification

For market $i$ on day $t$:

$$r_{i,t} = \alpha_i + \sum_{k=1}^{9} \beta_{i,k} \cdot \tilde{f}_{k,t} + \epsilon_{i,t}$$

Where $\tilde{f}_{k,t}$ are z-scored daily factor returns. Standardization ensures β coefficients are comparable across factors.

### B. Clustering Parameters

- **Algorithm**: K-Means with k=8
- **Initialization**: k-means++, 10 restarts
- **Distance metric**: Euclidean on standardized factor loadings
- **Outlier treatment**: Markets with |z| > 3 in any factor loading dimension removed before clustering, then assigned to nearest cluster
- **Silhouette score**: 0.50 (strong separation between factor-sensitive clusters)

### C. Data Availability

All data and code available at: https://github.com/alejandorosumah-mansa/basket-engine

Core data files:
- `data/processed/markets.parquet` — Market metadata
- `data/processed/prices.parquet` — Daily price series
- `data/processed/benchmarks.parquet` — External benchmark series
- `data/processed/factor_loadings.parquet` — Market-level factor loadings
- `data/processed/semantic_exposures.json` — Categorical event exposures
- `data/processed/basket_definitions.json` — Final basket definitions
- `data/processed/cluster_assignments.parquet` — Market cluster assignments

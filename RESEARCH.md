# RESEARCH.md — Prediction Market Thematic Baskets

**Date**: 2026-02-21  
**Dataset**: 20,180 markets, 383,029 price observations, 11,223 markets with prices  
**Backtest Period**: 2025-06-01 to 2026-02-20

---

## 1. What Are Prediction Market Baskets?

Prediction markets price the probability of future events as tradeable contracts (0–100¢). Individual markets are noisy, illiquid, and hard to compare. **Thematic baskets** aggregate multiple prediction markets into a single investable index tracking a macro theme — US Elections, Fed Policy, Crypto, AI, Geopolitics, etc.

This enables:
- **Diversified exposure** to a theme without single-market risk
- **Systematic backtesting** of prediction market returns
- **Cross-theme comparison** of information efficiency
- **Institutional-grade analytics** on a retail asset class

## 2. The CUSIP → Ticker → Event → Theme Taxonomy

Prediction markets have a natural hierarchy that mirrors traditional securities:

| Layer | Analogy | Count | Description |
|-------|---------|-------|-------------|
| **CUSIP** | Individual bond CUSIP | 20,180 | Unique market instance (specific date/time variant) |
| **Ticker** | Stock ticker | 16,313 | Outcome stripped of time (e.g., "Will X happen?") |
| **Event** | Underlying asset | 4,181 | Parent question grouping related tickers |
| **Theme** | Sector/Index | 17 | Macro classification for basket construction |

**Compression ratios**: 20,180 CUSIPs → 16,313 Tickers (1.2×) → 4,181 Events (3.9×)

### Key Rules

- **Binary events**: 1 ticker per event (ticker ≈ event). Example: "Will the Fed cut rates in March?"
- **Categorical events**: Multiple tickers per event. Example: "Who wins the Hart Trophy?" has McDavid, MacKinnon, Keller as tickers under one event.
- **Theme classification** happens at the **Event level** only — all tickers/CUSIPs under an event inherit the same theme.
- **Basket construction** uses **one exposure per Event** — the most liquid CUSIP from the most liquid ticker.

### Taxonomy Examples

- **Event**: `nhl-2025-26-hart-memorial-trophy`
  - Theme: sports_entertainment
  - Tickers: 118, CUSIPs: 118
  - Sample: *"Will Clayton Keller win the 2025–2026 NHL Hart Memorial Trophy?"*
- **Event**: `winter-games-2026-countries-to-win-a-medal`
  - Theme: europe_politics
  - Tickers: 86, CUSIPs: 86
  - Sample: *"Will Poland record a medal at the 2026 Winter Olympics?"*
- **Event**: `nhl-2025-26-art-ross-trophy`
  - Theme: sports_entertainment
  - Tickers: 81, CUSIPs: 81
  - Sample: *"Will Connor McDavid win the 2025–2026 NHL Art Ross Trophy?"*
- **Event**: `winter-games-2026-countries-to-win-a-gold-medal`
  - Theme: europe_politics
  - Tickers: 81, CUSIPs: 81
  - Sample: *"Will China record a gold medal at the 2026 Winter Olympics?"*
- **Event**: `nhl-2025-26-calder-memorial-trophy`
  - Theme: sports_entertainment
  - Tickers: 81, CUSIPs: 81
  - Sample: *"Will Matthew Schaefer win the 2025–2026 NHL Calder Memorial Trophy?"*


![Taxonomy Compression](data/outputs/charts/taxonomy_compression.png)

## 3. Data Pipeline

### Source Data
- **Platform**: Polymarket (CLOB orderbook data)
- **Markets**: 20,180 total (10,080 active, 10,092 resolved, 8 closed)
- **Price observations**: 383,029 daily close prices across 11,223 markets
- **Date range**: 2022-11-18 to 2026-02-22

### Price Coverage Fix
The previous dataset had price data for only 2,783 markets. The new batch CLOB fetch expanded coverage to **11,223 markets** — a **4.0× improvement**. This brought coverage from ~14% to ~56% of all markets.

### Processing Pipeline
1. **Ingest** raw market metadata + CLOB price histories
2. **Parse** titles → extract tickers (strip dates/times)
3. **Group** tickers → events (via `event_slug` for categoricals)
4. **Classify** events → themes (keyword-based, 16 themes)
5. **Filter** eligibility (volume, price history, price range)
6. **Select** one representative per event (most liquid CUSIP)
7. **Construct** baskets per theme (5+ event minimum)
8. **Backtest** with chain-link NAV, monthly rebalance

![Data Coverage Funnel](data/outputs/charts/data_coverage_funnel.png)

## 4. Classification Results

Events were classified into 17 themes using comprehensive keyword matching at the event level. The uncategorized rate is **16.0%** at the event level.

| Theme | Events | Share |
|-------|--------|-------|
| Sports Entertainment | 1,316 | 31.5% |
| Us Elections | 911 | 21.8% |
| Uncategorized | 668 | 16.0% |
| Crypto Digital | 410 | 9.8% |
| Middle East | 176 | 4.2% |
| Russia Ukraine | 156 | 3.7% |
| Ai Technology | 124 | 3.0% |
| Fed Monetary Policy | 74 | 1.8% |
| Climate Environment | 66 | 1.6% |
| Europe Politics | 57 | 1.4% |
| China Us | 53 | 1.3% |
| Energy Commodities | 37 | 0.9% |
| Us Economic | 33 | 0.8% |
| Space Frontier | 29 | 0.7% |
| Legal Regulatory | 25 | 0.6% |
| Pandemic Health | 23 | 0.6% |
| Pop Culture Misc | 23 | 0.6% |

![Theme Distribution](data/outputs/charts/theme_distribution_events.png)

## 5. Basket Construction Methodology

### Eligibility Criteria
| Filter | Threshold |
|--------|-----------|
| Minimum total volume | $10,000 |
| Minimum price history | 14 days (7 for resolved) |
| Price range | 5¢ – 95¢ |
| Minimum volume (resolved) | $5,000 |

### Basket Rules
- **Minimum events per basket**: 5
- **Maximum markets per basket**: 30
- **Maximum single weight**: 20%
- **Minimum single weight**: 2%
- **Rebalance frequency**: Monthly (1st of month)

### Weighting Methods

1. **Equal Weight**: 1/N allocation across all eligible events. Simple, transparent, no estimation error.
2. **Risk Parity (Liquidity-Capped)**: Weight inversely proportional to trailing 30-day volatility, capped at 2× liquidity share. Targets equal risk contribution.
3. **Volume-Weighted**: Weight proportional to total market volume. Reflects market conviction but concentrates in popular markets.

### Baskets Constructed

14 thematic baskets with 5+ eligible events:

| Theme | Events |
|-------|--------|
| Ai Technology | 46 |
| China Us | 21 |
| Climate Environment | 19 |
| Crypto Digital | 164 |
| Energy Commodities | 12 |
| Europe Politics | 12 |
| Fed Monetary Policy | 35 |
| Legal Regulatory | 8 |
| Middle East | 102 |
| Pandemic Health | 8 |
| Russia Ukraine | 65 |
| Space Frontier | 13 |
| Us Economic | 12 |
| Us Elections | 301 |

## 6. Backtest Results

### Combined Basket (All Serious Themes)

| Method | Total Return | Ann. Return | Sharpe | Max DD | Volatility | Calmar | Hit Rate | Avg Turnover |
|--------|-------------|-------------|--------|--------|------------|--------|----------|-------------|
| Equal | 6.1% | 8.6% | 0.34 | -45.3% | 77.0% | 0.19 | 32.2% | 33.7% |
| Risk Parity Liquidity Cap | -12.8% | -17.2% | -1.11 | -21.2% | 15.2% | -0.81 | 34.8% | 2.8% |
| Volume Weighted | -47.8% | -59.3% | -0.25 | -57.6% | 94.4% | -1.03 | 33.0% | 36.1% |

![NAV Time Series](data/outputs/charts/nav_time_series.png)

### Per-Theme Results (All Methods)

| Theme | Method | Total Return | Sharpe | Max DD | Volatility | Hit Rate |
|-------|--------|-------------|--------|--------|------------|----------|
| Ai Technology | Equal | 176.4% | 2.51 | -13.2% | 44.4% | 35.4% |
| Ai Technology | Risk Parity Liquidit | 1260.0% | 2.97 | -16.9% | 109.0% | 37.9% |
| Ai Technology | Volume Weighted | 618.9% | 2.44 | -17.0% | 102.8% | 36.7% |
| China Us | Equal | 31.5% | 0.63 | -26.4% | 59.0% | 29.9% |
| China Us | Risk Parity Liquidit | 75.9% | 0.86 | -30.2% | 101.7% | 27.7% |
| China Us | Volume Weighted | 76.2% | 0.86 | -30.2% | 101.7% | 27.7% |
| Climate Environment | Equal | -16.5% | -0.45 | -26.7% | 66.9% | 29.0% |
| Climate Environment | Risk Parity Liquidit | -33.5% | 0.23 | -56.3% | 190.4% | 4.0% |
| Climate Environment | Volume Weighted | -33.5% | 0.23 | -56.3% | 190.4% | 4.0% |
| Crypto Digital | Equal | -45.9% | -0.67 | -67.0% | 64.9% | 45.8% |
| Crypto Digital | Risk Parity Liquidit | 389.6% | 1.83 | -43.8% | 113.4% | 34.5% |
| Crypto Digital | Volume Weighted | 151.1% | 1.25 | -62.7% | 122.7% | 39.8% |
| Energy Commodities | Equal | 81.8% | 1.44 | -23.3% | 76.6% | 12.6% |
| Energy Commodities | Risk Parity Liquidit | 102.7% | 1.32 | -25.0% | 168.1% | 3.6% |
| Energy Commodities | Volume Weighted | 102.7% | 1.32 | -25.0% | 168.1% | 3.6% |
| Europe Politics | Equal | 54.1% | 0.91 | -38.7% | 59.1% | 12.1% |
| Europe Politics | Risk Parity Liquidit | 0.0% | 0.00 | 0.0% | 0.0% | 0.0% |
| Europe Politics | Volume Weighted | 54.1% | 0.91 | -38.7% | 59.1% | 12.1% |
| Fed Monetary Policy | Equal | 38.1% | 0.70 | -44.8% | 73.1% | 37.5% |
| Fed Monetary Policy | Risk Parity Liquidit | 35.1% | 0.68 | -44.9% | 78.9% | 32.6% |
| Fed Monetary Policy | Volume Weighted | -42.7% | -0.05 | -53.9% | 105.5% | 40.2% |
| Legal Regulatory | Equal | 12.9% | 0.45 | -13.9% | 29.6% | 8.8% |
| Legal Regulatory | Risk Parity Liquidit | 0.0% | 0.00 | 0.0% | 0.0% | 0.0% |
| Legal Regulatory | Volume Weighted | 0.0% | 0.00 | 0.0% | 0.0% | 0.0% |
| Middle East | Equal | -46.3% | -0.36 | -83.4% | 84.9% | 36.4% |
| Middle East | Risk Parity Liquidit | -62.2% | -1.53 | -64.9% | 54.1% | 27.3% |
| Middle East | Volume Weighted | -79.9% | -1.10 | -93.4% | 98.8% | 37.9% |
| Pandemic Health | Equal | -16.4% | -0.74 | -22.9% | 25.5% | 5.7% |
| Pandemic Health | Risk Parity Liquidit | 0.0% | 0.00 | 0.0% | 0.0% | 0.0% |
| Pandemic Health | Volume Weighted | -16.4% | -0.74 | -22.9% | 25.5% | 5.7% |
| Russia Ukraine | Equal | -90.4% | -2.02 | -91.0% | 91.6% | 36.4% |
| Russia Ukraine | Risk Parity Liquidit | -64.8% | -0.22 | -74.4% | 137.8% | 27.3% |
| Russia Ukraine | Volume Weighted | -88.0% | -0.77 | -91.2% | 147.3% | 40.5% |
| Space Frontier | Equal | 110.3% | 2.15 | -15.5% | 81.8% | 26.6% |
| Space Frontier | Risk Parity Liquidit | 0.0% | 0.00 | 0.0% | 0.0% | 0.0% |
| Space Frontier | Volume Weighted | 110.3% | 2.15 | -15.5% | 81.8% | 26.6% |
| Us Economic | Equal | -11.3% | 0.09 | -35.3% | 71.3% | 25.4% |
| Us Economic | Risk Parity Liquidit | -69.6% | -0.25 | -75.8% | 137.1% | 18.6% |
| Us Economic | Volume Weighted | -69.6% | -0.25 | -75.8% | 137.1% | 18.6% |
| Us Elections | Equal | 1555.9% | 2.45 | -20.3% | 135.0% | 51.1% |
| Us Elections | Risk Parity Liquidit | 38.3% | 0.73 | -24.0% | 55.5% | 33.0% |
| Us Elections | Volume Weighted | 128.9% | 1.29 | -33.4% | 82.9% | 48.1% |

![Sharpe Comparison](data/outputs/charts/sharpe_comparison.png)
![Max Drawdown Comparison](data/outputs/charts/max_drawdown_comparison.png)
![Methodology Comparison](data/outputs/charts/methodology_comparison.png)

### Cross-Basket Correlations

![Correlation Heatmap](data/outputs/charts/cross_basket_correlation.png)

Low cross-basket correlation indicates genuine thematic differentiation — baskets capture independent risk factors rather than common market noise.

## 7. Which Methodology Performs Best?

**Best overall: Equal** with a Sharpe of 0.34 and total return of 6.1%.

Key findings:
- **Equal weight** provides the most robust baseline with no estimation error. It tends to perform well when markets are similarly volatile.
- **Risk parity** reduces drawdowns by underweighting volatile markets but can underperform in trending regimes where volatile markets carry signal.
- **Volume-weighted** concentrates in high-conviction markets but risks overweighting crowded trades.

The prediction market context favors simpler methods: short histories, regime changes at resolution, and limited liquidity data make sophisticated estimation unreliable. **Equal weight is the recommended default** unless the universe is highly heterogeneous in volatility.

## 8. Limitations

1. **Survivorship bias**: We only observe markets that were listed. Failed/cancelled markets are underrepresented.
2. **Resolution discontinuity**: Markets jump to 0 or 100 at resolution, creating artificial return spikes. Current methodology handles this via chain-link pricing but residual effects remain.
3. **Liquidity**: Many markets have thin orderbooks. Volume-weighted approaches partially address this but real execution would face slippage.
4. **Short history**: Most markets exist for weeks to months, not years. Annualized metrics should be interpreted cautiously.
5. **Keyword classification**: No LLM in the loop — relies on comprehensive keyword matching. Edge cases exist (e.g., "Will Elon Musk tweet about Bitcoin?" spans AI, crypto, and pop culture).
6. **Single platform**: Only Polymarket data. Kalshi, Metaculus, and other platforms would improve coverage.
7. **No transaction costs**: Backtest assumes zero-cost rebalancing.

## 9. Next Steps

1. **LLM classification**: Use GPT-4o-mini for event classification to reduce uncategorized rate below 5%
2. **Multi-platform**: Integrate Kalshi and Metaculus data
3. **Live baskets**: Deploy real-time basket tracking with streaming prices
4. **Transaction cost model**: Incorporate bid-ask spreads and market impact
5. **Resolution-aware backtesting**: Properly handle market termination events
6. **Risk decomposition**: Factor analysis of basket returns (macro, sentiment, liquidity)
7. **Conditional rebalancing**: Trigger rebalances on resolution events, not just calendar
8. **Cross-basket portfolio**: Optimize allocation across thematic baskets

---

*Generated by basket-engine pipeline v2. 20,180 markets processed, 14 baskets constructed, 265 trading days backtested.*

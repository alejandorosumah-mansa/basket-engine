# basket-engine

Prediction market basket construction using correlation-based community detection. Ingests 20K+ markets from Polymarket, finds natural market communities through return correlations, and constructs diversified baskets with intuitive themes.

## What This Does

Takes raw prediction market data and answers: **can you build diversified, investable baskets of prediction markets?**

### Pipeline

1. **Ingest** 20,180 markets from Polymarket (+ 175 Kalshi)
2. **Classify** into a four-layer taxonomy: Theme → Event → Ticker → CUSIP
3. **Compute pairwise correlations** on daily price changes (2,666 x 2,666 matrix)
4. **Build correlation graph** (edges where |ρ| > 0.3)
5. **Louvain community detection** finds natural market clusters
6. **LLM labels** each community with an investable name
7. **41-factor model** characterizes each basket's macro exposure
8. **Risk-parity weighting** within each community
9. **Backtest** against traditional assets and factor-based approach

## Key Findings

### Correlation clustering beats factor clustering

The previous approach (k-means on factor loadings) put 69% of markets in one meaningless blob. Correlation clustering finds markets that actually move together:

| Metric | Correlation Method | Factor Method |
|--------|-------------------|---------------|
| Max drawdown | **-15.2%** | -46.9% |
| Volatility | **5.6%** | 11.6% |
| Largest basket | **32.5%** | 61.7% |
| Max basket weight | **29.8%** | 73.4% |
| Basket names | "2024 Election Outcomes" | "-DX_Y_NYB + +TNX" |

### Discovered communities

These emerged from the data, not imposed manually:

| Community | Markets | Label |
|-----------|---------|-------|
| 1 | 867 | 2024 Election Outcomes |
| 2 | 516 | 2026 Political & Economic Risks |
| 0 | 461 | 2025 Uncertainty Basket |
| 5 | 356 | Future Uncertainty Basket |
| 3 | 337 | Political & Economic Forecasts |
| 6 | 74 | Global Event Speculation |
| 9 | 52 | Market Uncertainty Dynamics |

Modularity score: 0.411 (strong community structure). Max inter-basket correlation: 0.276.

### 41-factor model confirms diversification

Expanded from 9 to 41 external factors (full US yield curve, global equities, global bonds, commodities). Uses Ridge regression to handle multicollinearity.

| Metric | 9 Factors | 41 Factors |
|--------|-----------|------------|
| Mean R² | 0.108 | 0.395 |
| High R² markets (>0.10) | 37% | 93% |

Even with 41 global factors, 60%+ of prediction market variance remains idiosyncratic. International markets explain more than US-only, but prediction markets are still fundamentally event-driven.

### Portfolio value

Adding 10% prediction markets to 60/40 reduces volatility by ~10% and max drawdown by ~10%, at the cost of ~1.7pp return drag. Near-zero correlation with all traditional assets (SPY: -0.097, GLD: -0.020, BTC: +0.009).

## Taxonomy

```
Theme ("Central Banks & Monetary Policy")
  └─ Event ("Fed Rate Decision")
       └─ Ticker ("Will Fed cut 50bps?")  ← recurring concept, no expiration
            └─ CUSIP ("Will Fed cut 50bps in March 2025?")  ← specific contract, resolves on date
```

A Ticker spawns new CUSIPs over time. When one CUSIP resolves, the next one for that Ticker is already trading. Baskets maintain Event/Ticker exposure by rolling CUSIPs.

## Semantic Exposure Layer

Categorical events (multiple outcomes per event) have directional economic implications:

**"Who will Trump nominate as Fed Chair?"**
- Kevin Warsh → hawkish → rates ↑, dollar ↑, equity ↓
- Kevin Hassett → dovish → rates ↓, dollar ↓, equity ↑

GPT-4o-mini maps outcomes to 8-dimensional factor vectors. Net event exposure = probability-weighted sum across outcomes. 31 events mapped.

## Factor Universe (41 factors)

| Category | Tickers | Count |
|----------|---------|-------|
| US Rates | IRX, SHY, FVX, TNX, TLH, TLT, TYX | 7 |
| Global Bonds | IGLT.L, IBGL.L, BNDX, EMB, BWX, IGOV | 6 |
| US Equity | SPY, QQQ | 2 |
| Global Equity Indices | FTSE, DAX, N225, Shanghai, CAC40, STOXX50E, HSI, SENSEX, BVSP, KOSPI | 10 |
| Country ETFs | EWC, EWA, EWW, EWT, EIDO, TUR, EZA, KSA, EWL, EWS | 10 |
| Commodities | GLD, USO, NG=F | 3 |
| Other | VIX, DXY, BTC | 3 |

Ridge regression (α=1.0) handles multicollinearity across correlated factors.

## Architecture

```
src/
├── ingestion/              # Polymarket + Kalshi API clients
├── classification/         # Four-layer taxonomy, LLM classifier
├── exposure/               # Side detection (long/short), normalization
├── analysis/
│   ├── correlation_clustering.py  # Pairwise correlations → Louvain communities
│   ├── factor_model.py            # 41-factor Ridge regression
│   ├── factor_clustering.py       # Legacy k-means (for comparison)
│   ├── semantic_exposure.py       # Categorical outcome → economic factors
│   ├── cross_asset.py             # Benchmark correlation analysis
│   └── regime.py                  # Risk-on/off regime detection
├── construction/
│   ├── correlation_baskets.py     # Community-based basket construction
│   ├── factor_baskets.py          # Legacy factor-based baskets
│   └── weighting.py               # Risk-parity, equal, market-cap
├── backtest/               # Backtest runner, return calculation
├── benchmarks/             # yfinance data fetching
└── validation/             # Sanity checks, stability tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add OPENAI_API_KEY for LLM classification
```

## Usage

```bash
# Full correlation-based pipeline
python3 run_correlation_pipeline.py

# Legacy factor-based pipeline
python3 scripts/full_pipeline.py

# Individual steps
python3 -m src.analysis.correlation_clustering
python3 -m src.analysis.market_factors
python3 -m src.backtest.factor_backtest
```

## Outputs

```
data/
├── processed/
│   ├── markets.parquet                # 20,180 market metadata
│   ├── prices.parquet                 # 383K daily price observations
│   ├── benchmarks.parquet             # 41 factors, 734 days
│   ├── correlation_matrix.parquet     # 2,666 x 2,666 pairwise correlations
│   ├── community_assignments.parquet  # Market → community mapping
│   ├── community_labels.json          # LLM-generated basket names
│   ├── factor_loadings.parquet        # Per-market factor betas (41 factors)
│   ├── semantic_exposures.json        # Categorical event factor vectors
│   └── final_classifications.csv      # LLM theme labels (6,769 events)
└── outputs/
    ├── charts/                        # Visualization PNGs
    ├── basket_returns.csv             # Daily returns (correlation baskets)
    ├── basket_correlations.csv        # Inter-basket correlations
    ├── basket_weights.csv             # Risk-parity weights
    └── basket_compositions.json       # Market IDs per basket
```

## Tests

```bash
pytest              # 239+ tests
pytest tests/ -v    # Verbose
```

## Known Limitations

1. **No contract rolling.** Current backtest treats CUSIPs as static holdings. When they resolve, they disappear instead of rolling into the next CUSIP for the same Ticker. This creates artificial negative drag and shrinking baskets. Major next step.
2. **Single platform.** Primarily Polymarket. Kalshi coverage is thin (175 markets).
3. **44% of active markets have no price history** (low-volume, AMM-only).
4. **Static correlations.** Should use rolling windows for regime adaptation.
5. **No entry/exit signals.** Backtest assumes continuous holding.

## Research

See **[RESEARCH.md](RESEARCH.md)** for full methodology and analysis.

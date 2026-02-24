# Basket Engine

**Prediction market basket construction using weighted hybrid clustering.** Combines correlation-based community detection with LLM theme categories to build semantically meaningful and data-driven market communities for diversified, investable prediction market portfolios.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete analysis pipeline
python run.py

# View results
open output/results.xlsx
open output/charts/
```

## What This Produces

- **📊 Excel Report**: `output/results.xlsx` with comprehensive analysis across multiple sheets
- **📈 Charts**: `output/charts/` containing all visualizations with descriptive names
- **🎯 Market Communities**: High-quality clusters of correlated prediction markets
- **🏷️ LLM Labels**: Human-readable names for each community (e.g., "Fed Chair and Rate Predictions")

## Methodology Summary

The pipeline transforms 20,000+ raw prediction markets into diversified, investable baskets through:

1. **Market Filtering**: Remove sports/entertainment using keyword filters and LLM classification
2. **Ticker Mapping**: Map individual contracts (CUSIPs) to recurring concepts (Tickers) using regex normalization  
3. **Time Series**: Build continuous price histories for each Ticker with rollover logic
4. **Correlation Matrix**: Compute pairwise correlations with strict quality filters
5. **Hybrid Clustering**: Weighted graph combining correlation evidence with theme structure
6. **Community Detection**: Louvain algorithm finds optimal market clusters
7. **LLM Naming**: GPT-4 generates semantic labels for each community

**Key Innovation**: **Weighted hybrid clustering** balances statistical evidence (markets that move together) with semantic coherence (markets that belong together conceptually).

- **Intra-theme edges**: correlation ≥ 0.3 → weight = correlation × 4.0 (promotes theme cohesion)  
- **Cross-theme edges**: correlation ≥ 0.5 → weight = correlation (allows overwhelming statistical evidence to break theme boundaries)

## Key Results

### Optimal Balance Achieved

Weighted hybrid clustering achieves the optimal balance between pure correlation clustering (too noisy) and rigid theme silos (ignores data):

| Metric | Hybrid Method | Pure Correlation | Theme-Only |
|--------|---------------|------------------|------------|
| Modularity | **0.668** | 0.411 | N/A |
| Communities | **61** | 7 | 19 |
| Theme purity | **81.2%** | ~40% | 100% |
| Theme cohesion | **83.6%** | ~60% | 100% |
| Max community size | **229 markets** | 867 | 800+ |

### Discovered Communities

**16 high-quality market communities** identified after filtering, each with strong internal correlation and semantic coherence:

| Community | Size | Theme | Volume | LLM Label |
|-----------|------|-------|---------|-----------|
| Political & Sports | 171 | Mixed | $502M | Political and Sports Predictions |
| Economic Events | 116 | Mixed | $806M | Economic and Political Events |
| 2025 Predictions | 115 | General | $593M | 2025 Predictions Basket |
| 2024 Elections | 110 | US Elections | $3.3B | 2024 US Elections Insights |
| Crypto Markets | 87 | Crypto/Digital | $389M | Cryptocurrency Market Predictions |
| Fed Policy | 85 | Monetary Policy | $1.1B | Fed Chair and Rate Predictions |

### Data Quality Improvements

**Critical correlation matrix fix implemented** (Feb 2026):

- **Problem**: Correlation calculation ignored `min_overlapping_days`, creating garbage correlations from 1-day overlaps
- **Solution**: Enforced 30+ overlapping days, raised threshold to 0.5, added LLM validation
- **Result**: 1,679 → 1,467 high-quality markets in 16 coherent communities

**Before/After Quality Metrics**:
- Spurious correlations: 100% → 0% (eliminated 1-day overlap noise)
- Market pairs with sufficient overlap: 0% → 40.8%  
- Communities with clear themes: Variable → 81.2% purity

## Taxonomy Structure

**Four-layer hierarchy** for systematic market organization:

```
Theme ("Central Banks & Monetary Policy")
  └─ Event ("Fed Rate Decision")  
       └─ Ticker ("Will Fed cut 50bps?")  ← recurring concept
            └─ CUSIP ("March 2025 50bp cut")  ← specific contract
```

**Key Distinctions**:
- **CUSIP**: Specific contract with expiration (what Polymarket calls a market_id)
- **Ticker**: Recurring concept without expiration (spawns multiple CUSIPs over time)  
- **Event**: Groups related Tickers (different rate cut sizes for same meeting)
- **Theme**: Macro category for portfolio allocation

**Classification Results**: 19 themes, 6,769 events classified via GPT-4, 0.5% uncategorized

## Detailed Methodology

### 1. Market Ingestion & Filtering

- **Raw Markets**: 20,180 (Polymarket + Kalshi)
- **With Price Data**: 11,223 (55.6% coverage)  
- **≥30 Days History**: 2,721 (minimum for analysis)
- **Final Eligible**: 1,679 (quality filters applied)

**Filters Applied**:
- Remove sports/entertainment via keyword matching
- Volume ≥ 25th percentile threshold
- Price variance ≥ 0.005 (avoid flat markets)
- ≥ 30 active trading days
- ≥ 10 unique price points

### 2. LLM Classification

Uses **GPT-4** to classify markets into 19 theme categories:

- `us_elections` - 2024/2028 presidential, congressional races
- `fed_monetary_policy` - Interest rates, Fed chair nominations  
- `crypto_digital` - Bitcoin/Ethereum prices, DeFi protocols
- `china_us` - Trade relations, tariffs, Taiwan
- `legal_regulatory` - Court cases, agency actions
- `global_politics` - International elections, conflicts
- `climate_energy` - Weather events, energy policy
- And 12 more specialized categories

**Few-shot examples** ensure consistent classification across similar markets.

### 3. Ticker Mapping

**Objective**: Map individual contracts (CUSIPs) to recurring concepts (Tickers) for continuous exposure.

**Method**: Regex normalization + exact string matching (no fuzzy matching to prevent over-grouping)

**Normalization Rules**:
- Replace specific dates with `DATE` placeholder
- Standardize "by March 15" → "by DATE" patterns  
- Remove extra whitespace, lowercase
- Keep outcome-specific details ("25bps" vs "50bps" remain distinct)

**Result**: 20,180 CUSIPs → 15,847 unique Tickers (many 1-to-1 mappings for one-off events)

### 4. Continuous Time Series

Build rollable chains for each Ticker using contract rollover logic:

1. **Load candle data** for each CUSIP (daily OHLC from order book)
2. **Sort contracts** by expiration date within each Ticker
3. **Chain construction**: Roll from expiring to next-available contract  
4. **Quality filters**: Minimum 30 days active, sufficient volume overlap

**Return Calculation**: Absolute probability differences (`diff()`), not percentage changes. A market moving 0.02 → 0.04 is a 2pp change, not 100% return.

### 5. Correlation Matrix with Quality Filters

**Strict quality requirements** to eliminate noise:

- **Minimum overlap**: 30+ days of simultaneous trading
- **Volume filter**: ≥25th percentile to avoid thin markets
- **Variance filter**: ≥0.005 to avoid flat/resolved markets  
- **Active days**: ≥30 days with price updates

**Result**: 1,679 × 1,679 correlation matrix with 40.8% valid pairs

**Correlation distribution**:
- Mean: 0.08 (low background correlation)
- 90th percentile: 0.31 
- 95th percentile: 0.45
- 99th percentile: 0.67 (strong relationships exist)

### 6. Weighted Hybrid Clustering

**Innovation**: Weighted graph that combines correlation evidence with theme structure.

**Graph Construction**:
1. **Nodes**: All markets in correlation matrix
2. **Intra-theme edges**: Same theme + correlation ≥ 0.3 → weight = correlation × 4.0
3. **Cross-theme edges**: Different themes + correlation ≥ 0.5 → weight = correlation  
4. **No edges**: Below thresholds (sparse graph)

**Rationale**: 
- Boost intra-theme connections to maintain semantic coherence
- Allow cross-theme connections only for overwhelming statistical evidence
- Prevent theme silos while preserving interpretability

**Community Detection**: Louvain algorithm optimizes weighted modularity

### 7. Community Validation & Naming

**Size Filtering**: Minimum 10 markets per community (remove noise clusters)

**LLM Naming**: GPT-4 analyzes each community's markets and generates:
- **Name**: Investment-ready label ("Fed Chair and Rate Predictions")
- **Theme**: Dominant category
- **Description**: 2-3 sentence explanation of market relationships

**Quality Metrics**:
- **Theme purity**: 81.2% of communities dominated by single theme  
- **Theme cohesion**: 83.6% of same-theme markets stay together
- **Modularity**: 0.668 (excellent community structure)

### 8. Results Export

**Excel Workbook** (`output/results.xlsx`):
- **Summary**: Pipeline statistics and key metrics
- **Communities**: Size, names, themes, top markets for each cluster
- **Ticker Mapping**: Complete CUSIP → Ticker mapping with counts
- **Correlation Matrix**: Filtered correlation data (sampled if large)  
- **Classifications**: All market theme assignments
- **Methodology**: Detailed parameter documentation

**Charts** (`output/charts/`):
- `01_market_filter_funnel.png` - Filtering stage breakdown
- `02_classification_summary.png` - Markets by theme
- `03_ticker_mapping_summary.png` - CUSIP/Ticker distribution  
- `04_correlation_distribution.png` - Correlation coefficient histogram
- `05_community_size_distribution.png` - Community sizes
- `06_network_visualization.png` - Graph structure visualization
- `07_community_theme_analysis.png` - Theme purity vs community size
- `08_correlation_heatmap_community_X.png` - Heatmaps for top communities

## Example Communities

### "Fed Chair and Rate Predictions" (85 markets, $1.1B volume)

**Theme Composition**: 88.2% `fed_monetary_policy`

**Top Markets**:
- Will Trump nominate Judy Shelton as the next Fed chair? ($92M)
- Fed decreases rates by 50+ bps after October 2025? ($53M) 
- Fed decreases rates by 50+ bps after July 2025? ($41M)
- Will Trump nominate Michelle Bowman as Fed chair? ($21M)

**Why This Works**: All markets directly tied to Federal Reserve decisions. High correlation because Fed policy affects all rate-related outcomes simultaneously.

### "Cryptocurrency Market Predictions" (87 markets, $389M volume)

**Theme Composition**: 73% `crypto_digital`, 15% `general_predictions`

**Top Markets**:
- Will Ethereum hit $10,000 by December 31? ($9.5M)
- Will Bitcoin reach $130,000 by December 31, 2025? ($13M)
- MicroStrategy sells any Bitcoin in 2025? ($18M)

**Why This Works**: Crypto markets move together due to shared risk factors (regulation, adoption, macroeconomic conditions). Cross-asset correlations captured naturally.

## Limitations & Future Work

### Current Limitations

1. **Price Data Coverage**: 45% of markets lack sufficient price history
2. **Resolution Bias**: Resolved markets excluded (survivorship bias)
3. **Time Decay**: No modeling of time-to-expiration effects
4. **Categorical Markets**: Multi-outcome events treated as independent binaries
5. **Factor Model**: 41-factor model explains only ~20% of variance

### Next Steps

1. **Real-Time Updates**: Integrate live price feeds for dynamic rebalancing
2. **Risk Models**: Factor decomposition for systematic vs idiosyncratic risk
3. **Portfolio Optimization**: Modern portfolio theory with prediction market constraints
4. **Backtesting**: Historical performance vs traditional assets
5. **Categorical Modeling**: Proper handling of mutually exclusive outcomes
6. **Alternative Data**: Social sentiment, news flow, expert forecasts

### Technical Improvements

1. **Scalability**: Sparse matrix operations for larger datasets
2. **Robustness**: Bootstrap confidence intervals for community assignments  
3. **Interpretability**: SHAP values for correlation drivers
4. **Validation**: Out-of-sample community stability testing

## Data Sources

- **Markets**: Polymarket API (20,005 markets) + Kalshi API (175 markets)
- **Prices**: CLOB (order book) data from both platforms
- **Classifications**: OpenAI GPT-4 with custom taxonomy
- **Benchmarks**: Traditional assets for factor model validation

## Technical Stack

- **Python 3.9+** with pandas, numpy, networkx, scikit-learn
- **Clustering**: `python-louvain` for community detection  
- **LLM**: OpenAI API (GPT-4) for classification and naming
- **Visualization**: matplotlib, seaborn for chart generation
- **Data**: Parquet files for efficient storage and retrieval

## Files Structure

```
basket-engine/
├── run.py                 # Complete pipeline script
├── README.md              # This documentation
├── output/
│   ├── charts/            # All generated visualizations  
│   └── results.xlsx       # Comprehensive Excel report
├── data/
│   ├── raw/               # Source data from APIs
│   └── processed/         # Intermediate pipeline outputs
├── src/                   # Modular source code
├── config/                # Settings and taxonomy definitions
├── archive/               # Original scripts (preserved)
└── requirements.txt       # Python dependencies
```

## Contributing

This is a research prototype. For production use:

1. **Data validation** - Add comprehensive input validation
2. **Error handling** - Robust error recovery and logging
3. **Performance** - Optimize for larger datasets (>100K markets)  
4. **Testing** - Unit tests for all pipeline components
5. **Documentation** - API docs for src/ modules

## License

Research and educational use. See `LICENSE` file for details.

---

*Generated by Basket Engine v1.0 - Prediction market portfolio construction using weighted hybrid clustering*
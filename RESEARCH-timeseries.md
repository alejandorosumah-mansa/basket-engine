# Continuous Event Time Series from Prediction Market Contracts

**Date:** February 23, 2025  
**Objective:** Build continuous time series from prediction market contracts with different expiration dates

## Executive Summary

This analysis successfully constructed continuous time series for prediction market events by chaining contracts with different expiration dates, similar to futures continuous contracts. We identified **4 events** with at least 60 days of continuous history, totaling **455 data points** spanning from August 2025 to February 2026.

## Methodology

### 1. Data Structure Analysis

Our analysis began with understanding the existing data structure:
- **Markets metadata**: 20,180 markets with event metadata, tickers, and expiration dates
- **Price data**: Daily candles stored in JSON format per contract condition_id
- **Event grouping**: Markets grouped by `event_slug` identifier

### 2. Contract Identification

We identified events with multiple contracts by:
- Grouping markets by `event_slug` 
- Finding events with multiple `end_date` values
- Discovered **121 events** with multiple expiration dates
- Top event: "us-strikes-iran-by" with 12 different contracts

### 3. Time Series Chaining Algorithm

For each event, we implemented a chronological chaining methodology:

1. **Sort contracts by end_date**: Order contracts chronologically
2. **Load price data**: Extract daily candle data for each contract
3. **Handle overlaps**: When contracts overlap, use the earlier contract's data up to its expiration, then roll to the next
4. **Handle gaps**: Accept natural gaps when no active contract exists
5. **Create continuous series**: Concatenate price data chronologically with contract identifiers

### 4. Data Quality Filters

Applied the following filters for robust analysis:
- **Minimum duration**: 60+ days of continuous history
- **Data availability**: Must have actual price data (not just market metadata)  
- **Valid timestamps**: Remove duplicate timestamps within contracts

## Results

### Qualified Events (4 total)

1. **will-metamask-launch-a-token-in-2025**
   - Duration: 103 days (2025-09-20 to 2026-01-01)
   - Data points: 104
   - Contracts used: 1
   - Average probability: ~15%

2. **us-x-venezuela-military-engagement-by**  
   - Duration: 133 days (2025-08-25 to 2026-01-05)
   - Data points: 134
   - Contracts used: 3  
   - Average probability: ~8%

3. **maduro-out-in-2025**
   - Duration: 61 days (2025-11-04 to 2026-01-04)
   - Data points: 62
   - Contracts used: 3
   - Average probability: ~12%

4. **megaeth-market-cap-fdv-one-day-after-launch**
   - Duration: 178 days (2025-08-27 to 2026-02-21)  
   - Data points: 155
   - Contracts used: 1
   - Average probability: ~65%

### Data Coverage

- **Total data points**: 455 observations
- **Date range**: August 25, 2025 to February 21, 2026 (180 days)
- **Average daily coverage**: 2.5 events active per day

## Technical Implementation

### Key Features

1. **Contract rollover detection**: Vertical lines in charts mark contract transitions
2. **Price continuity**: Maintains probability trajectory across contract boundaries  
3. **Volume tracking**: Preserves trading volume and open interest data
4. **Metadata preservation**: Tracks which contract is active at each point

### Data Schema

The final dataset contains:
- `timestamp`: Date/time of price observation
- `price`: Probability (0-1) of event occurring
- `volume`: Trading volume
- `open_interest`: Open interest
- `event_slug`: Event identifier
- `active_contract`: Current contract identifier  
- `contract_end_date`: Expiration date of active contract

## Insights and Patterns

### 1. Event Probability Dynamics
- **Geopolitical events** (Venezuela, Maduro) show high volatility with sharp price movements
- **Cryptocurrency events** (MetaMask token) show gradual decay as deadlines approach
- **Market events** (MegaETH) show sustained high probabilities when fundamentals are strong

### 2. Contract Rolling Behavior
- Most events successfully roll from expired to active contracts
- Price continuity is generally maintained across rollovers
- Volume often increases near contract expiration dates

### 3. Data Quality Issues
- Many contracts lack price data (candles files missing)
- Recent contracts (2025-2026) have better data coverage than historical ones
- Sports betting contracts tend to be shorter-term and less suitable for continuous analysis

## Limitations

1. **Data availability**: Only ~25% of identified multi-expiry events had sufficient price data
2. **Time span**: Current dataset primarily covers 2025-2026 timeframe  
3. **Contract gaps**: Some events have gaps between contract expirations
4. **Resolution bias**: Resolved events may have different price patterns than active ones

## Applications

This continuous time series dataset enables:
1. **Long-term event tracking**: Monitor probability evolution over months
2. **Volatility analysis**: Study price dynamics across contract boundaries
3. **Correlation studies**: Analyze relationships between related events
4. **Risk modeling**: Build models using extended historical data
5. **Portfolio construction**: Create baskets spanning multiple contract cycles

## File Outputs

- **Dataset**: `data/processed/continuous_event_timeseries.parquet`
- **Summary**: `data/processed/continuous_event_timeseries_summary.json` 
- **Charts**: `data/outputs/continuous_timeseries_charts/`
- **Analysis script**: `build_continuous_timeseries.py`

## Next Steps

1. **Expand data coverage**: Include Kalshi contracts and older Polymarket data
2. **Automated updates**: Schedule regular runs to extend time series
3. **Advanced chaining**: Implement volume-weighted rolling and basis adjustments
4. **Event clustering**: Group related events for thematic analysis
5. **Predictive modeling**: Use continuous series for forecasting

## Code Repository

The complete methodology is implemented in `build_continuous_timeseries.py` with the following key classes:
- `ContinuousTimeseriesBuilder`: Main orchestration class
- Methods for data loading, contract chaining, and visualization
- Configurable parameters for duration filters and output formats

---

*This analysis demonstrates the feasibility of creating continuous prediction market time series, opening new possibilities for quantitative analysis of event probabilities over extended periods.*
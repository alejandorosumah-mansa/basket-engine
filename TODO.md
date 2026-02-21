# Basket Engine - Priority Issues

All issues tracked on GitHub: [alejandorosumah-mansa/basket-engine](https://github.com/alejandorosumah-mansa/basket-engine/issues)

## P0 - Critical

1. **Price Data Coverage: Only 13.7% of markets have price history** ([#9](https://github.com/alejandorosumah-mansa/basket-engine/issues/9))
   - Root cause: Ingestion interrupted after 3,247/10,880 eligible markets
   - Dome API returns 400/404 for many condition_ids (possible token_id confusion)
   - Fallback price endpoint also fails for many markets
   - Impact: Entire pipeline bottlenecked by missing data

2. **High Uncategorized Rate in Theme Classification (~40%)** ([#10](https://github.com/alejandorosumah-mansa/basket-engine/issues/10))
   - Classification at wrong granularity (market-level vs event-level)
   - Event-level classification implemented but not fully integrated
   - Keyword fallback is too crude

3. **Four-Layer Taxonomy Not Fully Integrated Into Pipeline** ([#11](https://github.com/alejandorosumah-mansa/basket-engine/issues/11))
   - `four_layer_taxonomy.py` exists but isn't used in main pipeline
   - `run.py` still uses old `ticker_cusip.py` for normalization
   - Multiple competing classification approaches with no clear winner

## P1 - High

4. **Backtest Uses Wrong Market ID Mapping** ([#12](https://github.com/alejandorosumah-mansa/basket-engine/issues/12))
   - `run_backtest()` filters on `event_id` from classifications CSV
   - But prices/returns use `market_id` from markets.parquet
   - These don't match, causing silent data loss

5. **No Sports/Entertainment Theme in Taxonomy** ([#13](https://github.com/alejandorosumah-mansa/basket-engine/issues/13))
   - Code references `sports_entertainment` theme to filter it out
   - But taxonomy.yaml doesn't define it
   - Markets get classified as `uncategorized` instead of being properly excluded

6. **Redundant/Conflicting Classification Scripts** ([#14](https://github.com/alejandorosumah-mansa/basket-engine/issues/14))
   - `scripts/classify_full.py` vs `src/classification/llm_classifier.py`
   - Multiple cache files: `llm_classifications_cache.json`, `_cache_v2.json`
   - Multiple output files: `llm_classifications.csv`, `_full.csv`, `final_classifications.csv`

## P2 - Medium

7. **Kalshi Integration Incomplete** ([#15](https://github.com/alejandorosumah-mansa/basket-engine/issues/15))
   - Only 175 Kalshi markets ingested (vs 20K Polymarket)
   - Kalshi trade-based pricing is sparse
   - Cross-platform dedup exists but untested at scale

8. **No Transaction Cost Modeling** ([#16](https://github.com/alejandorosumah-mansa/basket-engine/issues/16))
   - Backtest assumes zero trading costs
   - Prediction markets have wide bid-ask spreads
   - Overstates achievable returns

9. **Return Calculation Uses Price Difference, Not Percentage** ([#17](https://github.com/alejandorosumah-mansa/basket-engine/issues/17))
   - `normalize.py` computes `return = close_price - prev_price` (absolute)
   - Should be `(close - prev) / prev` (percentage return)
   - Breaks volatility calculation and risk parity weighting

10. **`.env` File Security** ([#18](https://github.com/alejandorosumah-mansa/basket-engine/issues/18))
    - Ensure API keys are never committed
    - Rotate any exposed keys

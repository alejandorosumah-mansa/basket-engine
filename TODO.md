# Basket Engine - Priority Issues

All issues tracked on GitHub: [alejandorosumah-mansa/basket-engine](https://github.com/alejandorosumah-mansa/basket-engine/issues)

## P0 - Critical

1. **Price Data Coverage: Only 13.7% of markets have price history** (#1)
   - Root cause: Ingestion interrupted after 3,247/10,880 eligible markets
   - Dome API returns 400/404 for many condition_ids (possible token_id confusion)
   - Fallback price endpoint also fails for many markets
   - Impact: Entire pipeline bottlenecked by missing data

2. **High Uncategorized Rate in Theme Classification (~40%)** (#2)
   - Classification at wrong granularity (market-level vs event-level)
   - Event-level classification implemented but not fully integrated
   - Keyword fallback is too crude

3. **Four-Layer Taxonomy Not Fully Integrated Into Pipeline** (#3)
   - `four_layer_taxonomy.py` exists but isn't used in main pipeline
   - `run.py` still uses old `ticker_cusip.py` for normalization
   - Multiple competing classification approaches with no clear winner

## P1 - High

4. **Backtest Uses Wrong Market ID Mapping** (#4)
   - `run_backtest()` filters on `event_id` from classifications CSV
   - But prices/returns use `market_id` from markets.parquet
   - These don't match, causing silent data loss

5. **No Sports/Entertainment Theme in Taxonomy** (#5)
   - Code references `sports_entertainment` theme to filter it out
   - But taxonomy.yaml doesn't define it
   - Markets get classified as `uncategorized` instead of being properly excluded

6. **Redundant/Conflicting Classification Scripts** (#6)
   - `scripts/classify_full.py` vs `src/classification/llm_classifier.py`
   - Multiple cache files: `llm_classifications_cache.json`, `_cache_v2.json`
   - Multiple output files: `llm_classifications.csv`, `_full.csv`, `final_classifications.csv`

## P2 - Medium

7. **Kalshi Integration Incomplete** (#7)
   - Only 175 Kalshi markets ingested (vs 20K Polymarket)
   - Kalshi trade-based pricing is sparse
   - Cross-platform dedup exists but untested at scale

8. **No Transaction Cost Modeling** (#8)
   - Backtest assumes zero trading costs
   - Prediction markets have wide bid-ask spreads
   - Overstates achievable returns

9. **Return Calculation Uses Price Difference, Not Percentage** (#9)
   - `normalize.py` computes `return = close_price - prev_price` (absolute)
   - Should be `(close - prev) / prev` (percentage return)
   - Breaks volatility calculation and risk parity weighting

10. **`.env` File Committed to Repo** (#10)
    - Contains API keys (DOME_API_KEY, OPENAI_API_KEY)
    - Should be in `.gitignore` only, not tracked

## P3 - Low

11. **Missing `sports_entertainment` filter creates silent classification errors** (#11)
12. **Test suite uses mocked data only - no integration tests** (#12)
13. **No CI/CD pipeline** (#13)
14. **Notebook `04_classification_comparison.ipynb` - unclear if current** (#14)

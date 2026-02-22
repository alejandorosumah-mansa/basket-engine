# basket-engine

Thematic basket construction and backtesting for prediction markets (Polymarket). Implements a four-layer taxonomy (CUSIP → Ticker → Event → Theme), LLM-based classification, exposure normalization, and multiple weighting methodologies.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

## Run

```bash
python3 scripts/full_pipeline.py
```

This runs the full pipeline: data loading → taxonomy → classification → eligibility → basket construction → backtest → charts → RESEARCH.md generation.

**Outputs:**
- `data/outputs/charts/` — All visualization PNGs
- `data/outputs/backtest_metrics.csv` — Per-theme/method performance metrics
- `data/outputs/backtest_nav_series.csv` — Daily NAV series
- `data/outputs/basket_compositions.json` — Market IDs per basket
- `RESEARCH.md` — Complete research document (auto-generated)

## Tests

```bash
pytest
```

## Documentation

See **[RESEARCH.md](RESEARCH.md)** for the full methodology, results, and analysis.

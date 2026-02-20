# basket-engine

Prediction market basket construction engine for Aditis.

Replaces the manual, static basket system with a quantitative, rebalancing methodology.

## What This Does

1. **Ingests** all market data from Polymarket + Kalshi (metadata + full price history)
2. **Classifies** markets into investable themes using statistical clustering + LLM hybrid
3. **Constructs** baskets with risk parity weighting + liquidity caps
4. **Rebalances** monthly with full audit trail
5. **Backtests** historical performance with multiple methodology comparisons
6. **Validates** everything through automated tests + manual review

## Architecture

```
src/
├── ingestion/      # API data collection + caching
├── classification/  # Statistical clustering + LLM categorization + hybrid
├── construction/    # Eligibility, weighting, basket building
├── backtest/        # Historical replay + performance metrics
└── validation/      # Automated + manual verification
```

See [PLAN.md](./PLAN.md) for the full technical specification.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add API keys
```

## Status

🚧 In development. See GitHub Issues for workstreams.

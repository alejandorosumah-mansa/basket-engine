"""
Shared pytest fixtures and configuration for basket-engine tests.

Provides reusable sample data, mock API responses, and common test utilities.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
import yaml

from tests.test_data_generators import (
    generate_market,
    generate_market_batch,
    generate_price_series,
    generate_returns_matrix,
    generate_basket_weights,
    generate_nav_series,
    generate_classification_result,
    generate_rebalance_event,
)


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "integration: integration tests that may need external resources")
    config.addinivalue_line("markers", "slow: slow tests (clustering, backtests)")


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings_path():
    """Path to settings.yaml."""
    return Path(__file__).parent.parent / "config" / "settings.yaml"


@pytest.fixture
def taxonomy_path():
    """Path to taxonomy.yaml."""
    return Path(__file__).parent.parent / "config" / "taxonomy.yaml"


@pytest.fixture
def settings(settings_path):
    """Loaded settings dict."""
    with open(settings_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def taxonomy(taxonomy_path):
    """Loaded taxonomy themes dict."""
    with open(taxonomy_path) as f:
        data = yaml.safe_load(f)
    return data.get("themes", {})


# ---------------------------------------------------------------------------
# Market data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_market():
    """A single valid active market."""
    return generate_market(
        market_id="test_active_001",
        platform="polymarket",
        title="Will Bitcoin exceed $100K by end of 2026?",
        description="This market resolves YES if Bitcoin price exceeds $100,000 at any point before Dec 31, 2026.",
        start_date=datetime(2025, 6, 1),
        end_date=datetime(2026, 12, 31),
        total_volume=150000.0,
        daily_volume_7d=5000.0,
        liquidity=25000.0,
        tags=["crypto", "bitcoin"],
    )


@pytest.fixture
def resolved_market():
    """A resolved YES market."""
    return generate_market(
        market_id="test_resolved_001",
        platform="kalshi",
        title="Will the Fed cut rates in January 2026?",
        description="Resolves YES if the Federal Reserve announces a rate cut at the January 2026 FOMC meeting.",
        start_date=datetime(2025, 8, 1),
        end_date=datetime(2026, 1, 31),
        resolution="YES",
        resolution_value=1.0,
        total_volume=200000.0,
        daily_volume_7d=0.0,
        liquidity=0.0,
    )


@pytest.fixture
def illiquid_market():
    """A market that fails liquidity thresholds."""
    return generate_market(
        market_id="test_illiquid_001",
        platform="polymarket",
        title="Will a niche event happen?",
        description="Very low volume market.",
        total_volume=500.0,
        daily_volume_7d=10.0,
        liquidity=100.0,
    )


@pytest.fixture
def near_expiry_market():
    """A market expiring in <14 days."""
    return generate_market(
        market_id="test_near_expiry_001",
        platform="kalshi",
        title="Will X happen this week?",
        start_date=datetime(2025, 6, 1),
        end_date=datetime.now() + timedelta(days=5),
        total_volume=80000.0,
        daily_volume_7d=3000.0,
        liquidity=10000.0,
    )


@pytest.fixture
def market_batch():
    """A batch of 20 diverse markets."""
    return generate_market_batch(n=20, seed=42)


@pytest.fixture
def large_market_batch():
    """A large batch of 100 markets for stress testing."""
    return generate_market_batch(n=100, seed=123)


# ---------------------------------------------------------------------------
# Price series fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_price_series():
    """60-day price series with moderate volatility."""
    return generate_price_series(n_days=60, start_price=0.50, volatility=0.03, seed=42)


@pytest.fixture
def resolved_price_series():
    """Price series that resolves to YES (1.0)."""
    return generate_price_series(
        n_days=60, start_price=0.50, volatility=0.03,
        resolution=1.0, resolution_day=55, seed=42,
    )


@pytest.fixture
def gapped_price_series():
    """Price series with intentional gaps."""
    return generate_price_series(
        n_days=60, start_price=0.50, volatility=0.03,
        gaps=[10, 11, 12, 13, 14, 15, 16, 17, 18],  # 9-day gap
        seed=42,
    )


@pytest.fixture
def high_vol_price_series():
    """High-volatility price series."""
    return generate_price_series(n_days=60, start_price=0.50, volatility=0.10, seed=42)


@pytest.fixture
def low_vol_price_series():
    """Low-volatility price series."""
    return generate_price_series(n_days=60, start_price=0.50, volatility=0.005, seed=42)


# ---------------------------------------------------------------------------
# Returns / correlation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def returns_matrix():
    """Returns matrix with 3 correlated groups of markets."""
    return generate_returns_matrix(
        n_markets=12, n_days=60, n_correlated_groups=3,
        within_group_corr=0.6, seed=42,
    )


# ---------------------------------------------------------------------------
# Basket / weighting fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def equal_weights():
    """Equal-weight basket with 10 markets."""
    return generate_basket_weights(n_markets=10, method="equal")


@pytest.fixture
def random_weights():
    """Random-weight basket with 10 markets."""
    return generate_basket_weights(n_markets=10, method="random", seed=42)


@pytest.fixture
def concentrated_weights():
    """Concentrated basket (one market at 50%)."""
    return generate_basket_weights(n_markets=10, method="concentrated")


# ---------------------------------------------------------------------------
# NAV / backtest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_nav_series():
    """120-day NAV series with monthly rebalances."""
    return generate_nav_series(
        n_days=120, rebalance_days=[0, 30, 60, 90], seed=42,
    )


@pytest.fixture
def drawdown_nav_series():
    """NAV series with a significant drawdown."""
    return generate_nav_series(
        n_days=120, drawdown_at=50, drawdown_pct=0.30, seed=42,
    )


# ---------------------------------------------------------------------------
# Classification fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_classifications():
    """A set of classification results across multiple themes."""
    themes = [
        "us_elections", "fed_monetary_policy", "us_economic",
        "ai_technology", "crypto_digital", "russia_ukraine",
        "middle_east", "energy_commodities", "climate_environment",
    ]
    results = []
    for i, theme in enumerate(themes):
        for j in range(6):  # 6 markets per theme
            results.append(generate_classification_result(
                market_id=f"market_{theme}_{j}",
                primary_theme=theme,
                confidence=0.7 + 0.05 * j,
            ))
    return results


# ---------------------------------------------------------------------------
# Rebalance event fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rebalance_events():
    """Monthly rebalance events over 6 months."""
    events = []
    for m in range(6):
        date = datetime(2025, 8 + m, 1) if 8 + m <= 12 else datetime(2026, (8 + m) - 12, 1)
        events.append(generate_rebalance_event(
            date=date,
            basket_id="us_elections",
            turnover=0.15 + 0.05 * m,
            market_count=10 + m,
        ))
    return events


# ---------------------------------------------------------------------------
# Mock API fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_polymarket_api():
    """Mock Polymarket API client."""
    mock = MagicMock()
    mock.get_markets.return_value = generate_market_batch(n=10, platforms=["polymarket"], seed=100)
    mock.get_candlesticks.return_value = generate_price_series(seed=100)
    return mock


@pytest.fixture
def mock_kalshi_api():
    """Mock Kalshi API client."""
    mock = MagicMock()
    mock.get_markets.return_value = generate_market_batch(n=10, platforms=["kalshi"], seed=200)
    mock.get_price_history.return_value = generate_price_series(seed=200)
    return mock


# ---------------------------------------------------------------------------
# Eligibility settings fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def eligibility_settings():
    """Eligibility filter thresholds from settings."""
    return {
        "min_total_volume": 10000,
        "min_7d_avg_daily_volume": 500,
        "min_liquidity": 1000,
        "min_price_history_days": 14,
        "min_days_to_expiration": 14,
        "price_range_min": 0.05,
        "price_range_max": 0.95,
    }


@pytest.fixture
def basket_constraints():
    """Basket construction constraints from settings."""
    return {
        "min_markets": 5,
        "max_markets": 30,
        "max_single_weight": 0.20,
        "min_single_weight": 0.02,
    }

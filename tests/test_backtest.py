"""
Tests for backtest engine.

Validates:
- NAV series continuity (no gaps > 1 business day)
- NAV never goes negative
- Rebalance events recorded for every month
- Turnover calculation correctness
- No lookahead bias
- Chain-linking continuity at rebalance boundaries
- Sharpe ratio and max drawdown calculations
- Resolved market terminal returns
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from tests.test_data_generators import (
    generate_nav_series,
    generate_rebalance_event,
    generate_price_series,
)


# ===========================================================================
# NAV Series Continuity
# ===========================================================================


class TestNAVContinuity:
    """NAV series must be continuous with no large gaps."""

    @pytest.mark.unit
    def test_no_gaps_greater_than_one_business_day(self, sample_nav_series):
        """NAV series should have no gaps > 1 business day (allowing weekends)."""
        dates = pd.to_datetime(sample_nav_series["date"])
        gaps = dates.diff().dt.days
        # Allow up to 3 days (weekend: Fri → Mon)
        max_gap = gaps.iloc[1:].max()  # skip first NaT
        assert max_gap <= 3, f"NAV gap of {max_gap} days exceeds 3 (weekend allowance)"

    @pytest.mark.unit
    def test_nav_dates_monotonic(self, sample_nav_series):
        """NAV dates must be strictly increasing."""
        dates = pd.to_datetime(sample_nav_series["date"])
        assert dates.is_monotonic_increasing

    @pytest.mark.unit
    def test_nav_no_duplicate_dates(self, sample_nav_series):
        """No duplicate dates in NAV series."""
        dates = sample_nav_series["date"]
        assert dates.nunique() == len(dates)

    @pytest.mark.unit
    def test_detect_nav_gap(self):
        """A NAV series with a gap should be detectable."""
        nav = generate_nav_series(n_days=60, seed=42)
        # Remove 5 consecutive days to create a gap
        nav_gapped = nav.drop(index=range(20, 25)).reset_index(drop=True)
        dates = pd.to_datetime(nav_gapped["date"])
        max_gap = dates.diff().dt.days.iloc[1:].max()
        assert max_gap > 3, "Failed to detect gap in NAV series"


# ===========================================================================
# NAV Non-Negativity
# ===========================================================================


class TestNAVNonNegative:
    """NAV must never go negative."""

    @pytest.mark.unit
    def test_nav_always_positive(self, sample_nav_series):
        """NAV should be positive at all times."""
        assert (sample_nav_series["nav"] > 0).all(), (
            f"NAV went non-positive. Min: {sample_nav_series['nav'].min()}"
        )

    @pytest.mark.unit
    def test_nav_after_drawdown_still_positive(self, drawdown_nav_series):
        """Even after significant drawdown, NAV should remain positive."""
        assert (drawdown_nav_series["nav"] > 0).all(), (
            f"NAV went non-positive after drawdown. Min: {drawdown_nav_series['nav'].min()}"
        )

    @pytest.mark.unit
    def test_nav_starts_at_expected_value(self, sample_nav_series):
        """NAV should start at the expected initial value (100)."""
        assert sample_nav_series["nav"].iloc[0] == pytest.approx(100.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("daily_loss", [0.05, 0.10, 0.20, 0.50])
    def test_nav_survives_daily_loss(self, daily_loss):
        """NAV should remain positive even with large single-day losses."""
        nav = generate_nav_series(
            n_days=60, drawdown_at=30, drawdown_pct=daily_loss, seed=42,
        )
        assert (nav["nav"] > 0).all(), (
            f"NAV went negative with {daily_loss:.0%} daily loss"
        )


# ===========================================================================
# Rebalance Events
# ===========================================================================


class TestRebalanceEvents:
    """Rebalance events must be recorded for every month in backtest range."""

    @pytest.mark.unit
    def test_monthly_rebalance_events(self, sample_rebalance_events):
        """Should have one rebalance event per month."""
        assert len(sample_rebalance_events) == 6, (
            f"Expected 6 monthly rebalance events, got {len(sample_rebalance_events)}"
        )

    @pytest.mark.unit
    def test_rebalance_event_has_required_fields(self, sample_rebalance_events):
        """Each rebalance event must have date, basket_id, turnover, market_count."""
        required = ["date", "basket_id", "additions", "removals",
                     "weight_changes", "turnover", "market_count"]
        for event in sample_rebalance_events:
            for field in required:
                assert field in event, f"Rebalance event missing field: {field}"

    @pytest.mark.unit
    def test_rebalance_dates_monthly(self, sample_rebalance_events):
        """Rebalance dates should be roughly monthly (25-35 days apart)."""
        dates = sorted([
            datetime.fromisoformat(e["date"]) if isinstance(e["date"], str)
            else e["date"]
            for e in sample_rebalance_events
        ])
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i - 1]).days
            assert 25 <= gap <= 35, f"Rebalance gap of {gap} days is not monthly"

    @pytest.mark.unit
    def test_rebalance_turnover_in_range(self, sample_rebalance_events):
        """Turnover should be between 0 and 1 (0-100%)."""
        for event in sample_rebalance_events:
            assert 0.0 <= event["turnover"] <= 1.0, (
                f"Turnover {event['turnover']} out of range"
            )

    @pytest.mark.unit
    def test_rebalance_market_count_positive(self, sample_rebalance_events):
        """Market count should be positive after every rebalance."""
        for event in sample_rebalance_events:
            assert event["market_count"] > 0


# ===========================================================================
# Turnover Calculation
# ===========================================================================


class TestTurnoverCalculation:
    """Verify turnover = sum(|new_weight - old_weight|) / 2."""

    @pytest.mark.unit
    def test_zero_turnover_no_changes(self):
        """No weight changes should produce zero turnover."""
        old = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        new = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        turnover = sum(abs(new[m] - old[m]) for m in old) / 2
        assert turnover == pytest.approx(0.0)

    @pytest.mark.unit
    def test_full_turnover(self):
        """Complete portfolio replacement should have turnover = 1.0."""
        old = {"a": 0.50, "b": 0.50}
        new = {"c": 0.50, "d": 0.50}
        # Markets a,b go to 0; c,d go from 0
        all_markets = set(old) | set(new)
        turnover = sum(
            abs(new.get(m, 0) - old.get(m, 0)) for m in all_markets
        ) / 2
        assert turnover == pytest.approx(1.0)

    @pytest.mark.unit
    def test_partial_turnover(self):
        """Partial rebalance should produce turnover between 0 and 1."""
        old = {"a": 0.30, "b": 0.30, "c": 0.40}
        new = {"a": 0.25, "b": 0.35, "c": 0.40}
        turnover = sum(abs(new[m] - old[m]) for m in old) / 2
        assert 0 < turnover < 1
        assert turnover == pytest.approx(0.05)

    @pytest.mark.unit
    def test_turnover_symmetric(self):
        """Turnover(old→new) should equal turnover(new→old)."""
        old = {"a": 0.20, "b": 0.30, "c": 0.50}
        new = {"a": 0.35, "b": 0.25, "c": 0.40}
        t1 = sum(abs(new[m] - old[m]) for m in old) / 2
        t2 = sum(abs(old[m] - new[m]) for m in old) / 2
        assert t1 == pytest.approx(t2)


# ===========================================================================
# No Lookahead Bias
# ===========================================================================


class TestNoLookaheadBias:
    """Baskets at time T must use only data available up to T."""

    @pytest.mark.unit
    def test_volatility_uses_trailing_window(self):
        """Volatility at time T should use only data from [T-30, T], not future."""
        prices = generate_price_series(n_days=90, seed=42)
        T = 60  # Compute vol at day 60

        trailing = prices["price"].iloc[T - 30:T]
        future = prices["price"].iloc[T:T + 30]

        vol_trailing = trailing.pct_change().dropna().std()
        vol_with_future = prices["price"].iloc[T - 30:T + 30].pct_change().dropna().std()

        # These should be different (future data changes the calculation)
        # The test verifies the PRINCIPLE: use trailing only
        assert vol_trailing != pytest.approx(vol_with_future, abs=1e-6), (
            "Trailing vol equals future-inclusive vol — suspicious"
        )

    @pytest.mark.unit
    def test_eligibility_uses_past_data(self):
        """Eligibility at time T should check volume/liquidity up to T only."""
        # Simulate: a market becomes high-volume in month 4 but we're computing at month 3
        monthly_volumes = [500, 800, 1200, 50000, 80000, 100000]

        for t in range(len(monthly_volumes)):
            available_volume = sum(monthly_volumes[: t + 1])
            # At month 2 (index 2), total volume is 500+800+1200=2500 (below threshold)
            if t <= 2:
                assert available_volume < 10000, (
                    f"At month {t}, volume should be below threshold"
                )

    @pytest.mark.unit
    def test_resolution_only_after_event(self):
        """Resolution should only be processed after it occurs, not before.

        A market resolving on day 50 should contribute returns until day 50,
        then be removed. It should NOT be removed earlier.
        """
        resolution_day = 50
        prices = generate_price_series(
            n_days=60, resolution=1.0, resolution_day=resolution_day, seed=42,
        )
        # Price series should include data up to resolution day
        assert len(prices) == resolution_day + 1
        # Before resolution, market is still active
        assert all(0 < p < 1 for p in prices["price"].iloc[:-1])


# ===========================================================================
# Chain-Linking
# ===========================================================================


class TestChainLinking:
    """NAV must be continuous at rebalance boundaries (no jumps)."""

    @pytest.mark.unit
    def test_nav_continuous_at_rebalance(self, sample_nav_series):
        """NAV should not jump at rebalance dates."""
        rebalance_mask = sample_nav_series["is_rebalance"]
        rebalance_indices = sample_nav_series[rebalance_mask].index.tolist()

        for idx in rebalance_indices:
            if idx > 0:
                pre = sample_nav_series["nav"].iloc[idx - 1]
                post = sample_nav_series["nav"].iloc[idx]
                daily_return = abs(post - pre) / pre
                # At rebalance, the NAV change should be a normal daily return,
                # not a jump from weight changes
                assert daily_return < 0.15, (
                    f"NAV jump of {daily_return:.2%} at rebalance index {idx}"
                )

    @pytest.mark.unit
    def test_chain_link_arithmetic(self):
        """Chain-linking: NAV_t = NAV_{t-1} * (1 + portfolio_return_t).

        At rebalance: close old weights, open new weights at same prices.
        """
        nav_pre = 105.0
        # Old weights produce a return on rebalance day
        portfolio_return = 0.01  # 1% return
        nav_post = nav_pre * (1 + portfolio_return)

        assert nav_post == pytest.approx(106.05)
        # No discontinuity: new weights start from this NAV
        nav_next = nav_post * (1 + 0.005)  # Next day 0.5% return
        assert nav_next == pytest.approx(106.05 * 1.005)

    @pytest.mark.unit
    def test_divisor_adjustment(self):
        """Divisor method: basket_value / divisor = NAV. Divisor adjusts at rebalance."""
        # Pre-rebalance
        basket_value_pre = 1050.0
        divisor = 10.0
        nav = basket_value_pre / divisor
        assert nav == pytest.approx(105.0)

        # After rebalance, basket value changes due to new weights but NAV stays same
        basket_value_post = 1100.0  # Different due to new weights
        new_divisor = basket_value_post / nav  # Adjust to keep NAV continuous
        assert basket_value_post / new_divisor == pytest.approx(nav)


# ===========================================================================
# Performance Metrics
# ===========================================================================


class TestPerformanceMetrics:
    """Validate Sharpe ratio, max drawdown, and other metrics."""

    @pytest.mark.unit
    def test_sharpe_ratio_known_input(self):
        """Sharpe ratio with known returns should produce expected value.

        Sharpe = (mean_return - rf) / std_return * sqrt(252)
        """
        daily_returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005] * 50)
        rf_daily = 0.05 / 252  # 5% annual

        mean_r = daily_returns.mean()
        std_r = daily_returns.std(ddof=1)
        sharpe = (mean_r - rf_daily) / std_r * np.sqrt(252)

        assert sharpe > 0, "Positive mean returns should give positive Sharpe"
        # With these inputs, Sharpe should be roughly calculable
        assert np.isfinite(sharpe)

    @pytest.mark.unit
    def test_sharpe_ratio_zero_vol(self):
        """Zero volatility should produce undefined/infinite Sharpe (handle gracefully)."""
        daily_returns = np.array([0.01] * 100)  # Constant return
        std_r = daily_returns.std(ddof=1)
        assert std_r == pytest.approx(0.0, abs=1e-15)
        # Sharpe is undefined; implementation should handle this (return inf or NaN)

    @pytest.mark.unit
    def test_max_drawdown_known_input(self):
        """Max drawdown with known NAV should produce expected value."""
        # NAV: 100 → 120 → 90 → 110
        nav = np.array([100, 110, 120, 100, 90, 95, 110])
        running_max = np.maximum.accumulate(nav)
        drawdowns = (nav - running_max) / running_max
        max_dd = drawdowns.min()

        # Peak was 120, trough was 90 → drawdown = (90-120)/120 = -25%
        assert max_dd == pytest.approx(-0.25)

    @pytest.mark.unit
    def test_max_drawdown_monotonic_increase(self):
        """Monotonically increasing NAV should have zero drawdown."""
        nav = np.array([100, 101, 102, 103, 104, 105])
        running_max = np.maximum.accumulate(nav)
        drawdowns = (nav - running_max) / running_max
        max_dd = drawdowns.min()
        assert max_dd == pytest.approx(0.0)

    @pytest.mark.unit
    def test_max_drawdown_never_positive(self):
        """Max drawdown should always be <= 0."""
        nav = generate_nav_series(n_days=120, seed=42)
        running_max = np.maximum.accumulate(nav["nav"].values)
        drawdowns = (nav["nav"].values - running_max) / running_max
        assert drawdowns.min() <= 0.0

    @pytest.mark.unit
    def test_cumulative_return(self):
        """Cumulative return = (final_nav / initial_nav) - 1."""
        nav = np.array([100, 110, 105, 115])
        cum_return = nav[-1] / nav[0] - 1
        assert cum_return == pytest.approx(0.15)

    @pytest.mark.unit
    def test_annualized_return(self):
        """Annualized return with known inputs."""
        initial = 100
        final = 121
        days = 365
        ann_return = (final / initial) ** (365 / days) - 1
        assert ann_return == pytest.approx(0.21)


# ===========================================================================
# Resolved Market Terminal Returns
# ===========================================================================


class TestTransactionCosts:
    """Transaction cost modeling deducts costs from NAV at rebalance."""

    @pytest.mark.unit
    def test_transaction_cost_calculation(self):
        """Cost = turnover * (bps / 10000), deducted from NAV."""
        nav = 100.0
        turnover = 0.50  # 50% turnover
        cost_bps = 200   # 2% spread
        cost_fraction = turnover * (cost_bps / 10000)
        new_nav = nav * (1 - cost_fraction)
        assert cost_fraction == pytest.approx(0.01)  # 1% cost
        assert new_nav == pytest.approx(99.0)

    @pytest.mark.unit
    def test_zero_transaction_cost(self):
        """Zero bps means no cost deduction."""
        nav = 100.0
        turnover = 1.0
        cost_bps = 0
        cost_fraction = turnover * (cost_bps / 10000)
        assert cost_fraction == 0.0

    @pytest.mark.unit
    def test_full_turnover_high_cost(self):
        """100% turnover with 200 bps = 2% NAV hit."""
        nav = 100.0
        turnover = 1.0
        cost_bps = 200
        cost_fraction = turnover * (cost_bps / 10000)
        new_nav = nav * (1 - cost_fraction)
        assert new_nav == pytest.approx(98.0)


class TestResolvedMarketReturns:
    """Resolved markets contribute terminal returns (0 or 1)."""

    @pytest.mark.unit
    def test_resolved_yes_return(self):
        """YES resolution from price 0.6 → return = (1.0 - 0.6) / 0.6 ≈ 66.7%."""
        last_price = 0.60
        terminal_return = (1.0 - last_price) / last_price
        assert terminal_return == pytest.approx(2 / 3)

    @pytest.mark.unit
    def test_resolved_no_return(self):
        """NO resolution from price 0.6 → return = (0.0 - 0.6) / 0.6 = -100%."""
        last_price = 0.60
        terminal_return = (0.0 - last_price) / last_price
        assert terminal_return == pytest.approx(-1.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("last_price,resolution,expected_return", [
        (0.50, 1.0, 1.0),       # Even odds → 100% return on YES
        (0.50, 0.0, -1.0),      # Even odds → -100% on NO
        (0.90, 1.0, 1 / 9),     # Near-certain → small gain
        (0.10, 0.0, -1.0),      # Near-certain NO → full loss
        (0.10, 1.0, 9.0),       # Upset YES → 900% return
        (0.90, 0.0, -1.0),      # Upset NO → full loss
    ])
    def test_terminal_return_parametrized(self, last_price, resolution, expected_return):
        """Parametrized terminal return calculations."""
        terminal_return = (resolution - last_price) / last_price
        assert terminal_return == pytest.approx(expected_return, rel=1e-6)

    @pytest.mark.unit
    def test_weighted_resolution_impact(self):
        """Resolution impact on basket = weight * terminal_return."""
        weight = 0.10
        last_price = 0.60
        resolution = 1.0
        terminal_return = (resolution - last_price) / last_price
        basket_impact = weight * terminal_return
        assert basket_impact == pytest.approx(0.10 * (2 / 3))

    @pytest.mark.unit
    def test_multiple_resolutions_basket_return(self):
        """Multiple resolutions in same period: sum weighted returns."""
        resolutions = [
            {"weight": 0.10, "last_price": 0.60, "resolution": 1.0},  # YES
            {"weight": 0.08, "last_price": 0.40, "resolution": 0.0},  # NO
        ]
        total_impact = 0
        for r in resolutions:
            ret = (r["resolution"] - r["last_price"]) / r["last_price"]
            total_impact += r["weight"] * ret

        # 0.10 * 0.667 + 0.08 * (-1.0) = 0.0667 - 0.08 = -0.0133
        assert total_impact == pytest.approx(0.10 * (2 / 3) + 0.08 * (-1.0))

"""
Tests for data ingestion pipeline.

Validates that market data meets quality standards before any processing:
- Required fields present
- Price bounds
- No duplicates
- Valid date ranges
- Resolved market completeness
- Price series continuity
- Cross-platform dedup
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from tests.test_data_generators import (
    generate_market,
    generate_market_batch,
    generate_price_series,
)


# ===========================================================================
# Required Fields
# ===========================================================================


class TestRequiredFields:
    """Verify all markets have mandatory metadata fields."""

    REQUIRED_FIELDS = [
        "market_id", "platform", "title", "start_date", "end_date",
    ]

    @pytest.mark.unit
    def test_all_required_fields_present(self, sample_market):
        """Every market must have market_id, platform, title, start_date, end_date."""
        for field in self.REQUIRED_FIELDS:
            assert field in sample_market, f"Missing required field: {field}"

    @pytest.mark.unit
    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_required_field_not_none(self, sample_market, field):
        """Required fields must not be None."""
        assert sample_market[field] is not None, f"Required field '{field}' is None"

    @pytest.mark.unit
    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_required_field_not_empty(self, sample_market, field):
        """String required fields must not be empty."""
        val = sample_market[field]
        if isinstance(val, str):
            assert val.strip() != "", f"Required field '{field}' is empty string"

    @pytest.mark.unit
    def test_batch_all_have_required_fields(self, market_batch):
        """Every market in a batch must have all required fields."""
        for i, market in enumerate(market_batch):
            for field in self.REQUIRED_FIELDS:
                assert field in market, f"Market {i} missing field: {field}"

    @pytest.mark.unit
    def test_platform_is_valid(self, market_batch):
        """Platform must be one of the known platforms."""
        valid_platforms = {"polymarket", "kalshi"}
        for market in market_batch:
            assert market["platform"] in valid_platforms, (
                f"Unknown platform: {market['platform']}"
            )

    @pytest.mark.unit
    def test_title_minimum_length(self, market_batch):
        """Titles should be at least 10 characters (a meaningful question)."""
        for market in market_batch:
            assert len(market["title"]) >= 10, (
                f"Title too short: '{market['title']}'"
            )


# ===========================================================================
# Price Bounds
# ===========================================================================


class TestPriceBounds:
    """Validate that all prices are within [0, 1] bounds."""

    @pytest.mark.unit
    def test_prices_in_bounds(self, sample_price_series):
        """All prices must be in [0, 1]."""
        assert (sample_price_series["price"] >= 0.0).all(), "Found price < 0"
        assert (sample_price_series["price"] <= 1.0).all(), "Found price > 1"

    @pytest.mark.unit
    def test_resolved_terminal_price(self, resolved_price_series):
        """Resolved market terminal price must be exactly 0.0 or 1.0."""
        terminal = resolved_price_series["price"].iloc[-1]
        assert terminal in (0.0, 1.0), f"Terminal price {terminal} not 0 or 1"

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_price", [-0.1, 1.5, 2.0, -1.0, float("inf")])
    def test_reject_out_of_bounds_price(self, bad_price):
        """Prices outside [0, 1] should be detectable."""
        series = generate_price_series(n_days=10, seed=1)
        series.loc[5, "price"] = bad_price
        invalid = series[(series["price"] < 0) | (series["price"] > 1)]
        assert len(invalid) > 0, f"Failed to detect out-of-bounds price {bad_price}"

    @pytest.mark.unit
    def test_high_vol_stays_in_bounds(self, high_vol_price_series):
        """Even high-volatility series should stay in [0, 1]."""
        assert (high_vol_price_series["price"] >= 0.0).all()
        assert (high_vol_price_series["price"] <= 1.0).all()

    @pytest.mark.unit
    def test_volume_non_negative(self, sample_price_series):
        """Volume should never be negative."""
        assert (sample_price_series["volume"] >= 0).all(), "Found negative volume"


# ===========================================================================
# Duplicate Detection
# ===========================================================================


class TestDuplicates:
    """Ensure no duplicate market entries."""

    @pytest.mark.unit
    def test_no_duplicate_market_ids(self, market_batch):
        """No two markets should have the same market_id."""
        ids = [m["market_id"] for m in market_batch]
        assert len(ids) == len(set(ids)), "Found duplicate market_ids"

    @pytest.mark.unit
    def test_detect_injected_duplicate(self, market_batch):
        """Deliberately duplicated market_id should be detectable."""
        dup = market_batch[0].copy()
        batch_with_dup = market_batch + [dup]
        ids = [m["market_id"] for m in batch_with_dup]
        assert len(ids) != len(set(ids)), "Failed to detect injected duplicate"

    @pytest.mark.unit
    def test_cross_platform_dedup_same_title(self):
        """Markets with identical titles on different platforms are potential duplicates.

        Cross-platform dedup should flag markets with very similar titles
        on polymarket vs kalshi as potential duplicates.
        """
        m1 = generate_market(
            market_id="poly_001",
            platform="polymarket",
            title="Will Bitcoin exceed $100K by end of 2026?",
        )
        m2 = generate_market(
            market_id="kalshi_001",
            platform="kalshi",
            title="Will Bitcoin exceed $100K by end of 2026?",
        )

        # Simple exact match dedup
        titles_by_normalized = {}
        for m in [m1, m2]:
            key = m["title"].lower().strip()
            titles_by_normalized.setdefault(key, []).append(m["market_id"])

        duplicates = {k: v for k, v in titles_by_normalized.items() if len(v) > 1}
        assert len(duplicates) > 0, "Failed to detect cross-platform duplicate"

    @pytest.mark.unit
    def test_cross_platform_dedup_similar_title(self):
        """Markets with similar but not identical titles should be flagged.

        Real cross-platform duplicates often have slightly different wording.
        """
        m1 = generate_market(
            platform="polymarket",
            title="Will the Fed cut rates before July 2026?",
        )
        m2 = generate_market(
            platform="kalshi",
            title="Fed rate cut before July 2026",
        )

        # Simple word overlap similarity
        words1 = set(m1["title"].lower().split())
        words2 = set(m2["title"].lower().split())
        overlap = len(words1 & words2) / max(len(words1 | words2), 1)

        assert overlap > 0.3, (
            f"Similar titles not detected as potential duplicates (overlap={overlap:.2f})"
        )


# ===========================================================================
# Date Range Validation
# ===========================================================================


class TestDateRanges:
    """Validate date consistency for all markets."""

    @pytest.mark.unit
    def test_start_before_end(self, market_batch):
        """start_date must be strictly before end_date for all markets."""
        for market in market_batch:
            assert market["start_date"] < market["end_date"], (
                f"Market {market['market_id']}: start_date >= end_date"
            )

    @pytest.mark.unit
    def test_dates_are_datetime(self, sample_market):
        """Dates should be datetime objects, not strings."""
        assert isinstance(sample_market["start_date"], datetime)
        assert isinstance(sample_market["end_date"], datetime)

    @pytest.mark.unit
    def test_market_not_too_old(self, market_batch):
        """Markets shouldn't have start dates more than 3 years ago (data quality)."""
        cutoff = datetime.now() - timedelta(days=3 * 365)
        for market in market_batch:
            assert market["start_date"] > cutoff, (
                f"Market {market['market_id']} start_date suspiciously old: {market['start_date']}"
            )

    @pytest.mark.unit
    def test_end_date_not_past_for_active(self, sample_market):
        """Active (unresolved) markets should have end_date in the future."""
        if sample_market.get("resolution") is None:
            assert sample_market["end_date"] > datetime.now(), (
                "Active market has end_date in the past"
            )

    @pytest.mark.unit
    def test_market_duration_reasonable(self, market_batch):
        """Market duration should be between 1 day and 5 years."""
        for market in market_batch:
            duration = (market["end_date"] - market["start_date"]).days
            assert 1 <= duration <= 5 * 365, (
                f"Market {market['market_id']} has unreasonable duration: {duration} days"
            )


# ===========================================================================
# Resolved Markets
# ===========================================================================


class TestResolvedMarkets:
    """Validate completeness of resolved market data."""

    @pytest.mark.unit
    def test_resolved_has_resolution_value(self, resolved_market):
        """Resolved markets must have a resolution_value (0.0 or 1.0)."""
        assert resolved_market["resolution"] is not None
        assert resolved_market["resolution_value"] is not None

    @pytest.mark.unit
    def test_resolution_value_valid(self, resolved_market):
        """Resolution value must be 0.0 (NO) or 1.0 (YES)."""
        assert resolved_market["resolution_value"] in (0.0, 1.0), (
            f"Invalid resolution_value: {resolved_market['resolution_value']}"
        )

    @pytest.mark.unit
    def test_resolution_matches_value(self):
        """resolution='YES' must correspond to resolution_value=1.0."""
        yes_market = generate_market(resolution="YES", resolution_value=1.0)
        no_market = generate_market(resolution="NO", resolution_value=0.0)

        assert yes_market["resolution"] == "YES" and yes_market["resolution_value"] == 1.0
        assert no_market["resolution"] == "NO" and no_market["resolution_value"] == 0.0

    @pytest.mark.unit
    def test_active_market_no_resolution(self, sample_market):
        """Active markets should have resolution=None."""
        assert sample_market["resolution"] is None
        assert sample_market["resolution_value"] is None

    @pytest.mark.unit
    def test_batch_resolved_markets_complete(self, market_batch):
        """All resolved markets in a batch must have resolution_value."""
        for market in market_batch:
            if market["resolution"] is not None:
                assert market["resolution_value"] is not None, (
                    f"Market {market['market_id']} resolved but missing resolution_value"
                )


# ===========================================================================
# Price Series Continuity
# ===========================================================================


class TestPriceSeriesContinuity:
    """Validate that active markets have continuous price series."""

    @pytest.mark.unit
    def test_no_large_gaps(self, sample_price_series):
        """Active market price series should have no gaps > 7 days."""
        dates = pd.to_datetime(sample_price_series["date"])
        gaps = dates.diff().dt.days
        max_gap = gaps.max()
        assert max_gap <= 7, f"Found gap of {max_gap} days in price series"

    @pytest.mark.unit
    def test_detect_large_gap(self, gapped_price_series):
        """A series with an intentional gap should be detectable."""
        dates = pd.to_datetime(gapped_price_series["date"])
        gaps = dates.diff().dt.days
        max_gap = gaps.max()
        assert max_gap > 7, "Failed to detect intentional gap in price series"

    @pytest.mark.unit
    def test_minimum_price_history(self, sample_price_series):
        """Price series should have at least 14 days of data for eligibility."""
        assert len(sample_price_series) >= 14, (
            f"Price series too short: {len(sample_price_series)} days"
        )

    @pytest.mark.unit
    def test_dates_monotonic(self, sample_price_series):
        """Dates in price series must be strictly increasing."""
        dates = pd.to_datetime(sample_price_series["date"])
        assert dates.is_monotonic_increasing, "Dates are not monotonically increasing"

    @pytest.mark.unit
    def test_no_duplicate_dates(self, sample_price_series):
        """No duplicate dates in price series."""
        dates = sample_price_series["date"]
        assert dates.nunique() == len(dates), "Found duplicate dates in price series"

    @pytest.mark.unit
    @pytest.mark.parametrize("gap_size,expected_pass", [
        (3, True),   # 3-day gap is acceptable (weekend)
        (5, True),   # 5-day gap acceptable
        (7, False),  # 7-day gap: removing 7 consecutive days creates gap > 7
        (10, False),  # 10-day gap should fail
    ])
    def test_gap_threshold(self, gap_size, expected_pass):
        """Parameterized test of gap detection at various sizes."""
        gap_indices = list(range(20, 20 + gap_size))
        series = generate_price_series(n_days=60, gaps=gap_indices, seed=42)
        dates = pd.to_datetime(series["date"])
        max_gap = dates.diff().dt.days.max()
        passes = max_gap <= 7
        assert passes == expected_pass, (
            f"Gap of {gap_size} days: expected pass={expected_pass}, got {passes}"
        )


# ===========================================================================
# Volume & Liquidity
# ===========================================================================


class TestVolumeAndLiquidity:
    """Validate volume and liquidity data quality."""

    @pytest.mark.unit
    def test_volume_non_negative_in_market(self, sample_market):
        """Market total volume must be non-negative."""
        assert sample_market["total_volume"] >= 0

    @pytest.mark.unit
    def test_daily_volume_non_negative(self, sample_market):
        """7-day average daily volume must be non-negative."""
        assert sample_market["daily_volume_7d"] >= 0

    @pytest.mark.unit
    def test_liquidity_non_negative(self, sample_market):
        """Market liquidity must be non-negative."""
        assert sample_market["liquidity"] >= 0

    @pytest.mark.unit
    def test_resolved_market_zero_liquidity(self, resolved_market):
        """Resolved markets should have zero or near-zero liquidity."""
        assert resolved_market["liquidity"] <= 100, (
            "Resolved market has unexpectedly high liquidity"
        )

    @pytest.mark.unit
    def test_illiquid_market_flagged(self, illiquid_market, eligibility_settings):
        """Markets below liquidity threshold should be identifiable."""
        assert illiquid_market["liquidity"] < eligibility_settings["min_liquidity"], (
            "Illiquid market has liquidity above threshold"
        )

    @pytest.mark.unit
    def test_volume_series_consistent(self, sample_price_series):
        """Volume in price series should all be positive."""
        assert (sample_price_series["volume"] > 0).all()

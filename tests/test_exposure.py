"""Tests for the side detection and exposure system.

Validates:
- Phrasing polarity detection (positive/negative/neutral)
- Token side detection (YES/NO/categorical)
- Exposure direction computation
- Opposing exposure filtering
- Conflict detection
- Exposure report generation
- Edge cases: double negation, over/under, categorical markets
"""

import pytest
from src.exposure.side_detection import (
    detect_phrasing_polarity,
    detect_token_side,
    compute_exposure_direction,
    detect_side_batch,
)
from src.exposure.normalization import (
    ExposureInfo,
    normalize_exposures,
    adjust_return_for_exposure,
)
from src.exposure.basket_rules import (
    check_exposure_conflicts,
    filter_opposing_exposures,
)
from src.exposure.report import generate_exposure_report
import pandas as pd


# ===========================================================================
# Phrasing Polarity Detection
# ===========================================================================


class TestPhrasingPolarity:
    """Detect whether market titles are phrased positively or negatively."""

    @pytest.mark.unit
    @pytest.mark.parametrize("title,expected", [
        ("Will Bitcoin exceed $100K?", "positive"),
        ("Will the Fed cut rates?", "positive"),
        ("Will Harris win the election?", "positive"),
        ("Will GDP growth be above 3%?", "positive"),
        ("Will AI surpass human performance?", "positive"),
    ])
    def test_positive_phrasing(self, title, expected):
        assert detect_phrasing_polarity(title) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("title,expected", [
        ("Will the US NOT withdraw from NATO?", "negative"),
        ("Will Bitcoin fail to reach $100K?", "negative"),
        ("Will unemployment fall below 4%?", "negative"),
        ("Will the bill be vetoed?", "negative"),
        ("Will there be a recession in 2026?", "negative"),
        ("Will the government shutdown?", "negative"),
        ("Won't the Fed raise rates?", "negative"),
    ])
    def test_negative_phrasing(self, title, expected):
        assert detect_phrasing_polarity(title) == expected

    @pytest.mark.unit
    def test_double_negation_is_positive(self):
        """'Will Bitcoin NOT fall below 50K?' — negation of a negative = positive."""
        assert detect_phrasing_polarity("Will Bitcoin not fall below $50K?") == "positive"
        assert detect_phrasing_polarity("Won't unemployment drop below 3%?") == "positive"

    @pytest.mark.unit
    def test_empty_title_is_neutral(self):
        assert detect_phrasing_polarity("") == "neutral"
        assert detect_phrasing_polarity(None) == "neutral"

    @pytest.mark.unit
    def test_over_under_patterns(self):
        assert detect_phrasing_polarity("Will Bitcoin be above $100K?") == "positive"
        assert detect_phrasing_polarity("Will Bitcoin be below $100K?") == "negative"
        assert detect_phrasing_polarity("Over 2.5 goals in the match?") == "positive"
        assert detect_phrasing_polarity("Under 2.5 goals in the match?") == "negative"


# ===========================================================================
# Token Side Detection
# ===========================================================================


class TestTokenSide:
    @pytest.mark.unit
    def test_binary_default_yes(self):
        assert detect_token_side("Will X happen?") == "YES"

    @pytest.mark.unit
    def test_binary_no_token(self):
        assert detect_token_side("Will X happen?", ["Yes", "No"], tracked_token_index=1) == "NO"

    @pytest.mark.unit
    def test_categorical_outcome(self):
        outcomes = ["Harris", "Trump", "DeSantis", "Other"]
        assert detect_token_side("Who wins?", outcomes, tracked_token_index=0) == "Harris"
        assert detect_token_side("Who wins?", outcomes, tracked_token_index=1) == "Trump"


# ===========================================================================
# Exposure Direction
# ===========================================================================


class TestExposureDirection:
    @pytest.mark.unit
    def test_positive_yes_is_long(self):
        assert compute_exposure_direction("positive", "YES") == "long"

    @pytest.mark.unit
    def test_positive_no_is_short(self):
        assert compute_exposure_direction("positive", "NO") == "short"

    @pytest.mark.unit
    def test_negative_yes_is_short(self):
        """'Will US NOT withdraw?' YES = bad thing confirmed = short exposure."""
        assert compute_exposure_direction("negative", "YES") == "short"

    @pytest.mark.unit
    def test_negative_no_is_long(self):
        """'Will there be a recession?' NO = recession avoided = long."""
        assert compute_exposure_direction("negative", "NO") == "long"

    @pytest.mark.unit
    def test_neutral_yes_defaults_long(self):
        assert compute_exposure_direction("neutral", "YES") == "long"


# ===========================================================================
# Batch Side Detection
# ===========================================================================


class TestBatchDetection:
    @pytest.mark.unit
    def test_detect_side_batch_adds_columns(self):
        df = pd.DataFrame({
            "market_id": ["m1", "m2", "m3"],
            "title": [
                "Will Bitcoin exceed $100K?",
                "Will the US NOT withdraw from NATO?",
                "Will there be a recession?",
            ],
        })
        result = detect_side_batch(df)
        assert "phrasing_polarity" in result.columns
        assert "token_side" in result.columns
        assert "exposure_direction" in result.columns
        assert "normalized_direction" in result.columns

        assert result.loc[0, "phrasing_polarity"] == "positive"
        assert result.loc[0, "exposure_direction"] == "long"
        assert result.loc[1, "phrasing_polarity"] == "negative"
        assert result.loc[1, "exposure_direction"] == "short"
        assert result.loc[2, "phrasing_polarity"] == "negative"
        assert result.loc[2, "exposure_direction"] == "short"


# ===========================================================================
# Return Adjustment
# ===========================================================================


class TestReturnAdjustment:
    @pytest.mark.unit
    def test_long_positive_return(self):
        """Long position, price goes up → positive adjusted return."""
        assert adjust_return_for_exposure(0.05, 1.0) == 0.05

    @pytest.mark.unit
    def test_short_positive_return(self):
        """Short exposure, price goes up → negative adjusted return (bad outcome more likely)."""
        assert adjust_return_for_exposure(0.05, -1.0) == -0.05

    @pytest.mark.unit
    def test_short_negative_return(self):
        """Short exposure, price goes down → positive adjusted return (bad outcome less likely)."""
        assert adjust_return_for_exposure(-0.05, -1.0) == 0.05


# ===========================================================================
# Exposure Conflicts
# ===========================================================================


class TestExposureConflicts:
    def _make_exposure_map(self):
        return {
            "m1": ExposureInfo("m1", "event_A", "YES", "positive", "long", 1.0),
            "m2": ExposureInfo("m2", "event_A", "NO", "positive", "short", -1.0),
            "m3": ExposureInfo("m3", "event_B", "YES", "positive", "long", 1.0),
            "m4": ExposureInfo("m4", "event_C", "Harris", "neutral", "long", 1.0),
            "m5": ExposureInfo("m5", "event_C", "Trump", "neutral", "long", 1.0),
        }

    @pytest.mark.unit
    def test_detect_opposing_exposure(self):
        """Same event, YES + NO = opposing."""
        emap = self._make_exposure_map()
        conflicts = check_exposure_conflicts(["m1", "m2", "m3"], emap)
        assert len(conflicts) == 1
        assert conflicts[0]["type"] == "opposing_exposure"
        assert conflicts[0]["event_id"] == "event_A"

    @pytest.mark.unit
    def test_detect_categorical_overlap(self):
        """Same event, multiple categorical outcomes."""
        emap = self._make_exposure_map()
        conflicts = check_exposure_conflicts(["m4", "m5"], emap)
        assert len(conflicts) == 1
        assert conflicts[0]["type"] == "categorical_overlap"

    @pytest.mark.unit
    def test_no_conflict_different_events(self):
        emap = self._make_exposure_map()
        conflicts = check_exposure_conflicts(["m1", "m3"], emap)
        assert len(conflicts) == 0


# ===========================================================================
# Opposing Exposure Filtering
# ===========================================================================


class TestFilterOpposingExposures:
    @pytest.mark.unit
    def test_keeps_most_liquid(self):
        emap = {
            "m1": ExposureInfo("m1", "event_A", "YES", "positive", "long", 1.0),
            "m2": ExposureInfo("m2", "event_A", "NO", "positive", "short", -1.0),
            "m3": ExposureInfo("m3", "event_B", "YES", "positive", "long", 1.0),
        }
        liquidity = {"m1": 5000, "m2": 10000, "m3": 8000}
        result = filter_opposing_exposures(["m1", "m2", "m3"], emap, liquidity)
        assert "m2" in result  # more liquid
        assert "m1" not in result
        assert "m3" in result
        assert len(result) == 2

    @pytest.mark.unit
    def test_no_event_markets_pass_through(self):
        emap = {
            "m1": ExposureInfo("m1", None, "YES", "positive", "long", 1.0),
        }
        result = filter_opposing_exposures(["m1"], emap)
        assert result == ["m1"]


# ===========================================================================
# Exposure Report
# ===========================================================================


class TestExposureReport:
    @pytest.mark.unit
    def test_report_basic(self):
        weights = {"m1": 0.5, "m2": 0.3, "m3": 0.2}
        emap = {
            "m1": ExposureInfo("m1", "e1", "YES", "positive", "long", 1.0),
            "m2": ExposureInfo("m2", "e2", "YES", "negative", "short", -1.0),
            "m3": ExposureInfo("m3", "e3", "YES", "positive", "long", 1.0),
        }
        report = generate_exposure_report(weights, emap)
        assert report["long_weight"] == 0.7
        assert report["short_weight"] == 0.3
        assert report["net_exposure"] == pytest.approx(0.4)  # 0.7 - 0.3
        assert report["net_direction"] == "LONG"
        assert len(report["summary"]) > 0

    @pytest.mark.unit
    def test_report_with_conflicts(self):
        weights = {"m1": 0.5, "m2": 0.5}
        emap = {
            "m1": ExposureInfo("m1", "event_A", "YES", "positive", "long", 1.0),
            "m2": ExposureInfo("m2", "event_A", "NO", "positive", "short", -1.0),
        }
        report = generate_exposure_report(weights, emap)
        assert len(report["conflicts"]) == 1
        assert report["net_exposure"] == pytest.approx(0.0)
        assert report["net_direction"] == "NEUTRAL"

    @pytest.mark.unit
    def test_all_long_basket(self):
        weights = {"m1": 0.5, "m2": 0.5}
        emap = {
            "m1": ExposureInfo("m1", "e1", "YES", "positive", "long", 1.0),
            "m2": ExposureInfo("m2", "e2", "YES", "positive", "long", 1.0),
        }
        report = generate_exposure_report(weights, emap)
        assert report["net_direction"] == "LONG"
        assert report["short_weight"] == 0.0

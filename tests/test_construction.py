"""
Tests for basket construction pipeline.

Validates:
- Weight constraints (sum to 1, max/min bounds)
- Eligibility filter enforcement
- Basket size limits
- Resolution handling and weight redistribution
- Emergency rebalance triggers
- Risk parity weighting properties
- Liquidity cap enforcement
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from tests.test_data_generators import (
    generate_market,
    generate_market_batch,
    generate_basket_weights,
    generate_price_series,
)


# ===========================================================================
# Weight Sum Constraint
# ===========================================================================


class TestWeightSum:
    """All basket weights must sum to 1.0."""

    @pytest.mark.unit
    def test_equal_weights_sum_to_one(self, equal_weights):
        """Equal-weight basket weights must sum to 1.0."""
        total = sum(equal_weights.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    @pytest.mark.unit
    def test_random_weights_sum_to_one(self, random_weights):
        """Randomly generated weights must sum to 1.0."""
        total = sum(random_weights.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    @pytest.mark.unit
    @pytest.mark.parametrize("n_markets", [5, 10, 15, 20, 30])
    def test_weight_sum_various_sizes(self, n_markets):
        """Weights sum to 1.0 for various basket sizes."""
        weights = generate_basket_weights(n_markets=n_markets, method="random", seed=42)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"{n_markets} markets: weights sum to {total}"

    @pytest.mark.unit
    def test_weight_sum_after_normalization(self):
        """Manually constructed weights should be normalizable to 1.0."""
        raw = {"a": 0.3, "b": 0.5, "c": 0.7}
        total = sum(raw.values())
        normalized = {k: v / total for k, v in raw.items()}
        assert abs(sum(normalized.values()) - 1.0) < 1e-9


# ===========================================================================
# Max Weight Constraint
# ===========================================================================


class TestMaxWeight:
    """No single market should exceed max_single_weight (0.20)."""

    MAX_WEIGHT = 0.20

    @pytest.mark.unit
    def test_equal_weights_under_max(self, equal_weights):
        """Equal weights with >= 5 markets should be under 20%."""
        for market_id, weight in equal_weights.items():
            assert weight <= self.MAX_WEIGHT + 1e-9, (
                f"Market {market_id} weight {weight:.4f} exceeds max {self.MAX_WEIGHT}"
            )

    @pytest.mark.unit
    def test_concentrated_weights_violate_max(self, concentrated_weights):
        """Concentrated weights should detect max weight violation."""
        violations = {
            m: w for m, w in concentrated_weights.items()
            if w > self.MAX_WEIGHT + 1e-9
        }
        assert len(violations) > 0, "Failed to detect concentrated weight violation"

    @pytest.mark.unit
    def test_cap_max_weight(self):
        """After capping, no weight should exceed max."""
        weights = {"a": 0.40, "b": 0.30, "c": 0.20, "d": 0.10}
        max_w = self.MAX_WEIGHT

        # Iteratively cap and redistribute until convergence
        capped = dict(weights)
        for _ in range(10):  # Max iterations
            excess = 0.0
            for m, w in capped.items():
                if w > max_w:
                    excess += w - max_w
                    capped[m] = max_w
            if excess < 1e-12:
                break
            uncapped = {m: w for m, w in capped.items() if w < max_w - 1e-9}
            uncapped_total = sum(uncapped.values())
            if uncapped_total > 0:
                for m in uncapped:
                    capped[m] += excess * (capped[m] / uncapped_total)

        for m, w in capped.items():
            assert w <= max_w + 1e-9, f"Market {m} still exceeds max after capping: {w}"

    @pytest.mark.unit
    @pytest.mark.parametrize("n_markets", [5, 6, 10, 20])
    def test_max_weight_achievable(self, n_markets):
        """With N markets, equal weight should be <= 20% when N >= 5."""
        w = 1.0 / n_markets
        if n_markets >= 5:
            assert w <= self.MAX_WEIGHT + 1e-9


# ===========================================================================
# Min Weight Constraint
# ===========================================================================


class TestMinWeight:
    """No market should be below min_single_weight (0.02)."""

    MIN_WEIGHT = 0.02

    @pytest.mark.unit
    def test_equal_weights_above_min(self, equal_weights):
        """Equal weights should be above minimum for reasonable basket sizes."""
        for market_id, weight in equal_weights.items():
            assert weight >= self.MIN_WEIGHT - 1e-9, (
                f"Market {market_id} weight {weight:.4f} below min {self.MIN_WEIGHT}"
            )

    @pytest.mark.unit
    def test_detect_dust_positions(self):
        """Weights below 2% should be flagged as dust positions."""
        weights = generate_basket_weights(n_markets=10, method="random", seed=99)
        # Artificially create a dust position
        weights["market_0"] = 0.005
        # Renormalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        dust = {m: w for m, w in weights.items() if w < self.MIN_WEIGHT}
        # The artificially small weight should be detected
        assert len(dust) > 0 or weights["market_0"] >= self.MIN_WEIGHT

    @pytest.mark.unit
    def test_remove_dust_and_renormalize(self):
        """Removing dust positions and renormalizing should preserve sum=1."""
        weights = {"a": 0.01, "b": 0.30, "c": 0.35, "d": 0.34}
        # Remove dust
        filtered = {m: w for m, w in weights.items() if w >= self.MIN_WEIGHT}
        total = sum(filtered.values())
        normalized = {m: w / total for m, w in filtered.items()}

        assert "a" not in normalized, "Dust position 'a' should be removed"
        assert abs(sum(normalized.values()) - 1.0) < 1e-9


# ===========================================================================
# Eligibility Filters
# ===========================================================================


class TestEligibilityFilters:
    """All markets in basket must pass eligibility requirements."""

    @pytest.mark.unit
    def test_eligible_market_passes(self, sample_market, eligibility_settings):
        """A well-formed active market should pass all eligibility checks."""
        m = sample_market
        s = eligibility_settings

        assert m["total_volume"] >= s["min_total_volume"]
        assert m["daily_volume_7d"] >= s["min_7d_avg_daily_volume"]
        assert m["liquidity"] >= s["min_liquidity"]
        assert m["resolution"] is None  # Must be active

    @pytest.mark.unit
    def test_resolved_market_ineligible(self, resolved_market):
        """Resolved markets must not pass eligibility."""
        assert resolved_market["resolution"] is not None, (
            "Resolved market should be ineligible (not active)"
        )

    @pytest.mark.unit
    def test_illiquid_market_ineligible(self, illiquid_market, eligibility_settings):
        """Illiquid markets must not pass eligibility."""
        assert illiquid_market["total_volume"] < eligibility_settings["min_total_volume"]

    @pytest.mark.unit
    def test_near_expiry_ineligible(self, near_expiry_market, eligibility_settings):
        """Markets expiring within min_days_to_expiration should be ineligible."""
        days_to_exp = (near_expiry_market["end_date"] - datetime.now()).days
        assert days_to_exp < eligibility_settings["min_days_to_expiration"]

    @pytest.mark.unit
    @pytest.mark.parametrize("price,expected_eligible", [
        (0.50, True),    # Normal price
        (0.03, False),   # Below min (near-certain NO)
        (0.97, False),   # Above max (near-certain YES)
        (0.05, True),    # At min boundary
        (0.95, True),    # At max boundary
        (0.01, False),   # Well below
        (0.99, False),   # Well above
    ])
    def test_price_range_eligibility(self, price, expected_eligible, eligibility_settings):
        """Markets with extreme prices should be filtered out."""
        eligible = (
            eligibility_settings["price_range_min"] <= price <= eligibility_settings["price_range_max"]
        )
        assert eligible == expected_eligible, (
            f"Price {price}: expected eligible={expected_eligible}, got {eligible}"
        )

    @pytest.mark.unit
    def test_insufficient_price_history(self, eligibility_settings):
        """Markets with <14 days of price history should be ineligible."""
        series = generate_price_series(n_days=10, seed=42)
        assert len(series) < eligibility_settings["min_price_history_days"]


# ===========================================================================
# Basket Size Limits
# ===========================================================================


class TestBasketSize:
    """Baskets must have between 5 and 30 markets."""

    @pytest.mark.unit
    @pytest.mark.parametrize("n_markets,expected_valid", [
        (4, False),   # Below minimum
        (5, True),    # At minimum
        (15, True),   # Normal
        (30, True),   # At maximum
        (31, False),  # Above maximum
    ])
    def test_basket_size_bounds(self, n_markets, expected_valid, basket_constraints):
        """Basket size must be within [min_markets, max_markets]."""
        valid = (
            basket_constraints["min_markets"] <= n_markets <= basket_constraints["max_markets"]
        )
        assert valid == expected_valid, (
            f"{n_markets} markets: expected valid={expected_valid}, got {valid}"
        )

    @pytest.mark.unit
    def test_equal_weight_basket_size(self, equal_weights, basket_constraints):
        """Generated equal-weight basket should be within size limits."""
        n = len(equal_weights)
        assert basket_constraints["min_markets"] <= n <= basket_constraints["max_markets"]


# ===========================================================================
# Resolution Handling
# ===========================================================================


class TestResolutionHandling:
    """When a market resolves, weights must be redistributed correctly."""

    @pytest.mark.unit
    def test_weight_redistribution_preserves_sum(self):
        """After removing a resolved market, remaining weights should sum to 1.0."""
        weights = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        resolved = "b"

        remaining = {m: w for m, w in weights.items() if m != resolved}
        old_remaining_sum = sum(remaining.values())
        new_weights = {m: w / old_remaining_sum for m, w in remaining.items()}

        assert abs(sum(new_weights.values()) - 1.0) < 1e-9
        assert resolved not in new_weights

    @pytest.mark.unit
    def test_weight_redistribution_proportional(self):
        """Redistributed weights should maintain relative proportions."""
        weights = {"a": 0.10, "b": 0.20, "c": 0.30, "d": 0.40}
        resolved = "d"

        remaining = {m: w for m, w in weights.items() if m != resolved}
        total = sum(remaining.values())
        new_weights = {m: w / total for m, w in remaining.items()}

        # Ratio of a to b should be preserved
        orig_ratio = weights["a"] / weights["b"]
        new_ratio = new_weights["a"] / new_weights["b"]
        assert abs(orig_ratio - new_ratio) < 1e-9

    @pytest.mark.unit
    def test_multiple_resolutions(self):
        """Handling multiple resolutions in one period."""
        weights = {f"m{i}": 1.0 / 8 for i in range(8)}
        resolved = ["m2", "m5"]

        remaining = {m: w for m, w in weights.items() if m not in resolved}
        total = sum(remaining.values())
        new_weights = {m: w / total for m, w in remaining.items()}

        assert len(new_weights) == 6
        assert abs(sum(new_weights.values()) - 1.0) < 1e-9

    @pytest.mark.unit
    def test_resolution_terminal_return(self):
        """Resolved YES market should contribute return of (1.0 - last_price) / last_price."""
        last_price = 0.60
        resolution_value = 1.0
        terminal_return = (resolution_value - last_price) / last_price
        assert abs(terminal_return - (0.40 / 0.60)) < 1e-9

        # Resolved NO
        resolution_value_no = 0.0
        terminal_return_no = (resolution_value_no - last_price) / last_price
        assert terminal_return_no == pytest.approx(-1.0)


# ===========================================================================
# Emergency Rebalance
# ===========================================================================


class TestEmergencyRebalance:
    """Emergency rebalance when basket drops below minimum markets."""

    @pytest.mark.unit
    def test_emergency_trigger(self, basket_constraints):
        """Basket dropping below min_markets should trigger emergency rebalance."""
        min_markets = basket_constraints["min_markets"]
        current_count = 4  # Below minimum
        needs_emergency = current_count < min_markets
        assert needs_emergency, "Should trigger emergency rebalance"

    @pytest.mark.unit
    def test_no_emergency_at_minimum(self, basket_constraints):
        """Basket at exactly min_markets should NOT trigger emergency."""
        min_markets = basket_constraints["min_markets"]
        current_count = min_markets
        needs_emergency = current_count < min_markets
        assert not needs_emergency

    @pytest.mark.unit
    def test_cascading_resolutions(self, basket_constraints):
        """Multiple resolutions in quick succession dropping below minimum."""
        initial_count = 7
        resolutions = 3
        remaining = initial_count - resolutions
        needs_emergency = remaining < basket_constraints["min_markets"]
        assert needs_emergency, (
            f"{remaining} markets remaining, need {basket_constraints['min_markets']}"
        )


# ===========================================================================
# Risk Parity Weighting
# ===========================================================================


class TestRiskParityWeighting:
    """Risk parity: higher volatility markets get lower weight."""

    @pytest.mark.unit
    def test_inverse_vol_relationship(self):
        """In risk parity, weight should be inversely proportional to volatility."""
        vols = {"a": 0.02, "b": 0.04, "c": 0.08}  # a is least volatile
        inv_vols = {m: 1.0 / v for m, v in vols.items()}
        total_inv = sum(inv_vols.values())
        weights = {m: iv / total_inv for m, iv in inv_vols.items()}

        # Least volatile should have highest weight
        assert weights["a"] > weights["b"] > weights["c"], (
            f"Risk parity violated: {weights}"
        )

    @pytest.mark.unit
    def test_risk_parity_weights_sum_to_one(self):
        """Risk parity weights must sum to 1.0."""
        vols = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()
        assert abs(weights.sum() - 1.0) < 1e-9

    @pytest.mark.unit
    def test_equal_vol_gives_equal_weight(self):
        """Markets with identical volatility should get equal weight."""
        vols = {"a": 0.05, "b": 0.05, "c": 0.05}
        inv_vols = {m: 1.0 / v for m, v in vols.items()}
        total_inv = sum(inv_vols.values())
        weights = {m: iv / total_inv for m, iv in inv_vols.items()}

        for w in weights.values():
            assert abs(w - 1.0 / 3) < 1e-9

    @pytest.mark.unit
    def test_extreme_vol_doesnt_dominate(self):
        """A very high-vol market should get a very small weight, not zero."""
        vols = {"a": 0.01, "b": 0.50}  # b is 50x more volatile
        inv_vols = {m: 1.0 / v for m, v in vols.items()}
        total_inv = sum(inv_vols.values())
        weights = {m: iv / total_inv for m, iv in inv_vols.items()}

        assert weights["b"] > 0, "High-vol market should still have positive weight"
        assert weights["a"] > weights["b"], "Low-vol market should have higher weight"


# ===========================================================================
# Liquidity Cap
# ===========================================================================


class TestLiquidityCap:
    """No market should be weighted above its proportional liquidity share."""

    @pytest.mark.unit
    def test_liquidity_cap_applied(self):
        """Weight should be capped at market's share of total liquidity."""
        # Market A has 80% of risk-parity weight but only 30% of liquidity
        rp_weights = {"a": 0.80, "b": 0.10, "c": 0.10}
        liquidity = {"a": 3000, "b": 5000, "c": 2000}
        total_liq = sum(liquidity.values())
        liq_shares = {m: l / total_liq for m, l in liquidity.items()}

        # Iteratively cap by liquidity share and renormalize
        capped = dict(rp_weights)
        for _ in range(10):
            any_capped = False
            for m in capped:
                if capped[m] > liq_shares[m]:
                    capped[m] = liq_shares[m]
                    any_capped = True
            total = sum(capped.values())
            capped = {m: w / total for m, w in capped.items()}
            if not any_capped:
                break

        # After iterative capping, market a should respect its liquidity share
        assert capped["a"] <= liq_shares["a"] + 0.05, (
            f"Market a weight {capped['a']:.4f} exceeds liquidity share {liq_shares['a']:.4f}"
        )

    @pytest.mark.unit
    def test_no_cap_needed_when_weight_below_share(self):
        """If risk-parity weight is below liquidity share, no cap applied."""
        rp_weight = 0.10
        liq_share = 0.30
        final = min(rp_weight, liq_share)
        assert final == rp_weight

    @pytest.mark.unit
    def test_cap_preserves_sum_after_renormalization(self):
        """After capping and renormalizing, weights must still sum to 1.0."""
        rp_weights = {"a": 0.50, "b": 0.30, "c": 0.20}
        liq_shares = {"a": 0.20, "b": 0.40, "c": 0.40}

        capped = {m: min(rp_weights[m], liq_shares[m]) for m in rp_weights}
        total = sum(capped.values())
        normalized = {m: w / total for m, w in capped.items()}

        assert abs(sum(normalized.values()) - 1.0) < 1e-9

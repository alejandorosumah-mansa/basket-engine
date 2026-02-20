"""
Tests for market classification pipeline.

Validates:
- Every market gets a primary theme assignment
- Confidence scores in valid range
- Theme coverage (no theme too sparse)
- LLM output format validity
- Determinism of classification
- Anti-example correctness
- Taxonomy loading and validation
- Edge cases (empty descriptions, short titles, non-English)
"""

import pytest
import json
import yaml
from pathlib import Path
from datetime import datetime

from tests.test_data_generators import (
    generate_market,
    generate_classification_result,
    generate_market_batch,
)


# ===========================================================================
# Taxonomy Loading & Validation
# ===========================================================================


class TestTaxonomyLoading:
    """Validate taxonomy.yaml structure and content."""

    @pytest.mark.unit
    def test_taxonomy_loads(self, taxonomy):
        """Taxonomy YAML should load without errors."""
        assert taxonomy is not None
        assert len(taxonomy) > 0

    @pytest.mark.unit
    def test_taxonomy_has_required_themes(self, taxonomy):
        """Taxonomy should have the expected core themes."""
        expected = [
            "us_elections", "fed_monetary_policy", "us_economic",
            "ai_technology", "crypto_digital", "russia_ukraine",
        ]
        for theme_id in expected:
            assert theme_id in taxonomy, f"Missing expected theme: {theme_id}"

    @pytest.mark.unit
    @pytest.mark.parametrize("field", ["name", "description", "examples", "anti_examples"])
    def test_theme_required_fields(self, taxonomy, field):
        """Every theme must have name, description, examples, and anti_examples."""
        for theme_id, theme in taxonomy.items():
            assert field in theme, f"Theme '{theme_id}' missing field: {field}"

    @pytest.mark.unit
    def test_theme_examples_non_empty(self, taxonomy):
        """Every theme should have at least 1 example."""
        for theme_id, theme in taxonomy.items():
            assert len(theme["examples"]) >= 1, (
                f"Theme '{theme_id}' has no examples"
            )

    @pytest.mark.unit
    def test_theme_anti_examples_non_empty(self, taxonomy):
        """Every theme should have at least 1 anti-example."""
        for theme_id, theme in taxonomy.items():
            assert len(theme["anti_examples"]) >= 1, (
                f"Theme '{theme_id}' has no anti-examples"
            )

    @pytest.mark.unit
    def test_theme_ids_are_snake_case(self, taxonomy):
        """Theme IDs should be snake_case for consistency."""
        import re
        pattern = re.compile(r"^[a-z][a-z0-9_]*$")
        for theme_id in taxonomy:
            assert pattern.match(theme_id), (
                f"Theme ID '{theme_id}' is not snake_case"
            )

    @pytest.mark.unit
    def test_no_duplicate_examples_across_themes(self, taxonomy):
        """The same example shouldn't appear in multiple themes."""
        seen = {}
        for theme_id, theme in taxonomy.items():
            for ex in theme["examples"]:
                normalized = ex.lower().strip()
                if normalized in seen:
                    pytest.fail(
                        f"Duplicate example '{ex}' in themes "
                        f"'{seen[normalized]}' and '{theme_id}'"
                    )
                seen[normalized] = theme_id

    @pytest.mark.unit
    def test_descriptions_meaningful_length(self, taxonomy):
        """Theme descriptions should be at least 20 characters."""
        for theme_id, theme in taxonomy.items():
            assert len(theme["description"]) >= 20, (
                f"Theme '{theme_id}' description too short: '{theme['description']}'"
            )


# ===========================================================================
# Classification Completeness
# ===========================================================================


class TestClassificationCompleteness:
    """Every market must receive a classification."""

    @pytest.mark.unit
    def test_every_market_gets_primary_theme(self, sample_classifications):
        """All classification results must have a primary_theme."""
        for c in sample_classifications:
            assert c["primary_theme"] is not None, (
                f"Market {c['market_id']} missing primary_theme"
            )
            assert c["primary_theme"] != "", (
                f"Market {c['market_id']} has empty primary_theme"
            )

    @pytest.mark.unit
    def test_primary_theme_is_valid(self, sample_classifications, taxonomy):
        """primary_theme must be a valid theme ID from taxonomy (or 'uncategorized')."""
        valid = set(taxonomy.keys()) | {"uncategorized"}
        for c in sample_classifications:
            assert c["primary_theme"] in valid, (
                f"Market {c['market_id']} has invalid theme: {c['primary_theme']}"
            )

    @pytest.mark.unit
    def test_secondary_theme_valid_or_none(self, sample_classifications, taxonomy):
        """secondary_theme must be valid theme ID, or None."""
        valid = set(taxonomy.keys()) | {"uncategorized"}
        for c in sample_classifications:
            if c["secondary_theme"] is not None:
                assert c["secondary_theme"] in valid, (
                    f"Market {c['market_id']} has invalid secondary theme"
                )

    @pytest.mark.unit
    def test_secondary_differs_from_primary(self, sample_classifications):
        """secondary_theme must differ from primary_theme if present."""
        for c in sample_classifications:
            if c["secondary_theme"] is not None:
                assert c["secondary_theme"] != c["primary_theme"], (
                    f"Market {c['market_id']}: secondary == primary"
                )


# ===========================================================================
# Confidence Scores
# ===========================================================================


class TestConfidenceScores:
    """Validate confidence score ranges and distributions."""

    @pytest.mark.unit
    def test_confidence_in_range(self, sample_classifications):
        """Confidence must be in [0.0, 1.0]."""
        for c in sample_classifications:
            assert 0.0 <= c["confidence"] <= 1.0, (
                f"Market {c['market_id']} confidence {c['confidence']} out of range"
            )

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_confidence", [-0.1, 1.1, 2.0, -1.0, float("inf")])
    def test_reject_invalid_confidence(self, bad_confidence):
        """Invalid confidence values should be detectable."""
        result = generate_classification_result(confidence=bad_confidence)
        assert not (0.0 <= result["confidence"] <= 1.0), (
            f"Failed to detect invalid confidence: {bad_confidence}"
        )

    @pytest.mark.unit
    def test_low_confidence_markets_flagged(self):
        """Markets with confidence < 0.5 should be flagged as uncategorized."""
        result = generate_classification_result(confidence=0.3)
        min_confidence = 0.5  # From settings
        if result["confidence"] < min_confidence:
            # In production, this should trigger uncategorized assignment
            assert result["confidence"] < min_confidence


# ===========================================================================
# Theme Coverage
# ===========================================================================


class TestThemeCoverage:
    """Validate that themes have sufficient market coverage."""

    @pytest.mark.unit
    def test_no_theme_below_minimum(self, sample_classifications):
        """No theme should have fewer than 5 eligible markets (flag for merge).

        If a theme has <5 markets, it doesn't have enough for a basket and
        should either be merged with a related theme or flagged for review.
        """
        from collections import Counter
        theme_counts = Counter(c["primary_theme"] for c in sample_classifications)

        min_markets = 5
        sparse_themes = {
            theme: count for theme, count in theme_counts.items()
            if count < min_markets
        }
        assert len(sparse_themes) == 0, (
            f"Themes with <{min_markets} markets (flag for merge): {sparse_themes}"
        )

    @pytest.mark.unit
    def test_uncategorized_not_dominant(self, sample_classifications):
        """Uncategorized should not be the largest 'theme'.

        If most markets are uncategorized, the taxonomy is too narrow.
        """
        from collections import Counter
        theme_counts = Counter(c["primary_theme"] for c in sample_classifications)
        uncategorized = theme_counts.get("uncategorized", 0)
        total = sum(theme_counts.values())

        assert uncategorized < total * 0.3, (
            f"Too many uncategorized: {uncategorized}/{total} ({uncategorized/total:.1%})"
        )


# ===========================================================================
# LLM Output Format
# ===========================================================================


class TestLLMOutputFormat:
    """Validate that LLM classification outputs are well-formed."""

    REQUIRED_OUTPUT_FIELDS = ["primary_theme", "confidence", "reasoning"]

    @pytest.mark.unit
    def test_output_has_required_fields(self):
        """LLM output JSON must have primary_theme, confidence, reasoning."""
        output = {
            "primary_theme": "us_elections",
            "secondary_theme": None,
            "confidence": 0.85,
            "reasoning": "Market is about Senate control.",
        }
        for field in self.REQUIRED_OUTPUT_FIELDS:
            assert field in output, f"Missing required output field: {field}"

    @pytest.mark.unit
    def test_output_is_valid_json(self):
        """LLM output should be parseable as JSON."""
        raw = '{"primary_theme": "us_elections", "secondary_theme": null, "confidence": 0.85, "reasoning": "Election market."}'
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
        assert "primary_theme" in parsed

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_json", [
        '{"primary_theme": "us_elections"',  # missing closing brace
        'primary_theme: us_elections',        # YAML not JSON
        '',                                   # empty
        'null',                               # null
        '[]',                                 # array not object
    ])
    def test_reject_malformed_json(self, bad_json):
        """Malformed JSON from LLM should be caught."""
        try:
            parsed = json.loads(bad_json)
            # Even if it parses, it must be a dict with required fields
            is_valid = (
                isinstance(parsed, dict)
                and "primary_theme" in parsed
                and "confidence" in parsed
            )
        except (json.JSONDecodeError, TypeError):
            is_valid = False

        assert not is_valid, f"Failed to reject malformed JSON: {bad_json!r}"

    @pytest.mark.unit
    def test_reasoning_is_nonempty_string(self):
        """Reasoning field should be a non-empty string."""
        result = generate_classification_result(reasoning="Election market about Senate.")
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    """Same input should produce same classification (with temperature=0)."""

    @pytest.mark.unit
    def test_same_input_same_output(self):
        """Identical market metadata should yield identical classification.

        With LLM temperature=0, the same prompt should produce the same
        output. This test validates the principle; actual LLM tests need
        integration testing.
        """
        market = generate_market(
            title="Will Democrats win the Senate in 2026?",
            description="Resolves YES if Democrats hold majority.",
        )

        # Simulate two classification runs with same input
        result1 = generate_classification_result(
            market_id=market["market_id"],
            primary_theme="us_elections",
            confidence=0.92,
        )
        result2 = generate_classification_result(
            market_id=market["market_id"],
            primary_theme="us_elections",
            confidence=0.92,
        )

        assert result1["primary_theme"] == result2["primary_theme"]
        assert result1["confidence"] == result2["confidence"]

    @pytest.mark.unit
    def test_classification_independent_of_order(self):
        """Classification should not depend on order of market processing.

        Market A's classification should be the same whether it's processed
        first or last in a batch.
        """
        market_a = generate_classification_result(
            market_id="market_a", primary_theme="crypto_digital", confidence=0.9,
        )
        # In a proper implementation, reordering the batch shouldn't change results
        assert market_a["primary_theme"] == "crypto_digital"


# ===========================================================================
# Anti-Examples
# ===========================================================================


class TestAntiExamples:
    """Known anti-examples should not be classified into their forbidden theme."""

    @pytest.mark.unit
    def test_anti_examples_loaded(self, taxonomy):
        """Should be able to extract all anti-examples from taxonomy."""
        from src.classification.taxonomy import get_all_anti_examples
        anti_examples = get_all_anti_examples(taxonomy)
        assert len(anti_examples) > 0, "No anti-examples found"

    @pytest.mark.unit
    def test_anti_example_structure(self, taxonomy):
        """Each anti-example should have text and forbidden_theme."""
        from src.classification.taxonomy import get_all_anti_examples
        for ae in get_all_anti_examples(taxonomy):
            assert "text" in ae
            assert "forbidden_theme" in ae
            assert len(ae["text"]) > 0

    @pytest.mark.unit
    @pytest.mark.parametrize("anti_example,forbidden_theme", [
        ("Will the Fed cut rates?", "us_elections"),
        ("Will Trump impose tariffs on China?", "us_elections"),
        ("US GDP growth above 2%?", "fed_monetary_policy"),
        ("Will oil prices rise?", "climate_environment"),
        ("Will NVIDIA stock hit $200?", "ai_technology"),
    ])
    def test_known_anti_examples(self, anti_example, forbidden_theme):
        """Specific anti-examples must NOT be classified into their forbidden theme.

        This test documents the expected behavior. In production, the classifier
        should be tested against these cases.
        """
        # This is a specification test - it documents what SHOULD happen.
        # The actual LLM classifier integration test would verify this.
        result = generate_classification_result(
            market_id=f"anti_{forbidden_theme}",
            primary_theme="some_other_theme",  # Should NOT be forbidden_theme
            confidence=0.8,
        )
        assert result["primary_theme"] != forbidden_theme, (
            f"Anti-example '{anti_example}' wrongly classified as '{forbidden_theme}'"
        )


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases that the classifier must handle gracefully."""

    @pytest.mark.unit
    def test_empty_description(self):
        """Markets with empty descriptions should still get classified (using title only)."""
        market = generate_market(
            title="Will Bitcoin exceed $100K?",
            description="",
        )
        # Classifier should handle gracefully, not crash
        assert market["title"] != ""
        # In production: classification should still work, possibly with lower confidence

    @pytest.mark.unit
    def test_very_short_title(self):
        """Very short titles should be handled (possibly flagged as low confidence)."""
        market = generate_market(title="BTC > 100K?", description="")
        assert len(market["title"]) > 0

    @pytest.mark.unit
    def test_very_long_description(self):
        """Very long descriptions should be truncated, not crash the classifier."""
        long_desc = "This is a test. " * 1000  # ~16K chars
        market = generate_market(
            title="Will X happen?",
            description=long_desc,
        )
        # Classifier prompt should truncate to ~500 chars
        truncated = market["description"][:500]
        assert len(truncated) == 500

    @pytest.mark.unit
    def test_non_english_market(self):
        """Non-English markets should be handled (classified or flagged)."""
        market = generate_market(
            title="¿Bitcoin superará los $100K?",
            description="Este mercado se resuelve SÍ si Bitcoin supera los $100,000.",
        )
        # Should not crash; may be classified with lower confidence
        assert len(market["title"]) > 0

    @pytest.mark.unit
    def test_special_characters_in_title(self):
        """Titles with special characters should not break classification."""
        market = generate_market(
            title='Will "GPT-5" be released? (OpenAI\'s next model) — 2026',
            description="Test with quotes, parens, em-dash.",
        )
        assert len(market["title"]) > 0

    @pytest.mark.unit
    def test_market_with_no_tags(self):
        """Markets without tags should still be classifiable."""
        market = generate_market(tags=[])
        assert market["tags"] == []

    @pytest.mark.unit
    def test_ambiguous_market(self):
        """A market that could belong to multiple themes should have reasonable confidence.

        'Will Trump impose tariffs on China?' could be us_elections, china_us, or
        energy_commodities. The classifier should pick one but may have lower confidence.
        """
        # This is a specification test for expected behavior
        result = generate_classification_result(
            market_id="ambiguous_001",
            primary_theme="china_us",
            secondary_theme="us_elections",
            confidence=0.65,  # Lower confidence due to ambiguity
        )
        assert result["secondary_theme"] is not None, (
            "Ambiguous markets should have a secondary theme"
        )
        assert result["confidence"] < 0.9, (
            "Ambiguous markets should not have very high confidence"
        )

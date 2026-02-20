"""
Stability analysis for classification and basket construction.

Tests whether results are robust to:
- Different time windows for clustering
- Random removal of 10% of markets
- Prompt variation for LLM classification
- Sensitivity to eligibility thresholds

Usage:
    python -m src.validation.stability --markets data/processed/markets.parquet
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class StabilityResult:
    """Result of a single stability test."""
    test_name: str
    stability_score: float  # 0-1, higher = more stable
    detail: str = ""
    raw_data: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "stability_score": round(self.stability_score, 4),
            "detail": self.detail,
        }


@dataclass
class StabilityReport:
    """Full stability analysis report."""
    results: list[StabilityResult] = field(default_factory=list)

    @property
    def avg_stability(self) -> float:
        if not self.results:
            return 0.0
        return np.mean([r.stability_score for r in self.results])

    def to_dict(self) -> dict:
        return {
            "avg_stability": round(self.avg_stability, 4),
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        lines = [f"Stability Analysis: avg score = {self.avg_stability:.2%}"]
        for r in self.results:
            icon = "✅" if r.stability_score > 0.7 else "⚠️" if r.stability_score > 0.5 else "❌"
            lines.append(f"  {icon} {r.test_name}: {r.stability_score:.2%}")
            if r.detail:
                lines.append(f"      {r.detail}")
        return "\n".join(lines)


def clustering_time_window_stability(
    returns_matrix: pd.DataFrame,
    cluster_fn: Callable[[pd.DataFrame], dict[str, int]],
    windows: Optional[list[tuple[int, int]]] = None,
) -> StabilityResult:
    """Test clustering stability across different time windows.

    Re-runs clustering with different trailing windows and measures
    how much cluster assignments change.

    Args:
        returns_matrix: DataFrame indexed by date, columns are market IDs.
        cluster_fn: Function that takes a returns matrix and returns
                    {market_id: cluster_id} dict.
        windows: List of (start_offset, end_offset) tuples from matrix end.
                 Defaults to 60-day, 90-day, and 120-day windows.

    Returns:
        StabilityResult with agreement score.
    """
    n_days = len(returns_matrix)
    if windows is None:
        windows = [
            (max(0, n_days - 60), n_days),
            (max(0, n_days - 90), n_days),
            (max(0, n_days - 120), n_days),
        ]

    all_assignments = []
    for start, end in windows:
        subset = returns_matrix.iloc[start:end]
        if len(subset) < 20:
            continue
        assignments = cluster_fn(subset)
        all_assignments.append(assignments)

    if len(all_assignments) < 2:
        return StabilityResult(
            "clustering_time_windows", 1.0,
            "Not enough windows to compare",
        )

    # Measure pairwise agreement using Adjusted Rand Index concept
    # Simplified: for each pair of markets, check if they're in same cluster
    # across all windows
    markets = list(all_assignments[0].keys())
    agreements = 0
    comparisons = 0

    for i in range(len(markets)):
        for j in range(i + 1, len(markets)):
            m_i, m_j = markets[i], markets[j]
            same_cluster_counts = []
            for assignments in all_assignments:
                if m_i in assignments and m_j in assignments:
                    same_cluster_counts.append(
                        assignments[m_i] == assignments[m_j]
                    )

            if len(same_cluster_counts) >= 2:
                # All windows agree on whether i,j are in same cluster?
                if len(set(same_cluster_counts)) == 1:
                    agreements += 1
                comparisons += 1

    score = agreements / comparisons if comparisons > 0 else 1.0

    return StabilityResult(
        "clustering_time_windows",
        score,
        f"Pairwise agreement across {len(all_assignments)} windows: "
        f"{agreements}/{comparisons} pairs consistent",
    )


def market_removal_stability(
    returns_matrix: pd.DataFrame,
    cluster_fn: Callable[[pd.DataFrame], dict[str, int]],
    removal_fraction: float = 0.10,
    n_trials: int = 5,
    seed: int = 42,
) -> StabilityResult:
    """Test clustering stability when 10% of markets are randomly removed.

    Args:
        returns_matrix: Returns matrix.
        cluster_fn: Clustering function.
        removal_fraction: Fraction of markets to remove.
        n_trials: Number of random removal trials.
        seed: Random seed.

    Returns:
        StabilityResult.
    """
    rng = np.random.default_rng(seed)
    markets = list(returns_matrix.columns)
    n_remove = max(1, int(len(markets) * removal_fraction))

    base_assignments = cluster_fn(returns_matrix)
    trial_agreements = []

    for trial in range(n_trials):
        remove = rng.choice(markets, size=n_remove, replace=False)
        remaining = [m for m in markets if m not in remove]
        subset = returns_matrix[remaining]
        trial_assignments = cluster_fn(subset)

        # Compare remaining markets' assignments
        agree = 0
        total = 0
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                m_i, m_j = remaining[i], remaining[j]
                if m_i in base_assignments and m_j in base_assignments:
                    base_same = base_assignments[m_i] == base_assignments[m_j]
                    trial_same = trial_assignments[m_i] == trial_assignments[m_j]
                    if base_same == trial_same:
                        agree += 1
                    total += 1

        if total > 0:
            trial_agreements.append(agree / total)

    avg_agreement = np.mean(trial_agreements) if trial_agreements else 1.0

    return StabilityResult(
        "market_removal",
        avg_agreement,
        f"Avg pairwise agreement after removing {removal_fraction:.0%} markets "
        f"across {n_trials} trials",
    )


def llm_prompt_variant_stability(
    markets: list[dict],
    classify_fn: Callable[[dict, str], dict],
    prompt_variants: Optional[list[str]] = None,
) -> StabilityResult:
    """Test LLM classification agreement across prompt variants.

    Args:
        markets: List of market dicts.
        classify_fn: Function(market, prompt_variant) → classification dict.
        prompt_variants: List of prompt variant identifiers.

    Returns:
        StabilityResult with inter-prompt agreement.
    """
    if prompt_variants is None:
        prompt_variants = ["default", "detailed", "concise"]

    if not markets:
        return StabilityResult("llm_prompt_variants", 1.0, "No markets to test")

    agreements = 0
    total = 0

    for market in markets:
        themes = []
        for variant in prompt_variants:
            result = classify_fn(market, variant)
            themes.append(result.get("primary_theme"))

        # Check if all variants agree
        if len(set(themes)) == 1:
            agreements += 1
        total += 1

    score = agreements / total if total > 0 else 1.0

    return StabilityResult(
        "llm_prompt_variants",
        score,
        f"{agreements}/{total} markets had unanimous classification "
        f"across {len(prompt_variants)} prompt variants",
    )


def eligibility_threshold_sensitivity(
    markets: list[dict],
    eligibility_fn: Callable[[dict, dict], bool],
    base_thresholds: dict,
    perturbation: float = 0.20,
) -> StabilityResult:
    """Sensitivity analysis: how much do baskets change when thresholds shift ±20%?

    Args:
        markets: List of market dicts.
        eligibility_fn: Function(market, thresholds) → bool.
        base_thresholds: Base eligibility thresholds dict.
        perturbation: Fraction to perturb thresholds by.

    Returns:
        StabilityResult.
    """
    base_eligible = set()
    for m in markets:
        if eligibility_fn(m, base_thresholds):
            base_eligible.add(m["market_id"])

    # Test tighter thresholds (+20%)
    tight = {k: v * (1 + perturbation) if isinstance(v, (int, float)) else v
             for k, v in base_thresholds.items()}
    tight_eligible = set()
    for m in markets:
        if eligibility_fn(m, tight):
            tight_eligible.add(m["market_id"])

    # Test looser thresholds (-20%)
    loose = {k: v * (1 - perturbation) if isinstance(v, (int, float)) else v
             for k, v in base_thresholds.items()}
    loose_eligible = set()
    for m in markets:
        if eligibility_fn(m, loose):
            loose_eligible.add(m["market_id"])

    # Jaccard similarity between base and perturbed
    jaccard_tight = (
        len(base_eligible & tight_eligible) / len(base_eligible | tight_eligible)
        if base_eligible | tight_eligible else 1.0
    )
    jaccard_loose = (
        len(base_eligible & loose_eligible) / len(base_eligible | loose_eligible)
        if base_eligible | loose_eligible else 1.0
    )

    avg_jaccard = (jaccard_tight + jaccard_loose) / 2

    return StabilityResult(
        "eligibility_sensitivity",
        avg_jaccard,
        f"Base: {len(base_eligible)} eligible, "
        f"Tight(+{perturbation:.0%}): {len(tight_eligible)}, "
        f"Loose(-{perturbation:.0%}): {len(loose_eligible)}. "
        f"Avg Jaccard similarity: {avg_jaccard:.2%}",
    )


def run_full_stability_analysis(
    returns_matrix: pd.DataFrame,
    cluster_fn: Callable[[pd.DataFrame], dict[str, int]],
    markets: list[dict],
    eligibility_fn: Callable[[dict, dict], bool],
    base_thresholds: dict,
    classify_fn: Optional[Callable] = None,
) -> StabilityReport:
    """Run all stability tests and return report.

    Args:
        returns_matrix: Returns matrix.
        cluster_fn: Clustering function.
        markets: Market metadata.
        eligibility_fn: Eligibility filter function.
        base_thresholds: Eligibility thresholds.
        classify_fn: Optional LLM classification function.

    Returns:
        StabilityReport.
    """
    report = StabilityReport()

    report.results.append(
        clustering_time_window_stability(returns_matrix, cluster_fn)
    )
    report.results.append(
        market_removal_stability(returns_matrix, cluster_fn)
    )
    report.results.append(
        eligibility_threshold_sensitivity(markets, eligibility_fn, base_thresholds)
    )

    if classify_fn:
        report.results.append(
            llm_prompt_variant_stability(markets[:20], classify_fn)  # Sample for cost
        )

    return report

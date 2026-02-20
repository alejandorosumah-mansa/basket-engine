"""
Classification audit tool for human review.

Randomly samples N markets stratified by theme, presents each with its
classification for human verification, tracks results, and exports an
accuracy report per theme.

Usage:
    python -m src.validation.classification_audit --n-per-theme 5 --output audit_report.json
"""

import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional


class ClassificationAuditor:
    """Audits classification results via stratified sampling and human review."""

    VALID_JUDGMENTS = {"correct", "incorrect", "borderline"}

    def __init__(
        self,
        classifications: list[dict],
        markets: Optional[list[dict]] = None,
    ):
        """Initialize auditor.

        Args:
            classifications: List of dicts with market_id, primary_theme, confidence, reasoning.
            markets: Optional list of market metadata dicts (for showing title/description).
        """
        self.classifications = classifications
        self.markets_by_id = {}
        if markets:
            self.markets_by_id = {m["market_id"]: m for m in markets}

        self.audit_results: list[dict] = []

    def stratified_sample(self, n_per_theme: int = 5, seed: int = 42) -> list[dict]:
        """Sample N markets per theme for review.

        Args:
            n_per_theme: Number of markets to sample from each theme.
            seed: Random seed for reproducibility.

        Returns:
            List of classification dicts to review.
        """
        rng = random.Random(seed)
        by_theme = defaultdict(list)
        for c in self.classifications:
            by_theme[c["primary_theme"]].append(c)

        sample = []
        for theme, items in sorted(by_theme.items()):
            k = min(n_per_theme, len(items))
            sample.extend(rng.sample(items, k))

        return sample

    def format_for_review(self, classification: dict) -> str:
        """Format a single classification for human review display."""
        market = self.markets_by_id.get(classification["market_id"], {})
        lines = [
            "=" * 60,
            f"Market ID:   {classification['market_id']}",
            f"Title:       {market.get('title', 'N/A')}",
            f"Description: {market.get('description', 'N/A')[:200]}",
            f"Platform:    {market.get('platform', 'N/A')}",
            f"Tags:        {market.get('tags', [])}",
            "",
            f"→ Primary Theme: {classification['primary_theme']}",
            f"→ Secondary:     {classification.get('secondary_theme', 'None')}",
            f"→ Confidence:    {classification['confidence']:.2f}",
            f"→ Reasoning:     {classification.get('reasoning', 'N/A')}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def record_judgment(
        self,
        classification: dict,
        judgment: str,
        correct_theme: Optional[str] = None,
        notes: str = "",
    ) -> dict:
        """Record a human judgment for a classification.

        Args:
            classification: The classification being judged.
            judgment: One of 'correct', 'incorrect', 'borderline'.
            correct_theme: If incorrect, what should the theme be.
            notes: Optional reviewer notes.

        Returns:
            Audit result dict.
        """
        if judgment not in self.VALID_JUDGMENTS:
            raise ValueError(f"Invalid judgment: {judgment}. Must be one of {self.VALID_JUDGMENTS}")

        result = {
            "market_id": classification["market_id"],
            "assigned_theme": classification["primary_theme"],
            "confidence": classification["confidence"],
            "judgment": judgment,
            "correct_theme": correct_theme,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
        }
        self.audit_results.append(result)
        return result

    def calculate_accuracy(self) -> dict:
        """Calculate accuracy metrics per theme and overall.

        Returns:
            Dict with per-theme accuracy, overall accuracy, and confusion details.
        """
        if not self.audit_results:
            return {"error": "No audit results recorded"}

        by_theme = defaultdict(lambda: {"correct": 0, "incorrect": 0, "borderline": 0, "total": 0})
        overall = {"correct": 0, "incorrect": 0, "borderline": 0, "total": 0}

        for r in self.audit_results:
            theme = r["assigned_theme"]
            judgment = r["judgment"]
            by_theme[theme][judgment] += 1
            by_theme[theme]["total"] += 1
            overall[judgment] += 1
            overall["total"] += 1

        theme_accuracy = {}
        for theme, counts in by_theme.items():
            if counts["total"] > 0:
                # Treat borderline as 0.5 correct
                acc = (counts["correct"] + 0.5 * counts["borderline"]) / counts["total"]
                theme_accuracy[theme] = {
                    "accuracy": round(acc, 3),
                    "correct": counts["correct"],
                    "incorrect": counts["incorrect"],
                    "borderline": counts["borderline"],
                    "total": counts["total"],
                }

        overall_acc = (
            (overall["correct"] + 0.5 * overall["borderline"]) / overall["total"]
            if overall["total"] > 0
            else 0
        )

        return {
            "overall_accuracy": round(overall_acc, 3),
            "overall_counts": overall,
            "per_theme": theme_accuracy,
            "audit_date": datetime.now().isoformat(),
            "n_audited": overall["total"],
        }

    def export_report(self, path: str) -> None:
        """Export full audit report to JSON.

        Args:
            path: Output file path.
        """
        report = {
            "accuracy": self.calculate_accuracy(),
            "individual_results": self.audit_results,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def run_interactive_audit(self, n_per_theme: int = 5, seed: int = 42) -> dict:
        """Run an interactive audit session in the terminal.

        Args:
            n_per_theme: Markets to review per theme.
            seed: Random seed.

        Returns:
            Accuracy report dict.
        """
        sample = self.stratified_sample(n_per_theme, seed)
        print(f"\n🔍 Classification Audit: {len(sample)} markets to review\n")

        for i, c in enumerate(sample, 1):
            print(f"\n[{i}/{len(sample)}]")
            print(self.format_for_review(c))

            while True:
                judgment = input("Judgment (correct/incorrect/borderline/skip): ").strip().lower()
                if judgment == "skip":
                    break
                if judgment in self.VALID_JUDGMENTS:
                    correct_theme = None
                    if judgment == "incorrect":
                        correct_theme = input("  Correct theme: ").strip() or None
                    notes = input("  Notes (optional): ").strip()
                    self.record_judgment(c, judgment, correct_theme, notes)
                    break
                print(f"  Invalid. Choose from: correct, incorrect, borderline, skip")

        report = self.calculate_accuracy()
        print(f"\n📊 Overall Accuracy: {report['overall_accuracy']:.1%}")
        print(f"   Audited: {report['n_audited']} markets")
        for theme, stats in report.get("per_theme", {}).items():
            print(f"   {theme}: {stats['accuracy']:.1%} ({stats['total']} reviewed)")

        return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit classification results")
    parser.add_argument("--classifications", required=True, help="Path to classifications JSON")
    parser.add_argument("--markets", help="Path to markets JSON (optional)")
    parser.add_argument("--n-per-theme", type=int, default=5)
    parser.add_argument("--output", default="data/outputs/audit_report.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.classifications) as f:
        classifications = json.load(f)

    markets = None
    if args.markets:
        with open(args.markets) as f:
            markets = json.load(f)

    auditor = ClassificationAuditor(classifications, markets)
    report = auditor.run_interactive_audit(args.n_per_theme, args.seed)
    auditor.export_report(args.output)
    print(f"\n✅ Report saved to {args.output}")

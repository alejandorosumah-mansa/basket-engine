"""
Backtest sanity checks.

Cross-validates backtest results to ensure they're plausible:
- Cross-basket correlation (should be low)
- Max drawdown limits
- Turnover ranges
- Comparison to equal-weight benchmark
- Anomaly detection

Usage:
    python -m src.validation.backtest_sanity --results-dir data/outputs/performance/
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SanityCheckResult:
    """Result of a single sanity check."""
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": round(self.value, 4),
            "threshold": round(self.threshold, 4),
            "detail": self.detail,
        }


@dataclass
class SanityReport:
    """Full sanity check report across all baskets."""
    checks: list[SanityCheckResult] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(c.passed for c in self.checks)

    @property
    def n_failed(self) -> int:
        return sum(not c.passed for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "checks": [c.to_dict() for c in self.checks],
            "anomalies": self.anomalies,
        }

    def summary(self) -> str:
        lines = [
            f"Sanity Check: {'✅ PASSED' if self.all_passed else '❌ FAILED'}",
            f"  {self.n_passed} passed, {self.n_failed} failed",
        ]
        for c in self.checks:
            icon = "✅" if c.passed else "❌"
            lines.append(f"  {icon} {c.name}: {c.value:.4f} (threshold: {c.threshold:.4f})")
        if self.anomalies:
            lines.append("  ⚠️ Anomalies:")
            for a in self.anomalies:
                lines.append(f"    - {a}")
        return "\n".join(lines)


class BacktestSanityChecker:
    """Runs sanity checks on backtest results."""

    def __init__(
        self,
        basket_navs: dict[str, pd.DataFrame],
        basket_rebalances: Optional[dict[str, list[dict]]] = None,
        benchmark_nav: Optional[pd.DataFrame] = None,
    ):
        """Initialize checker.

        Args:
            basket_navs: Dict mapping basket_id to DataFrame with 'date' and 'nav' columns.
            basket_rebalances: Dict mapping basket_id to list of rebalance event dicts.
            benchmark_nav: Equal-weight benchmark NAV DataFrame.
        """
        self.basket_navs = basket_navs
        self.basket_rebalances = basket_rebalances or {}
        self.benchmark_nav = benchmark_nav

    def check_cross_correlation(self, max_avg_corr: float = 0.3) -> SanityCheckResult:
        """Cross-basket correlation should be low (baskets are independent themes).

        Args:
            max_avg_corr: Maximum acceptable average pairwise correlation.

        Returns:
            SanityCheckResult.
        """
        if len(self.basket_navs) < 2:
            return SanityCheckResult(
                "cross_correlation", True, 0.0, max_avg_corr,
                "Only 1 basket, skipping correlation check",
            )

        # Compute daily returns for each basket
        returns = {}
        for basket_id, nav_df in self.basket_navs.items():
            r = nav_df.set_index("date")["nav"].pct_change().dropna()
            returns[basket_id] = r

        returns_df = pd.DataFrame(returns)
        corr_matrix = returns_df.corr(method="spearman")

        # Average off-diagonal correlation
        n = len(corr_matrix)
        if n < 2:
            return SanityCheckResult("cross_correlation", True, 0.0, max_avg_corr)

        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr = corr_matrix.values[mask].mean()

        return SanityCheckResult(
            "cross_correlation",
            abs(avg_corr) < max_avg_corr,
            avg_corr,
            max_avg_corr,
            f"Avg pairwise Spearman correlation across {n} baskets",
        )

    def check_max_drawdown(self, max_allowed_dd: float = 0.80) -> list[SanityCheckResult]:
        """No basket should have > max_allowed_dd max drawdown.

        Args:
            max_allowed_dd: Maximum allowed drawdown (e.g., 0.80 = 80%).

        Returns:
            List of SanityCheckResult, one per basket.
        """
        results = []
        for basket_id, nav_df in self.basket_navs.items():
            nav = nav_df["nav"].values
            running_max = np.maximum.accumulate(nav)
            drawdowns = (nav - running_max) / running_max
            max_dd = abs(drawdowns.min())

            results.append(SanityCheckResult(
                f"max_drawdown_{basket_id}",
                max_dd < max_allowed_dd,
                max_dd,
                max_allowed_dd,
                f"Max drawdown for basket '{basket_id}'",
            ))
        return results

    def check_turnover(
        self,
        min_turnover: float = 0.05,
        max_turnover: float = 0.50,
    ) -> list[SanityCheckResult]:
        """Turnover should be within expected range.

        Too low = baskets never change (stale).
        Too high = baskets are unstable.

        Args:
            min_turnover: Minimum expected average turnover.
            max_turnover: Maximum expected average turnover (warning at 50%).

        Returns:
            List of SanityCheckResult, one per basket.
        """
        results = []
        for basket_id, events in self.basket_rebalances.items():
            if not events:
                results.append(SanityCheckResult(
                    f"turnover_{basket_id}", False, 0.0, max_turnover,
                    "No rebalance events recorded",
                ))
                continue

            turnovers = [e.get("turnover", 0) for e in events]
            avg_turnover = np.mean(turnovers)

            passed = min_turnover <= avg_turnover <= max_turnover
            results.append(SanityCheckResult(
                f"turnover_{basket_id}",
                passed,
                avg_turnover,
                max_turnover,
                f"Avg monthly turnover for '{basket_id}' ({len(events)} rebalances)",
            ))
        return results

    def check_vs_benchmark(self) -> Optional[SanityCheckResult]:
        """Compare basket returns to equal-weight benchmark.

        Baskets don't need to outperform, but returns should be in a
        reasonable range relative to benchmark.

        Returns:
            SanityCheckResult or None if no benchmark available.
        """
        if self.benchmark_nav is None or not self.basket_navs:
            return None

        bench_return = (
            self.benchmark_nav["nav"].iloc[-1] / self.benchmark_nav["nav"].iloc[0] - 1
        )

        basket_returns = []
        for basket_id, nav_df in self.basket_navs.items():
            r = nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1
            basket_returns.append(r)

        avg_basket_return = np.mean(basket_returns)

        # Basket returns should be within 50pp of benchmark
        # (this is a very loose check — just flagging extreme divergence)
        diff = abs(avg_basket_return - bench_return)
        max_diff = 0.50

        return SanityCheckResult(
            "vs_benchmark",
            diff < max_diff,
            diff,
            max_diff,
            f"Avg basket return: {avg_basket_return:.2%}, Benchmark: {bench_return:.2%}",
        )

    def run_all_checks(self) -> SanityReport:
        """Run all sanity checks and return report."""
        report = SanityReport()

        # Cross-correlation
        report.checks.append(self.check_cross_correlation())

        # Max drawdown per basket
        report.checks.extend(self.check_max_drawdown())

        # Turnover per basket
        report.checks.extend(self.check_turnover())

        # Benchmark comparison
        bench_result = self.check_vs_benchmark()
        if bench_result:
            report.checks.append(bench_result)

        # Anomaly detection
        for basket_id, nav_df in self.basket_navs.items():
            nav = nav_df["nav"].values
            daily_returns = np.diff(nav) / nav[:-1]

            # Flag days with > 20% move
            extreme_days = np.where(np.abs(daily_returns) > 0.20)[0]
            for day_idx in extreme_days:
                report.anomalies.append(
                    f"Basket '{basket_id}' day {day_idx}: "
                    f"{daily_returns[day_idx]:.2%} daily return"
                )

            # Flag if NAV goes below 50% of starting value
            if nav.min() < nav[0] * 0.50:
                report.anomalies.append(
                    f"Basket '{basket_id}' NAV dropped below 50% of starting value"
                )

        return report

    def export_report(self, path: str) -> None:
        """Run checks and export report to JSON."""
        report = self.run_all_checks()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest sanity checks")
    parser.add_argument("--results-dir", required=True, help="Directory with basket NAV CSVs")
    parser.add_argument("--output", default="data/outputs/sanity_report.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    basket_navs = {}
    for csv_path in results_dir.glob("*.csv"):
        basket_id = csv_path.stem
        df = pd.read_csv(csv_path, parse_dates=["date"])
        basket_navs[basket_id] = df

    checker = BacktestSanityChecker(basket_navs)
    report = checker.run_all_checks()
    print(report.summary())
    checker.export_report(args.output)
    print(f"\n✅ Report saved to {args.output}")

"""
Run all economic/financial analysis modules.

Usage:
    python -m scripts.run_economic_analysis [--fetch-benchmarks]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-benchmarks", action="store_true",
                        help="Fetch fresh benchmark data from Yahoo Finance")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip benchmark fetch (use cached data)")
    args = parser.parse_args()
    
    output_dir = Path("data/outputs/charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Fetch benchmarks
    if not args.skip_fetch:
        print("\n" + "=" * 60)
        print("STEP 1: FETCHING BENCHMARK DATA")
        print("=" * 60)
        from src.benchmarks.fetch import fetch_benchmarks, save_benchmarks
        df = fetch_benchmarks()
        save_benchmarks(df)
    
    # Step 2: Cross-asset correlation
    print("\n")
    from src.analysis.cross_asset import run_cross_asset_analysis
    cross_results = run_cross_asset_analysis(output_dir)
    
    # Step 3: Internal factor model
    print("\n")
    from src.analysis.factor_model import run_factor_model
    factor_results = run_factor_model(output_dir)
    
    # Step 4: Macro factor model
    print("\n")
    from src.analysis.macro_factors import run_macro_factor_model
    macro_results = run_macro_factor_model(output_dir)
    
    # Step 5: Regime analysis
    print("\n")
    from src.analysis.regime import run_regime_analysis
    regime_results = run_regime_analysis(output_dir)
    
    print("\n" + "=" * 60)
    print("ALL ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Charts saved to: {output_dir}")
    print(f"Files generated:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()

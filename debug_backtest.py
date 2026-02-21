"""Debug the backtest bug: identical results for risk parity and volume weighted."""

import pandas as pd
import numpy as np
from src.construction.weighting import compute_weights
import logging

logging.basicConfig(level=logging.INFO)

# Create test data
def create_test_data():
    # Mock market data
    market_ids = ["market_A", "market_B", "market_C", "market_D", "market_E"]
    
    # Mock returns (different volatilities)
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    
    returns_data = []
    for market in market_ids:
        # Different volatility per market to test risk parity
        vol_multiplier = {"market_A": 0.5, "market_B": 1.0, "market_C": 1.5, 
                         "market_D": 2.0, "market_E": 0.8}[market]
        
        for date in dates:
            ret = np.random.normal(0, 0.02 * vol_multiplier)
            returns_data.append({
                "market_id": market,
                "date": date.strftime("%Y-%m-%d"),
                "return": ret
            })
    
    returns_df = pd.DataFrame(returns_data)
    
    # Mock volumes (different volumes to test volume weighting)
    volumes = {
        "market_A": 1000000,   # $1M
        "market_B": 5000000,   # $5M  
        "market_C": 2000000,   # $2M
        "market_D": 500000,    # $500K
        "market_E": 10000000,  # $10M
    }
    
    return market_ids, returns_df, volumes


def debug_weighting_methods():
    print("=== DEBUGGING WEIGHTING METHODS ===")
    
    market_ids, returns_df, volumes = create_test_data()
    
    print(f"Test data: {len(market_ids)} markets, {len(returns_df)} return observations")
    print(f"Volumes: {volumes}")
    print()
    
    # Test equal weights
    eq_weights = compute_weights(
        method="equal",
        market_ids=market_ids,
        returns_df=returns_df,
        volumes=volumes,
        liquidity=volumes
    )
    print("Equal weights:")
    for m, w in sorted(eq_weights.items()):
        print(f"  {m}: {w:.4f}")
    print(f"  Sum: {sum(eq_weights.values()):.4f}")
    print()
    
    # Test volume weighted  
    vol_weights = compute_weights(
        method="volume_weighted",
        market_ids=market_ids,
        returns_df=returns_df,
        volumes=volumes,
        liquidity=volumes
    )
    print("Volume weighted:")
    for m, w in sorted(vol_weights.items()):
        print(f"  {m}: {w:.4f} (volume: {volumes[m]:,})")
    print(f"  Sum: {sum(vol_weights.values()):.4f}")
    print()
    
    # Test risk parity
    rp_weights = compute_weights(
        method="risk_parity",
        market_ids=market_ids,
        returns_df=returns_df,
        volumes=volumes,
        liquidity=volumes
    )
    print("Risk parity:")
    for m, w in sorted(rp_weights.items()):
        mret = returns_df[returns_df["market_id"] == m]["return"]
        vol = mret.std()
        print(f"  {m}: {w:.4f} (vol: {vol:.4f})")
    print(f"  Sum: {sum(rp_weights.values()):.4f}")
    print()
    
    # Test risk parity with liquidity cap
    rp_lc_weights = compute_weights(
        method="risk_parity_liquidity_cap",
        market_ids=market_ids,
        returns_df=returns_df,
        volumes=volumes,
        liquidity=volumes,
        liquidity_cap=True
    )
    print("Risk parity with liquidity cap:")
    for m, w in sorted(rp_lc_weights.items()):
        mret = returns_df[returns_df["market_id"] == m]["return"]
        vol = mret.std()
        print(f"  {m}: {w:.4f} (vol: {vol:.4f}, volume: {volumes[m]:,})")
    print(f"  Sum: {sum(rp_lc_weights.values()):.4f}")
    print()
    
    # Check if any are identical
    print("=== COMPARISON ===")
    if vol_weights == rp_weights:
        print("🚨 BUG: Volume weighted and risk parity are IDENTICAL!")
    elif vol_weights == rp_lc_weights:
        print("🚨 BUG: Volume weighted and risk parity liquidity cap are IDENTICAL!")
    elif rp_weights == rp_lc_weights:
        print("🚨 BUG: Risk parity and risk parity liquidity cap are IDENTICAL!")
    else:
        print("✅ All methods produce different results")
    
    # Calculate some differences
    vol_vs_rp = sum(abs(vol_weights.get(m, 0) - rp_weights.get(m, 0)) for m in market_ids)
    vol_vs_rp_lc = sum(abs(vol_weights.get(m, 0) - rp_lc_weights.get(m, 0)) for m in market_ids)
    
    print(f"Volume vs Risk Parity total weight difference: {vol_vs_rp:.6f}")
    print(f"Volume vs Risk Parity LC total weight difference: {vol_vs_rp_lc:.6f}")


if __name__ == "__main__":
    debug_weighting_methods()
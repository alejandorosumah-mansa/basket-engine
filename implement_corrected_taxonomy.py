#!/usr/bin/env python3
"""
Implement the CORRECTED 4-Layer Taxonomy: CUSIP → Ticker → Event → Theme

This replaces the previous incorrect implementation with the proper bottom-up hierarchy:

1. CUSIP: Individual market instance with specific time/date
2. Ticker: Outcome level, stripped of time  
3. Event: Parent question (can have multiple tickers for categorical events)
4. Theme: Classification applied at EVENT level for basket construction

KEY RULES:
- Binary events: 1 ticker per event (ticker = event)
- Categorical events: multiple tickers per event
- Theme classification happens at EVENT level, not ticker/CUSIP level
- Basket construction: one exposure per EVENT
- CUSIPs are time variants of tickers
"""

import sys
import logging
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.four_layer_taxonomy import FourLayerTaxonomy, process_markets_with_corrected_taxonomy
from src.ingestion.run import load_all_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the corrected 4-layer taxonomy implementation."""
    
    logger.info("=== IMPLEMENTING CORRECTED 4-LAYER TAXONOMY ===")
    logger.info("Hierarchy: CUSIP → Ticker → Event → Theme")
    
    # Load markets data
    logger.info("Loading markets data...")
    markets_path = PROJECT_ROOT / "data" / "processed" / "markets.parquet"
    
    if not markets_path.exists():
        logger.error(f"Markets file not found: {markets_path}")
        logger.info("Please run the ingestion pipeline first:")
        logger.info("  python -m src.ingestion.run")
        sys.exit(1)
    
    markets_df = pd.read_parquet(markets_path)
    logger.info(f"Loaded {len(markets_df):,} markets")
    
    # Apply corrected taxonomy
    logger.info("\n=== PROCESSING THROUGH 4-LAYER TAXONOMY ===")
    taxonomy_df = process_markets_with_corrected_taxonomy(markets_df)
    
    # Save results
    output_path = PROJECT_ROOT / "data" / "processed" / "markets_corrected_taxonomy.parquet"
    taxonomy_df.to_parquet(output_path, index=False)
    logger.info(f"Saved taxonomy results to: {output_path}")
    
    # Generate detailed analysis
    logger.info("\n=== DETAILED ANALYSIS ===")
    analyze_corrected_taxonomy(taxonomy_df)
    
    # Generate event-level baskets
    logger.info("\n=== GENERATING EVENT-LEVEL BASKETS ===")
    generate_event_baskets(taxonomy_df)
    
    logger.info("\n=== CORRECTED TAXONOMY IMPLEMENTATION COMPLETE ===")


def analyze_corrected_taxonomy(taxonomy_df: pd.DataFrame):
    """Analyze the results of the corrected taxonomy."""
    
    print("=== CORRECTED TAXONOMY ANALYSIS ===")
    
    # Layer summary
    total_cusips = len(taxonomy_df)
    unique_tickers = taxonomy_df['ticker'].nunique()
    unique_events = taxonomy_df['event'].nunique()
    unique_themes = taxonomy_df['theme'].nunique()
    
    print(f"Layer 1 (CUSIPs): {total_cusips:,} individual markets")
    print(f"Layer 2 (Tickers): {unique_tickers:,} unique tickers")
    print(f"Layer 3 (Events): {unique_events:,} unique events")  
    print(f"Layer 4 (Themes): {unique_themes} themes")
    
    print(f"\nCompression ratios:")
    print(f"  CUSIP→Ticker: {total_cusips/unique_tickers:.1f}x")
    print(f"  Ticker→Event: {unique_tickers/unique_events:.1f}x")
    print(f"  CUSIP→Event: {total_cusips/unique_events:.1f}x (overall)")
    
    # Theme distribution
    print(f"\nTheme distribution:")
    theme_counts = taxonomy_df['theme'].value_counts()
    for theme, count in theme_counts.items():
        pct = count / total_cusips * 100
        print(f"  {theme}: {count:,} CUSIPs ({pct:.1f}%)")
    
    # Event analysis
    print(f"\nEvent analysis:")
    event_sizes = taxonomy_df.groupby('event')['ticker'].nunique()
    binary_events = (event_sizes == 1).sum()
    categorical_events = (event_sizes > 1).sum()
    
    print(f"  Binary events (1 ticker): {binary_events:,}")
    print(f"  Categorical events (>1 ticker): {categorical_events:,}")
    
    if categorical_events > 0:
        max_tickers = event_sizes.max()
        avg_categorical = event_sizes[event_sizes > 1].mean()
        print(f"  Largest categorical event: {max_tickers} tickers")
        print(f"  Average categorical event size: {avg_categorical:.1f} tickers")
        
        # Show examples of largest categorical events
        print(f"\nTop categorical events by ticker count:")
        top_events = event_sizes.sort_values(ascending=False).head(5)
        for event, ticker_count in top_events.items():
            example_tickers = taxonomy_df[taxonomy_df['event'] == event]['ticker'].unique()[:3]
            print(f"  {event}: {ticker_count} tickers")
            print(f"    Examples: {', '.join(example_tickers)}")
            if ticker_count > 3:
                print(f"    (and {ticker_count-3} others)")
    
    # Time variant analysis (CUSIPs per ticker)
    print(f"\nTime variant analysis (CUSIPs per ticker):")
    cusip_per_ticker = taxonomy_df.groupby('ticker').size()
    print(f"  Average CUSIPs per ticker: {cusip_per_ticker.mean():.1f}")
    print(f"  Max CUSIPs per ticker: {cusip_per_ticker.max()}")
    
    # Show examples of high time fragmentation
    print(f"\nMost fragmented tickers (most time variants):")
    top_fragmented = cusip_per_ticker.sort_values(ascending=False).head(5)
    for ticker, cusip_count in top_fragmented.items():
        print(f"  {ticker}: {cusip_count} CUSIPs")
        examples = taxonomy_df[taxonomy_df['ticker'] == ticker]['title'].head(3).tolist()
        for i, example in enumerate(examples, 1):
            print(f"    {i}. {example}")
        if cusip_count > 3:
            print(f"    (and {cusip_count-3} other time variants)")


def generate_event_baskets(taxonomy_df: pd.DataFrame):
    """Generate event-level baskets following the one-exposure-per-event rule."""
    
    taxonomy = FourLayerTaxonomy()
    
    print("=== EVENT-LEVEL BASKET CONSTRUCTION ===")
    print("Rule: One exposure per EVENT (not per ticker or CUSIP)")
    
    # Filter for eligible markets (basic criteria)
    eligible_df = taxonomy_df[
        (taxonomy_df['theme'] != 'uncategorized') &
        (taxonomy_df['volume'] > 10000) &  # Minimum volume threshold
        (taxonomy_df['status'] == 'active')
    ].copy()
    
    print(f"\nEligible markets: {len(eligible_df):,} CUSIPs")
    eligible_events = eligible_df['event'].nunique()
    print(f"Eligible events: {eligible_events:,}")
    
    # Generate baskets by theme
    basket_summary = []
    
    for theme in eligible_df['theme'].unique():
        if theme == 'uncategorized':
            continue
            
        basket_df = taxonomy.construct_event_level_baskets(
            eligible_df, 
            theme, 
            selection_method='most_liquid'
        )
        
        if len(basket_df) > 0:
            total_volume = basket_df['volume'].sum()
            avg_volume = basket_df['volume'].mean()
            
            basket_summary.append({
                'theme': theme,
                'events': len(basket_df),
                'total_volume': total_volume,
                'avg_volume': avg_volume
            })
            
            # Save basket
            basket_path = PROJECT_ROOT / "data" / "outputs" / f"basket_{theme}_events.parquet"
            basket_path.parent.mkdir(exist_ok=True)
            basket_df.to_parquet(basket_path, index=False)
            
            print(f"\n{theme.upper()} BASKET:")
            print(f"  Events: {len(basket_df)}")
            print(f"  Total volume: ${total_volume:,.0f}")
            print(f"  Average volume per event: ${avg_volume:,.0f}")
            print(f"  Saved to: {basket_path}")
            
            # Show examples
            sample_basket = basket_df.sample(min(3, len(basket_df)))
            for _, row in sample_basket.iterrows():
                print(f"    • {row['event']} ({row['ticker']}) - ${row['volume']:,.0f}")
    
    # Overall summary
    total_basket_events = sum(b['events'] for b in basket_summary)
    total_basket_volume = sum(b['total_volume'] for b in basket_summary)
    
    print(f"\n=== OVERALL BASKET SUMMARY ===")
    print(f"Total basket events: {total_basket_events:,}")
    print(f"Total basket volume: ${total_basket_volume:,.0f}")
    print(f"Utilization: {total_basket_events}/{eligible_events} eligible events ({total_basket_events/eligible_events*100:.1f}%)")
    
    # Save summary
    summary_df = pd.DataFrame(basket_summary)
    summary_path = PROJECT_ROOT / "data" / "outputs" / "basket_summary_corrected.parquet"
    summary_df.to_parquet(summary_path, index=False)
    print(f"Basket summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
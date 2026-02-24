#!/usr/bin/env python3
"""
Build CUSIP → Ticker mapping for all prediction markets.

This script implements the methodology described in RESEARCH-timeseries.md
for mapping individual market contracts (CUSIPs) to recurring concepts (Tickers).

The hierarchy is: Theme → Event → Ticker → CUSIP
- CUSIP = specific contract with expiration (what Polymarket calls a market/condition_id)  
- Ticker = recurring concept without expiration ("Will Fed cut 25bps?")
- Same Ticker spawns multiple CUSIPs across time periods

Methodology:
1. Normalize titles to Ticker concepts using regex
2. Fuzzy deduplication of normalized titles
3. Assign Ticker IDs and map CUSIPs
4. Build rollable chains sorted by end_date
5. Output mapping files and statistics
6. Validate results

NO LLM for matching - Pure regex + fuzzy string matching.
"""

import pandas as pd
import numpy as np
import re
import json
import uuid
from datetime import datetime
from pathlib import Path
from rapidfuzz import fuzz
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TickerMapper:
    """Build CUSIP → Ticker mapping using regex normalization and fuzzy matching."""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.markets = None
        self.ticker_mapping = None
        self.ticker_chains = None
        
    def load_data(self):
        """Load markets data."""
        logger.info("Loading markets data...")
        markets_path = self.data_dir / 'processed' / 'markets.parquet'
        self.markets = pd.read_parquet(markets_path)
        logger.info(f"Loaded {len(self.markets)} markets")
        
        # Convert end_date to datetime for sorting
        self.markets['end_date_parsed'] = pd.to_datetime(self.markets['end_date'], errors='coerce')
        logger.info(f"Parsed {self.markets['end_date_parsed'].notna().sum()} end dates successfully")
        
    def normalize_title(self, title):
        """
        Normalize market title to recurring Ticker concept.
        
        Strips time-specific parts:
        - Month names with optional year
        - Years (2024, 2025, 2026, etc.)
        - Quarter references (Q1, Q2, etc.)
        - Relative time phrases
        - Meeting-specific references
        """
        if pd.isna(title):
            return ""
            
        normalized = str(title)
        
        # Month names with optional year patterns
        month_patterns = [
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b'
        ]
        
        for pattern in month_patterns:
            normalized = re.sub(pattern, 'MONTH', normalized, flags=re.IGNORECASE)
            
        # Year patterns (2024, 2025, 2026, etc.)
        normalized = re.sub(r'\b20[2-9]\d\b', 'YEAR', normalized)
        
        # Quarter references
        normalized = re.sub(r'\bQ[1-4]\b', 'QUARTER', normalized, flags=re.IGNORECASE)
        
        # Relative time phrases
        time_phrases = [
            r'\bafter\s+the\s+MONTH\s+meeting\b',
            r'\bafter\s+the\s+MONTH\s+YEAR\s+meeting\b', 
            r'\bby\s+MONTH\b',
            r'\bby\s+MONTH\s+YEAR\b',
            r'\bin\s+QUARTER\b',
            r'\bin\s+YEAR\b',
            r'\bfor\s+YEAR\b',
            r'\bof\s+YEAR\b',
            r'\bduring\s+YEAR\b',
            r'\bbefore\s+YEAR\b',
            r'\bafter\s+YEAR\b'
        ]
        
        for phrase_pattern in time_phrases:
            normalized = re.sub(phrase_pattern, 'TIMEREF', normalized, flags=re.IGNORECASE)
            
        # Generic time references
        normalized = re.sub(r'\bYEAR–YEAR\b', 'YEARRANGE', normalized)
        normalized = re.sub(r'\bYEAR-YEAR\b', 'YEARRANGE', normalized)
        
        # Clean up multiple spaces and trim
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Specific phrase normalizations for common patterns
        normalized = re.sub(r'\bafter\s+meeting\b', 'after meeting', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bMONTH\s+meeting\b', 'meeting', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bthe\s+YEAR\s+', 'the ', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bTIMEREF\s+', '', normalized)
        
        # Remove standalone time tokens
        normalized = re.sub(r'\b(MONTH|YEAR|QUARTER|TIMEREF|YEARRANGE)\b', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
        
    def fuzzy_deduplicate(self, normalized_titles, threshold=90):
        """
        Group near-identical normalized titles using fuzzy matching.
        
        Returns dict mapping original normalized title to canonical form.
        """
        logger.info(f"Fuzzy deduplicating {len(normalized_titles)} normalized titles...")
        
        # Get unique normalized titles
        unique_titles = list(set(normalized_titles))
        logger.info(f"Found {len(unique_titles)} unique normalized titles before fuzzy matching")
        
        # Group similar titles
        title_groups = {}  # canonical -> [similar_titles]
        title_to_canonical = {}  # title -> canonical
        
        for title in unique_titles:
            if not title:  # Skip empty titles
                continue
                
            # Find best match among existing canonicals
            best_match = None
            best_score = 0
            
            for canonical in title_groups.keys():
                score = fuzz.ratio(title, canonical)
                if score > best_score:
                    best_score = score
                    best_match = canonical
                    
            # If good match found, add to that group
            if best_score >= threshold and best_match:
                title_groups[best_match].append(title)
                title_to_canonical[title] = best_match
            else:
                # Create new group
                title_groups[title] = [title]
                title_to_canonical[title] = title
                
        logger.info(f"Fuzzy matching reduced {len(unique_titles)} titles to {len(title_groups)} groups")
        
        # Return mapping for all original titles (including duplicates)
        result = {}
        for original_title in normalized_titles:
            if original_title in title_to_canonical:
                result[original_title] = title_to_canonical[original_title]
            else:
                result[original_title] = original_title
                
        return result
        
    def assign_ticker_ids(self):
        """
        Process all markets to create CUSIP → Ticker mapping.
        """
        logger.info("Starting ticker ID assignment process...")
        
        # Step 1: Normalize all titles
        logger.info("Step 1: Normalizing titles...")
        self.markets['normalized_title'] = self.markets['title'].apply(self.normalize_title)
        
        # Step 2: Fuzzy deduplicate
        logger.info("Step 2: Fuzzy deduplication...")
        normalized_titles = self.markets['normalized_title'].tolist()
        title_to_canonical = self.fuzzy_deduplicate(normalized_titles, threshold=90)
        
        # Apply canonical mapping
        self.markets['ticker_name'] = self.markets['normalized_title'].map(title_to_canonical)
        
        # Step 3: Assign ticker IDs
        logger.info("Step 3: Assigning ticker IDs...")
        unique_tickers = self.markets['ticker_name'].dropna().unique()
        ticker_id_map = {ticker_name: f"ticker_{i:06d}" for i, ticker_name in enumerate(sorted(unique_tickers))}
        
        self.markets['ticker_id'] = self.markets['ticker_name'].map(ticker_id_map)
        
        # Filter out markets without valid ticker mapping
        valid_mapping = self.markets[self.markets['ticker_id'].notna()].copy()
        
        logger.info(f"Successfully mapped {len(valid_mapping)} markets to {len(unique_tickers)} tickers")
        
        # Create final mapping dataframe
        self.ticker_mapping = valid_mapping[[
            'market_id', 'ticker_id', 'ticker_name', 'event_slug', 
            'end_date', 'end_date_parsed', 'title'
        ]].copy()
        
    def build_rollable_chains(self):
        """
        Build rollable chains for each ticker with multiple CUSIPs.
        """
        logger.info("Building rollable chains...")
        
        chains = {}
        stats = {
            'total_tickers': 0,
            'rollable_tickers': 0,
            'cusips_per_ticker': []
        }
        
        for ticker_id in self.ticker_mapping['ticker_id'].unique():
            ticker_markets = self.ticker_mapping[
                self.ticker_mapping['ticker_id'] == ticker_id
            ].copy()
            
            # Sort by end_date for proper rolling order
            ticker_markets = ticker_markets.sort_values('end_date_parsed', na_position='last')
            
            # Build chain info
            chain_info = {
                'ticker_id': ticker_id,
                'ticker_name': ticker_markets['ticker_name'].iloc[0],
                'market_count': len(ticker_markets),
                'markets': []
            }
            
            for _, market in ticker_markets.iterrows():
                market_info = {
                    'market_id': market['market_id'],
                    'title': market['title'],
                    'end_date': market['end_date'],
                    'event_slug': market['event_slug']
                }
                chain_info['markets'].append(market_info)
                
            chains[ticker_id] = chain_info
            
            # Update stats
            stats['total_tickers'] += 1
            market_count = len(ticker_markets)
            stats['cusips_per_ticker'].append(market_count)
            
            if market_count >= 2:
                stats['rollable_tickers'] += 1
                
        self.ticker_chains = chains
        
        logger.info(f"Built {stats['total_tickers']} ticker chains")
        logger.info(f"Rollable tickers (2+ CUSIPs): {stats['rollable_tickers']}")
        logger.info(f"Average CUSIPs per ticker: {np.mean(stats['cusips_per_ticker']):.1f}")
        logger.info(f"Median CUSIPs per ticker: {np.median(stats['cusips_per_ticker']):.1f}")
        
        return stats
        
    def save_outputs(self):
        """
        Save ticker mapping and chains to files.
        """
        logger.info("Saving output files...")
        
        # Ensure output directory exists
        output_dir = self.data_dir / 'processed'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ticker mapping as parquet
        mapping_path = output_dir / 'ticker_mapping.parquet'
        self.ticker_mapping.to_parquet(mapping_path, index=False)
        logger.info(f"Saved ticker mapping to {mapping_path}")
        
        # Save ticker chains as JSON
        chains_path = output_dir / 'ticker_chains.json'
        with open(chains_path, 'w') as f:
            json.dump(self.ticker_chains, f, indent=2, default=str)
        logger.info(f"Saved ticker chains to {chains_path}")
        
    def generate_stats(self):
        """
        Generate and display comprehensive statistics.
        """
        logger.info("Generating statistics...")
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_markets': len(self.markets),
            'mapped_markets': len(self.ticker_mapping),
            'total_tickers': len(self.ticker_chains),
            'rollable_tickers': sum(1 for chain in self.ticker_chains.values() if chain['market_count'] >= 2),
            'cusip_distribution': {}
        }
        
        # CUSIP distribution
        cusip_counts = [chain['market_count'] for chain in self.ticker_chains.values()]
        stats['cusip_distribution'] = {
            'min': min(cusip_counts),
            'max': max(cusip_counts),
            'mean': np.mean(cusip_counts),
            'median': np.median(cusip_counts),
            'std': np.std(cusip_counts)
        }
        
        # Rollable distribution
        rollable_counts = [count for count in cusip_counts if count >= 2]
        if rollable_counts:
            stats['rollable_distribution'] = {
                'count': len(rollable_counts),
                'mean': np.mean(rollable_counts),
                'median': np.median(rollable_counts),
                'max': max(rollable_counts)
            }
        
        # Save stats
        stats_path = self.data_dir / 'processed' / 'ticker_mapping_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Saved statistics to {stats_path}")
        
        return stats
        
    def validate_results(self, top_n=20):
        """
        Validate results by showing top tickers and examples.
        """
        logger.info("Validating results...")
        
        # Top tickers by CUSIP count
        ticker_sizes = [(ticker_id, chain['market_count'], chain['ticker_name']) 
                       for ticker_id, chain in self.ticker_chains.items()]
        ticker_sizes.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {top_n} Tickers by CUSIP Count:")
        print("=" * 80)
        
        for i, (ticker_id, count, name) in enumerate(ticker_sizes[:top_n]):
            print(f"{i+1:2d}. {name[:50]:50s} ({count} CUSIPs)")
            
            # Show first few markets in chain
            chain = self.ticker_chains[ticker_id]
            for j, market in enumerate(chain['markets'][:3]):
                end_date = market['end_date'] or 'No end date'
                title = market['title'][:60] + "..." if len(market['title']) > 60 else market['title']
                print(f"    {j+1}. {title} (end: {end_date})")
            if len(chain['markets']) > 3:
                print(f"    ... and {len(chain['markets'])-3} more")
            print()
            
        # Show examples across categories using event_slug patterns
        print("\nExamples Across Categories:")
        print("=" * 80)
        
        category_patterns = [
            ('Federal Reserve', ['fed', 'federal-reserve', 'interest-rate']),
            ('Elections', ['election', 'presidential', 'democratic', 'republican']),
            ('Crypto', ['bitcoin', 'crypto', 'btc', 'ethereum']),
            ('Geopolitics', ['war', 'ukraine', 'china', 'iran', 'israel'])
        ]
        
        for category_name, patterns in category_patterns:
            print(f"\n{category_name}:")
            print("-" * 40)
            
            # Find tickers matching patterns
            matching_tickers = []
            for ticker_id, chain in self.ticker_chains.items():
                if chain['market_count'] >= 2:  # Only rollable ones
                    for market in chain['markets']:
                        event_slug = market.get('event_slug', '').lower()
                        ticker_name = chain['ticker_name'].lower()
                        if any(pattern in event_slug or pattern in ticker_name for pattern in patterns):
                            matching_tickers.append((ticker_id, chain['market_count'], chain['ticker_name']))
                            break
                            
            # Show top 3 for this category
            matching_tickers.sort(key=lambda x: x[1], reverse=True)
            for i, (ticker_id, count, name) in enumerate(matching_tickers[:3]):
                print(f"  {i+1}. {name} ({count} CUSIPs)")
                
    def run(self):
        """
        Execute the complete ticker mapping pipeline.
        """
        logger.info("Starting ticker mapping pipeline...")
        
        # Load data
        self.load_data()
        
        # Build mapping
        self.assign_ticker_ids()
        
        # Build rollable chains  
        chain_stats = self.build_rollable_chains()
        
        # Save outputs
        self.save_outputs()
        
        # Generate stats
        stats = self.generate_stats()
        
        # Validate results
        self.validate_results()
        
        logger.info("Ticker mapping pipeline completed successfully!")
        return stats

def main():
    """Main execution function."""
    mapper = TickerMapper()
    stats = mapper.run()
    
    # Print summary
    print("\n" + "="*80)
    print("TICKER MAPPING SUMMARY")
    print("="*80)
    print(f"Total markets processed: {stats['total_markets']:,}")
    print(f"Markets successfully mapped: {stats['mapped_markets']:,}")
    print(f"Total unique tickers: {stats['total_tickers']:,}")
    print(f"Rollable tickers (2+ CUSIPs): {stats['rollable_tickers']:,}")
    print(f"Average CUSIPs per ticker: {stats['cusip_distribution']['mean']:.1f}")
    print(f"Maximum CUSIPs per ticker: {stats['cusip_distribution']['max']:,}")
    
    rollable_pct = 100 * stats['rollable_tickers'] / stats['total_tickers']
    print(f"Percentage of rollable tickers: {rollable_pct:.1f}%")

if __name__ == "__main__":
    main()
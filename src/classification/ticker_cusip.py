"""Ticker/CUSIP classification system for markets.

This module extracts base market names (tickers) and assigns unique CUSIPs to individual markets.
Many markets represent the same underlying but different time periods or conditions.

Example:
- "Bitcoin Up or Down - February 22, 5:40AM-5:45AM ET" → ticker: BTC-UP-DOWN, cusip: unique_id
- "Bitcoin Up or Down - February 22, 5:50AM-5:55AM ET" → ticker: BTC-UP-DOWN, cusip: unique_id_2
"""

import re
import hashlib
import logging
from typing import Tuple, Dict, List
import pandas as pd
from datetime import datetime
from fuzzywuzzy import fuzz
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "ticker_patterns.yaml"


def normalize_title(title: str) -> str:
    """Basic title normalization."""
    if not title:
        return ""
    # Convert to lowercase, remove extra whitespace
    normalized = re.sub(r'\s+', ' ', title.lower().strip())
    return normalized


def extract_ticker_regex(title: str) -> str:
    """Extract ticker using regex patterns for common formats."""
    if not title:
        return "UNKNOWN"
    
    title_norm = normalize_title(title)
    
    # Common patterns to remove (dates, times, specific periods)
    patterns_to_remove = [
        # Dates: "February 22", "Feb 22", "2026-02-22", "22/02/2026"
        r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b',
        r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        
        # Times: "5:40AM-5:45AM ET", "12:30 PM", "3:00PM-4:00PM"
        r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\s*-?\s*\d{1,2}:\d{2}\s*(?:am|pm)?\s*(?:et|est|edt|pt|pst|pdt)?\b',
        
        # Periods and ranges: "Q1", "Q4", "week of", "by march", "through april"
        r'\b(?:q1|q2|q3|q4)\s*\d{4}?\b',
        r'\b(?:week|month)\s+of\b',
        r'\b(?:by|through|before|after|until)\s+\w+\b',
        
        # Specific instances: "game 1", "round 2", "week 5"
        r'\b(?:game|round|week|day|match)\s+\d+\b',
        
        # Event-specific: "2024 election", "2026 world cup"
        r'\b\d{4}\s+(?:election|midterms?|primary|general)\b',
        
        # Question markers and punctuation cleanup
        r'[?!.]+$',
        r'^(?:will|does|is|can|should|would|could)\s+',
    ]
    
    cleaned = title_norm
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, ' ', cleaned)
    
    # Clean up extra spaces and punctuation
    cleaned = re.sub(r'[^\w\s-]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Convert to ticker format: uppercase, spaces to hyphens, limit length
    ticker = cleaned.upper().replace(' ', '-')
    ticker = re.sub(r'-+', '-', ticker)  # Remove multiple hyphens
    ticker = ticker.strip('-')
    
    # Truncate if too long
    if len(ticker) > 50:
        ticker = ticker[:50].rstrip('-')
    
    return ticker if ticker else "UNKNOWN"


def extract_ticker_fuzzy(title: str, existing_tickers: Dict[str, List[str]]) -> str:
    """Use fuzzy matching to find similar existing tickers."""
    if not title or not existing_tickers:
        return extract_ticker_regex(title)
    
    title_norm = normalize_title(title)
    
    # Check similarity against existing ticker examples
    best_match = None
    best_score = 0
    threshold = 85  # Similarity threshold
    
    for ticker, examples in existing_tickers.items():
        for example in examples:
            score = fuzz.ratio(title_norm, normalize_title(example))
            if score > best_score and score >= threshold:
                best_score = score
                best_match = ticker
    
    if best_match:
        logger.debug(f"Fuzzy match: '{title}' → '{best_match}' (score: {best_score})")
        return best_match
    
    return extract_ticker_regex(title)


def generate_cusip(market_id: str, title: str, platform: str) -> str:
    """Generate unique CUSIP for individual market instance."""
    # Use hash of market_id + title for deterministic but unique CUSIP
    content = f"{platform}:{market_id}:{title}"
    hash_obj = hashlib.sha256(content.encode('utf-8'))
    cusip = hash_obj.hexdigest()[:12].upper()
    return cusip


def classify_tickers_cusips(markets_df: pd.DataFrame, use_fuzzy: bool = True) -> pd.DataFrame:
    """Add ticker and cusip columns to markets DataFrame."""
    logger.info(f"Classifying tickers/CUSIPs for {len(markets_df)} markets...")
    
    results = []
    ticker_examples = {}  # ticker -> list of example titles
    
    for idx, row in markets_df.iterrows():
        title = row.get('title', '')
        market_id = row.get('market_id', '')
        platform = row.get('platform', '')
        
        # Extract ticker
        if use_fuzzy:
            ticker = extract_ticker_fuzzy(title, ticker_examples)
        else:
            ticker = extract_ticker_regex(title)
        
        # Generate CUSIP
        cusip = generate_cusip(market_id, title, platform)
        
        # Track examples for fuzzy matching
        if ticker not in ticker_examples:
            ticker_examples[ticker] = []
        if len(ticker_examples[ticker]) < 5:  # Keep top 5 examples
            ticker_examples[ticker].append(title)
        
        results.append({
            'market_id': market_id,
            'ticker': ticker,
            'cusip': cusip,
            'title': title,
            'platform': platform
        })
    
    results_df = pd.DataFrame(results)
    
    # Add to original DataFrame
    markets_df = markets_df.copy()
    markets_df = markets_df.merge(
        results_df[['market_id', 'ticker', 'cusip']], 
        on='market_id', 
        how='left'
    )
    
    # Summary statistics
    unique_tickers = markets_df['ticker'].nunique()
    logger.info(f"Classification complete: {len(markets_df)} markets → {unique_tickers} unique tickers")
    
    # Log top tickers by market count
    top_tickers = markets_df['ticker'].value_counts().head(10)
    logger.info("Top 10 tickers by market count:")
    for ticker, count in top_tickers.items():
        logger.info(f"  {ticker}: {count} markets")
    
    return markets_df


def deduplicate_by_ticker(markets_df: pd.DataFrame, 
                         prices_df: pd.DataFrame,
                         dedup_method: str = 'most_liquid') -> pd.DataFrame:
    """Select one representative market per ticker for basket construction.
    
    Args:
        markets_df: Markets with ticker/cusip columns
        prices_df: Price history for volume/liquidity calculations
        dedup_method: 'most_liquid', 'highest_volume', 'longest_history'
    
    Returns:
        DataFrame with one market per ticker
    """
    logger.info(f"Deduplicating by ticker using method: {dedup_method}")
    
    if 'ticker' not in markets_df.columns:
        logger.warning("No ticker column found, skipping deduplication")
        return markets_df
    
    # Calculate metrics for deduplication
    volume_by_market = prices_df.groupby('market_id')['volume'].sum().to_dict()
    history_days_by_market = prices_df.groupby('market_id').size().to_dict()
    
    markets_df = markets_df.copy()
    markets_df['total_price_volume'] = markets_df['market_id'].map(volume_by_market).fillna(0)
    markets_df['price_history_days'] = markets_df['market_id'].map(history_days_by_market).fillna(0)
    
    deduped = []
    for ticker, group in markets_df.groupby('ticker'):
        if dedup_method == 'most_liquid':
            # Use total volume as proxy for liquidity
            best = group.loc[group['volume'].idxmax()]
        elif dedup_method == 'highest_volume':
            # Use metadata volume
            best = group.loc[group['volume'].idxmax()]  
        elif dedup_method == 'longest_history':
            # Use market with most price history
            best = group.loc[group['price_history_days'].idxmax()]
        else:
            # Default: just take first
            best = group.iloc[0]
        
        deduped.append(best)
    
    deduped_df = pd.DataFrame(deduped)
    original_count = len(markets_df)
    deduped_count = len(deduped_df)
    
    logger.info(f"Deduplication: {original_count} markets → {deduped_count} unique tickers "
                f"({original_count - deduped_count} duplicates removed)")
    
    return deduped_df


if __name__ == "__main__":
    # Test with sample data
    test_data = pd.DataFrame([
        {"market_id": "poly_1", "title": "Bitcoin Up or Down - February 22, 5:40AM-5:45AM ET", "platform": "polymarket", "volume": 1000000},
        {"market_id": "poly_2", "title": "Bitcoin Up or Down - February 22, 5:50AM-5:55AM ET", "platform": "polymarket", "volume": 2000000},
        {"market_id": "kalshi_1", "title": "Will Trump win 2028 election?", "platform": "kalshi", "volume": 5000000},
        {"market_id": "kalshi_2", "title": "Trump to win 2028 presidential race", "platform": "kalshi", "volume": 3000000},
    ])
    
    result = classify_tickers_cusips(test_data)
    print(result[['market_id', 'title', 'ticker', 'cusip']])
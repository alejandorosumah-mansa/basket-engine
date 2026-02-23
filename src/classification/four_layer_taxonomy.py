"""Four Layer Taxonomy: Theme → Event → Ticker → CUSIP

CORRECTED HIERARCHY (Top-Down):
1. Theme: Macro category (e.g., "Central Banks & Monetary Policy") 
2. Event: Broader question (e.g., "Fed Rate Decision")
3. Ticker: Recurring market concept without expiration (e.g., "Will Fed cut 50bps?")  
4. CUSIP: Specific contract WITH expiration date (e.g., "Will Fed cut 50bps in March 2025?")

BOTTOM-UP CONSTRUCTION:
- CUSIP: Individual market instance with specific time/date (what Polymarket calls market/condition_id)
- Ticker: Outcome level, stripped of time (recurring concept)
- Event: Parent question (can have multiple tickers for categorical events)
- Theme: Classification applied at EVENT level for basket construction

KEY RULES:
- Binary events: 1 ticker per event (ticker = event)
- Categorical events: multiple tickers per event
- Theme classification happens at EVENT level, not ticker/CUSIP level  
- Basket construction: one exposure per EVENT
- CUSIPs are time variants of tickers
"""

import re
import hashlib
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import yaml

from .taxonomy import load_taxonomy
import os
import json
import time

# Optional OpenAI import for LLM classification
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(Path.home() / ".openclaw" / ".env")
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)


class FourLayerTaxonomy:
    """Implements the corrected Theme → Event → Ticker → CUSIP hierarchy (bottom-up construction)."""
    
    def __init__(self):
        self.taxonomy = load_taxonomy()
        
        # Regex patterns for time/date stripping (CUSIP → Ticker)
        self.time_patterns = [
            # Dates: "February 28th", "by March 31", "before April 2026"
            r'\b(?:by|before|after|until|through)\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}?\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b',
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?\b',
            
            # Times: "5:40AM-5:45AM", "12:30 PM ET"
            r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\s*-?\s*\d{1,2}:\d{2}\s*(?:am|pm)?\s*(?:et|est|edt|pt|pst|pdt)?\b',
            
            # Meeting references: "after the March 2026 meeting"
            r'\bafter\s+the\s+\w+\s+\d{4}\s+meeting\b',
            
            # Quarterly: "in Q1", "Q4 2026"
            r'\b(?:in\s+)?q[1-4]\s*\d{4}?\b',
            
            # Year references: "in 2026", "2028 election"  
            r'\bin\s+\d{4}\b',
            r'\b\d{4}\s+(?:election|primary|midterms?)\b',
            
            # Generic time references
            r'\b(?:week|month)\s+of\s+\w+\b',
            r'\b(?:game|round|week|day|match)\s+\d+\b'
        ]
        
        # Patterns for categorical event detection
        self.categorical_patterns = [
            # Person wins something: "Harris wins nomination"
            r'^(.+?)\s+wins\s+(.+)$',
            # Team/player performance: "McDavid wins Hart Trophy"  
            r'^(.+?)\s+wins\s+(.+?)(?:\s+trophy|\s+award)?$',
            # Country outcomes: "China wins a gold medal"
            r'^(.+?)\s+wins?\s+a?\s*(.+?)(?:\s+medal)?$',
            # Over/under: "Ravens Team Total O/U 24.5"
            r'^(.+?)\s+(?:total|score):\s*o/u\s+(\d+\.?\d*)$'
        ]
        
    def cusip_to_ticker(self, cusip_title: str) -> str:
        """Layer 1→2: Strip time/date information from CUSIP to get Ticker."""
        if not cusip_title:
            return "UNKNOWN"
            
        ticker = cusip_title
        
        # Remove time patterns
        for pattern in self.time_patterns:
            ticker = re.sub(pattern, '', ticker, flags=re.IGNORECASE)
            
        # Clean up
        ticker = re.sub(r'\s+', ' ', ticker).strip()
        ticker = ticker.strip('.,!?-').strip()
        
        return ticker if ticker else "UNKNOWN"
    
    def group_tickers_into_events(self, tickers: List[str]) -> Dict[str, List[str]]:
        """Layer 2→3: Group related tickers into events."""
        events = {}
        processed_tickers = set()
        
        # First pass: detect categorical events using patterns
        for ticker in tickers:
            if ticker in processed_tickers:
                continue
                
            # Check if this ticker matches categorical patterns
            categorical_match = self._find_categorical_event(ticker, tickers)
            
            if categorical_match:
                event_name, related_tickers = categorical_match
                events[event_name] = related_tickers
                processed_tickers.update(related_tickers)
            else:
                # Binary event: ticker = event
                events[ticker] = [ticker]
                processed_tickers.add(ticker)
                
        return events
    
    def _find_categorical_event(self, ticker: str, all_tickers: List[str]) -> Optional[Tuple[str, List[str]]]:
        """Find if this ticker is part of a categorical event."""
        
        # Pattern 1: "[PERSON] wins [SAME_THING]"
        match = re.match(r'^(.+?)\s+wins\s+(.+)$', ticker, re.IGNORECASE)
        if match:
            person, award = match.groups()
            award = award.strip()
            
            # Find other tickers with same award
            related_tickers = []
            event_pattern = rf'^(.+?)\s+wins\s+{re.escape(award)}$'
            for other_ticker in all_tickers:
                if re.match(event_pattern, other_ticker, re.IGNORECASE):
                    related_tickers.append(other_ticker)
                    
            if len(related_tickers) > 1:
                return (award, related_tickers)
        
        # Pattern 2: "[TEAM] Total O/U [DIFFERENT_NUMBERS]"  
        match = re.match(r'^(.+?)\s+total.*o/u\s+(\d+\.?\d*)$', ticker, re.IGNORECASE)
        if match:
            team, number = match.groups()
            team = team.strip()
            
            # Find other O/U lines for same team
            related_tickers = []
            team_pattern = rf'^{re.escape(team)}\s+total.*o/u\s+\d+\.?\d*$'
            for other_ticker in all_tickers:
                if re.match(team_pattern, other_ticker, re.IGNORECASE):
                    related_tickers.append(other_ticker)
                    
            if len(related_tickers) > 1:
                return (f"{team} Total", related_tickers)
                
        # Pattern 3: Use existing event_slug structure if available
        # (This would require passing in the event_slug data)
        
        return None
    
    def classify_event_themes(self, events: Dict[str, List[str]]) -> Dict[str, str]:
        """Layer 3→4: Classify events into themes."""
        event_themes = {}
        
        for event_name, event_tickers in events.items():
            # Create context for LLM classification
            context = f"Event: {event_name}\n"
            if len(event_tickers) == 1:
                context += f"Type: Binary event\nTicker: {event_tickers[0]}"
            else:
                context += f"Type: Categorical event with {len(event_tickers)} outcomes\n"
                context += f"Sample tickers: {', '.join(event_tickers[:5])}"
                if len(event_tickers) > 5:
                    context += f" (and {len(event_tickers)-5} others)"
                    
            # Use LLM to classify the EVENT (not individual tickers)
            theme = self._classify_event_with_llm(event_name, context)
            event_themes[event_name] = theme
            
        return event_themes
    
    def _classify_event_with_llm(self, event_name: str, context: str) -> str:
        """Use LLM to classify an event into a theme, with fallback to keyword-based classification."""
        theme_options = list(self.taxonomy.keys())
        
        # Try LLM classification if available
        if HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
            try:
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                
                prompt = f"""
                You are classifying EVENTS (not individual markets) into thematic baskets.
                
                Available themes: {', '.join(theme_options)}
                
                Event to classify:
                {context}
                
                Instructions:
                - Classify the EVENT (the parent question), not individual outcomes
                - For categorical events, focus on what the event is fundamentally about
                - Binary events: the ticker usually represents the event directly
                - Return only the theme name from the available themes list
                - If no theme fits well, return "uncategorized"
                
                Return ONLY the theme name, nothing else.
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": "You are a market classifier. Return only the theme name."},
                        {"role": "user", "content": prompt.strip()}
                    ]
                )
                
                theme = response.choices[0].message.content.strip()
                
                # Validate theme is in our taxonomy
                if theme in theme_options:
                    return theme
                else:
                    logger.warning(f"LLM returned invalid theme '{theme}' for event '{event_name}', falling back to keyword classification")
                    
            except Exception as e:
                logger.warning(f"LLM classification failed for event '{event_name}': {e}, falling back to keyword classification")
        
        # Fallback to keyword-based classification
        return self._classify_event_with_keywords(event_name, context)
    
    def _classify_event_with_keywords(self, event_name: str, context: str) -> str:
        """Fallback keyword-based classification."""
        text = f"{event_name} {context}".lower()
        
        # Simple keyword matching
        if any(word in text for word in ['election', 'nomination', 'president', 'senate', 'congress', 'governor']):
            return 'us_elections'
        elif any(word in text for word in ['bitcoin', 'crypto', 'ethereum', 'ai', 'nvidia', 'technology']):
            return 'ai_technology'  
        elif any(word in text for word in ['iran', 'israel', 'middle east', 'strike']):
            return 'middle_east'
        elif any(word in text for word in ['fed', 'rate', 'inflation', 'cpi', 'economic']):
            return 'fed_monetary_policy'
        elif any(word in text for word in ['nhl', 'nfl', 'nba', 'sports', 'trophy', 'medal', 'olympics']):
            return 'sports_entertainment'
        else:
            return 'uncategorized'
    
    def process_markets(self, markets_df: pd.DataFrame, 
                       use_event_slug: bool = True) -> pd.DataFrame:
        """Process all markets through the 4-layer taxonomy."""
        logger.info(f"Processing {len(markets_df)} markets through 4-layer taxonomy...")
        
        # Layer 1: Markets are already CUSIPs (individual market instances)
        markets_df = markets_df.copy()
        
        # Layer 1→2: CUSIP to Ticker (strip time/date)
        logger.info("Layer 1→2: Converting CUSIPs to Tickers...")
        markets_df['ticker'] = markets_df['title'].apply(self.cusip_to_ticker)
        
        # Use event_slug as shortcut for event grouping if available
        if use_event_slug and 'event_slug' in markets_df.columns:
            logger.info("Using event_slug for Event grouping...")
            # event_slug essentially gives us the Event level
            markets_df['event'] = markets_df['event_slug'].fillna(markets_df['ticker'])
            
            # Group by event_slug to get event→tickers mapping
            event_to_tickers = {}
            for _, row in markets_df.iterrows():
                event = row['event']
                ticker = row['ticker'] 
                if event not in event_to_tickers:
                    event_to_tickers[event] = []
                if ticker not in event_to_tickers[event]:
                    event_to_tickers[event].append(ticker)
        else:
            # Layer 2→3: Group Tickers into Events  
            logger.info("Layer 2→3: Grouping Tickers into Events...")
            unique_tickers = markets_df['ticker'].unique().tolist()
            event_to_tickers = self.group_tickers_into_events(unique_tickers)
            
            # Create ticker→event mapping
            ticker_to_event = {}
            for event, tickers in event_to_tickers.items():
                for ticker in tickers:
                    ticker_to_event[ticker] = event
                    
            markets_df['event'] = markets_df['ticker'].map(ticker_to_event)
        
        # Layer 3→4: Classify Events into Themes
        logger.info("Layer 3→4: Classifying Events into Themes...")
        event_themes = self.classify_event_themes(event_to_tickers)
        
        # Add theme column
        markets_df['theme'] = markets_df['event'].map(event_themes).fillna('uncategorized')
        
        # Add CUSIP identifier (using market_id as unique identifier)
        markets_df['cusip'] = markets_df['market_id']
        
        # Summary statistics
        self._log_taxonomy_summary(markets_df, event_to_tickers)
        
        return markets_df
    
    def _log_taxonomy_summary(self, markets_df: pd.DataFrame, event_to_tickers: Dict):
        """Log summary statistics of the taxonomy process."""
        total_markets = len(markets_df)
        unique_tickers = markets_df['ticker'].nunique()
        unique_events = len(event_to_tickers)
        theme_counts = markets_df['theme'].value_counts()
        
        logger.info("=== TAXONOMY SUMMARY ===")
        logger.info(f"Layer 1 (CUSIPs): {total_markets:,} individual markets")
        logger.info(f"Layer 2 (Tickers): {unique_tickers:,} unique tickers")
        logger.info(f"Layer 3 (Events): {unique_events:,} unique events")
        logger.info(f"Layer 4 (Themes): {len(theme_counts)} themes")
        
        logger.info("\nTheme distribution:")
        for theme, count in theme_counts.head(10).items():
            pct = count / total_markets * 100
            logger.info(f"  {theme}: {count:,} markets ({pct:.1f}%)")
            
        # Event size analysis
        event_sizes = [len(tickers) for tickers in event_to_tickers.values()]
        binary_events = sum(1 for size in event_sizes if size == 1)
        categorical_events = sum(1 for size in event_sizes if size > 1)
        
        logger.info(f"\nEvent types:")
        logger.info(f"  Binary events (1 ticker): {binary_events:,}")
        logger.info(f"  Categorical events (>1 ticker): {categorical_events:,}")
        
        if categorical_events > 0:
            max_tickers = max(event_sizes)
            avg_categorical_size = sum(size for size in event_sizes if size > 1) / categorical_events
            logger.info(f"  Largest categorical event: {max_tickers} tickers")
            logger.info(f"  Average categorical event size: {avg_categorical_size:.1f} tickers")
    
    def construct_event_level_baskets(self, markets_df: pd.DataFrame, 
                                    theme: str,
                                    selection_method: str = 'most_liquid') -> pd.DataFrame:
        """Construct baskets with one exposure per EVENT (not per ticker/CUSIP).
        
        This implements the key rule: one exposure per underlying event.
        """
        theme_markets = markets_df[markets_df['theme'] == theme].copy()
        
        if len(theme_markets) == 0:
            return pd.DataFrame()
            
        logger.info(f"Constructing {theme} basket with one exposure per event...")
        
        basket_positions = []
        
        for event_name, event_group in theme_markets.groupby('event'):
            
            # For categorical events, pick ONE ticker (outcome) 
            if event_group['ticker'].nunique() > 1:
                # Multiple tickers per event - pick the best one
                if selection_method == 'most_liquid':
                    # Pick ticker with highest total volume
                    ticker_volumes = event_group.groupby('ticker')['volume'].sum()
                    best_ticker = ticker_volumes.idxmax()
                    ticker_group = event_group[event_group['ticker'] == best_ticker]
                else:
                    # Default: pick first ticker alphabetically for consistency
                    best_ticker = sorted(event_group['ticker'].unique())[0]
                    ticker_group = event_group[event_group['ticker'] == best_ticker]
            else:
                # Binary event - only one ticker
                ticker_group = event_group
            
            # Now pick ONE CUSIP from the selected ticker
            if len(ticker_group) > 1:
                # Multiple CUSIPs (time variants) - pick the best one
                if selection_method == 'most_liquid':
                    best_cusip = ticker_group.loc[ticker_group['volume'].idxmax()]
                else:
                    # Default: pick first one
                    best_cusip = ticker_group.iloc[0]
            else:
                best_cusip = ticker_group.iloc[0]
            
            basket_positions.append(best_cusip)
            
        basket_df = pd.DataFrame(basket_positions)
        
        # Verify one position per event
        assert basket_df['event'].nunique() == len(basket_df), f"Duplicate events in {theme} basket!"
        
        events_count = len(basket_df)
        original_markets = len(theme_markets)
        
        logger.info(f"{theme} basket: {events_count} events from {original_markets} original markets")
        
        return basket_df


def process_markets_with_corrected_taxonomy(markets_df: pd.DataFrame) -> pd.DataFrame:
    """Main entry point for processing markets with corrected 4-layer taxonomy."""
    taxonomy = FourLayerTaxonomy()
    return taxonomy.process_markets(markets_df)


if __name__ == "__main__":
    # Test with sample data
    test_data = pd.DataFrame([
        # Binary event with time variants (CUSIPs)
        {"market_id": "1", "title": "US strikes Iran by February 28th", "volume": 100000, "platform": "polymarket"},
        {"market_id": "2", "title": "US strikes Iran by March 31st", "volume": 150000, "platform": "polymarket"},
        {"market_id": "3", "title": "US strikes Iran by June 30th", "volume": 200000, "platform": "polymarket"},
        
        # Categorical event with multiple outcomes (tickers) and time variants (CUSIPs)
        {"market_id": "4", "title": "Harris wins nomination by March 2028", "volume": 300000, "platform": "polymarket"},
        {"market_id": "5", "title": "Clinton wins nomination by March 2028", "volume": 250000, "platform": "polymarket"},
        {"market_id": "6", "title": "Harris wins nomination by June 2028", "volume": 280000, "platform": "polymarket"},
        {"market_id": "7", "title": "Clinton wins nomination by June 2028", "volume": 220000, "platform": "polymarket"},
        
        # Sports categorical event
        {"market_id": "8", "title": "McDavid wins Hart Trophy", "volume": 50000, "platform": "polymarket", "event_slug": "nhl-hart-trophy"},
        {"market_id": "9", "title": "MacKinnon wins Hart Trophy", "volume": 40000, "platform": "polymarket", "event_slug": "nhl-hart-trophy"},
        {"market_id": "10", "title": "Keller wins Hart Trophy", "volume": 30000, "platform": "polymarket", "event_slug": "nhl-hart-trophy"},
    ])
    
    taxonomy = FourLayerTaxonomy()
    result = taxonomy.process_markets(test_data)
    
    print("=== TAXONOMY RESULTS ===")
    print(result[['market_id', 'title', 'ticker', 'event', 'theme']].to_string())
    
    print("\n=== EVENT-LEVEL BASKETS ===")
    for theme in result['theme'].unique():
        if theme != 'uncategorized':
            basket = taxonomy.construct_event_level_baskets(result, theme)
            print(f"\n{theme.upper()} BASKET ({len(basket)} events):")
            print(basket[['event', 'ticker', 'title', 'volume']].to_string())
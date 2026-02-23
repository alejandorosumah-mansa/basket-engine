#!/usr/bin/env python3
"""
LLM-based market classification script.
Classifies all 20,180 markets using GPT-4o-mini in batches.
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI()

# Constants
BATCH_SIZE = 35  # 30-40 as requested
RATE_LIMIT_DELAY = 0.5  # 0.5 seconds between batches
MODEL = "gpt-4o-mini"
CACHE_FILE = "data/processed/llm_market_categories.json"
MARKETS_FILE = "data/processed/markets.parquet"
OUTPUT_FILE = "data/processed/market_classifications.parquet"
FILTERED_FILE = "data/processed/markets_filtered.parquet"

# Classification prompt template
CLASSIFICATION_PROMPT = """Classify each prediction market into exactly ONE category. Categories:

- sports (any sport: NBA, NFL, soccer, tennis, golf, Olympics, esports, player stats, game results, fantasy sports)
- entertainment (movies, music, TV shows, celebrities, awards shows, reality TV, gaming releases like GTA, social media posting counts)
- us_elections (US presidential, congressional, gubernatorial, mayoral elections and primaries)
- fed_monetary_policy (Fed rate decisions, Fed chair nominations, FOMC meetings)
- crypto_digital (Bitcoin, Ethereum, altcoins, token launches, FDV predictions, crypto exchanges, DeFi, NFTs)
- ai_technology (AI models, AI companies, self-driving, robotics, tech product launches)
- us_economic (GDP, inflation, unemployment, stock prices, earnings, IPOs, M&A, tariffs, debt, market cap)
- middle_east (Israel, Iran, Hamas, Hezbollah, Yemen, Gaza, Saudi Arabia conflicts and diplomacy)
- russia_ukraine (Russia-Ukraine war, ceasefire, NATO-Russia)
- china_geopolitics (China-Taiwan, China-US trade, TikTok, South China Sea)
- global_politics (non-US elections, international leaders, EU/NATO/BRICS/UN politics)
- venezuela (Venezuela, Maduro, US-Venezuela)
- climate_environment (temperature records, earthquakes, natural disasters, climate change, weather)
- pandemic_health (COVID, new pandemics, vaccines, disease outbreaks)
- legal_regulatory (indictments, SCOTUS, legislation, regulatory actions, criminal charges)
- energy_commodities (oil, gold, silver, natural gas, commodities trading)
- space_frontier (SpaceX, NASA, space missions, aliens/UFOs, Artemis)
- us_military (US military strikes, US troops deployment, defense policy)
- other (anything that doesn't clearly fit above)

Reply as JSON array: [{"title": "...", "category": "..."}]
Only return the JSON, no other text."""


def load_cache() -> Dict[str, str]:
    """Load existing classifications from cache file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
    return {}


def save_cache(cache: Dict[str, str]) -> None:
    """Save classifications to cache file."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def classify_batch(titles: List[str]) -> Optional[Dict[str, str]]:
    """Classify a batch of market titles using OpenAI API."""
    
    # Format titles for the prompt
    titles_text = "\n".join([f"- {title}" for title in titles])
    full_prompt = f"{CLASSIFICATION_PROMPT}\n\nMarkets to classify:\n{titles_text}"
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent classifications
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response (handle markdown wrapping)
        try:
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]   # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            
            content = content.strip()
            classifications = json.loads(content)
            
            # Convert to dict for easier lookup
            result = {}
            for item in classifications:
                if isinstance(item, dict) and "title" in item and "category" in item:
                    result[item["title"]] = item["category"]
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Response content: {content}")
            return None
            
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None


def main():
    """Main classification pipeline."""
    
    logger.info("Loading markets data...")
    markets = pd.read_parquet(MARKETS_FILE)
    logger.info(f"Loaded {len(markets)} markets")
    
    # Load existing cache
    cache = load_cache()
    logger.info(f"Loaded {len(cache)} cached classifications")
    
    # Get unique titles that need classification
    all_titles = markets['title'].unique()
    titles_to_classify = [title for title in all_titles if title not in cache]
    logger.info(f"Need to classify {len(titles_to_classify)} new titles")
    
    if titles_to_classify:
        # Process in batches
        total_batches = (len(titles_to_classify) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(0, len(titles_to_classify), BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            batch_titles = titles_to_classify[i:i + BATCH_SIZE]
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_titles)} titles)")
            
            # Classify this batch
            batch_results = classify_batch(batch_titles)
            
            if batch_results:
                # Update cache with results
                cache.update(batch_results)
                logger.info(f"Successfully classified {len(batch_results)} titles")
                
                # Save cache after each batch (for resume support)
                save_cache(cache)
            else:
                logger.warning(f"Failed to classify batch {batch_num}")
                # Continue with next batch instead of stopping
            
            # Rate limiting
            if batch_num < total_batches:  # Don't delay after last batch
                time.sleep(RATE_LIMIT_DELAY)
    
    logger.info(f"Total classifications: {len(cache)}")
    
    # Apply classifications to markets DataFrame
    logger.info("Applying classifications to markets...")
    markets['category'] = markets['title'].map(cache)
    
    # Handle any missing classifications
    missing_classifications = markets['category'].isna().sum()
    if missing_classifications > 0:
        logger.warning(f"Missing classifications for {missing_classifications} markets")
        markets.loc[markets['category'].isna(), 'category'] = 'other'
    
    # Save full results
    classification_df = markets[['market_id', 'title', 'category']].copy()
    classification_df.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Saved classifications to {OUTPUT_FILE}")
    
    # Print category distribution
    logger.info("\nCategory distribution:")
    category_counts = markets['category'].value_counts()
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count}")
    
    # Filter out sports and entertainment
    logger.info("\nFiltering out sports and entertainment categories...")
    filtered_markets = markets[~markets['category'].isin(['sports', 'entertainment'])].copy()
    
    sports_count = (markets['category'] == 'sports').sum()
    entertainment_count = (markets['category'] == 'entertainment').sum()
    
    logger.info(f"Filtered out {sports_count} sports markets")
    logger.info(f"Filtered out {entertainment_count} entertainment markets")
    logger.info(f"Remaining markets: {len(filtered_markets)}")
    
    # Save filtered markets (overwrite existing file)
    filtered_markets.to_parquet(FILTERED_FILE, index=False)
    logger.info(f"Saved filtered markets to {FILTERED_FILE}")
    
    logger.info("Classification complete!")


if __name__ == "__main__":
    main()
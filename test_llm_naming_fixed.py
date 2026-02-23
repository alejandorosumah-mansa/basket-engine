#!/usr/bin/env python3

import pandas as pd
import json
from openai import OpenAI
import os
import time

def test_llm_naming():
    """Test LLM naming on just the top 3 communities"""
    
    print("=== TESTING LLM NAMING (TOP 3 COMMUNITIES) ===")
    
    # Set up OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized")
    
    # Load community assignments
    print("Loading community assignments...")
    community_assignments = pd.read_parquet('data/processed/community_assignments.parquet')
    print(f"Loaded {len(community_assignments)} market assignments")
    
    # Load markets data for titles
    print("Loading markets data...")
    markets_df = pd.read_parquet('data/processed/markets_filtered.parquet')
    markets_df = markets_df.set_index('market_id')
    print(f"Loaded {len(markets_df)} market titles")
    
    # Get top 3 largest communities
    community_sizes = community_assignments['community'].value_counts().head(3)
    print(f"Testing on top 3 communities: {community_sizes.tolist()}")
    
    # Process each community
    for i, (community_id, size) in enumerate(community_sizes.items()):
        print(f"\n--- Testing Community {community_id} - Size: {size} ---")
        
        # Get market IDs in this community
        market_ids = community_assignments[community_assignments['community'] == community_id].index.tolist()
        
        # Get market titles (limit to first 10 for testing)
        market_titles = []
        for market_id in market_ids[:10]:
            if market_id in markets_df.index:
                title = markets_df.loc[market_id, 'title']
                market_titles.append(title)
        
        print(f"Sample titles:")
        for title in market_titles:
            print(f"  - {title}")
        
        # Prepare simple prompt
        titles_text = "\n".join([f"- {title}" for title in market_titles])
        
        prompt = f"""These are prediction markets in a basket. They should be FINANCIAL, POLITICAL, or ECONOMIC only.

Market titles:
{titles_text}

Give:
1) A short name (max 5 words)
2) One main problem with this grouping

Format:
NAME: [name]
PROBLEM: [issue]"""
        
        print("Calling OpenAI API...")
        
        # Call OpenAI API with new format
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            print(f"✅ Response: {result}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(1)  # Brief pause between calls
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_llm_naming()
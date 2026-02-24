#!/usr/bin/env python3
"""
LLM validation of communities to identify and remove outliers.
For each community, ask LLM which markets don't belong.
"""

import pandas as pd
import os
import openai
from typing import Dict, List
import json
import re

def validate_community_with_llm(market_titles: List[str], community_id: int, client, model: str = "gpt-4o-mini") -> Dict:
    """Ask LLM to validate a community and identify outliers."""
    
    titles_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(market_titles)])
    
    prompt = f"""You are analyzing a cluster of prediction markets that are grouped together because they have correlated price movements. Your job is to identify any markets that don't belong - markets that seem unrelated to the main theme or likely don't share the same underlying risk factors.

Community {community_id} Markets:
{titles_text}

Please:
1. Identify the main theme/topic that most markets are about
2. List any markets that seem unrelated or don't fit the main theme
3. Suggest a name for this community (2-4 words)

Respond in this exact format:
THEME: [what most markets are about - 1-2 sentences]
OUTLIERS: [comma-separated list of market numbers that don't belong, or "None" if all fit]
NAME: [2-4 word descriptive name]

Be strict about outliers - only include markets that clearly don't share the same underlying risk/event as the majority."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse response
        theme = "Unknown theme"
        outliers = []
        name = f"Community_{community_id}"
        
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('THEME:'):
                theme = line[6:].strip()
            elif line.startswith('OUTLIERS:'):
                outliers_text = line[9:].strip()
                if outliers_text.lower() != "none":
                    # Extract numbers from comma-separated list
                    outlier_numbers = re.findall(r'\d+', outliers_text)
                    outliers = [int(n) - 1 for n in outlier_numbers if n.isdigit()]  # Convert to 0-indexed
            elif line.startswith('NAME:'):
                name = line[5:].strip()
        
        return {
            "theme": theme,
            "outlier_indices": outliers,
            "name": name,
            "raw_response": response_text
        }
        
    except Exception as e:
        print(f"Error calling LLM for community {community_id}: {e}")
        return {
            "theme": "Error analyzing community",
            "outlier_indices": [],
            "name": f"Community_{community_id}",
            "raw_response": f"Error: {e}"
        }

def main():
    print("=== LLM Validation of Communities ===")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    client = openai.OpenAI(api_key=api_key)
    
    # Load community assignments
    assignments = pd.read_parquet("data/processed/strict_community_assignments.parquet")
    print(f"Loaded {len(assignments)} market assignments")
    
    # Load market metadata
    markets = pd.read_parquet("data/processed/markets_filtered.parquet")
    print(f"Loaded {len(markets)} market metadata")
    
    # Group markets by community
    communities = {}
    for market_id, row in assignments.iterrows():
        comm_id = row['community']
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(market_id)
    
    print(f"Found {len(communities)} communities to validate")
    
    # Validate each community
    validation_results = {}
    cleaned_partition = {}
    
    for comm_id, market_ids in communities.items():
        print(f"\nValidating Community {comm_id} ({len(market_ids)} markets)...")
        
        # Get market titles
        comm_markets = markets[markets["market_id"].isin(market_ids)]
        if len(comm_markets) == 0:
            print(f"Warning: No metadata found for community {comm_id}")
            continue
            
        market_titles = comm_markets["title"].tolist()
        market_ids_with_titles = comm_markets["market_id"].tolist()
        
        # Validate with LLM
        validation = validate_community_with_llm(market_titles, comm_id, client)
        validation_results[comm_id] = validation
        
        print(f"Theme: {validation['theme']}")
        print(f"Name: {validation['name']}")
        print(f"Outliers: {len(validation['outlier_indices'])} markets")
        
        # Remove outliers from community
        clean_market_ids = []
        outlier_indices = set(validation['outlier_indices'])
        
        for i, market_id in enumerate(market_ids_with_titles):
            if i not in outlier_indices:
                clean_market_ids.append(market_id)
                cleaned_partition[market_id] = comm_id
            else:
                print(f"  Removing outlier: {market_titles[i]}")
        
        print(f"Cleaned community size: {len(clean_market_ids)}")
    
    # Filter out communities that are too small after cleaning
    min_size_after_cleaning = 8
    final_partition = {}
    final_community_info = {}
    
    community_sizes = {}
    for market_id, comm_id in cleaned_partition.items():
        community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
    
    for market_id, comm_id in cleaned_partition.items():
        if community_sizes[comm_id] >= min_size_after_cleaning:
            final_partition[market_id] = comm_id
            if comm_id not in final_community_info:
                final_community_info[comm_id] = validation_results[comm_id]
    
    # Save cleaned assignments
    if final_partition:
        cleaned_assignments = pd.DataFrame([
            {"market_id": market_id, "community": comm_id} 
            for market_id, comm_id in final_partition.items()
        ]).set_index("market_id")
        
        output_path = "data/processed/llm_validated_assignments.parquet"
        cleaned_assignments.to_parquet(output_path)
        print(f"\nSaved LLM-validated assignments to {output_path}")
    
    # Save validation results
    with open("data/processed/llm_validation_results.json", "w") as f:
        json.dump(validation_results, f, indent=2)
    
    # Print final summary
    final_community_sizes = {}
    for comm_id in final_community_info:
        final_community_sizes[comm_id] = sum(1 for c in final_partition.values() if c == comm_id)
    
    print(f"\n=== Final Results ===")
    print(f"Communities after validation: {len(final_community_info)}")
    print(f"Markets in final communities: {len(final_partition)}")
    print(f"Markets removed as outliers: {len(assignments) - len(final_partition)}")
    
    print(f"\nFinal community sizes:")
    for comm_id, size in sorted(final_community_sizes.items(), key=lambda x: -x[1]):
        info = final_community_info[comm_id]
        print(f"  {info['name']}: {size} markets")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
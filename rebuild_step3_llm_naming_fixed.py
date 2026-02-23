#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
from pathlib import Path
from openai import OpenAI
import os
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_llm_naming_and_critique():
    """Run LLM naming and critique for all communities using OpenAI GPT-4o-mini"""
    
    print("=== STEP 3: LLM NAMING + CRITIQUE (ALL COMMUNITIES) ===")
    
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
    print(f"Found {community_assignments['community'].nunique()} communities")
    
    # Load markets data for titles
    print("Loading markets data...")
    markets_df = pd.read_parquet('data/processed/markets_filtered.parquet')
    markets_df = markets_df.set_index('market_id')
    print(f"Loaded {len(markets_df)} market titles")
    
    # Prepare results storage
    community_names_critique = []
    optimal_community_info = {}
    community_labels = {}
    
    # Get community sizes for processing order (largest first)
    community_sizes = community_assignments['community'].value_counts().sort_values(ascending=False)
    print(f"Processing {len(community_sizes)} communities...")
    
    # Process each community
    for i, (community_id, size) in enumerate(community_sizes.items()):
        print(f"\n--- Processing Community {community_id} ({i+1}/{len(community_sizes)}) - Size: {size} ---")
        
        # Get market IDs in this community
        market_ids = community_assignments[community_assignments['community'] == community_id].index.tolist()
        
        # Get market titles
        market_titles = []
        for market_id in market_ids:
            if market_id in markets_df.index:
                title = markets_df.loc[market_id, 'title']
                market_titles.append(title)
            else:
                market_titles.append(f"[Missing title for {market_id}]")
        
        print(f"Got {len(market_titles)} market titles")
        
        # Prepare prompt (limit to 30 titles for token efficiency)
        titles_text = "\n".join([f"- {title}" for title in market_titles[:30]])
        if len(market_titles) > 30:
            titles_text += f"\n... and {len(market_titles) - 30} more similar markets"
        
        prompt = f"""These are prediction markets in a basket. They should be FINANCIAL, POLITICAL, or ECONOMIC only (no sports/entertainment).

Market titles:
{titles_text}

Please provide:
1) A short name (max 5 words) for this market basket
2) List 2-3 problems with this grouping. If you see any sports or entertainment markets that don't belong, flag them.

Format your response as:
NAME: [your name here]
PROBLEMS:
- [problem 1]
- [problem 2]
- [problem 3 if applicable]"""
        
        # Call OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in financial markets and prediction market analysis. Be concise and critical in your assessment."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            print(f"Got response: {result[:100]}...")
            
            # Parse response
            lines = result.split('\n')
            name = "Unknown Basket"
            problems = []
            
            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith('NAME:'):
                    name = line.replace('NAME:', '').strip()
                elif line.startswith('PROBLEMS:'):
                    current_section = 'problems'
                elif current_section == 'problems' and line.startswith('-'):
                    problems.append(line[1:].strip())
            
            print(f"Parsed name: {name}")
            print(f"Parsed problems: {len(problems)}")
            
        except Exception as e:
            print(f"❌ Error calling OpenAI API: {e}")
            name = f"Community {community_id}"
            problems = [f"API Error: {str(e)}"]
        
        # Store results
        community_info = {
            'community_id': int(community_id),
            'name': name,
            'size': int(size),
            'problems': problems,
            'market_count': len(market_titles),
            'sample_markets': market_titles[:10]  # Keep top 10 as samples
        }
        
        optimal_community_info[str(community_id)] = community_info
        community_labels[str(community_id)] = name
        
        # Add to markdown content
        community_names_critique.append({
            'id': community_id,
            'name': name,
            'size': size,
            'problems': problems,
            'markets': market_titles
        })
        
        # Rate limiting - small delay between API calls
        time.sleep(0.5)
        
        # Progress update every 25 communities
        if (i + 1) % 25 == 0:
            print(f"✅ Progress: {i+1}/{len(community_sizes)} communities processed")
    
    # Create outputs directory if it doesn't exist
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    # Save community names and critique as markdown
    markdown_content = f"""# Community Names and Critique
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Resolution: 5.0
Pipeline: Rebuilt from 8,426 clean markets

Total communities: {len(community_sizes)}
Total markets: {len(community_assignments)}

## Summary
- Largest community: {community_sizes.iloc[0]} markets
- Median community size: {community_sizes.median():.1f} markets
- Communities with 1 market: {(community_sizes == 1).sum()}
- Communities with >10 markets: {(community_sizes > 10).sum()}

## Communities (sorted by size)

"""
    
    for item in community_names_critique:
        markdown_content += f"""### Community {item['id']}: {item['name']} ({item['size']} markets)

**Problems:**
"""
        for problem in item['problems']:
            markdown_content += f"- {problem}\n"
        
        markdown_content += f"\n**Sample Markets:**\n"
        for market in item['markets'][:5]:  # Show top 5
            markdown_content += f"- {market}\n"
        
        if len(item['markets']) > 5:
            markdown_content += f"- ... and {len(item['markets']) - 5} more\n"
        
        markdown_content += "\n---\n\n"
    
    # Save markdown file
    markdown_path = outputs_dir / 'community_names_critique.md'
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    print(f"✅ Saved critique to {markdown_path}")
    
    # Save JSON files
    json_path = outputs_dir / 'optimal_community_info.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(optimal_community_info, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved community info to {json_path}")
    
    # Update community labels
    labels_path = Path('data/processed/community_labels.json')
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(community_labels, f, indent=2, ensure_ascii=False)
    print(f"✅ Updated community labels at {labels_path}")
    
    print(f"\n✅ LLM naming and critique completed successfully!")
    print(f"Summary:")
    print(f"  - Communities processed: {len(community_sizes)}")
    print(f"  - Files created: {markdown_path.name}, {json_path.name}")
    print(f"  - Labels updated: {labels_path.name}")
    
    return community_names_critique, optimal_community_info, community_labels

if __name__ == "__main__":
    critique, info, labels = run_llm_naming_and_critique()
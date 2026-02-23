#!/usr/bin/env python3
"""
LLM-based community naming and critique for basket-engine clustering.
"""

import pandas as pd
import json
import os
import sys
import time
from openai import OpenAI
from pathlib import Path

def load_data():
    """Load and join community assignments with market data."""
    print("Loading data...")
    sys.stdout.flush()
    
    # Load community assignments (market_id as index)
    ca = pd.read_parquet('data/processed/community_assignments.parquet')
    print(f"Community assignments: {ca.shape[0]} markets, {ca['community'].nunique()} communities")
    sys.stdout.flush()
    
    # Load filtered markets
    markets = pd.read_parquet('data/processed/markets_filtered.parquet')
    print(f"Markets filtered: {markets.shape[0]} markets")
    sys.stdout.flush()
    
    # Join on market_id
    ca_reset = ca.reset_index()  # market_id becomes a column
    joined = ca_reset.merge(markets[['market_id', 'title']], on='market_id', how='inner')
    print(f"Joined data: {joined.shape[0]} markets with titles")
    sys.stdout.flush()
    
    return joined

def get_community_titles(data, community_id):
    """Get all market titles for a specific community."""
    community_data = data[data['community'] == community_id]
    titles = community_data['title'].tolist()
    return titles, len(titles)

def query_openai(titles, community_id):
    """Query OpenAI GPT-4o-mini for community naming and critique."""
    client = OpenAI()  # Uses OPENAI_API_KEY env var
    
    titles_text = "\n".join([f"- {title}" for title in titles])
    
    prompt = f"""Here are the prediction markets in this basket. These are financial, political, and economic markets (no sports/entertainment). Give a short descriptive name (max 5 words) and list 2-3 potential problems with this grouping.

Markets in this community ({len(titles)} total):
{titles_text}

Please respond in this exact format:
NAME: [your short name here]
PROBLEMS:
1. [problem 1]
2. [problem 2]
3. [problem 3 if applicable]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing prediction markets and identifying thematic patterns. Be concise and specific."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying OpenAI for community {community_id}: {e}")
        return f"ERROR: {e}"

def parse_openai_response(response_text):
    """Parse the OpenAI response to extract name and problems."""
    lines = response_text.split('\n')
    name = ""
    problems = []
    
    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith('NAME:'):
            name = line.replace('NAME:', '').strip()
        elif line.startswith('PROBLEMS:'):
            current_section = 'problems'
        elif current_section == 'problems' and line.startswith(('1.', '2.', '3.')):
            problem = line.split('.', 1)[1].strip()
            problems.append(problem)
    
    return name, problems

def main():
    print("Starting LLM community naming script...")
    sys.stdout.flush()
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    print("Created outputs directory")
    sys.stdout.flush()
    
    # Load and join data
    data = load_data()
    print("Data loaded successfully")
    sys.stdout.flush()
    
    # Get unique communities
    communities = sorted(data['community'].unique())
    print(f"Processing {len(communities)} communities...")
    sys.stdout.flush()
    
    # For testing, process only first 5 communities
    # communities = communities[:5]
    # print(f"Testing with first {len(communities)} communities")
    
    # Results storage
    all_results = {}
    markdown_content = "# Community Names and Critique\n\n"
    markdown_content += f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    markdown_content += f"Total communities: {len(communities)}\n\n"
    
    for i, community_id in enumerate(communities):
        print(f"Processing community {community_id} ({i+1}/{len(communities)})...")
        sys.stdout.flush()
        
        # Get titles for this community
        titles, count = get_community_titles(data, community_id)
        
        if count == 0:
            print(f"  Warning: Community {community_id} has no titles!")
            sys.stdout.flush()
            continue
            
        print(f"  {count} markets")
        sys.stdout.flush()
        
        # Query OpenAI
        response = query_openai(titles, community_id)
        
        # Parse response
        name, problems = parse_openai_response(response)
        
        # Store results
        all_results[str(community_id)] = {
            'community_id': int(community_id),
            'name': name,
            'problems': problems,
            'market_count': count,
            'raw_response': response,
            'sample_titles': titles[:5]  # First 5 titles as sample
        }
        
        # Add to markdown
        markdown_content += f"## Community {community_id}: {name}\n\n"
        markdown_content += f"**Markets:** {count}\n\n"
        markdown_content += "**Problems:**\n"
        for j, problem in enumerate(problems, 1):
            markdown_content += f"{j}. {problem}\n"
        markdown_content += "\n**Sample titles:**\n"
        for title in titles[:5]:
            markdown_content += f"- {title}\n"
        markdown_content += "\n" + "-"*80 + "\n\n"
        
        print(f"  Name: {name}")
        print(f"  Problems: {len(problems)} identified")
        sys.stdout.flush()
        
        # Rate limiting - be nice to OpenAI
        time.sleep(0.5)
    
    # Save results
    print("\nSaving results...")
    
    # Save markdown
    with open('outputs/community_names_critique.md', 'w') as f:
        f.write(markdown_content)
    print("  Saved: outputs/community_names_critique.md")
    
    # Save JSON
    with open('outputs/optimal_community_info.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("  Saved: outputs/optimal_community_info.json")
    
    # Create community labels JSON for charts
    community_labels = {}
    for comm_id, info in all_results.items():
        community_labels[comm_id] = info['name']
    
    with open('data/processed/community_labels.json', 'w') as f:
        json.dump(community_labels, f, indent=2, ensure_ascii=False)
    print("  Updated: data/processed/community_labels.json")
    
    print(f"\nCompleted! Processed {len(all_results)} communities.")

if __name__ == "__main__":
    main()
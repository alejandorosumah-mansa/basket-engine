#!/usr/bin/env python3
"""
Run LLM naming and critique for all communities using GPT-4o-mini.
"""

import pandas as pd
import os
import openai
from datetime import datetime

def generate_community_names_and_critiques():
    print("=== Running LLM Community Naming and Critique ===")
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    client = openai.OpenAI(api_key=api_key)
    
    # Load community assignments
    assignments = pd.read_parquet("data/processed/community_assignments.parquet")
    print(f"Loaded community assignments for {len(assignments)} markets")
    
    # Load filtered markets
    markets = pd.read_parquet("data/processed/markets_filtered.parquet")
    print(f"Loaded {len(markets)} filtered markets (FINANCIAL, POLITICAL, ECONOMIC only)")
    
    # Group markets by community
    communities = {}
    for market_id, row in assignments.iterrows():
        comm_id = row["community"]
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(market_id)
    
    print(f"Processing {len(communities)} communities...")
    
    results = []
    
    for comm_id, market_ids in sorted(communities.items()):
        print(f"Processing Community {comm_id} ({len(market_ids)} markets)...")
        
        # Get market titles for this community
        comm_markets = markets[markets["market_id"].isin(market_ids)]
        
        if len(comm_markets) == 0:
            print(f"  Warning: No market data found for Community {comm_id}")
            results.append({
                "community_id": comm_id,
                "name": f"Community_{comm_id}",
                "critique": "No market data available",
                "n_markets": len(market_ids),
                "sample_titles": []
            })
            continue
        
        # Get all titles for this community
        market_titles = comm_markets["title"].tolist()
        titles_text = "\n".join([f"- {title}" for title in market_titles])
        
        # Create prompt for naming and critique
        prompt = f"""Here are prediction markets in this basket. These are FINANCIAL, POLITICAL, and ECONOMIC markets only (no sports/entertainment). Give a short descriptive name (max 5 words) and list 2-3 potential problems.

Markets:
{titles_text}

Respond in this exact format:
NAME: [short descriptive name, max 5 words]
PROBLEMS:
- [problem 1]
- [problem 2]
- [problem 3 if applicable]"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse response
            lines = response_text.split('\n')
            name = f"Community_{comm_id}"
            problems = []
            
            for line in lines:
                line = line.strip()
                if line.startswith("NAME:"):
                    name = line.replace("NAME:", "").strip()
                elif line.startswith("-"):
                    problems.append(line[1:].strip())
            
            critique = "\n".join([f"- {problem}" for problem in problems])
            
            results.append({
                "community_id": comm_id,
                "name": name,
                "critique": critique,
                "n_markets": len(market_ids),
                "sample_titles": market_titles[:5]  # First 5 for reference
            })
            
            print(f"  Community {comm_id}: '{name}' ({len(market_ids)} markets)")
            
        except Exception as e:
            print(f"  ERROR processing Community {comm_id}: {e}")
            results.append({
                "community_id": comm_id,
                "name": f"Community_{comm_id}",
                "critique": f"Error generating critique: {str(e)}",
                "n_markets": len(market_ids),
                "sample_titles": market_titles[:5] if market_titles else []
            })
    
    # Generate markdown report
    print("\nGenerating community names and critique report...")
    
    markdown_content = f"""# Community Names and Critique

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total communities: {len(results)}
Total markets clustered: {sum(r['n_markets'] for r in results)}

**Note**: These are FINANCIAL, POLITICAL, and ECONOMIC prediction markets only. Sports and entertainment markets have been filtered out.

"""
    
    # Sort by community size (descending)
    results_sorted = sorted(results, key=lambda x: x['n_markets'], reverse=True)
    
    for result in results_sorted:
        markdown_content += f"""
## Community {result['community_id']}: {result['name']}

**Size**: {result['n_markets']} markets

**Potential Problems**:
{result['critique']}

**Sample Markets**:
"""
        for title in result['sample_titles']:
            markdown_content += f"- {title}\n"
        
        markdown_content += "\n---\n"
    
    # Save the report
    output_path = "outputs/community_names_critique.md"
    with open(output_path, "w") as f:
        f.write(markdown_content)
    
    print(f"Saved community names and critique to {output_path}")
    
    # Also save as JSON for programmatic use
    import json
    json_path = "data/processed/community_labels.json"
    labels_dict = {str(r['community_id']): r['name'] for r in results}
    with open(json_path, "w") as f:
        json.dump(labels_dict, f, indent=2)
    
    print(f"Saved community labels to {json_path}")
    print("Done!")

if __name__ == "__main__":
    generate_community_names_and_critiques()
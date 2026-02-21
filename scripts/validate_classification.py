"""Validation checks for LLM classification results."""

import json
import os
import sys
import time
import random
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path.home() / ".openclaw" / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classification.taxonomy import load_taxonomy, format_taxonomy_for_prompt, list_themes, get_all_anti_examples

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def check_coverage(df):
    """Every market should be classified."""
    total = len(df)
    classified = df['primary_theme'].notna().sum()
    print(f"✓ Coverage: {classified}/{total} ({classified/total*100:.1f}%)")
    return classified == total


def check_confidence_distribution(df):
    """Most should be >0.7."""
    above_07 = (df['confidence'] >= 0.7).sum()
    pct = above_07 / len(df) * 100
    status = "✓" if pct > 70 else "✗"
    print(f"{status} Confidence ≥ 0.7: {above_07}/{len(df)} ({pct:.1f}%)")
    print(f"  Mean: {df['confidence'].mean():.3f}, Median: {df['confidence'].median():.3f}")
    return pct > 70


def check_theme_balance(df):
    """No theme should have 0 or >500 markets."""
    themes = load_taxonomy()
    counts = df[df['primary_theme'] != 'uncategorized']['primary_theme'].value_counts()
    
    all_ok = True
    for theme_id in themes:
        count = counts.get(theme_id, 0)
        if count == 0:
            print(f"  ✗ {theme_id}: 0 markets!")
            all_ok = False
        elif count > 500:
            print(f"  ✗ {theme_id}: {count} markets (>500)")
            all_ok = False
    
    print(f"{'✓' if all_ok else '✗'} Theme balance: min={counts.min()} max={counts.max()}")
    return all_ok


def check_anti_examples(df):
    """Anti-examples should NOT be classified into their forbidden theme."""
    anti_examples = get_all_anti_examples()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    themes = load_taxonomy()
    taxonomy_text = format_taxonomy_for_prompt(themes)
    theme_ids = list_themes(themes)
    
    system_prompt = f"""You are a prediction market classifier. Assign the market to a theme.

TAXONOMY:
{taxonomy_text}

VALID THEME IDS: {json.dumps(theme_ids)}

Return ONLY JSON: {{"primary_theme": "id", "confidence": 0.85, "reasoning": "one sentence"}}"""
    
    violations = 0
    total = len(anti_examples)
    
    for ae in anti_examples:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini", temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify: {ae['text']}\nReturn JSON only."}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(resp.choices[0].message.content)
            assigned = result.get("primary_theme", "")
            if assigned == ae["forbidden_theme"]:
                print(f"  ✗ VIOLATION: \"{ae['text']}\" classified as {assigned} (forbidden)")
                violations += 1
            else:
                print(f"  ✓ \"{ae['text'][:50]}\" → {assigned} (not {ae['forbidden_theme']})")
        except Exception as e:
            print(f"  ? Error testing \"{ae['text'][:40]}\": {e}")
        time.sleep(0.05)
    
    print(f"{'✓' if violations == 0 else '✗'} Anti-example test: {violations}/{total} violations")
    return violations == 0


def check_determinism(df, n_samples=20, n_runs=3):
    """Classify random markets multiple times, check agreement >90%."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    themes = load_taxonomy()
    taxonomy_text = format_taxonomy_for_prompt(themes)
    theme_ids = list_themes(themes)
    
    # Different prompt phrasings
    prompts = [
        f"You are classifying prediction markets into themes.\n\nTAXONOMY:\n{taxonomy_text}\n\nVALID IDS: {json.dumps(theme_ids)}\n\nReturn JSON: {{\"primary_theme\": \"id\", \"confidence\": 0.85, \"reasoning\": \"why\"}}",
        f"Classify this prediction market into one investment theme.\n\nAvailable themes:\n{taxonomy_text}\n\nValid theme IDs: {json.dumps(theme_ids)}\n\nRespond with JSON: {{\"primary_theme\": \"theme_id\", \"confidence\": 0.0-1.0, \"reasoning\": \"explanation\"}}",
        f"As an analyst, categorize this market.\n\nTheme taxonomy:\n{taxonomy_text}\n\nChoose from: {json.dumps(theme_ids)}\n\nOutput JSON: {{\"primary_theme\": \"chosen_id\", \"confidence\": 0.85, \"reasoning\": \"brief\"}}",
    ]
    
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    agreements = 0
    total_comparisons = 0
    
    for idx, row in sample.iterrows():
        title = str(row.get('event_title', ''))
        desc = str(row.get('description', ''))[:300]
        tags = str(row.get('tags', ''))
        user_msg = f"Title: {title}\nDescription: {desc}\nTags: {tags}"
        
        results = []
        for prompt in prompts[:n_runs]:
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0,
                    messages=[{"role": "system", "content": prompt},
                              {"role": "user", "content": user_msg}],
                    response_format={"type": "json_object"}
                )
                r = json.loads(resp.choices[0].message.content)
                results.append(r.get("primary_theme", "unknown"))
            except:
                results.append("error")
            time.sleep(0.05)
        
        # Check pairwise agreement
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                total_comparisons += 1
                if results[i] == results[j]:
                    agreements += 1
        
        all_same = len(set(results)) == 1
        status = "✓" if all_same else "~"
        print(f"  {status} \"{title[:50]}\" → {results}")
    
    agreement_rate = agreements / total_comparisons if total_comparisons > 0 else 0
    print(f"{'✓' if agreement_rate > 0.9 else '✗'} Determinism: {agreement_rate*100:.1f}% agreement ({agreements}/{total_comparisons})")
    return agreement_rate > 0.9


def main():
    df = pd.read_csv(DATA_DIR / "llm_classifications.csv")
    print(f"Loaded {len(df)} classified markets\n")
    
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    print("\n1. COVERAGE")
    check_coverage(df)
    
    print("\n2. CONFIDENCE DISTRIBUTION")
    check_confidence_distribution(df)
    
    print("\n3. THEME BALANCE")
    check_theme_balance(df)
    
    print("\n4. ANTI-EXAMPLE TEST")
    check_anti_examples(df)
    
    print("\n5. DETERMINISM TEST (20 random markets, 3 prompt variants)")
    check_determinism(df)
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")


if __name__ == "__main__":
    main()

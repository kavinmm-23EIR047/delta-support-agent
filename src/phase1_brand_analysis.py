"""
Phase 1: Candidate Brand Selection & Thread-Completeness Profiling
Analyzes the Kaggle Twitter Customer Support Dataset (twcs.csv) to evaluate
multi-turn thread density, conversational depth, template deflection rates,
and contextual resolution suitability across multiple candidate brands.
"""

import os
import re
import csv
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any
import json

DATA_PATH = r"d:\hiver_task\data\twcs\twcs.csv"

def inspect_dataset_overview():
    print("=" * 80)
    print("PHASE 1: DATASET OVERVIEW & CANDIDATE BRAND EXTRACTION")
    print("=" * 80)
    
    # Load dataset in chunks to get top author_ids where inbound is False (brands)
    brand_counts = Counter()
    total_rows = 0
    
    print("Scanning twcs.csv for brand volume...")
    for chunk in pd.read_csv(DATA_PATH, chunksize=100000, usecols=['author_id', 'inbound']):
        total_rows += len(chunk)
        brands = chunk[chunk['inbound'] == False]['author_id']
        brand_counts.update(brands)
        
    print(f"Total rows in dataset: {total_rows:,}")
    print("\nTop 15 Brand Accounts by Outbound Response Volume:")
    for rank, (brand, count) in enumerate(brand_counts.most_common(15), 1):
        print(f"  {rank:2d}. {brand:<20} : {count:,} replies")
        
    return [b for b, _ in brand_counts.most_common(15)]

def profile_candidate_brands(candidate_brands: List[str]):
    print("\n" + "=" * 80)
    print("PROFILING CANDIDATE BRANDS FOR MULTI-TURN DEPTH & GROUNDING FIDELITY")
    print("=" * 80)
    
    # We will load all tweets associated with these candidate brands
    # Both outbound (author_id == brand) and inbound mentioning or replying to/from them
    cand_set = set(candidate_brands)
    
    print(f"Loading conversations for {len(cand_set)} candidate brands: {sorted(cand_set)}")
    
    # Pass 1: Collect tweet metadata
    # We store: tweet_id -> {author_id, inbound, in_response_to_tweet_id, response_tweet_id, text, created_at}
    # To save memory, let's process brand by brand or gather for target candidates
    
    # Target candidates for in-depth comparative profiling:
    target_brands = [
        "AmazonHelp",     # Top 1 tutorial brand
        "AppleSupport",   # Top 2 tutorial brand
        "SpotifyCares",   # Top 3 tutorial brand
        "Delta",          # Airline: High-stakes operations, luggage, flight disruption
        "AmericanAir",    # Airline: High-volume multi-turn logistics
        "Uber_Support",   # Ride-hailing: Fare disputes, route issues, driver/rider safety
        "XboxSupport",    # Tech/Gaming: Complex multi-step diagnostics, error codes
        "Tesco",          # Grocery/Retail: Delivery, in-store, returns, refunds
        "SprintCare",     # Telecom: Account, billing, network connectivity
    ]
    
    target_set = set(target_brands)
    
    # Data containers per brand
    # tweets[brand][tweet_id] = (author_id, inbound, in_reply_to, text)
    brand_tweets = {b: {} for b in target_brands}
    
    # Pattern for agent signatures e.g. ^JD, ^AM, -Sam, _Alex, /Sarah, [CS]
    sig_pattern = re.compile(r'(\^[A-Za-z]{1,4}|-[A-Za-z]{2,4}|/[A-Za-z]{2,4}|\b\^[A-Z]{1,3}\b)')
    
    # Pattern for immediate boilerplate deflection spam e.g., "Please DM us", "send us a DM with your account details"
    dm_deflection_pattern = re.compile(
        r'(please\s+(send|dm|direct\s+message)|send\s+us\s+a\s+dm|dm\s+us\s+your|via\s+dm|inbox\s+us|reach\s+out\s+in\s+dm|follow\s+and\s+dm|dm\s+for\s+assistance)',
        re.IGNORECASE
    )
    
    print("Reading full dataset to extract brand conversation graphs...")
    chunk_size = 150000
    for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size, 
                             usecols=['tweet_id', 'author_id', 'inbound', 'in_response_to_tweet_id', 'response_tweet_id', 'text', 'created_at']):
        # Filter for brand outbound tweets
        brand_out = chunk[chunk['author_id'].isin(target_set)]
        for _, row in brand_out.iterrows():
            b = row['author_id']
            tid = int(row['tweet_id'])
            in_reply = int(row['in_response_to_tweet_id']) if pd.notna(row['in_response_to_tweet_id']) else None
            brand_tweets[b][tid] = {
                'tweet_id': tid,
                'author_id': b,
                'inbound': False,
                'in_reply_to': in_reply,
                'text': str(row['text']),
                'created_at': row['created_at']
            }
            
    print(f"Extracted outbound tweets per brand:")
    for b in target_brands:
        print(f"  - {b}: {len(brand_tweets[b]):,} outbound replies")
        
    # Pass 2: Extract inbound tweets that were replied to or initiated conversations
    print("\nExtracting corresponding inbound user tweets...")
    # Collect all needed in_reply_to IDs
    needed_inbound_ids = defaultdict(set)
    for b in target_brands:
        for tid, d in brand_tweets[b].items():
            if d['in_reply_to']:
                needed_inbound_ids[b].add(d['in_reply_to'])
                
    all_needed_inbounds = set().union(*needed_inbound_ids.values())
    print(f"Total unique inbound root/parent tweets needed: {len(all_needed_inbounds):,}")
    
    for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size,
                             usecols=['tweet_id', 'author_id', 'inbound', 'in_response_to_tweet_id', 'text', 'created_at']):
        inbound_chunk = chunk[chunk['tweet_id'].isin(all_needed_inbounds)]
        for _, row in inbound_chunk.iterrows():
            tid = int(row['tweet_id'])
            in_reply = int(row['in_response_to_tweet_id']) if pd.notna(row['in_response_to_tweet_id']) else None
            # Find which brand needed this
            for b in target_brands:
                if tid in needed_inbound_ids[b]:
                    brand_tweets[b][tid] = {
                        'tweet_id': tid,
                        'author_id': str(row['author_id']),
                        'inbound': bool(row['inbound']),
                        'in_reply_to': in_reply,
                        'text': str(row['text']),
                        'created_at': row['created_at']
                    }

    # Now compute thread metrics per brand
    results = []
    
    for b in target_brands:
        b_dict = brand_tweets[b]
        outbound_tweets = [t for t in b_dict.values() if not t['inbound']]
        
        # 1. Deflection / Canned template rate
        dm_deflections = 0
        agent_sigs = set()
        signature_count = 0
        all_words = []
        bigrams = set()
        total_bigrams = 0
        
        for t in outbound_tweets:
            txt = t['text']
            if dm_deflection_pattern.search(txt):
                dm_deflections += 1
            sigs = sig_pattern.findall(txt)
            if sigs:
                signature_count += 1
                for s in sigs:
                    agent_sigs.add(s.strip())
            words = re.findall(r'\b[a-z]{2,}\b', txt.lower())
            all_words.extend(words)
            for i in range(len(words)-1):
                bigrams.add((words[i], words[i+1]))
                total_bigrams += 1
                
        dm_deflection_rate = dm_deflections / max(1, len(outbound_tweets))
        distinct_1 = len(set(all_words)) / max(1, len(all_words)) if all_words else 0
        distinct_2 = len(bigrams) / max(1, total_bigrams) if total_bigrams else 0
        
        # 2. Reconstruct thread trees / paths
        # Children mapping
        children = defaultdict(list)
        for tid, t in b_dict.items():
            if t['in_reply_to'] and t['in_reply_to'] in b_dict:
                children[t['in_reply_to']].append(tid)
                
        # Find roots (tweets with no in_reply_to in b_dict or in_reply_to is None)
        roots = [tid for tid, t in b_dict.items() if (t['in_reply_to'] is None or t['in_reply_to'] not in b_dict)]
        
        # Calculate depth for all root-to-leaf paths
        thread_depths = []
        multi_turn_threads = 0 # depth >= 3 (e.g. User -> Brand -> User -> ...)
        deep_threads = 0 # depth >= 4
        handoff_threads = 0
        
        def traverse(curr_tid, path, distinct_agents_in_path):
            curr = b_dict[curr_tid]
            if not curr['inbound']:
                sigs = sig_pattern.findall(curr['text'])
                if sigs:
                    distinct_agents_in_path.add(sigs[0])
            
            curr_children = children.get(curr_tid, [])
            if not curr_children:
                # Leaf reached
                depth = len(path)
                thread_depths.append(depth)
                if len(distinct_agents_in_path) >= 2:
                    return 1 # had handoff
                return 0
            
            sub_handoffs = 0
            for child in curr_children:
                sub_handoffs += traverse(child, path + [child], set(distinct_agents_in_path))
            return 1 if sub_handoffs > 0 else 0

        for r in roots:
            curr = b_dict[r]
            init_agents = set()
            if not curr['inbound']:
                sigs = sig_pattern.findall(curr['text'])
                if sigs:
                    init_agents.add(sigs[0])
            has_handoff = traverse(r, [r], init_agents)
            if has_handoff:
                handoff_threads += 1

        total_threads = len(thread_depths)
        threads_ge_2 = sum(1 for d in thread_depths if d >= 2)
        threads_ge_3 = sum(1 for d in thread_depths if d >= 3)
        threads_ge_4 = sum(1 for d in thread_depths if d >= 4)
        avg_depth = np.mean(thread_depths) if thread_depths else 0
        median_depth = np.median(thread_depths) if thread_depths else 0
        max_depth = max(thread_depths) if thread_depths else 0
        
        results.append({
            'brand': b,
            'outbound_replies': len(outbound_tweets),
            'reconstructed_threads': total_threads,
            'avg_thread_depth': round(float(avg_depth), 2),
            'median_thread_depth': round(float(median_depth), 1),
            'max_thread_depth': int(max_depth),
            'pct_depth_ge_3 (multi-turn)': round(float(threads_ge_3 / max(1, total_threads) * 100), 2),
            'pct_depth_ge_4 (deep multi-turn)': round(float(threads_ge_4 / max(1, total_threads) * 100), 2),
            'dm_deflection_rate_pct': round(float(dm_deflection_rate * 100), 2),
            'distinct_2_bigram_ratio': round(float(distinct_2), 4),
            'unique_agent_sigs': len(agent_sigs),
            'handoff_threads_pct': round(float(handoff_threads / max(1, total_threads) * 100), 2)
        })
        
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("PHASE 1 COMPARATIVE ANALYSIS RESULTS TABLE")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    # Save results to json and csv
    os.makedirs(r"d:\hiver_task\artifacts", exist_ok=True)
    df_results.to_csv(r"d:\hiver_task\artifacts\phase1_brand_comparison.csv", index=False)
    with open(r"d:\hiver_task\artifacts\phase1_brand_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nArtifacts saved to d:\\hiver_task\\artifacts\\phase1_brand_comparison.csv")
    return df_results

if __name__ == "__main__":
    top_brands = inspect_dataset_overview()
    profile_candidate_brands(top_brands)

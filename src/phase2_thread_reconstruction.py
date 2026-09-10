"""
Phase 2 (Optimized JSONL Output): Thread Reconstruction with Adversarial Cleaning & Semantic PII Scrubbing
Reconstructs conversation trees for @Delta from twcs.csv and streams out as line-delimited JSON (JSONL)
for O(1) memory efficiency.
"""

import os
import re
import json
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

DATA_PATH = r"d:\hiver_task\data\twcs\twcs.csv"
ARTIFACTS_DIR = r"d:\hiver_task\artifacts"

class SemanticPIIScrubber:
    def __init__(self):
        self.email_re = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
        self.phone_re = re.compile(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)(\d{3}[-.\s]?\d{4})\b')
        self.card_re = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
        self.pnr_re = re.compile(r'(?:confirmation|record\s+locator|conf|booking\s+ref|pnr|reservation\s+#?|code)[:\s#]+([A-Z0-9]{6})\b', re.IGNORECASE)
        self.ticket_re = re.compile(r'\b(?:006|\d{3})[- ]?\d{10}\b')
        self.flight_re = re.compile(r'\b(?:DL|DAL|Flight|Flt|Delta flight|Flight#|FLT#)[:\s#]*(\d{1,4})\b', re.IGNORECASE)
        self.handle_re = re.compile(r'@([A-Za-z0-9_]+)')
        self.url_re = re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_+.~#?&/=]*)')
        self.agent_sig_re = re.compile(r'(\*[A-Za-z]{1,4}|\^[A-Za-z]{1,4}|-[A-Za-z]{2,4}|/[A-Za-z]{2,4})\s*$')

    def extract_agent_signature(self, text: str) -> Tuple[Optional[str], str]:
        match = self.agent_sig_re.search(text)
        if match:
            sig = match.group(1).strip()
            clean_text = text[:match.start()].strip()
            return sig, clean_text
        return None, text

    def scrub(self, text: str, is_brand: bool = False) -> Tuple[str, Dict[str, int]]:
        counts = defaultdict(int)
        
        def url_sub(m):
            counts['urls'] += 1
            u = m.group(0).lower()
            if 'delta.com' in u:
                return '<DELTA_OFFICIAL_URL>'
            elif 'bit.ly' in u or 't.co' in u:
                return '<SHORT_LINK>'
            return '<EXTERNAL_URL>'
        scrubbed = self.url_re.sub(url_sub, text)
        scrubbed = self.email_re.sub(lambda m: '<EMAIL_ADDRESS>', scrubbed)
        scrubbed = self.phone_re.sub(lambda m: '<PHONE_NUMBER>', scrubbed)
        scrubbed = self.card_re.sub(lambda m: '<CARD_NUMBER>', scrubbed)
        scrubbed = self.pnr_re.sub(lambda m: '<PNR_CONFIRMATION_CODE>', scrubbed)
        scrubbed = self.ticket_re.sub(lambda m: '<TICKET_NUMBER>', scrubbed)
        scrubbed = self.flight_re.sub(lambda m: f'<FLIGHT_NUM:DL{m.group(1)}>', scrubbed)
        
        def handle_sub(m):
            handle = m.group(1)
            if handle.lower() == 'delta':
                return '@Delta'
            counts['handles'] += 1
            return '<CUSTOMER_HANDLE>'
        scrubbed = self.handle_re.sub(handle_sub, scrubbed)
        scrubbed = re.sub(r'\s+', ' ', scrubbed).strip()
        return scrubbed, dict(counts)

class MacroDetector:
    def __init__(self):
        self.dm_deflection_patterns = [
            re.compile(r'please\s+(?:send|dm|direct\s+message)\s+us', re.IGNORECASE),
            re.compile(r'dm\s+us\s+your\s+(?:confirmation|name|ticket|details|pnr|info)', re.IGNORECASE),
            re.compile(r'reach\s+out\s+(?:to\s+us\s+)?via\s+dm', re.IGNORECASE),
            re.compile(r'follow\s+and\s+dm', re.IGNORECASE),
            re.compile(r'click\s+here\s+to\s+dm', re.IGNORECASE),
            re.compile(r'send\s+a\s+dm\s+with', re.IGNORECASE)
        ]
        self.canned_acknowledgments = [
            re.compile(r'thanks\s+for\s+reaching\s+out\s+to\s+delta', re.IGNORECASE),
            re.compile(r'we\s+appreciate\s+your\s+patience', re.IGNORECASE),
            re.compile(r'sorry\s+for\s+any\s+inconvenience\s+caused', re.IGNORECASE)
        ]

    def is_canned_reply(self, text: str) -> Tuple[bool, str]:
        for pat in self.dm_deflection_patterns:
            if pat.search(text):
                return True, "pure_dm_deflection" if len(text.split()) <= 25 else "partial_dm_deflection"
        for pat in self.canned_acknowledgments:
            if pat.search(text) and len(text.split()) <= 15:
                return True, "generic_acknowledgment"
        return False, "substantive_contextual"

def parse_twitter_date(date_str: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        return None

def reconstruct_and_stream_delta_threads(brand: str = "Delta"):
    print("=" * 80)
    print(f"RECONSTRUCTING & STREAMING CONVERSATION TREES FOR @{brand}")
    print("=" * 80)
    
    scrubber = SemanticPIIScrubber()
    macro_detector = MacroDetector()
    
    brand_tweets = {}
    needed_inbound_ids = set()
    chunk_size = 200000
    
    print("Scanning outbound brand replies...")
    for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size,
                             usecols=['tweet_id', 'author_id', 'inbound', 'in_response_to_tweet_id', 'text', 'created_at']):
        outbound = chunk[chunk['author_id'] == brand]
        for _, row in outbound.iterrows():
            tid = int(row['tweet_id'])
            in_reply = int(row['in_response_to_tweet_id']) if pd.notna(row['in_response_to_tweet_id']) else None
            brand_tweets[tid] = {
                'tweet_id': tid,
                'author_id': brand,
                'inbound': False,
                'in_reply_to': in_reply,
                'raw_text': str(row['text']),
                'created_at_str': str(row['created_at']),
                'dt': parse_twitter_date(str(row['created_at']))
            }
            if in_reply:
                needed_inbound_ids.add(in_reply)
                
    print(f"Found {len(brand_tweets):,} outbound tweets. Scanning ancestors...")
    all_tweets = dict(brand_tweets)
    remaining_needed = set(needed_inbound_ids)
    
    for iteration in range(1, 4):
        if not remaining_needed:
            break
        next_needed = set()
        for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size,
                                 usecols=['tweet_id', 'author_id', 'inbound', 'in_response_to_tweet_id', 'text', 'created_at']):
            matched = chunk[chunk['tweet_id'].isin(remaining_needed)]
            for _, row in matched.iterrows():
                tid = int(row['tweet_id'])
                in_reply = int(row['in_response_to_tweet_id']) if pd.notna(row['in_response_to_tweet_id']) else None
                all_tweets[tid] = {
                    'tweet_id': tid,
                    'author_id': str(row['author_id']),
                    'inbound': bool(row['inbound']),
                    'in_reply_to': in_reply,
                    'raw_text': str(row['text']),
                    'created_at_str': str(row['created_at']),
                    'dt': parse_twitter_date(str(row['created_at']))
                }
                if in_reply and in_reply not in all_tweets:
                    next_needed.add(in_reply)
        remaining_needed = next_needed

    children_map = defaultdict(list)
    for tid, t in all_tweets.items():
        p = t['in_reply_to']
        if p and p in all_tweets:
            children_map[p].append(tid)
            
    roots = [tid for tid, t in all_tweets.items() if (t['in_reply_to'] is None or t['in_reply_to'] not in all_tweets)]
    print(f"Reconstructing {len(roots):,} conversation trees to JSONL...")
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    jsonl_path = os.path.join(ARTIFACTS_DIR, "delta_reconstructed_threads.jsonl")
    sample_jsonl_path = os.path.join(ARTIFACTS_DIR, "delta_reconstructed_threads_sample.jsonl")
    
    count = 0
    with open(jsonl_path, "w", encoding="utf-8") as f_out, open(sample_jsonl_path, "w", encoding="utf-8") as f_sample:
        for r in roots:
            stack = [[r]]
            while stack:
                path = stack.pop()
                curr = path[-1]
                children = children_map.get(curr, [])
                if not children:
                    # Leaf reached -> linear thread
                    turns = []
                    agent_sigs = []
                    time_gaps = []
                    prev_dt = None
                    has_brand = False
                    
                    for idx, tid in enumerate(path):
                        t = all_tweets[tid]
                        is_inbound = t['inbound']
                        sig = None
                        if not is_inbound:
                            has_brand = True
                            sig, clean_raw = scrubber.extract_agent_signature(t['raw_text'])
                            if sig:
                                agent_sigs.append(sig)
                            is_canned, m_type = macro_detector.is_canned_reply(t['raw_text'])
                        else:
                            clean_raw = t['raw_text']
                            is_canned, m_type = False, None
                            
                        scrubbed_text, _ = scrubber.scrub(clean_raw, is_brand=not is_inbound)
                        gap = 0.0
                        if prev_dt and t['dt']:
                            gap = round(max(0.0, (t['dt'] - prev_dt).total_seconds() / 3600.0), 2)
                            time_gaps.append(gap)
                        prev_dt = t['dt']
                        
                        turns.append({
                            'turn_index': idx,
                            'tweet_id': tid,
                            'speaker': 'customer' if is_inbound else 'brand',
                            'scrubbed_text': scrubbed_text,
                            'is_canned': is_canned,
                            'gap_from_prev_hours': gap
                        })
                        
                    if has_brand and len(turns) >= 2:
                        distinct_agents = sorted(list(set(agent_sigs)))
                        canned_count = sum(1 for trn in turns if trn['is_canned'])
                        brand_turn_count = sum(1 for trn in turns if trn['speaker'] == 'brand')
                        is_pure_canned = (canned_count == brand_turn_count)
                        
                        thread_obj = {
                            'thread_id': f"delta_thread_{path[0]}",
                            'root_tweet_id': path[0],
                            'num_turns': len(turns),
                            'customer_inquiry_text': turns[0]['scrubbed_text'],
                            'initial_brand_reply': next((trn['scrubbed_text'] for trn in turns if trn['speaker'] == 'brand'), ""),
                            'turns': turns,
                            'metadata': {
                                'agent_signatures': distinct_agents,
                                'flags': {
                                    'multi_agent_handoff': len(distinct_agents) >= 2,
                                    'temporal_gap_gt_24h': (max(time_gaps) >= 24.0) if time_gaps else False,
                                    'is_pure_canned_deflection': is_pure_canned
                                }
                            }
                        }
                        line = json.dumps(thread_obj, ensure_ascii=False) + "\n"
                        f_out.write(line)
                        if count < 5000:
                            f_sample.write(line)
                        count += 1
                else:
                    for ch in children:
                        stack.append(path + [ch])
                        
    print(f"Successfully streamed {count:,} reconstructed threads to {jsonl_path} and sample {sample_jsonl_path}")

if __name__ == "__main__":
    reconstruct_and_stream_delta_threads()

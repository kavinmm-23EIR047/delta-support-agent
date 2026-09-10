"""
Interactive Human Gold Labeling CLI for @Delta Customer Support
Streams real customer inquiries from reconstructed threads, provides an interactive
terminal interface for human annotation, and saves genuine human gold evaluation sets.

Usage:
  python src/labeling_interface.py --sample-size 60 --mode label
  python src/labeling_interface.py --resume
  python src/labeling_interface.py --mode relabel-blind
"""

import os
import sys
import json
import argparse
import random
import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Ensure UTF-8 stdout encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
JSONL_SAMPLE = os.path.join(ARTIFACTS_DIR, "delta_reconstructed_threads_sample.jsonl")
HUMAN_GOLD_PATH = os.path.join(ARTIFACTS_DIR, "human_gold_dataset.json")
BLIND_AUDIT_PATH = os.path.join(ARTIFACTS_DIR, "human_blind_relabel_audit.json")

INTENT_CHOICES = {
    "1": "FLIGHT_DISRUPTIONS",
    "2": "BAGGAGE_AND_CARGO",
    "3": "RESERVATIONS_AND_TICKETING",
    "4": "REFUNDS_CREDITS_COMPENSATION",
    "5": "SKYMILES_AND_LOYALTY",
    "6": "AIRPORT_AND_ONBOARD_EXPERIENCE",
    "7": "TIER0_CRITICAL_EMERGENCY",
    "8": "GENERAL_INQUIRY"
}

def is_bare_numeric(text: str) -> bool:
    """Checks if a string is purely numeric or bare punctuation/digits (e.g. '2', '4.5')."""
    s = text.strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        pass
    if len(s) <= 3 and all(c.isdigit() or c in '.,+- ' for c in s):
        return True
    return False

def print_escalation_rubric_reminder():
    """Prints a clear rubric checklist right above the escalation decision prompt."""
    print("-" * 60)
    print("Quick check before you answer:")
    print(" - Is this a real customer issue, or a compliment / marketing post /")
    print("   no actual ask? -> likely 0")
    print(" - Could a grounded bot reply safely resolve this alone? -> 0")
    print(" - Is there anger + unresolved issue, safety/legal/ADA concern,")
    print("   ambiguous multi-part request, or need for a policy exception? -> 1")
    print("-" * 60)

def load_real_candidate_inquiries(n: int = 200) -> List[Dict[str, Any]]:
    """Loads substantive real customer inquiries from reconstructed threads deterministically."""
    if not os.path.exists(JSONL_SAMPLE):
        print(f"Error: {JSONL_SAMPLE} not found. Please run run_pipeline.py first.")
        sys.exit(1)
        
    candidates = []
    with open(JSONL_SAMPLE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                inq = t.get('customer_inquiry_text', '')
                reply = t.get('initial_brand_reply', '')
                if len(inq.split()) >= 6 and len(reply.split()) >= 6:
                    candidates.append({
                        'thread_id': t['thread_id'],
                        'customer_inquiry': inq,
                        'actual_brand_reply': reply,
                        'num_turns': t.get('num_turns', 2)
                    })
                    
    # Deterministic shuffle so sample order is stable across runs
    rng = random.Random(42)
    rng.shuffle(candidates)
    return candidates[:max(n, 500)]

def run_labeling_session(sample_size: int = 50, force_resume: bool = True):
    print("=" * 80)
    print("INTERACTIVE HUMAN GOLD DATASET LABELING SESSION")
    print("=" * 80)
    print(f"Goal: Annotate {sample_size} real @Delta customer tweets for Ground Truth evaluation.")
    print("For each tweet, decide:")
    print("  [1] Escalation: Should an automated bot reply (0) or MUST this go to a human (1)?")
    print("  [2] Primary Intent: Pick from 1-8")
    print("  [3] Optional Notes: Why you made this decision (e.g. 'Safety hazard', 'Vague complaint')")
    print("-" * 80)
    
    # Load existing labels if resuming
    labeled_data = []
    if os.path.exists(HUMAN_GOLD_PATH):
        try:
            with open(HUMAN_GOLD_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    loaded = json.loads(content)
                    if isinstance(loaded, list):
                        labeled_data = loaded
        except Exception as e:
            print(f"[Warning] Could not parse existing {HUMAN_GOLD_PATH}: {e}")
            labeled_data = []
            
    print(f"Session Status: Found {len(labeled_data)} already labeled samples in {HUMAN_GOLD_PATH}.")
            
    labeled_ids = set(d.get('thread_id') for d in labeled_data if 'thread_id' in d)
    candidates = load_real_candidate_inquiries(n=sample_size + len(labeled_ids) + 50)
    remaining = [c for c in candidates if c['thread_id'] not in labeled_ids]
    
    if len(labeled_data) >= sample_size:
        print(f"\nAll requested {sample_size} samples (or more) are already annotated ({len(labeled_data)} total) in {HUMAN_GOLD_PATH}!")
        print("To annotate more, pass a larger --sample-size (e.g. --sample-size 100).")
        return
        
    needed = sample_size - len(labeled_data)
    remaining = remaining[:needed]
    session_labeled_count = 0
    
    for idx, c in enumerate(remaining, start=len(labeled_data) + 1):
        print("\n" + "=" * 80)
        print(f"SAMPLE {idx} of {sample_size} | Thread: {c['thread_id']}")
        print("=" * 80)
        print(f"\n[CUSTOMER TWEET]:\n  \"{c['customer_inquiry']}\"\n")
        print(f"[ACTUAL DELTA REPLY IN DATASET]:\n  \"{c['actual_brand_reply']}\"\n")
        
        # 1. Escalation Decision (with continuous rubric reminder)
        print_escalation_rubric_reminder()
        while True:
            esc_input = input(">> SHOULD THIS ESCALATE TO HUMAN? (1 = Yes/Human, 0 = No/Bot, s = Skip, q = Save & Quit): ").strip().lower()
            if esc_input in ['0', '1']:
                should_escalate = int(esc_input)
                break
            elif esc_input == 's':
                should_escalate = None
                break
            elif esc_input == 'q':
                print(f"\nSaving progress ({len(labeled_data)} samples) to {HUMAN_GOLD_PATH}...")
                with open(HUMAN_GOLD_PATH, "w", encoding="utf-8") as f:
                    json.dump(labeled_data, f, indent=2)
                print("Session saved. You can resume anytime!")
                return
            else:
                print("Invalid input. Please type 1 (Escalate), 0 (Bot), s (Skip), or q (Quit).")
                
        if should_escalate is None:
            continue
            
        # 2. Intent Classification
        print("\nPick Intent Category:")
        for k, v in INTENT_CHOICES.items():
            print(f"  [{k}] {v}")
            
        while True:
            intent_input = input(">> INTENT (1-8): ").strip()
            if intent_input in INTENT_CHOICES:
                chosen_intent = INTENT_CHOICES[intent_input]
                break
            else:
                print("Invalid choice. Please select a number from 1 to 8.")
                
        # 3. Notes with validation against numeric pollution
        while True:
            notes = input(">> OPTIONAL NOTE / RATIONALE (text only, or press Enter to skip): ").strip()
            if not notes:
                break
            if is_bare_numeric(notes):
                print("  [Notice] That looks like a rating, not a note — please enter a short reason, or press Enter to skip.")
                continue
            break
        
        record = {
            'id': f"GOLD_{idx:03d}",
            'thread_id': c['thread_id'],
            'text': c['customer_inquiry'],
            'actual_historical_reply': c['actual_brand_reply'],
            'ground_truth_escalate': should_escalate,
            'ground_truth_intent': chosen_intent,
            'human_rationale': notes,
            'labeled_at': datetime.utcnow().isoformat()
        }
        labeled_data.append(record)
        
        # Save after every entry
        with open(HUMAN_GOLD_PATH, "w", encoding="utf-8") as f:
            json.dump(labeled_data, f, indent=2)
            
        session_labeled_count += 1
        if session_labeled_count % 10 == 0:
            print(f"\n[SESSION PROGRESS]: {session_labeled_count}/{needed} labeled this session ({len(labeled_data)}/{sample_size} total). Keep going or press 'q' to save and quit anytime.\n")
            
    print("\n" + "=" * 80)
    print(f"CONGRATULATIONS! Successfully labeled {len(labeled_data)} real customer cases.")
    print(f"Saved to {HUMAN_GOLD_PATH}")
    print("=" * 80)

def run_review_session():
    """Interactive review pass allowing the annotator to inspect and correct existing annotations."""
    print("=" * 80)
    print("HUMAN GOLD DATASET — ANNOTATION REVIEW & CORRECTION SESSION")
    print("=" * 80)
    print("Review each labeled sample. Press [Enter] or 'k' to keep as-is, 'e' to edit, or 'q' to save & quit.")
    print("-" * 80)
    
    if not os.path.exists(HUMAN_GOLD_PATH):
        print(f"Error: {HUMAN_GOLD_PATH} does not exist.")
        return
        
    with open(HUMAN_GOLD_PATH, "r", encoding="utf-8") as f:
        try:
            labeled_data = json.load(f)
        except Exception as e:
            print(f"Error reading {HUMAN_GOLD_PATH}: {e}")
            return
            
    if not labeled_data:
        print(f"No samples found in {HUMAN_GOLD_PATH}.")
        return
        
    total_len = len(labeled_data)
    corrected_count = 0
    
    for idx, item in enumerate(labeled_data, start=1):
        print("\n" + "=" * 80)
        print(f"REVIEW ITEM {idx} of {total_len} | ID: {item.get('id', 'N/A')} | Thread: {item.get('thread_id', 'N/A')}")
        print("=" * 80)
        print(f"\n[CUSTOMER TWEET]:\n  \"{item.get('text', '')}\"\n")
        print(f"[ACTUAL DELTA REPLY]:\n  \"{item.get('actual_historical_reply', '')}\"\n")
        
        esc_val = item.get('ground_truth_escalate')
        esc_desc = "1 (Human/Escalate)" if esc_val == 1 else ("0 (Bot/Self-Service)" if esc_val == 0 else str(esc_val))
        print("[CURRENT ANNOTATION]:")
        print(f"  - Escalation: {esc_desc}")
        print(f"  - Intent:     {item.get('ground_truth_intent', 'N/A')}")
        print(f"  - Note:       \"{item.get('human_rationale', '')}\"")
        if 'labeled_at' in item:
            print(f"  - Labeled At: {item['labeled_at']}")
        if 'corrected_at' in item:
            print(f"  - Corrected:  {item['corrected_at']}")
        print("-" * 80)
        
        while True:
            action = input(">> Action: [Enter/k] Keep as-is | [e] Edit / Re-label | [q] Save & Quit: ").strip().lower()
            if action in ['', 'k', 'e', 'q']:
                break
            print("Invalid choice. Press Enter or 'k' to keep, 'e' to edit, or 'q' to quit.")
            
        if action == 'q':
            print(f"\nSaving progress ({corrected_count} corrections made) to {HUMAN_GOLD_PATH}...")
            with open(HUMAN_GOLD_PATH, "w", encoding="utf-8") as f:
                json.dump(labeled_data, f, indent=2)
            print("Review session saved.")
            return
            
        if action in ['', 'k']:
            continue
            
        # Re-labeling item
        print("\n--- EDITING ANNOTATION ---")
        print_escalation_rubric_reminder()
        
        # 1. Escalation Decision
        while True:
            esc_input = input(f">> NEW ESCALATE? (1 = Yes/Human, 0 = Bot, default={esc_val}): ").strip().lower()
            if not esc_input and esc_val in [0, 1]:
                new_escalate = esc_val
                break
            elif esc_input in ['0', '1']:
                new_escalate = int(esc_input)
                break
            else:
                print("Invalid input. Enter 1 (Escalate), 0 (Bot), or press Enter to keep current.")
                
        # 2. Intent Classification
        print("\nPick Intent Category:")
        for k, v in INTENT_CHOICES.items():
            print(f"  [{k}] {v}")
        curr_intent = item.get('ground_truth_intent', '')
        while True:
            intent_input = input(f">> NEW INTENT (1-8, press Enter to keep '{curr_intent}'): ").strip()
            if not intent_input and curr_intent:
                new_intent = curr_intent
                break
            elif intent_input in INTENT_CHOICES:
                new_intent = INTENT_CHOICES[intent_input]
                break
            else:
                print("Invalid choice. Select a number from 1 to 8 or press Enter to keep current.")
                
        # 3. Notes with validation
        curr_notes = item.get('human_rationale', '')
        while True:
            note_prompt = ">> NEW NOTE / RATIONALE (text reason, Enter to keep current, '-' to clear): "
            notes_input = input(note_prompt).strip()
            if not notes_input:
                new_notes = curr_notes
                break
            elif notes_input == '-':
                new_notes = ""
                break
            elif is_bare_numeric(notes_input):
                print("  [Notice] That looks like a rating, not a note — please enter a short text reason, Enter to keep, or '-' to clear.")
                continue
            else:
                new_notes = notes_input
                break
                
        # Update record
        item['ground_truth_escalate'] = new_escalate
        item['ground_truth_intent'] = new_intent
        item['human_rationale'] = new_notes
        item['corrected_at'] = datetime.utcnow().isoformat()
        corrected_count += 1
        
        # Save after every correction
        with open(HUMAN_GOLD_PATH, "w", encoding="utf-8") as f:
            json.dump(labeled_data, f, indent=2)
        print(f"-> Updated item {item.get('id', 'N/A')} and saved to {HUMAN_GOLD_PATH}.")
        
    print("\n" + "=" * 80)
    print(f"REVIEW SESSION COMPLETE: Checked {total_len} samples ({corrected_count} updated).")
    print(f"Saved to {HUMAN_GOLD_PATH}")
    print("=" * 80)

def run_blind_relabel_audit(sample_pct: float = 0.20):
    """Runs a blind re-annotation audit on a 20% random slice to measure human consistency."""
    if not os.path.exists(HUMAN_GOLD_PATH):
        print(f"\n[FAIL-LOUD ERROR: Missing Gold Dataset] File '{HUMAN_GOLD_PATH}' does not exist.")
        print("Please annotate samples first:\n  python src/labeling_interface.py --sample-size 40 --mode label\n")
        return
        
    with open(HUMAN_GOLD_PATH, "r", encoding="utf-8") as f:
        try:
            gold_data = json.load(f)
        except Exception:
            gold_data = []
        
    if not gold_data or len(gold_data) < 15:
        total_found = len(gold_data) if gold_data else 0
        needed = 15 - total_found
        print(f"\n[FAIL-LOUD ERROR: Insufficient Gold Data for Blind Audit]")
        print(f"Found {total_found} samples, need at least 15 before a 20% blind relabel is statistically meaningful.")
        print(f"You need {needed} more labeled samples. Please run:")
        print(f"  python src/labeling_interface.py --sample-size 40 --mode label\n")
        return
        
    total_len = len(gold_data)
    # Sane sample size: at least 3, at most total_len, default 20%
    audit_n = max(3, min(total_len, int(math.ceil(total_len * sample_pct))))
    
    rng = random.Random(99)
    audit_slice = rng.sample(gold_data, audit_n)
    
    print("=" * 80)
    print("BLIND INTRA-ANNOTATOR CONSISTENCY AUDIT")
    print("=" * 80)
    print(f"You will re-label {audit_n} samples (out of {total_len} total) BLIND to your original choices.")
    print("-" * 80)
    
    reannotated = []
    for idx, item in enumerate(audit_slice, 1):
        print(f"\n[BLIND AUDIT {idx}/{audit_n}]:")
        print(f"  \"{item['text']}\"\n")
        
        while True:
            esc = input(">> SHOULD THIS ESCALATE? (1 = Yes/Human, 0 = No/Bot): ").strip()
            if esc in ['0', '1']:
                re_esc = int(esc)
                break
                
        reannotated.append({
            'id': item['id'],
            'text': item['text'],
            'original_escalate': item['ground_truth_escalate'],
            'blind_relabel_escalate': re_esc,
            'is_consistent': (item['ground_truth_escalate'] == re_esc)
        })
        
    agreements = sum(1 for r in reannotated if r['is_consistent'])
    rate = agreements / audit_n
    print("\n" + "=" * 80)
    print(f"BLIND AUDIT COMPLETE: Human Self-Consistency = {rate:.1%} ({agreements}/{audit_n})")
    print("=" * 80)
    
    with open(BLIND_AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            'audit_size': audit_n,
            'self_consistency_rate': rate,
            'details': reannotated
        }, f, indent=2)
    print(f"Saved audit log to {BLIND_AUDIT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Gold Labeling Interface")
    parser.add_argument("--mode", choices=["label", "relabel-blind", "review"], default="label")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of samples to annotate")
    parser.add_argument("--resume", action="store_true", help="Explicit flag to resume labeling session (default behavior)")
    args = parser.parse_args()
    
    if args.mode == "review":
        run_review_session()
    elif args.mode == "label" or args.resume:
        run_labeling_session(sample_size=args.sample_size, force_resume=True)
    else:
        run_blind_relabel_audit()

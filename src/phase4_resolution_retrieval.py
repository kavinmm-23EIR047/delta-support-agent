"""
Phase 4: Retrieval-Grounded Generation with Resolution Success Weighting
Constructs a Resolution-Success Labeler that evaluates post-reply thread trajectories,
weights retrieval candidates toward proven successful resolutions, and compares
naive top-k semantic search against resolution-weighted retrieval.
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ARTIFACTS_DIR = r"d:\hiver_task\artifacts"
THREADS_JSON = os.path.join(ARTIFACTS_DIR, "delta_reconstructed_threads.json")

# ---------------------------------------------------------------------------
# 1. Resolution-Success Labeler
# ---------------------------------------------------------------------------

class ResolutionSuccessLabeler:
    """
    Evaluates whether a historical brand reply successfully resolved the customer's issue.
    Analyzes post-reply conversational signals, terminal gratitude, customer rebuttal sentiment,
    and agent handoff corrections.
    """
    def __init__(self):
        # Explicit gratitude and closure patterns
        self.gratitude_re = re.compile(
            r'\b(thank\s+you|thanks|thx|appreciate|helpful|sorted|resolved|perfect|awesome|great\s+service|got\s+it|all\s+set|kudos)\b',
            re.IGNORECASE
        )
        
        # Frustration, persistent complaint, and escalation patterns
        self.frustration_re = re.compile(
            r'\b(ridiculous|unacceptable|useless|horrible|worst|still\s+(not|waiting|broken|delayed)|never\s+again|lawyer|sue|complaint|terrible|incompetent|disgusted|joke|unhelpful)\b',
            re.IGNORECASE
        )
        
        # Follow-up question patterns (subsequent turn needed clarification)
        self.clarification_re = re.compile(r'\?|how|why|when|what|where|who', re.IGNORECASE)

    def evaluate_thread_resolution(self, thread: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates resolution success score for brand replies in a thread.
        Returns detailed scoring and behavioral rationale.
        """
        turns = thread['turns']
        num_turns = len(turns)
        flags = thread['metadata']['flags']
        
        # If thread is pure canned deflection
        if flags.get('is_pure_canned_deflection', False):
            return {
                'resolution_score': 0.20,
                'resolution_label': 'DEFLECTED_UNVERIFIED',
                'rationale': 'Pure canned DM redirect; resolution occurred off-channel without public verification.',
                'signals': {'gratitude': False, 'frustration': False, 'terminal_brand': False}
            }
            
        # Find the primary brand reply and subsequent customer turns
        brand_turn_idx = None
        for i, trn in enumerate(turns):
            if trn['speaker'] == 'brand':
                brand_turn_idx = i
                break
                
        if brand_turn_idx is None:
            return {
                'resolution_score': 0.0,
                'resolution_label': 'NO_BRAND_REPLY',
                'rationale': 'No brand response found in thread.',
                'signals': {}
            }
            
        # Analyze customer reaction after brand reply
        subsequent_cust_turns = [trn for trn in turns[brand_turn_idx+1:] if trn['speaker'] == 'customer']
        
        # Case A: Customer replied with gratitude / satisfaction
        has_gratitude = any(self.gratitude_re.search(trn['scrubbed_text']) for trn in subsequent_cust_turns)
        has_frustration = any(self.frustration_re.search(trn['scrubbed_text']) for trn in subsequent_cust_turns)
        
        if has_gratitude and not has_frustration:
            score = 0.95
            label = "EXPLICIT_SUCCESS"
            rationale = "Customer explicitly expressed gratitude/satisfaction after brand response."
        elif has_frustration:
            score = 0.10
            label = "FAILED_ESCALATION"
            rationale = "Customer expressed continued frustration, anger, or escalated complaint post-reply."
        elif len(subsequent_cust_turns) == 0:
            # Case B: Terminal brand reply without customer follow-up complaint (Silent resolution)
            brand_text = turns[brand_turn_idx]['scrubbed_text']
            word_count = len(brand_text.split())
            
            # If substantive detailed answer (> 15 words and not canned)
            if word_count >= 15 and not turns[brand_turn_idx]['is_canned']:
                score = 0.75
                label = "IMPLICIT_SUCCESSFUL_CLOSURE"
                rationale = "Substantive brand response with no further customer rebuttal (issue closed)."
            else:
                score = 0.40
                label = "AMBIGUOUS_SILENT_CLOSURE"
                rationale = "Short or template reply with no follow-up (risk of customer abandonment)."
        else:
            # Case C: Customer had multiple non-angry follow-up turns
            score = 0.60
            label = "MULTI_TURN_CONVERSATIONAL"
            rationale = "Active conversational resolution across multiple turns."
            
        return {
            'resolution_score': score,
            'resolution_label': label,
            'rationale': rationale,
            'signals': {
                'has_gratitude': has_gratitude,
                'has_frustration': has_frustration,
                'subsequent_turns_count': len(subsequent_cust_turns),
                'multi_agent_handoff': flags.get('multi_agent_handoff', False)
            }
        }

# ---------------------------------------------------------------------------
# 2. Resolution-Weighted Knowledge Base & Retriever
# ---------------------------------------------------------------------------

class ResolutionGroundedRetriever:
    """
    Retrieves grounded historical customer support Q&A pairs.
    Supports both Naive Top-K Semantic Retrieval and Resolution-Weighted Retrieval.
    """
    def __init__(self, threads: List[Dict[str, Any]], alpha_weight: float = 0.65):
        self.labeler = ResolutionSuccessLabeler()
        self.alpha_weight = alpha_weight # Weight on semantic similarity vs resolution score
        
        # Build Knowledge Base of (Customer Query, Brand Resolution) pairs with Resolution Scores
        self.kb = []
        for t in threads:
            cust_text = t.get('customer_inquiry_text', '')
            brand_reply = t.get('initial_brand_reply', '')
            
            # Need substantive text on both sides
            if len(cust_text.split()) >= 4 and len(brand_reply.split()) >= 5:
                res_eval = self.labeler.evaluate_thread_resolution(t)
                self.kb.append({
                    'thread_id': t['thread_id'],
                    'customer_inquiry': cust_text,
                    'brand_reply': brand_reply,
                    'resolution_score': res_eval['resolution_score'],
                    'resolution_label': res_eval['resolution_label'],
                    'rationale': res_eval['rationale'],
                    'metadata': t['metadata']
                })
                
        print(f"Knowledge Base Index built with {len(self.kb):,} grounded resolution pairs.")
        
        # Fit vectorizer over historical customer inquiries with float32 lightweight memory
        self.vectorizer = TfidfVectorizer(
            max_features=2500,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=3,
            dtype=np.float32
        )
        all_inquiries = [k['customer_inquiry'] for k in self.kb]
        self.tfidf_matrix = self.vectorizer.fit_transform(all_inquiries)

    def retrieve(self, query: str, top_k: int = 3, mode: str = "resolution_weighted") -> List[Dict[str, Any]]:
        """
        Retrieves top_k historical examples.
        mode='naive': Pure cosine similarity.
        mode='resolution_weighted': Composite score = alpha * CosineSim + (1-alpha) * ResolutionScore.
        """
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        
        results = []
        for idx, sim in enumerate(sims):
            item = self.kb[idx]
            res_score = item['resolution_score']
            
            if mode == "naive":
                final_score = float(sim)
            else: # resolution_weighted
                # Heavily penalize failed escalations or unverified deflections
                if res_score <= 0.20:
                    final_score = float(sim) * 0.2
                else:
                    final_score = float(self.alpha_weight * sim + (1 - self.alpha_weight) * res_score)
                    
            results.append({
                'thread_id': item['thread_id'],
                'customer_inquiry': item['customer_inquiry'],
                'brand_reply': item['brand_reply'],
                'cosine_similarity': round(float(sim), 4),
                'resolution_score': round(float(res_score), 4),
                'resolution_label': item['resolution_label'],
                'composite_score': round(float(final_score), 4),
                'rationale': item['rationale']
            })
            
        # Sort by final score
        results = sorted(results, key=lambda x: x['composite_score'], reverse=True)
        return results[:top_k]

# ---------------------------------------------------------------------------
# 3. Comparative Evaluation of Retrieval Quality
# ---------------------------------------------------------------------------

def load_threads_from_jsonl(max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads reconstructed threads from JSONL with streaming memory efficiency."""
    jsonl_path = os.path.join(ARTIFACTS_DIR, "delta_reconstructed_threads_sample.jsonl")
    if not os.path.exists(jsonl_path):
        jsonl_path = os.path.join(ARTIFACTS_DIR, "delta_reconstructed_threads.jsonl")
        
    threads = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                threads.append(json.loads(line))
                if max_samples and len(threads) >= max_samples:
                    break
    return threads

def run_phase4_grounding_evaluation():
    print("=" * 80)
    print("PHASE 4: RESOLUTION-SUCCESS WEIGHTED RETRIEVAL EVALUATION")
    print("=" * 80)
    
    threads = load_threads_from_jsonl(max_samples=5000)
    retriever = ResolutionGroundedRetriever(threads)
    
    # Analyze Knowledge Base Resolution Distribution
    res_labels = Counter(item['resolution_label'] for item in retriever.kb)
    print("\nKnowledge Base Resolution State Distribution:")
    for lbl, cnt in res_labels.most_common():
        print(f"  - {lbl:<30} : {cnt:6,d} ({cnt/len(retriever.kb)*100:5.2f}%)")
        
    # Test Queries comparing Naive Top-K vs Resolution-Weighted Top-K
    test_queries = [
        "Flight DL194 delayed 3 hours in Atlanta, missed connection to Boston. Can I get meal voucher or change to morning flight?",
        "Checked bag DL4829 didn't show up on carousel at LGA. Where is the baggage service desk located?",
        "Can I bring my cat in the cabin on a flight from JFK to LAX? What is the pet fee?",
        "How do I apply my SkyMiles eCredit voucher to book a new flight online?"
    ]
    
    comparison_log = []
    
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE RETRIEVAL COMPARISON (NAIVE VS RESOLUTION-WEIGHTED)")
    print("=" * 80)
    
    for q in test_queries:
        print(f"\n[QUERY]: \"{q}\"")
        naive_hits = retriever.retrieve(q, top_k=2, mode="naive")
        weighted_hits = retriever.retrieve(q, top_k=2, mode="resolution_weighted")
        
        print("\n  --- NAIVE TOP-K RETRIEVAL (Cosine Sim Only) ---")
        for i, hit in enumerate(naive_hits, 1):
            print(f"    Hit {i} [Sim={hit['cosine_similarity']:.3f} | ResScore={hit['resolution_score']:.2f} ({hit['resolution_label']})]:")
            print(f"      Q: {hit['customer_inquiry'][:80]}...")
            print(f"      A: {hit['brand_reply'][:90]}...")
            
        print("\n  --- RESOLUTION-WEIGHTED RETRIEVAL (Composite Scoring) ---")
        for i, hit in enumerate(weighted_hits, 1):
            print(f"    Hit {i} [CompScore={hit['composite_score']:.3f} | Sim={hit['cosine_similarity']:.3f} | ResScore={hit['resolution_score']:.2f} ({hit['resolution_label']})]:")
            print(f"      Q: {hit['customer_inquiry'][:80]}...")
            print(f"      A: {hit['brand_reply'][:90]}...")
            
        comparison_log.append({
            'query': q,
            'naive_hits': naive_hits,
            'weighted_hits': weighted_hits
        })
        
    # Save artifacts
    out_eval_path = os.path.join(ARTIFACTS_DIR, "phase4_retrieval_comparison.json")
    with open(out_eval_path, "w", encoding="utf-8") as f:
        json.dump(comparison_log, f, indent=2)
        
    print(f"\nArtifacts saved to {out_eval_path}")
    return comparison_log

if __name__ == "__main__":
    run_phase4_grounding_evaluation()

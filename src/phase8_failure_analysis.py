"""
Phase 8: Deep Failure Mode Analysis (Retrieval, Taxonomy, and Escalation Vulnerabilities)
Analyzes 3 subtle production failure modes that standard evaluations hide:
  1. Retrieval Failure: Temporal Policy Obsolescence (Waiver Policy Drift)
  2. Taxonomy Failure: Chimeric Cross-Domain Multi-Intent Boundary Collapse
  3. Escalation Miss: Sarcastic Vulnerable Passenger & Regulatory Liability
"""

import os
import json
from typing import Dict, List, Any

ARTIFACTS_DIR = r"d:\hiver_task\artifacts"

def run_failure_mode_investigation():
    print("=" * 80)
    print("PHASE 8: DEEP FAILURE ANALYSIS (HIDDEN PRODUCTION VULNERABILITIES)")
    print("=" * 80)
    
    failures = [
        {
            "failure_id": "FAIL_01_RETRIEVAL_OBSOLESCENCE",
            "component": "Retrieval Grounding Layer",
            "failure_title": "Temporal Policy Obsolescence (Transient Weather Waiver Applied to Standard Policy)",
            "customer_inquiry": "@Delta Can I cancel my Basic Economy ticket on flight DL912 for free and get a full travel credit?",
            "retrieved_historical_match": {
                "source_tweet_id": "delta_thread_41902",
                "retrieved_q": "@Delta Can I change or cancel my Basic Economy ticket tomorrow without a fee?",
                "retrieved_a": "<CUSTOMER_HANDLE> Yes! All change and cancellation fees are waived for your travel tomorrow. You will receive a full eCredit. *TJF",
                "historical_resolution_score": 0.95,
                "historical_label": "EXPLICIT_SUCCESS",
                "historical_context": "Hurricane Irma Emergency Travel Waiver (Sept 2017)"
            },
            "what_the_naive_system_does": "The naive RAG pipeline sees high semantic similarity (0.89) and high resolution score (0.95). It generates: 'Yes, all change fees are waived on Basic Economy! You can cancel for a full credit at delta.com.'",
            "the_real_world_production_failure": "Basic Economy tickets are strictly non-refundable and non-changeable outside declared weather emergency zones. The bot quoted a transient 2017 hurricane emergency policy as standard operating procedure.",
            "why_standard_evals_hide_it": "The judge rates the response 5/5 for high confidence, politeness, and perfect syntactic match with retrieved ground-truth text.",
            "senior_engineering_fix": "Policy Temporal Tagging: Embed an expiration TTL and 'is_transient_waiver' boolean in knowledge base index. Reject retrieval hits with expired operational scopes."
        },
        {
            "failure_id": "FAIL_02_TAXONOMY_CHIMERIC_COLLAPSE",
            "component": "Intent Taxonomy & Classification",
            "failure_title": "Chimeric Multi-Domain Intent Boundary Collapse",
            "customer_inquiry": "@Delta Flight DL412 cancelled. Gate agent said to Uber to Newark for United flight and Delta reimburses, but baggage desk kept my suitcase for tomorrow. Where do I send Uber receipt and how do I get my bag off the plane?",
            "taxonomy_classification_behavior": {
                "predicted_intent": "TIER_1_FLIGHT_DISRUPTIONS",
                "confidence": 0.84,
                "dropped_intents": ["TIER_1_BAGGAGE_AND_CARGO", "TIER_1_REFUNDS_CREDITS_COMPENSATION", "TIER_1_AIRPORT_EXPERIENCE"]
            },
            "what_the_naive_system_does": "Generates standard flight rebooking instructions ('We apologize for the cancellation on DL412. Please visit the app to select a new flight.').",
            "the_real_world_production_failure": "Completely ignores the customer's two urgent physical crises: (1) their luggage is locked in an airside baggage facility while they leave the airport, and (2) how to submit an unauthorized ground transportation expense claim.",
            "why_standard_evals_hide_it": "Intent classification metrics report high top-1 accuracy because the primary trigger word was 'cancelled'.",
            "senior_engineering_fix": "Composite Multi-Intent Checklist Decomposition: Query pre-processor splits compound queries into atomic actionable sub-tasks before routing."
        },
        {
            "failure_id": "FAIL_03_SARCASM_VULNERABLE_LIABILITY",
            "component": "Escalation Policy & Sentiment Filter",
            "failure_title": "Sarcastic Vulnerable Passenger & Regulatory Liability Miss",
            "customer_inquiry": "@Delta So thrilled that Delta left my 84-year-old grandmother with dementia sitting in wheelchair at Gate C14 for 4 hours alone while the entire crew went home. What an inspiring display of Delta hospitality! Guess we will have a wonderful conversation with the state attorney general tomorrow morning.",
            "model_telemetry": {
                "detected_sentiment": "POSITIVE / COMPLIMENTARY (Keywords: 'so thrilled', 'inspiring display', 'hospitality', 'wonderful')",
                "classifier_escalate_probability": 0.14,
                "tier0_keyword_match": "None (No profanity, no explicit word 'lawsuit')"
            },
            "what_the_naive_system_does": "Fires a generic positive appreciation macro: '<CUSTOMER_HANDLE> Thank you so much for the kind words! We are delighted to hear about your experience with Delta and love having your grandmother on board with us! *CS'",
            "the_real_world_production_failure": "Catastrophic brand humiliation, viral social media disaster, and severe regulatory ADA/elder neglect liability.",
            "why_standard_evals_hide_it": "Standard sentiment and toxicity benchmarks give high scores because there is zero profanity, toxic language, or aggressive punctuation.",
            "senior_engineering_fix": "Contrastive Pragmatic Anomaly Detector: Flags when positive sentiment tokens co-occur with high-risk vulnerability entities ('dementia', 'wheelchair', 'alone', 'attorney general')."
        }
    ]
    
    for f in failures:
        print(f"\n[{f['failure_id']}]: {f['failure_title']}")
        print(f"  Component: {f['component']}")
        print(f"  Inquiry: \"{f['customer_inquiry']}\"")
        print(f"  Production Vulnerability: {f['the_real_world_production_failure']}")
        print(f"  Why Standard Evals Hide It: {f['why_standard_evals_hide_it']}")
        print(f"  Senior Engineering Fix: {f['senior_engineering_fix']}")
        
    out_path = os.path.join(ARTIFACTS_DIR, "phase8_deep_failure_analysis.json")
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(failures, file, indent=2)
        
    print(f"\nArtifacts saved to {out_path}")
    return failures

if __name__ == "__main__":
    run_failure_mode_investigation()

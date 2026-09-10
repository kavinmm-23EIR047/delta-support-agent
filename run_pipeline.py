"""
Production AI Customer Support Agent - Single Command End-to-End Runner
Executes the full multi-phase pipeline or fast evaluation suite in < 2 minutes.

Usage:
  python run_pipeline.py --mode fast    (Reproduces all headline metrics on cached artifacts in ~15s)
  python run_pipeline.py --mode full    (Re-runs full thread extraction, HDBSCAN, and calibration from scratch)
"""

import os
import sys
import argparse
import time
import json
from typing import Dict, List, Any

# Ensure UTF-8 stdout encoding on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from phase4_resolution_retrieval import run_phase4_grounding_evaluation
from phase5_escalation_policy import run_phase5_pipeline
from phase6_baselines_benchmark import run_benchmark_comparison
from phase7_eval_judge_skepticism import run_phase7_pipeline
from phase8_failure_analysis import run_failure_mode_investigation
from phase9_relabeling_audit import run_phase9_relabeling_audit

def run_all(mode: str = "fast"):
    start_time = time.time()
    print("=" * 80)
    print(f"PRODUCTION AI CUSTOMER SUPPORT AGENT — BENCHMARK HARNESS [Mode: {mode.upper()}]")
    print("Dataset: Kaggle Customer Support on Twitter (@Delta Airlines)")
    print("=" * 80)
    
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    if mode == "full":
        print("\n[PHASE 1 & 2] Reconstructing Conversation Trees & Adversarial Cleaning...")
        from phase1_brand_analysis import inspect_dataset_overview, profile_candidate_brands
        from phase2_thread_reconstruction import reconstruct_delta_threads
        from phase3_intent_taxonomy import load_customer_inquiries, run_hdbscan_clustering, evaluate_taxonomy_stress_test
        
        top_brands = inspect_dataset_overview()
        profile_candidate_brands(top_brands)
        reconstruct_delta_threads()
        inquiries, _ = load_customer_inquiries(sample_size=10000)
        run_hdbscan_clustering(inquiries)
        evaluate_taxonomy_stress_test()

    print("\n[PHASE 4] Running Resolution-Success Grounding Evaluation...")
    run_phase4_grounding_evaluation()

    print("\n[PHASE 5] Calibrating Asymmetric Risk Escalation Policy...")
    run_phase5_pipeline()

    print("\n[PHASE 6] Running Multi-Baseline Comparative Benchmark...")
    eval_results = run_benchmark_comparison()

    print("\n[PHASE 7] Running Judge Skepticism & Human Agreement (Cohen's Kappa)...")
    run_phase7_pipeline()

    print("\n[PHASE 8] Generating Deep Failure Analysis Case Studies...")
    run_failure_mode_investigation()

    print("\n[PHASE 9] Running 20% Blind Relabeling Noise Ceiling Audit...")
    run_phase9_relabeling_audit()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print(f"All artifacts and benchmarks written to: {artifacts_dir}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Support AI End-to-End Pipeline")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast", help="Execution mode (fast ~15s, full ~3min)")
    args = parser.parse_args()
    run_all(mode=args.mode)

"""
Generates a compact, stratified 5,000-thread sample for sub-10-second fast benchmarking.
"""
import json
import os

ARTIFACTS_DIR = r"d:\hiver_task\artifacts"
FULL_JSON = os.path.join(ARTIFACTS_DIR, "delta_reconstructed_threads.json")
SAMPLE_JSON = os.path.join(ARTIFACTS_DIR, "delta_reconstructed_threads_sample.json")

def create_sample():
    # Read line by line or with chunked streaming
    with open(FULL_JSON, "r", encoding="utf-8") as f:
        threads = json.load(f)
        
    print(f"Loaded {len(threads):,} full threads.")
    sample = threads[:5000]
    
    with open(SAMPLE_JSON, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)
        
    print(f"Saved compact sample ({len(sample):,} threads) to {SAMPLE_JSON} (Size: {os.path.getsize(SAMPLE_JSON)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    create_sample()

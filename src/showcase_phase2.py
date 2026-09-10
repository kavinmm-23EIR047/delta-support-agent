"""
Extracts concrete showcase examples of each adversarial cleaning category
and semantic PII scrubbing from delta_reconstructed_threads.json
"""
import json

with open(r"d:\hiver_task\artifacts\delta_reconstructed_threads.json", "r", encoding="utf-8") as f:
    threads = json.load(f)

examples = {
    'multi_agent_handoff': [],
    'temporal_gap_gt_24h': [],
    'canned_deflection': [],
    'pii_scrubbed_pnr_flight': []
}

for t in threads:
    m = t['metadata']
    flags = m['flags']
    
    if flags['multi_agent_handoff'] and len(examples['multi_agent_handoff']) < 2:
        examples['multi_agent_handoff'].append(t)
        
    if flags['temporal_gap_gt_24h'] and len(examples['temporal_gap_gt_24h']) < 2:
        examples['temporal_gap_gt_24h'].append(t)
        
    if flags['is_pure_canned_deflection'] and len(examples['canned_deflection']) < 2:
        examples['canned_deflection'].append(t)
        
    if m['pii_scrubbed_counts'].get('pnrs', 0) > 0 and m['pii_scrubbed_counts'].get('flights', 0) > 0 and len(examples['pii_scrubbed_pnr_flight']) < 2:
        examples['pii_scrubbed_pnr_flight'].append(t)

with open(r"d:\hiver_task\artifacts\phase2_showcase_examples.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=2)

print("Extracted showcase examples successfully.")

"""
Sample inspection script: compares qualitative grounding complexity and canned deflection examples
between Delta, XboxSupport, AppleSupport, and Uber_Support.
"""
import pandas as pd
import json

DATA_PATH = r"d:\hiver_task\data\twcs\twcs.csv"

def inspect_sample_threads():
    brands = ["Delta", "XboxSupport", "AppleSupport", "Uber_Support", "SpotifyCares"]
    samples = {b: [] for b in brands}
    
    # Load chunks and find multi-turn threads
    for chunk in pd.read_csv(DATA_PATH, chunksize=100000, 
                             usecols=['tweet_id', 'author_id', 'inbound', 'in_response_to_tweet_id', 'text', 'created_at']):
        for b in brands:
            sub = chunk[(chunk['author_id'] == b) & (chunk['in_response_to_tweet_id'].notna())]
            for _, row in sub.head(10).iterrows():
                if len(samples[b]) < 3:
                    samples[b].append({
                        'tweet_id': int(row['tweet_id']),
                        'in_response_to': int(row['in_response_to_tweet_id']),
                        'text': str(row['text'])
                    })
                    
    with open(r"d:\hiver_task\artifacts\brand_sample_replies.json", "w") as f:
        json.dump(samples, f, indent=2)
    print("Saved brand reply samples.")

if __name__ == "__main__":
    inspect_sample_threads()

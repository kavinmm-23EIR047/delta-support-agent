"""
Granular cluster analysis and taxonomy mapping script
Extracts fine-grained topic distributions across the full 33,111 Delta inquiries.
"""
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

with open(r"d:\hiver_task\artifacts\delta_reconstructed_threads.json", "r", encoding="utf-8") as f:
    threads = json.load(f)

inquiries = []
for t in threads:
    cust = [trn['scrubbed_text'] for trn in t['turns'] if trn['speaker'] == 'customer']
    if cust and len(cust[0].split()) >= 4:
        inquiries.append(cust[0])

# Fit NMF topic model for granular sub-intent density discovery
tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1,2), stop_words='english', min_df=5, max_df=0.6)
X = tfidf.fit_transform(inquiries)

nmf = NMF(n_components=8, random_state=42)
W = nmf.fit_transform(X)
H = nmf.components_
feature_names = tfidf.get_feature_names_out()

topics = []
for topic_idx, topic in enumerate(H):
    top_features = [feature_names[i] for i in topic.argsort()[:-12 - 1:-1]]
    dominant_count = int(np.sum(W.argmax(axis=1) == topic_idx))
    topics.append({
        'topic_id': topic_idx,
        'dominant_inquiries_count': dominant_count,
        'pct': round(dominant_count / len(inquiries) * 100, 2),
        'top_keywords': top_features
    })

df_topics = pd.DataFrame(topics)
print(df_topics.to_string(index=False))

with open(r"d:\hiver_task\artifacts\phase3_topic_distribution.json", "w", encoding="utf-8") as f:
    json.dump(topics, f, indent=2)

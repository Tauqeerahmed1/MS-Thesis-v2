# Model Card - Context-Aware PR Distribution Recommender

## Purpose
Offline research prototype for ranking media endpoints for press releases using lexical baselines, compact embedding fallbacks, CTR prediction, and PageRank-enhanced hybrid scoring.

## Data Mode
proxy_from_available_pr_titles_and_outlet_engagement

## Selected Components
- Best semantic signal: tfidf_similarity
- Hybrid weights: {'semantic': 0.2, 'ctr': 0.49999999999999994, 'pagerank': 0.3000000000000001, 'val_ndcg@10': 0.8637795073115179}
- Split protocol: {'train_end': '2024-04-11', 'validation_end': '2024-12-18', 'protocol': 'time_based_by_press_release_post_date'}

## Current Limitation
The workspace does not contain a true PR-endpoint interaction log with per-PR exposure, impression, click, or propensity columns. The notebook therefore runs a proxy backtesting mode using PR titles and outlet-level engagement summaries. Replace the proxy builder with real logs before treating the results as final thesis evidence.

## Operational Notes
The implementation is dependency-light and uses pandas/numpy only. If local transformer or sentence-transformer packages are added later, the notebook can be extended to swap hash fallback encoders for cached SLM embeddings.
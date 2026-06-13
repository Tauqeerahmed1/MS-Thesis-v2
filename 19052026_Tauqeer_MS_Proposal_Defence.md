# Introduction

Traditional press release (PR) distribution platforms typically rely on static pricing tiers that optimize for broad reach rather than semantic relevance. In practice, this reach-first approach can target irrelevant media endpoints and yield low engagement. This research proposes a context-aware recommendation engine that dynamically ranks media endpoints according to the semantic intent of a PR and the historical performance profile of each outlet.

While LLMs can provide strong semantic reasoning, their inference latency, operational cost, and deployment complexity can limit feasibility for real-time, high-volume recommendation workflows. The research instead investigates open-source Small Language Models (SLMs) as resource-efficient semantic encoders for press-release routing. The SLM set for this study includes Llama, Ministral, Phi, Qwen, and SmolLM. The proposed architecture extracts dense semantic embeddings from PR text using these SLM encoders and combines them with structured features (time, geography, outlet niche, and outlet metadata) using a Deep Factorization Machine (DeepFM) to predict engagement, i.e., click-through rate (CTR).

Evaluation will be performed via offline backtesting and counterfactual evaluation on historical distribution logs and recommendations where available, benchmarking recommendation precision, engagement quality, and operational efficiency against static tier baselines, while also measuring latency and memory footprint to quantify the advantage of SLM-based inference.

# Problem Statement

PR distribution services frequently sell bundles based on number of outlets or broad audience size. Generally, a PR is published on media outlets whose editorial focus does not match the PR topic and audience. Hence, this produces a relevance gap that leads to weaker user engagement.

In addition, modern ML solutions often assume always-on access to frontier LLM APIs. In many PR industry settings (compute budgets, latency requirements, and tight integration needs), relying on external LLM APIs can be impractical. The aim is to build a PR distribution recommender that:

- Matches PR content to outlets with higher topical alignment.
- Predicts engagement outcomes (CTR).
- Generates stronger ranked recommendations than static tier-based distribution while using resource-efficient SLMs rather than expensive LLM inference.

# Research Questions

- To what extent do open-source SLM embeddings improve PR-endpoint semantic matching over lexical and distilled encoder baselines?
- To what extent does integrating semantic embeddings with structured metadata through DeepFM improve engagement prediction?
- How effective is a PageRank-enhanced ranking mechanism compared to relevance-only and CTR-only ranking strategies for PR-endpoint recommendation?

# Statement of Objectives

The following objectives are intended to be achieved through this research:

- Define and implement baseline methods (rule/keyword matching and TF-IDF similarity) inspired by prior constrained text-based recommenders.
- Train and compare compact semantic encoders (distilled encoders vs. open-source SLM embeddings from Llama, Ministral, Phi, Qwen, and SmolLM) for PR-endpoint relevance.
- Train a CTR predictor (SLM embeddings + structured metadata) using DeepFM and compare against simpler predictors.
- Evaluate ranked endpoint recommendations offline against static-tier baselines.
- Design and test a PageRank-based authority signal for endpoint reranking and hybrid score fusion.

# Literature Review

Press-release distribution can be framed as a contextual recommender problem where the items are media endpoints and the context is PR text plus campaign metadata. Recommender surveys show that modern systems must balance semantic relevance with operational constraints such as sparsity and non-stationarity (Li et al., 2023; Zhou et al., 2025). Classical collaborative-filtering baselines like matrix factorization remain strong for implicit-feedback settings, but they suffer from cold-start and limited interpretability (Koren et al., 2009). For PR-endpoint matching, lexical baselines (keyword rules and TF-IDF) are efficient yet brittle to paraphrase and nuanced intent. LLM-enhanced recommender research demonstrates how language models can improve candidate generation, ranking, and explanations (Liu et al., 2024), but API dependency and inference cost can be prohibitive for real-time, high-volume routing. This motivates open-source SLM encoders and hybrid rankers that fuse text representations with structured signals.

Beyond pure ranking, PR distribution requires practical decision support, where publishers need recommendations that are relevant, measurable, and operationally feasible. Hybrid neural-symbolic approaches illustrate how logical constraints can be integrated with learned relevance signals, supporting rules such as industry AND geography (Chen et al., 2021). Finally, measurement guidance in PR practice emphasizes engagement and downstream outcomes (e.g., CTR, pickups, and conversion tracking), providing concrete targets for learning and evaluation (Aslam, 2025). For deployment, incremental updating methods help keep outlet models fresh (Zhang & Kim, 2023). Together, these works justify an SLM-based semantic layer combined with outcome prediction.

Prior constrained text-based recommender systems show that classical content-based approaches (TF-IDF, shallow embedding similarity such as Word2Vec/Doc2Vec-style, and topic models) can be feasible and competitive when labels and resources are limited. However, these approaches often lack:

- Strong domain-adapted semantic representations for complex intent.
- A robust outcome-prediction layer that links relevance to engagement.

This thesis extends prior work on building text-based recommenders under limited labels and resources into the press-release distribution domain by:

- Using compact SLM-derived embeddings as the semantic signal (instead of only TF-IDF or shallow embeddings).
- Adding CTR prediction via a combined model (DeepFM) that fuses text embeddings with structured metadata.
- Prioritizing engagement-aware ranking rather than static distribution tiers.

This study is grounded in global prior work and does not claim invention of SLMs, DeepFM, or PageRank as standalone methods. Instead, its novelty lies in task-specific integration and validation for PR endpoint recommendation. In this sense, the contribution is methodological (integrated framework), empirical (benchmark evidence against established baselines), and practical (resource-efficient open-source encoder evaluation and inference).

| Approach family | Semantic signal | Supervision need | Typical output | Strength/limitation |
|---|---|---|---|---|
| Rule/keyword matching | Sparse lexical | None | Ranked list | Fast and explainable; weak semantics |
| TF-IDF similarity | Sparse lexical | None | Ranked list | Strong baseline; brittle to paraphrase |
| Topic models (LDA-style) | Topic distribution | None | Ranked list | Interpretable; coarse topics |
| Shallow embeddings (Word2Vec/Doc2Vec-style) | Dense shallow | Weak/self-supervised | Ranked list | Better semantics; domain sensitivity |
| Distilled encoders (e.g., DistilBERT) | Dense contextual | Weak/self-supervised + optional fine-tune | Ranked list | Strong semantics; still heavier |
| Open-source SLM encoders (Llama, Ministral, Phi, Qwen, and SmolLM) | Dense contextual | Weak/self-supervised + PEFT | Ranked list + features | Strong semantics with lower operational cost |
| SLM + DeepFM | Embeddings + structured | Supervised on logs | CTR + ranking | Best for outcome prediction; needs logs |
| PageRank-enhanced hybrid ranking | Relevance + CTR + graph authority | Supervised + graph structure | Final ranked list | Improves authority-aware ordering; depends on graph quality |

# Research Contribution in Proposed Framework

The proposed framework contributes to research in three complementary ways:

- Methodological contribution: a unified pipeline that combines resource-efficient SLM semantic encoding, DeepFM-based engagement prediction, and PageRank-enhanced reranking in one end-to-end recommendation architecture.
- Empirical contribution: a reproducible benchmark evaluation against lexical, distilled-encoder, metadata-only, and historical-prior baselines using temporal splits and standard ranking/prediction metrics.
- Practical contribution: demonstration that open-source compact SLM encoders can provide strong recommendation quality with feasible latency and memory usage for near-real-time PR routing.

The contribution is positioned as a gap-bridging extension of existing global literature: prior studies establish individual benefits of semantic encoders, CTR prediction, and graph-based ranking, while this work evaluates their integrated effect in the PR distribution domain under a common protocol.

# Proposed Methodology of Research

Proposed methodology of research is explained in the ensuing paragraphs.

## Protocol for Data Collection

This study will leverage historical press-release (PR) distribution logs of a US-based firm, Evertise AI (https://evertise.net), as a case study to establish a linkage between PR content and endpoint-level engagement outcomes. The primary data will comprise:

- PR text fields (headline, subheading, and body, with optional tags or industry labels where available).
- Endpoint metadata (e.g., outlet name, category or niche, geography, and reach/authority proxies).
- Distribution logs capturing which endpoints were selected for each PR, timestamps, and observed exposure/engagement signals (impressions where available, and clicks).

The unit of analysis will be a PR-endpoint interaction observed within a defined time window, represented as a row containing text-derived features, structured metadata, and an engagement label. Engagement will be operationalized primarily through click-through rate (CTR). When impression counts are available, CTR will be computed as:

$$CTR = \frac{Clicks}{Impressions}$$

When impression counts are not available, a binary click indicator (clicked vs. not clicked) will be used for supervised learning, and CTR will be approximated during evaluation where feasible.

To improve robustness, low-signal observations (e.g., endpoints with very small numbers of impressions) will be handled using filtering and smoothing procedures, such as minimum-impression thresholds and Bayesian (or Empirical-Bayes) smoothing, to reduce variance and mitigate noise. Text preprocessing will include cleaning and normalization (e.g., removing boilerplate and standardizing casing/whitespace) while preserving named entities and domain-specific terms that are likely to carry targeting signal. Where available, an endpoint profile text (e.g., outlet descriptions, typical categories, or recent headlines) will be constructed to support similarity-based baselines.

Finally, to reflect real deployment conditions and minimize temporal leakage, the study will employ time-based splitting (training on earlier campaigns and validating/testing on later campaigns).

## Experimental Setup

The experimental design follows a staged recommender pipeline:

- Represent PR content for PR-endpoint matching.
- Predict engagement outcomes.
- Generate ranked endpoint recommendations.

In Stage 1 (candidate ranking), multiple matching strategies will be compared. Lexical baselines will include rule/keyword matching and TF-IDF cosine similarity, with optional topic-based similarity (LDA-style) when endpoint profile text is sufficiently informative. Compact embedding baselines will include distilled sentence encoders (e.g., MiniLM/Distil* families). The proposed representation will use a set of open-source SLM encoders (Llama, Ministral, Phi, Qwen, and SmolLM) to generate dense semantic vectors for PR text, denoted as $e_{pr}$. Where justified by data availability, parameter-efficient domain adaptation (e.g., LoRA) and/or contrastive alignment may be applied using positive and negative pairs derived from historical outcomes. Each SLM encoder will be trained/evaluated under the same split and preprocessing protocol so that model-level comparisons remain fair and directly attributable to encoder choice rather than pipeline variation.

In Stage 2 (outcome modeling), the goal is to predict engagement (CTR or click probability) for PR-endpoint pairs. Baselines will include:

- Historical priors such as smoothed endpoint CTR.
- A metadata-only predictor.

The proposed model is a Deep Factorization Machine (DeepFM) that fuses semantic features (e.g., $e_{pr}$ and optional endpoint representations) with structured variables such as endpoint identifiers, category/niche, time, geography, and authority/reach proxies.

In Stage 3 (recommendation generation with PageRank), endpoints will be ranked using a hybrid score that combines semantic relevance, predicted engagement, and graph-based authority. An outlet graph will be constructed where nodes are endpoints and edges represent hyperlink/citation relationships (or co-pickup/co-mention proxies when direct link data is unavailable). PageRank will be computed over this graph and normalized as $PR_i$. Final ranking score will be computed as:

$$Score_i = \lambda_1 Rel_i + \lambda_2 \widehat{CTR}_i + \lambda_3 PR_i$$

where $Rel_i$ is semantic relevance, $\widehat{CTR}_i$ is predicted engagement, and $PR_i$ is authority score. Comparative experiments will evaluate relevance-only, CTR-only, and hybrid scoring, with and without PageRank reranking.

## Benchmarking Strategy and Success Criteria

To ensure rigorous comparison, all methods will be evaluated under a unified benchmark protocol:

- Baselines: rule/keyword matching, TF-IDF, distilled encoder baseline, historical-prior CTR baseline, and metadata-only CTR predictor.
- Protocol: identical preprocessing, identical candidate pools, and time-based train/validation/test splits.
- Metrics: Precision@K, Recall@K, MAP@K, NDCG@K, AUC, log-loss, ECE, latency, throughput, and peak memory.
- Statistical validation: repeated runs where applicable, confidence intervals, and significance testing for key metric differences.
- Success criteria: the proposed framework should outperform lexical and distilled baselines on ranking and prediction metrics while meeting practical efficiency constraints for open-source SLM inference and adaptation.

## Implementation Plan and Reproducibility

To make the research directly executable, implementation will proceed through the following work packages:

- WP1 - Data Engineering and Labeling: ingest logs, standardize schema, build PR-endpoint interaction table, apply smoothing/filters, and freeze dataset versions.
- WP2 - Baseline Reproduction: implement keyword/rule, TF-IDF, and distilled-encoder baselines with shared candidate generation and ranking interfaces.
- WP3 - Open-Source SLM Encoder Experiments: run Llama, Ministral, Phi, Qwen, and SmolLM under identical train/validation/test splits; compare embedding quality, inference latency, memory footprint, and adaptation feasibility.
- WP4 - CTR Modeling: train metadata-only and DeepFM predictors; calibrate probabilities and export calibrated scores for ranking.
- WP5 - Graph Authority Module: construct outlet graph, compute PageRank, and evaluate score-fusion and reranking strategies.
- WP6 - Evaluation and Robustness: run benchmark, ablation, statistical tests, and sensitivity checks (including limited-data and cold-start slices).
- WP7 - Prototype Integration: expose the final pipeline as a reproducible offline recommendation workflow for replay/backtesting.

Reproducibility controls will include fixed random seeds, configuration versioning, dataset snapshots, and experiment tracking logs for every run (model version, hyperparameters, metrics, and runtime profile). All models will be evaluated with the same candidate sets and temporal split boundaries. Result tables and figures will be generated from saved experiment artifacts to ensure full traceability.

Implementation outputs will include:

- A benchmark matrix (all models x all metrics).
- An ablation matrix (encoder, predictor, ranking, and authority variants).
- A final model card summarizing selected pipeline, assumptions, and operational limits.
- A reproducible runbook describing end-to-end execution steps.

Operational acceptance criteria for implementation completion are: statistically significant gains on key ranking metrics versus strongest baseline, stable calibration for CTR prediction, and inference performance suitable for near-real-time recommendation using resource-efficient open-source SLMs on target hardware.

## Method of Analysis and Validation

Evaluation will be conducted offline using held-out historical logs and, where feasible, counterfactual estimators to reduce exposure bias inherent in logged decision policies. Predictive performance will be assessed using AUC and log-loss, supplemented by calibration diagnostics (reliability plots and expected calibration error, ECE).

Recommendation quality will be measured using standard Top-K ranking metrics, including Precision@K, Recall@K, MAP@K, and NDCG@K. These will be computed using an offline relevance signal derived from observed outcomes within the logged candidate set. Business-oriented outcomes will be assessed using engagement-focused indicators such as expected clicks at K, observed CTR of recommended endpoints, pickup rate, and comparative utility relative to static tier baselines. Operational feasibility will be quantified through latency per recommendation, peak memory footprint, throughput, model size, and hardware requirements for open-source SLM inference.

The offline evaluation strategy will include replay/backtesting over observed candidates, comparing predicted rankings against the endpoints actually used and their realized outcomes. If propensities (or approximate exposure probabilities) can be recovered or modeled, counterfactual evaluation will be performed using IPS/SNIPS estimators of policy value, with Doubly Robust (DR) estimation applied when both propensity and outcome models are available. Where propensities cannot be reliably recovered, sensitivity analyses will be conducted to understand how conclusions vary under plausible exposure assumptions.

Finally, an ablation study will be used to isolate the contribution of each major component:

- Encoder ablation (keywords/TF-IDF vs. distilled encoders vs. open-source SLM encoders: Llama, Ministral, Phi, Qwen, and SmolLM).
- Predictor ablation (similarity-only vs. metadata-only vs. SLM + DeepFM).
- Ranking ablation (semantic-only vs. CTR-only vs. hybrid relevance + CTR ranking).
- Authority ablation (without PageRank vs. PageRank as feature vs. PageRank reranking).
- Significance testing of improvements over strongest baseline models.

# Expected Results

This research is expected to demonstrate that a context-aware PR distribution recommender, built using compact semantic encoders and outcome-aware ranking, can outperform static tier-based distribution strategies on relevance and engagement quality. Specifically, the study anticipates that semantic matching models (distilled encoders and open-source SLM embeddings from Llama, Ministral, Phi, Qwen, and SmolLM) will produce substantially more topically aligned endpoint rankings than lexical baselines (rule/keyword matching and TF-IDF), particularly in cases where PRs use paraphrasing, domain jargon, or implicit intent. In turn, this improved alignment is expected to translate into higher engagement under logged evaluation, as the recommended endpoints are better matched to the PR topic and audience.

For engagement prediction, it is expected that the proposed SLM-embedding + DeepFM model will outperform metadata-only and prior-based predictors in discriminating high-response endpoints, yielding improved AUC and log-loss, and better-calibrated probabilities. Better calibration is expected to improve ranking reliability and the consistency of high-performing recommendations across campaigns.

For ranking quality, it is expected that adding PageRank-based authority to semantic relevance and predicted CTR will improve top-K ordering, especially when multiple candidates have similar semantic scores but different publication influence profiles. Operationally, open-source compact SLM encoders are expected to provide latency, memory, transparency, and cost advantages over frontier LLM-based alternatives where target hardware permits. Overall, the expected result is a validated, practical methodology showing that small, efficient language models can deliver a competitive relevance-efficiency trade-off for PR endpoint recommendation, while the addition of CTR prediction yields stronger recommendation performance compared to static distribution tiers.

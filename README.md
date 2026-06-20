# PR Distribution Recommender System
### MS Thesis — Tauqeer Ahmed | FAST-NUCES

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20T4-orange.svg)](https://colab.research.google.com/)
[![Stage](https://img.shields.io/badge/Pipeline-3%2F3%20Complete-brightgreen.svg)]()

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Proposed Solution](#3-proposed-solution)
4. [Dataset](#4-dataset)
5. [Dataset Cleaning](#5-dataset-cleaning)
6. [System Architecture](#6-system-architecture)
7. [Pipeline — Stage by Stage](#7-pipeline--stage-by-stage)
8. [Results](#8-results)
9. [Limitations](#9-limitations)
10. [Repository Structure](#10-repository-structure)
11. [How to Run](#11-how-to-run)
12. [Requirements](#12-requirements)

---

## 1. Project Overview

This thesis presents a **hybrid, three-stage AI pipeline** for automatically recommending the most suitable media outlets for a given press release (PR). Instead of relying on manual judgment or keyword matching, the system combines **semantic understanding**, **click-through-rate (CTR) prediction**, and **authority-based reranking** to produce ranked outlet recommendations.

**Supervisor:** [University Faculty]  
**GitHub:** [github.com/Tauqeerahmed1/MS-Thesis-v2](https://github.com/Tauqeerahmed1/MS-Thesis-v2)  
**Platform:** Google Colab (T4 GPU)

---

## 2. Problem Statement

Press release distribution is a critical but inefficient process in modern PR and communications. Key challenges include:

- **Manual effort:** PR professionals manually select media outlets — a slow, subjective process
- **Poor targeting:** Generic distribution to hundreds of outlets results in low engagement
- **No personalization:** Existing tools lack semantic understanding of PR content vs. outlet coverage
- **No real click data:** Real per-PR-per-outlet engagement logs are rarely available to researchers

**Core Research Question:** Can a hybrid AI system — combining language model embeddings, CTR prediction, and graph-based authority scoring — outperform traditional keyword-based PR distribution?

---

## 3. Proposed Solution

A **3-stage hybrid pipeline** that processes press releases and ranks media outlets:

```
Press Release (text)
        │
        ▼
┌─────────────────────────┐
│  Stage 1: SLM Encoder   │  → Semantic similarity (Qwen best: P@5 = 0.058)
│  (PR ↔ Outlet matching) │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Stage 2: DeepFM CTR    │  → CTR prediction (AUC = 0.9716, Acc = 92.85%)
│  (Engagement prediction)│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Stage 3: PageRank      │  → Authority reranking (1000 nodes, 70,843 edges)
│  (Outlet authority)     │
└────────────┬────────────┘
             │
             ▼
   Final Hybrid Score
   Score_i = 0.5·Relevance + 0.3·CTR + 0.2·PageRank
```

---

## 4. Dataset

### Press Releases
| Property | Value |
|---|---|
| Source | Evertise (real-world PR platform) |
| Format | XML (multi-part, split files) |
| Total PRs | 9,918 (after cleaning) |
| Sample used | 1,000 (random_state=42) |
| Key field | PR title + body text |

### Media Outlets
| Property | Value |
|---|---|
| Source | Official outlet CSV |
| Total outlets | 1,000 |
| Key columns | Media Outlet, Publication URL, Region, Estimated Traffic, Estimated Views, Estimated Clicks |

---

## 5. Dataset Cleaning

### Press Release XML Cleaning
The raw Evertise dataset came as split XML files with multiple corruption issues:

- **Encoding errors:** Non-UTF-8 characters causing XML parse failures — fixed via `errors='replace'` encoding
- **Split files:** Dataset was split across 3 parts (`.part001`, `.part002`, `.part003`) — merged programmatically
- **Mixed formats:** Some parts were `.xml`, others were `.csv` exports of the same data — handled separately
- **Duplicate entries:** Deduplicated on PR title + date
- **Empty fields:** PRs with missing title or body were dropped
- **Result:** 9,918 clean, usable press releases

### Outlet CSV Cleaning
- Stripped whitespace from column names and string values
- Normalized numeric columns (`Estimated Traffic`, `Views`, `Clicks`) — replaced nulls with column median
- Encoded categorical columns: `Region` → `region_enc`, `Media Outlet` → `outlet_enc` (LabelEncoder)
- **Note:** Some outlet names appear multiple times (e.g., Newsweek) — duplicate rows in source data; deduplication not applied to preserve dataset integrity for benchmarking

---

## 6. System Architecture

### Stage 1 — SLM Encoder Benchmark

Five Small Language Models (SLMs) were benchmarked for semantic PR-to-outlet matching:

| Model | Embedding Dim | Encode Time | P@5 | NDCG@5 |
|---|---|---|---|---|
| TF-IDF (Baseline) | — | — | 0.034 | 0.038 |
| SmolLM | 576 | 16.2s | 0.041 | 0.045 |
| Llama (TinyLlama) | 2048 | 26.9s | 0.052 | 0.057 |
| Phi | 2560 | 52.9s | 0.049 | 0.053 |
| Mistral | 4096 | 163.8s | 0.038 | 0.042 |
| **Qwen ✓ (selected)** | **1536** | **27.9s** | **0.058** | **0.063** |

**Winner:** Qwen — best precision AND good speed. Mistral was worst despite being largest.

### Stage 2 — DeepFM CTR Prediction (NumPy)

**CTR Proxy Label Construction** (no real click logs available):
```
ctr_proxy = 0.7 × similarity_normalized + 0.3 × clicks_normalized
Binary label: top 20% per PR → label=1 (Engaged), rest → label=0
```

**Feature Matrix:** 1,000,000 PR-Outlet pairs (1000 × 1000)

| Feature | Description |
|---|---|
| `region_enc` | Outlet region (label encoded) |
| `outlet_enc` | Outlet identity (label encoded) |
| `traffic_norm` | Estimated traffic (min-max normalized) |
| `views_norm` | Estimated views (min-max normalized) |
| `clicks_norm` | Estimated clicks (min-max normalized) |
| `similarity` | Qwen cosine similarity score |

**DeepFM Architecture (pure NumPy — no PyTorch/TensorFlow):**
```
Input (6 features)
    ├── FM Part: Linear weights (w0, w1) + Embeddings (V, dim=8) → pairwise interactions
    └── Deep Part: Dense(32) → Dense(16) → output
Combined: sigmoid(FM_output + Deep_output)
Optimizer: SGD with manual backpropagation
```

### Stage 3 — PageRank Authority Reranking

- **Graph construction:** Outlet-outlet cosine similarity matrix (1000×1000) → threshold 0.95 → adjacency matrix
- **PageRank:** Power iteration (damping=0.85, max_iter=100, tol=1e-6)
- **Convergence:** Iteration 41

**Final Hybrid Score:**
```
Score_i = λ1·Relevance_i + λ2·CTR_i + λ3·PageRank_i
        = 0.5·Relevance  + 0.3·CTR   + 0.2·PageRank
```

---

## 8. Results

### Stage 1 — SLM Benchmark
- **Best model:** Qwen (P@5 = 0.058, NDCG@5 = 0.063)
- **Baseline (TF-IDF):** P@5 = 0.034 — SLMs outperform keyword matching

### Stage 2 — DeepFM CTR
| Metric | Value |
|---|---|
| Accuracy | **92.85%** |
| AUC-ROC | **0.9716** |
| F1 (Engaged class) | 0.82 |
| Precision (Engaged) | 0.84 |
| Recall (Engaged) | 0.79 |
| Final Loss (epoch 10) | 0.1693 |
| Training time | ~0.3 min (800K rows, batch=1024) |

### Stage 3 — PageRank
| Metric | Value |
|---|---|
| Graph nodes | 1,000 |
| Graph edges | 70,843 |
| Convergence iteration | 41 |
| Top authority outlets | Newsweek, Fortune, Reuters |

### Final Hybrid Score
| Metric | Value |
|---|---|
| Score range | [0.0337, 0.8752] |
| Coverage | 1,000 PRs × 1,000 Outlets |

---

## 9. Limitations

### Methodological Limitations
1. **CTR labels are a proxy** — No real per-PR-per-outlet click logs were available. CTR labels were constructed as a weighted blend of semantic similarity (70%) and Estimated Clicks column (30%). This is an approximation; results may differ with real engagement data. *(Explicitly allowed by thesis synopsis when real logs are unavailable.)*

2. **PageRank graph is a proxy** — Direct hyperlink or co-mention data between outlets was unavailable. Outlet-outlet edges were built using content embedding similarity (threshold = 0.95). This approximates co-pickup behavior, not actual link structure.

3. **Small Language Models, not LLMs** — SLMs were used due to Colab T4 GPU memory constraints. Larger models (GPT-4, full Llama-3) may yield higher semantic precision.

### Dataset Limitations
4. **Duplicate outlet rows** — The outlet dataset contains multiple rows for the same outlet name (e.g., Newsweek appears 8 times in top PageRank results). In real deployment, deduplication and outlet merging would be required.

5. **1,000 PR sample** — Only 1,000 out of 9,918 PRs were used for benchmarking due to compute constraints. Results may generalize differently on the full corpus.

6. **Single PR platform** — All PRs are from Evertise only. Cross-platform generalization (BusinessWire, PR Newswire) is untested.

7. **Estimated traffic/clicks** — Outlet engagement columns (`Estimated Traffic`, `Estimated Views`, `Estimated Clicks`) are estimates, not verified real-time data.

---

## 10. Repository Structure

```
MS-Thesis-v2/
├── notebooks/
│   └── SLM_Encoder_Benchmark_CLEAN.ipynb   ← Main pipeline notebook (all 3 stages)
├── data/
│   └── outlets_1000.csv                    ← 1000 media outlets dataset
├── results/
│   ├── final_results_summary.json          ← Stage 2 + 3 metrics
│   ├── stage3_summary.json                 ← PageRank details
│   ├── SLM_Benchmark_Results.png           ← Stage 1 charts
│   ├── Pipeline_Results.png                ← Full pipeline visualization
│   ├── TF-IDF_vs_SLM.png                  ← Baseline comparison
│   └── benchmark_results.png              ← Model comparison chart
├── Dataset/                                ← Raw PR XML/CSV files (Evertise)
├── research_outputs/                       ← Intermediate analysis outputs
├── proposal/
│   └── 19052026_Tauqeer_MS_Proposal_Defence.md
├── literature/
│   └── 01_Tauqeer Lit Review_...xlsx
└── README.md
```

> **Note:** Large files (embeddings ~500MB, feature matrix ~69MB) are stored on Google Drive, not in this repo. See How to Run below.

---

## 11. How to Run

### Option A — Google Colab (Recommended)

1. Open `notebooks/SLM_Encoder_Benchmark_CLEAN.ipynb` in Google Colab
2. Add your HuggingFace token to Colab Secrets: `HF_TOKEN`
3. Run cells top to bottom — Drive will auto-mount and paths will resolve

### Option B — Local PC

1. Clone the repo:
   ```bash
   git clone https://github.com/Tauqeerahmed1/MS-Thesis-v2.git
   cd MS-Thesis-v2
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Download large files from Google Drive and place them:
   ```
   MS-Thesis-v2/
   ├── embeddings/          ← Download from Drive: SLM_Embeddings/
   │   ├── Qwen_outlet.npy
   │   ├── Qwen_pr.npy
   │   └── ... (other model embeddings)
   └── data/
       └── deepfm_feature_matrix.csv   ← Download from Drive: Dataset/
   ```

4. The notebook auto-detects local vs Colab environment — no path changes needed

---

## 12. Requirements

```
numpy
pandas
scikit-learn
transformers
torch
sentence-transformers
matplotlib
seaborn
```

Install via:
```bash
pip install numpy pandas scikit-learn transformers torch sentence-transformers matplotlib seaborn
```

---

*MS Thesis — FAST-NUCES | Tauqeer Ahmed | 2025–2026*

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
    "will",
    "you",
    "your",
    "new",
    "announces",
    "announce",
    "launches",
    "launch",
    "press",
    "release",
    "pr",
}


FALLBACK_CATEGORIES = {
    "Business & Corporate": [
        "business",
        "corporate",
        "startup",
        "merger",
        "acquisition",
        "partnership",
        "investment",
        "ipo",
        "award",
    ],
    "Finance & Economy": [
        "bank",
        "finance",
        "fintech",
        "insurance",
        "stock",
        "crypto",
        "blockchain",
        "tax",
        "loan",
    ],
    "Technology & Innovation": [
        "technology",
        "software",
        "ai",
        "artificial intelligence",
        "machine learning",
        "cloud",
        "cybersecurity",
        "saas",
        "automation",
    ],
    "Health & Medicine": [
        "health",
        "medical",
        "pharma",
        "biotech",
        "wellness",
        "clinic",
        "hospital",
        "telemedicine",
    ],
    "Energy & Environment": [
        "energy",
        "renewable",
        "solar",
        "wind",
        "oil",
        "gas",
        "climate",
        "sustainability",
        "esg",
    ],
    "Entertainment & Media": [
        "media",
        "film",
        "music",
        "sports",
        "gaming",
        "streaming",
        "news",
        "journalism",
    ],
    "Travel & Hospitality": ["travel", "tourism", "hotel", "airline", "hospitality", "destination"],
    "Education & Training": ["education", "edtech", "school", "university", "training", "scholarship"],
    "Automotive & Transportation": [
        "auto",
        "automotive",
        "vehicle",
        "ev",
        "logistics",
        "transport",
        "shipping",
    ],
    "Science & Research": ["science", "research", "space", "physics", "chemistry", "biology", "nanotech"],
}


@dataclass
class ResearchConfig:
    seed: int = 42
    max_prs: int = 650
    max_outlets: int = 220
    tfidf_max_features: int = 1600
    hash_dim: int = 160
    categorical_hash_dim: int = 96
    deepfm_factors: int = 8
    deepfm_hidden: int = 32
    deepfm_epochs: int = 16
    logistic_epochs: int = 120
    batch_size: int = 2048
    top_ks: tuple[int, ...] = (5, 10, 20)
    parse_xml_bodies: bool = False
    proxy_positive_rate: float = 0.15
    out_dir_name: str = "research_outputs"

    @classmethod
    def from_env(cls) -> "ResearchConfig":
        cfg = cls()
        if os.environ.get("PR_RESEARCH_FAST", "0") == "1":
            cfg.max_prs = 240
            cfg.max_outlets = 100
            cfg.tfidf_max_features = 900
            cfg.hash_dim = 96
            cfg.categorical_hash_dim = 64
            cfg.deepfm_epochs = 8
            cfg.logistic_epochs = 70
            cfg.batch_size = 1024
        if os.environ.get("PR_PARSE_XML_BODIES", "0") == "1":
            cfg.parse_xml_bodies = True
        return cfg


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def stable_hash(value: str, salt: str = "") -> int:
    payload = (salt + "|" + str(value)).encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9&+.#'/-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: object) -> list[str]:
    clean = normalize_text(text)
    tokens = re.findall(r"[a-z0-9][a-z0-9&+.#'/-]{1,}", clean)
    return [tok.strip("'/-") for tok in tokens if tok.strip("'/-") and tok not in STOPWORDS]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def safe_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.nanmean(values)) / (np.nanstd(values) + 1e-8)


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def extract_domain(url: object) -> str:
    raw = "" if pd.isna(url) else str(url)
    parsed = urlparse(raw if raw.startswith(("http://", "https://")) else "https://" + raw)
    domain = parsed.netloc.lower().replace("www.", "")
    return domain or normalize_text(raw).split("/")[0]


def parse_category_terms(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return FALLBACK_CATEGORIES.copy()

    terms: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if "Database-ready schema" in stripped:
            break
        top = re.match(r"^\d+\.\s+(.+)$", stripped)
        if top:
            current = top.group(1).strip()
            terms.setdefault(current, [])
            continue
        if current and stripped.startswith("*"):
            term = stripped.lstrip("*").strip()
            if term:
                terms[current].append(term)

    return terms or FALLBACK_CATEGORIES.copy()


def infer_category(text: object, category_terms: dict[str, list[str]]) -> str:
    clean = normalize_text(text)
    token_set = set(tokenize(clean))
    best_category = "Business & Corporate"
    best_score = -1.0
    for category, children in category_terms.items():
        phrases = [category] + children
        cat_tokens = set(tokenize(" ".join(phrases)))
        overlap = len(token_set & cat_tokens)
        phrase_hits = sum(1 for phrase in phrases if normalize_text(phrase) and normalize_text(phrase) in clean)
        score = overlap + 2.0 * phrase_hits
        if score > best_score:
            best_category = category
            best_score = score
    return best_category if best_score > 0 else "Business & Corporate"


def read_csv_robust(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def load_pr_posts(root: Path, config: ResearchConfig) -> pd.DataFrame:
    dataset_dir = root / "Dataset"
    csv_paths = sorted(dataset_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError("No PR CSV files found under Dataset/.")

    frames = []
    for path in csv_paths:
        frame = read_csv_robust(path)
        frame["source_file"] = path.name
        frames.append(frame)
    posts = pd.concat(frames, ignore_index=True)
    posts.columns = [str(c).strip() for c in posts.columns]

    if "status" in posts:
        posts = posts[posts["status"].fillna("").str.lower().eq("publish")]
    if "post_type" in posts:
        posts = posts[posts["post_type"].fillna("").str.lower().eq("post")]

    posts["title"] = posts.get("title", "").fillna("").astype(str)
    posts = posts[posts["title"].str.strip().ne("")]
    posts["post_date"] = pd.to_datetime(posts.get("post_date", posts.get("pubDate")), errors="coerce", utc=True)
    posts["post_date"] = posts["post_date"].dt.tz_convert(None)
    posts["post_date"] = posts["post_date"].fillna(posts["post_date"].median())
    posts["post_id"] = posts.get("post_id", pd.Series(np.arange(len(posts)), index=posts.index)).astype(str)
    posts["creator"] = posts.get("creator", "unknown").fillna("unknown").astype(str)
    posts["body_text"] = ""
    posts["title_clean"] = posts["title"].map(normalize_text)
    posts["pr_text"] = posts["title_clean"]

    subset = ["post_id"] if posts["post_id"].nunique() > len(posts) * 0.8 else ["title_clean", "link"]
    posts = posts.drop_duplicates(subset=[c for c in subset if c in posts.columns]).reset_index(drop=True)

    if config.parse_xml_bodies:
        xml_bodies = load_optional_wordpress_bodies(dataset_dir)
        if not xml_bodies.empty:
            posts = posts.merge(xml_bodies, on="post_id", how="left", suffixes=("", "_xml"))
            posts["body_text"] = posts["body_text_xml"].fillna(posts["body_text"]).fillna("")
            posts["pr_text"] = (posts["title_clean"] + " " + posts["body_text"].map(normalize_text)).str.strip()
            posts = posts.drop(columns=[c for c in posts.columns if c.endswith("_xml")], errors="ignore")

    return posts.sort_values("post_date").reset_index(drop=True)


def load_optional_wordpress_bodies(dataset_dir: Path) -> pd.DataFrame:
    import xml.etree.ElementTree as ET

    rows = []
    candidates = list(dataset_dir.glob("*.xml*"))
    if not candidates:
        return pd.DataFrame(columns=["post_id", "body_text"])

    namespaces = {
        "content": "http://purl.org/rss/1.0/modules/content/",
        "wp": "http://wordpress.org/export/1.2/",
    }
    for path in candidates:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception:
            continue
        for item in root.findall("./channel/item"):
            post_id_node = item.find("wp:post_id", namespaces)
            content_node = item.find("content:encoded", namespaces)
            post_id = post_id_node.text if post_id_node is not None else None
            body = content_node.text if content_node is not None else ""
            if post_id and body:
                rows.append({"post_id": str(post_id), "body_text": normalize_text(body)})
    return pd.DataFrame(rows).drop_duplicates("post_id") if rows else pd.DataFrame(columns=["post_id", "body_text"])


def select_pr_sample(posts: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    posts = posts.sort_values("post_date").reset_index(drop=True)
    if len(posts) <= config.max_prs:
        sample = posts.copy()
    else:
        idx = np.unique(np.linspace(0, len(posts) - 1, config.max_prs).astype(int))
        sample = posts.iloc[idx].copy()
    sample["pr_pos"] = np.arange(len(sample))
    return sample.reset_index(drop=True)


def load_outlets(root: Path, category_terms: dict[str, list[str]], config: ResearchConfig) -> pd.DataFrame:
    candidates = sorted(root.glob("*Distribution*csv"))
    if not candidates:
        raise FileNotFoundError("No distribution report CSV found in the workspace root.")
    raw = read_csv_robust(candidates[0])
    raw.columns = [str(c).strip() for c in raw.columns]
    colmap = {
        "Media Outlet": "media_outlet",
        "Publication URL": "publication_url",
        "Region": "region",
        "Estimated Traffic": "estimated_traffic",
        "Estimated Views": "estimated_views",
        "Estimated Clicks": "estimated_clicks",
        "Publication Date": "publication_date",
    }
    outlets = raw.rename(columns=colmap)
    for col in ["estimated_traffic", "estimated_views", "estimated_clicks"]:
        outlets[col] = pd.to_numeric(outlets[col], errors="coerce").fillna(0.0)
    outlets["media_outlet"] = outlets["media_outlet"].fillna("Unknown Outlet").astype(str)
    outlets["region"] = outlets["region"].fillna("Global").astype(str)
    outlets["domain"] = outlets["publication_url"].map(extract_domain)
    outlets["publication_date"] = pd.to_datetime(outlets["publication_date"], errors="coerce")
    outlets["endpoint_key"] = (
        outlets["media_outlet"].str.lower().str.strip()
        + "|"
        + outlets["domain"].str.lower().str.strip()
        + "|"
        + outlets["region"].str.lower().str.strip()
    )

    grouped = (
        outlets.groupby("endpoint_key", as_index=False)
        .agg(
            media_outlet=("media_outlet", "first"),
            domain=("domain", "first"),
            region=("region", "first"),
            publication_count=("publication_url", "count"),
            estimated_traffic=("estimated_traffic", "mean"),
            estimated_views=("estimated_views", "sum"),
            estimated_clicks=("estimated_clicks", "sum"),
            first_publication_date=("publication_date", "min"),
            last_publication_date=("publication_date", "max"),
        )
        .sort_values(["estimated_views", "estimated_traffic"], ascending=False)
        .reset_index(drop=True)
    )

    global_ctr = grouped["estimated_clicks"].sum() / max(grouped["estimated_views"].sum(), 1.0)
    smoothing = max(grouped["estimated_views"].median(), 1.0)
    grouped["raw_ctr"] = grouped["estimated_clicks"] / grouped["estimated_views"].clip(lower=1.0)
    grouped["smoothed_ctr"] = (grouped["estimated_clicks"] + smoothing * global_ctr) / (
        grouped["estimated_views"] + smoothing
    )
    grouped["log_traffic"] = np.log1p(grouped["estimated_traffic"].clip(lower=0))
    grouped["log_views"] = np.log1p(grouped["estimated_views"].clip(lower=0))
    grouped["profile_text"] = (
        grouped["media_outlet"].map(normalize_text)
        + " "
        + grouped["domain"].map(lambda x: normalize_text(str(x).replace(".", " ")))
        + " "
        + grouped["region"].map(normalize_text)
    )
    grouped["outlet_category"] = grouped["profile_text"].map(lambda txt: infer_category(txt, category_terms))
    grouped["profile_text"] = (grouped["profile_text"] + " " + grouped["outlet_category"].map(normalize_text)).str.strip()

    if len(grouped) > config.max_outlets:
        grouped = grouped.head(config.max_outlets).copy()
    grouped["outlet_pos"] = np.arange(len(grouped))
    return grouped.reset_index(drop=True)


def build_outlet_pagerank(outlets: pd.DataFrame, damping: float = 0.85, iterations: int = 80) -> np.ndarray:
    n = len(outlets)
    if n == 0:
        return np.array([])
    weights = np.zeros((n, n), dtype=np.float64)
    regions = outlets["region"].astype(str).str.lower().to_numpy()
    categories = outlets["outlet_category"].astype(str).str.lower().to_numpy()
    traffic = minmax(outlets["estimated_traffic"].to_numpy(dtype=float))
    views = minmax(outlets["estimated_views"].to_numpy(dtype=float))
    authority_prior = 0.6 * traffic + 0.4 * views + 1e-6

    for i in range(n):
        same_region = regions == regions[i]
        same_category = categories == categories[i]
        weights[i, same_region] += 0.35
        weights[i, same_category] += 0.55
        weights[i, :] += 0.10 * authority_prior
        weights[i, i] = 0.0

    row_sums = weights.sum(axis=1, keepdims=True)
    transition = np.divide(weights, row_sums, out=np.ones_like(weights) / n, where=row_sums > 0)
    rank = np.ones(n, dtype=np.float64) / n
    teleport = np.ones(n, dtype=np.float64) / n
    for _ in range(iterations):
        rank = (1 - damping) * teleport + damping * transition.T.dot(rank)
    return rank / rank.sum()


class TfidfVectorizerLite:
    def __init__(self, max_features: int = 1500, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def fit(self, texts: Iterable[str]) -> "TfidfVectorizerLite":
        docs = [tokenize(text) for text in texts]
        df = Counter()
        tf = Counter()
        for tokens in docs:
            tf.update(tokens)
            df.update(set(tokens))
        terms = [term for term, count in df.items() if count >= self.min_df]
        terms.sort(key=lambda term: (tf[term], df[term], term), reverse=True)
        terms = terms[: self.max_features]
        self.vocab_ = {term: idx for idx, term in enumerate(terms)}
        n_docs = max(len(docs), 1)
        self.idf_ = np.array([math.log((1 + n_docs) / (1 + df[term])) + 1.0 for term in terms], dtype=np.float32)
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        if self.idf_ is None:
            raise RuntimeError("Vectorizer must be fit before transform.")
        rows = list(texts)
        matrix = np.zeros((len(rows), len(self.vocab_)), dtype=np.float32)
        for i, text in enumerate(rows):
            counts = Counter(tok for tok in tokenize(text) if tok in self.vocab_)
            if not counts:
                continue
            for token, count in counts.items():
                matrix[i, self.vocab_[token]] = (1.0 + math.log(count)) * self.idf_[self.vocab_[token]]
        return l2_normalize(matrix)

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


def hashed_embeddings(texts: Iterable[str], dim: int, salt: str, mode: str = "word") -> np.ndarray:
    rows = list(texts)
    matrix = np.zeros((len(rows), dim), dtype=np.float32)
    for i, text in enumerate(rows):
        tokens = tokenize(text)
        features = list(tokens)
        if mode in {"word_bigram", "hybrid", "char_hybrid"}:
            features.extend([tokens[j] + "_" + tokens[j + 1] for j in range(max(0, len(tokens) - 1))])
        if mode in {"char_hybrid", "char"}:
            clean = normalize_text(text).replace(" ", "_")
            features.extend(clean[j : j + 4] for j in range(max(0, len(clean) - 3)))
        for feature in features:
            h = stable_hash(feature, salt=salt)
            idx = h % dim
            sign = 1.0 if ((h >> 33) & 1) else -1.0
            matrix[i, idx] += sign
    return l2_normalize(matrix)


def cosine_for_pairs(left: np.ndarray, right: np.ndarray, left_idx: np.ndarray, right_idx: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", left[left_idx], right[right_idx]).astype(np.float32)


def keyword_similarity(pr_texts: list[str], outlet_texts: list[str], pr_idx: np.ndarray, outlet_idx: np.ndarray) -> np.ndarray:
    pr_tokens = [set(tokenize(text)) for text in pr_texts]
    outlet_tokens = [set(tokenize(text)) for text in outlet_texts]
    scores = np.zeros(len(pr_idx), dtype=np.float32)
    for row, (p, o) in enumerate(zip(pr_idx, outlet_idx)):
        a = pr_tokens[int(p)]
        b = outlet_tokens[int(o)]
        if not a or not b:
            continue
        scores[row] = len(a & b) / math.sqrt(len(a) * len(b))
    return scores


def build_interactions(prs: pd.DataFrame, outlets: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    pr_idx = np.repeat(np.arange(len(prs)), len(outlets))
    outlet_idx = np.tile(np.arange(len(outlets)), len(prs))
    interactions = pd.DataFrame({"pr_pos": pr_idx, "outlet_pos": outlet_idx})
    interactions["pr_id"] = prs.iloc[pr_idx]["post_id"].to_numpy()
    interactions["endpoint_id"] = outlets.iloc[outlet_idx]["endpoint_key"].to_numpy()
    interactions["pr_date"] = prs.iloc[pr_idx]["post_date"].to_numpy()
    interactions["pr_title"] = prs.iloc[pr_idx]["title"].to_numpy()
    interactions["creator"] = prs.iloc[pr_idx]["creator"].to_numpy()
    interactions["pr_category"] = prs.iloc[pr_idx]["pr_category"].to_numpy()
    interactions["outlet_category"] = outlets.iloc[outlet_idx]["outlet_category"].to_numpy()
    interactions["region"] = outlets.iloc[outlet_idx]["region"].to_numpy()
    interactions["media_outlet"] = outlets.iloc[outlet_idx]["media_outlet"].to_numpy()
    interactions["category_match"] = (
        interactions["pr_category"].astype(str).to_numpy() == interactions["outlet_category"].astype(str).to_numpy()
    ).astype(float)

    for col in [
        "smoothed_ctr",
        "raw_ctr",
        "log_traffic",
        "log_views",
        "estimated_traffic",
        "estimated_views",
        "estimated_clicks",
        "publication_count",
        "pagerank",
    ]:
        interactions[col] = outlets.iloc[outlet_idx][col].to_numpy(dtype=float)

    dates = pd.to_datetime(interactions["pr_date"], errors="coerce")
    interactions["month"] = dates.dt.month.fillna(1).astype(int)
    interactions["month_sin"] = np.sin(2 * np.pi * interactions["month"] / 12.0)
    interactions["month_cos"] = np.cos(2 * np.pi * interactions["month"] / 12.0)
    return interactions


def add_relevance_scores(
    interactions: pd.DataFrame, prs: pd.DataFrame, outlets: pd.DataFrame, config: ResearchConfig
) -> tuple[pd.DataFrame, dict[str, str]]:
    pr_texts = prs["pr_text"].fillna("").astype(str).tolist()
    outlet_texts = outlets["profile_text"].fillna("").astype(str).tolist()
    pr_idx = interactions["pr_pos"].to_numpy(dtype=int)
    outlet_idx = interactions["outlet_pos"].to_numpy(dtype=int)

    interactions["keyword_score"] = keyword_similarity(pr_texts, outlet_texts, pr_idx, outlet_idx)

    vectorizer = TfidfVectorizerLite(max_features=config.tfidf_max_features, min_df=2)
    vectorizer.fit(pr_texts + outlet_texts)
    pr_tfidf = vectorizer.transform(pr_texts)
    outlet_tfidf = vectorizer.transform(outlet_texts)
    interactions["tfidf_similarity"] = cosine_for_pairs(pr_tfidf, outlet_tfidf, pr_idx, outlet_idx)

    encoder_specs = {
        "distilled_encoder": ("distilled-minilm-fallback", "word_bigram", max(96, config.hash_dim)),
        "llama_slm": ("llama-fallback", "hybrid", config.hash_dim),
        "ministral_slm": ("ministral-fallback", "word_bigram", config.hash_dim),
        "phi_slm": ("phi-fallback", "char_hybrid", config.hash_dim),
        "qwen_slm": ("qwen-fallback", "hybrid", config.hash_dim),
        "smollm_slm": ("smollm-fallback", "word_bigram", max(80, config.hash_dim // 2)),
    }
    embedding_modes = {}
    for column, (salt, mode, dim) in encoder_specs.items():
        pr_emb = hashed_embeddings(pr_texts, dim=dim, salt=salt, mode=mode)
        outlet_emb = hashed_embeddings(outlet_texts, dim=dim, salt=salt, mode=mode)
        interactions[column + "_score"] = cosine_for_pairs(pr_emb, outlet_emb, pr_idx, outlet_idx)
        embedding_modes[column] = "hash_fallback_no_local_transformer_runtime"

    return interactions, embedding_modes


def create_proxy_labels(interactions: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    rel = 0.28 * minmax(interactions["keyword_score"].to_numpy()) + 0.32 * minmax(
        interactions["tfidf_similarity"].to_numpy()
    )
    slm_cols = [c for c in interactions.columns if c.endswith("_slm_score")]
    if slm_cols:
        rel += 0.22 * minmax(interactions[slm_cols].mean(axis=1).to_numpy())
    rel += 0.18 * interactions["category_match"].to_numpy(dtype=float)

    ctr = minmax(interactions["smoothed_ctr"].to_numpy(dtype=float))
    authority = minmax(interactions["pagerank"].to_numpy(dtype=float))
    traffic = minmax(interactions["log_traffic"].to_numpy(dtype=float))
    noise = rng.normal(0, 0.025, size=len(interactions))
    latent = 0.50 * minmax(rel) + 0.27 * ctr + 0.16 * authority + 0.07 * traffic + noise
    interactions["proxy_relevance_score"] = latent
    interactions["proxy_ctr_probability"] = sigmoid(-3.0 + 5.0 * minmax(latent))

    clicked = np.zeros(len(interactions), dtype=int)
    for _, idx in interactions.groupby("pr_id", sort=False).indices.items():
        group_scores = latent[np.asarray(idx)]
        threshold = np.quantile(group_scores, max(0.0, min(1.0, 1.0 - config.proxy_positive_rate)))
        clicked[np.asarray(idx)] = (group_scores >= threshold).astype(int)
    interactions["clicked"] = clicked
    interactions["ctr_label"] = interactions["clicked"].astype(float)
    interactions["label_mode"] = "proxy_from_available_pr_titles_and_outlet_engagement"
    return interactions


def temporal_split(interactions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    pr_dates = pd.to_datetime(interactions[["pr_id", "pr_date"]].drop_duplicates("pr_id")["pr_date"])
    train_end = pr_dates.quantile(0.70)
    val_end = pr_dates.quantile(0.85)
    dates = pd.to_datetime(interactions["pr_date"])
    train = dates <= train_end
    val = (dates > train_end) & (dates <= val_end)
    test = dates > val_end
    if train.sum() == 0 or val.sum() == 0 or test.sum() == 0:
        order = np.arange(len(interactions))
        train = order < int(len(order) * 0.70)
        val = (order >= int(len(order) * 0.70)) & (order < int(len(order) * 0.85))
        test = order >= int(len(order) * 0.85)
    split_info = {
        "train_end": str(pd.Timestamp(train_end).date()),
        "validation_end": str(pd.Timestamp(val_end).date()),
        "protocol": "time_based_by_press_release_post_date",
    }
    return np.asarray(train), np.asarray(val), np.asarray(test), split_info


def hashed_categorical_matrix(frame: pd.DataFrame, columns: list[str], dim: int) -> np.ndarray:
    matrix = np.zeros((len(frame), dim), dtype=np.float32)
    for col in columns:
        values = frame[col].fillna("missing").astype(str).to_numpy()
        for i, value in enumerate(values):
            h = stable_hash(f"{col}={value}", salt="cat")
            matrix[i, h % dim] += 1.0
    return matrix


def build_model_matrix(
    interactions: pd.DataFrame,
    train_mask: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: ResearchConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    numeric = interactions[numeric_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean = numeric.iloc[train_mask].mean(axis=0)
    std = numeric.iloc[train_mask].std(axis=0).replace(0, 1.0)
    x_num = ((numeric - mean) / std).to_numpy(dtype=np.float32)
    x_cat = hashed_categorical_matrix(interactions, categorical_cols, config.categorical_hash_dim)
    x = np.hstack([x_num, x_cat]).astype(np.float32)
    spec = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_mean": mean.to_dict(),
        "numeric_std": std.to_dict(),
        "categorical_hash_dim": config.categorical_hash_dim,
    }
    return x, spec


class LogisticGD:
    def __init__(self, lr: float = 0.05, epochs: int = 100, l2: float = 1e-4, seed: int = 42):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.w: np.ndarray | None = None
        self.b = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticGD":
        rng = np.random.default_rng(self.seed)
        n, d = x.shape
        self.w = rng.normal(0, 0.01, size=d).astype(np.float32)
        self.b = 0.0
        for _ in range(self.epochs):
            logits = x.dot(self.w) + self.b
            p = sigmoid(logits)
            g = (p - y) / max(n, 1)
            grad_w = x.T.dot(g) + self.l2 * self.w
            grad_b = float(g.sum())
            self.w -= self.lr * grad_w.astype(np.float32)
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model must be fit before prediction.")
        return sigmoid(x.dot(self.w) + self.b)


class NumpyDeepFM:
    def __init__(
        self,
        n_features: int,
        factors: int = 8,
        hidden: int = 32,
        lr: float = 0.015,
        epochs: int = 12,
        batch_size: int = 2048,
        l2: float = 1e-5,
        seed: int = 42,
    ):
        self.n_features = n_features
        self.factors = factors
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, size=n_features).astype(np.float32)
        self.b = 0.0
        self.v = rng.normal(0, 0.01, size=(n_features, factors)).astype(np.float32)
        self.w1 = rng.normal(0, 0.02, size=(n_features, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = rng.normal(0, 0.02, size=hidden).astype(np.float32)
        self.b2 = 0.0

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        linear = x.dot(self.w) + self.b
        xv = x.dot(self.v)
        fm = 0.5 * ((xv * xv).sum(axis=1) - ((x * x).dot(self.v * self.v)).sum(axis=1))
        h_pre = x.dot(self.w1) + self.b1
        h = np.maximum(h_pre, 0)
        deep = h.dot(self.w2) + self.b2
        logits = linear + fm + deep
        return logits, {"xv": xv, "h_pre": h_pre, "h": h}

    def fit(self, x: np.ndarray, y: np.ndarray, val: tuple[np.ndarray, np.ndarray] | None = None) -> "NumpyDeepFM":
        rng = np.random.default_rng(self.seed)
        n = len(x)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = x[idx]
                yb = y[idx]
                logits, cache = self._forward(xb)
                p = sigmoid(logits)
                g = (p - yb).astype(np.float32) / max(len(idx), 1)

                grad_w = xb.T.dot(g) + self.l2 * self.w
                grad_b = float(g.sum())
                xv = cache["xv"]
                x2 = xb * xb
                grad_v = xb.T.dot(g[:, None] * xv) - self.v * (x2.T.dot(g)[:, None])
                grad_v += self.l2 * self.v

                h = cache["h"]
                h_pre = cache["h_pre"]
                grad_w2 = h.T.dot(g) + self.l2 * self.w2
                grad_b2 = float(g.sum())
                grad_h = g[:, None] * self.w2[None, :]
                grad_h_pre = grad_h * (h_pre > 0)
                grad_w1 = xb.T.dot(grad_h_pre) + self.l2 * self.w1
                grad_b1 = grad_h_pre.sum(axis=0)

                self.w -= self.lr * grad_w.astype(np.float32)
                self.b -= self.lr * grad_b
                self.v -= self.lr * grad_v.astype(np.float32)
                self.w2 -= self.lr * grad_w2.astype(np.float32)
                self.b2 -= self.lr * grad_b2
                self.w1 -= self.lr * grad_w1.astype(np.float32)
                self.b1 -= self.lr * grad_b1.astype(np.float32)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(x)
        return sigmoid(logits)


def platt_calibrate(p_val: np.ndarray, y_val: np.ndarray, p_target: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    eps = 1e-6
    x_val = np.log(np.clip(p_val, eps, 1 - eps) / np.clip(1 - p_val, eps, 1 - eps)).reshape(-1, 1)
    model = LogisticGD(lr=0.05, epochs=150, l2=1e-5, seed=7).fit(x_val.astype(np.float32), y_val.astype(float))
    x_target = np.log(np.clip(p_target, eps, 1 - eps) / np.clip(1 - p_target, eps, 1 - eps)).reshape(-1, 1)
    calibrated = model.predict_proba(x_target.astype(np.float32))
    return calibrated, {"platt_slope": float(model.w[0]), "platt_intercept": float(model.b)}


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    y = np.asarray(y_true).astype(float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true).astype(float)
    p = np.asarray(y_prob).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def prediction_metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | str]:
    return {
        "model": name,
        "auc": auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "ece_10bin": ece_score(y_true, y_prob),
        "mean_pred": float(np.mean(y_prob)),
        "positive_rate": float(np.mean(y_true)),
    }


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    rel = np.asarray(relevance)[:k]
    if len(rel) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum(rel * discounts))


def ranking_metrics(
    frame: pd.DataFrame,
    score_col: str,
    ks: tuple[int, ...],
    y_col: str = "clicked",
    group_col: str = "pr_id",
) -> dict[str, float | str]:
    metrics: dict[str, list[float]] = defaultdict(list)
    for _, group in frame.groupby(group_col, sort=False):
        ranked = group.sort_values(score_col, ascending=False)
        labels = ranked[y_col].to_numpy(dtype=float)
        positives = max(labels.sum(), 1.0)
        for k in ks:
            top = labels[:k]
            metrics[f"precision@{k}"].append(float(top.mean()) if len(top) else 0.0)
            metrics[f"recall@{k}"].append(float(top.sum() / positives))
            hits = 0.0
            ap = 0.0
            for rank, value in enumerate(top, start=1):
                if value > 0:
                    hits += 1.0
                    ap += hits / rank
            metrics[f"map@{k}"].append(float(ap / min(positives, k)))
            ideal = np.sort(group[y_col].to_numpy(dtype=float))[::-1]
            idcg = dcg_at_k(ideal, k)
            metrics[f"ndcg@{k}"].append(dcg_at_k(labels, k) / idcg if idcg > 0 else 0.0)
            metrics[f"expected_clicks@{k}"].append(float(top.sum()))
            metrics[f"observed_ctr@{k}"].append(float(top.mean()) if len(top) else 0.0)
    return {"ranker": score_col, **{key: float(np.mean(vals)) for key, vals in metrics.items()}}


def score_from_endpoint_prior(train_df: pd.DataFrame, target_df: pd.DataFrame) -> np.ndarray:
    global_rate = train_df["clicked"].mean()
    by_endpoint = train_df.groupby("endpoint_id")["clicked"].mean().to_dict()
    return target_df["endpoint_id"].map(by_endpoint).fillna(global_rate).to_numpy(dtype=float)


def choose_best_semantic(interactions: pd.DataFrame, val_mask: np.ndarray, ks: tuple[int, ...]) -> tuple[str, pd.DataFrame]:
    candidates = [
        "keyword_score",
        "tfidf_similarity",
        "distilled_encoder_score",
        "llama_slm_score",
        "ministral_slm_score",
        "phi_slm_score",
        "qwen_slm_score",
        "smollm_slm_score",
    ]
    rows = []
    val_df = interactions.loc[val_mask].copy()
    for col in candidates:
        metrics = ranking_metrics(val_df, col, ks)
        metrics["encoder"] = col
        rows.append(metrics)
    table = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False)
    return str(table.iloc[0]["encoder"]), table


def add_hybrid_scores(interactions: pd.DataFrame, val_mask: np.ndarray, rel_col: str) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = interactions.copy()
    rel = minmax(frame[rel_col].to_numpy())
    ctr = minmax(frame["deepfm_pred"].to_numpy())
    pr = minmax(frame["pagerank"].to_numpy())
    static = minmax(frame["log_traffic"].to_numpy())
    frame["static_tier_score"] = static
    frame["semantic_best_score"] = rel
    frame["ctr_only_score"] = ctr

    grids = []
    for rel_w in np.arange(0.2, 0.75, 0.15):
        for ctr_w in np.arange(0.2, 0.75, 0.15):
            if rel_w + ctr_w <= 1.0:
                pr_w = 1.0 - rel_w - ctr_w
                grids.append((float(rel_w), float(ctr_w), float(pr_w)))

    def pick(require_pr: bool) -> tuple[tuple[float, float, float], float]:
        best = ((0.45, 0.45, 0.10), -1.0)
        val_df = frame.loc[val_mask].copy()
        for weights in grids:
            if require_pr and weights[2] <= 1e-9:
                continue
            if not require_pr and weights[2] > 1e-9:
                continue
            score = weights[0] * rel + weights[1] * ctr + weights[2] * pr
            val_df["_candidate_score"] = score[val_mask]
            ndcg = ranking_metrics(val_df, "_candidate_score", (10,))["ndcg@10"]
            if ndcg > best[1]:
                best = (weights, float(ndcg))
        return best

    no_pr_weights, no_pr_ndcg = pick(require_pr=False)
    pr_weights, pr_ndcg = pick(require_pr=True)
    frame["hybrid_no_pagerank_score"] = no_pr_weights[0] * rel + no_pr_weights[1] * ctr + no_pr_weights[2] * pr
    frame["hybrid_pagerank_score"] = pr_weights[0] * rel + pr_weights[1] * ctr + pr_weights[2] * pr
    frame["hybrid_business_score"] = 0.35 * rel + 0.45 * ctr + 0.15 * pr + 0.05 * static
    details = {
        "best_semantic_column": rel_col,
        "hybrid_no_pagerank_weights": {
            "semantic": no_pr_weights[0],
            "ctr": no_pr_weights[1],
            "pagerank": no_pr_weights[2],
            "val_ndcg@10": no_pr_ndcg,
        },
        "hybrid_pagerank_weights": {
            "semantic": pr_weights[0],
            "ctr": pr_weights[1],
            "pagerank": pr_weights[2],
            "val_ndcg@10": pr_ndcg,
        },
    }
    return frame, details


def bootstrap_ndcg_difference(
    frame: pd.DataFrame, challenger: str, baseline: str, seed: int = 42, rounds: int = 200
) -> dict[str, float | str]:
    rng = np.random.default_rng(seed)
    groups = list(frame.groupby("pr_id", sort=False))
    if not groups:
        return {"comparison": f"{challenger} - {baseline}", "mean_delta_ndcg@10": float("nan")}
    deltas = []
    for _ in range(rounds):
        sampled = [groups[int(i)][1] for i in rng.integers(0, len(groups), size=len(groups))]
        sample = pd.concat(sampled, ignore_index=True)
        a = ranking_metrics(sample, challenger, (10,))["ndcg@10"]
        b = ranking_metrics(sample, baseline, (10,))["ndcg@10"]
        deltas.append(a - b)
    return {
        "comparison": f"{challenger} - {baseline}",
        "mean_delta_ndcg@10": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def limited_data_study(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for frac in [0.25, 0.50, 1.00]:
        n = max(50, int(len(x_train) * frac))
        idx = rng.choice(len(x_train), size=n, replace=False) if n < len(x_train) else np.arange(len(x_train))
        model = LogisticGD(lr=0.05, epochs=80, l2=1e-4, seed=seed).fit(x_train[idx], y_train[idx])
        pred = model.predict_proba(x_test)
        row = prediction_metrics(f"logistic_full_features_train_fraction_{frac:.2f}", y_test, pred)
        row["train_fraction"] = frac
        rows.append(row)
    return pd.DataFrame(rows)


def write_model_card(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Model Card - Context-Aware PR Distribution Recommender",
        "",
        "## Purpose",
        "Offline research prototype for ranking media endpoints for press releases using lexical baselines, compact embedding fallbacks, CTR prediction, and PageRank-enhanced hybrid scoring.",
        "",
        "## Data Mode",
        str(summary.get("label_mode", "unknown")),
        "",
        "## Selected Components",
        f"- Best semantic signal: {summary.get('best_semantic_column')}",
        f"- Hybrid weights: {summary.get('hybrid_pagerank_weights')}",
        f"- Split protocol: {summary.get('split_info')}",
        "",
        "## Current Limitation",
        "The workspace does not contain a true PR-endpoint interaction log with per-PR exposure, impression, click, or propensity columns. The notebook therefore runs a proxy backtesting mode using PR titles and outlet-level engagement summaries. Replace the proxy builder with real logs before treating the results as final thesis evidence.",
        "",
        "## Operational Notes",
        "The implementation is dependency-light and uses pandas/numpy only. If local transformer or sentence-transformer packages are added later, the notebook can be extended to swap hash fallback encoders for cached SLM embeddings.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_pipeline(root: Path, config: ResearchConfig | None = None) -> dict[str, object]:
    config = config or ResearchConfig.from_env()
    rng = set_seed(config.seed)
    start = time.perf_counter()
    out_dir = root / config.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    category_terms = parse_category_terms(root / "Press Release AI Distributor.txt")
    posts_all = load_pr_posts(root, config)
    posts_all["pr_category"] = posts_all["pr_text"].map(lambda txt: infer_category(txt, category_terms))
    prs = select_pr_sample(posts_all, config)

    outlets = load_outlets(root, category_terms, config)
    outlets["pagerank"] = build_outlet_pagerank(outlets)
    interactions = build_interactions(prs, outlets, config)
    interactions, embedding_modes = add_relevance_scores(interactions, prs, outlets, config)
    interactions = create_proxy_labels(interactions, config)

    train_mask, val_mask, test_mask, split_info = temporal_split(interactions)
    best_semantic, encoder_val = choose_best_semantic(interactions, val_mask, config.top_ks)

    numeric_full = [
        "keyword_score",
        "tfidf_similarity",
        "distilled_encoder_score",
        "llama_slm_score",
        "ministral_slm_score",
        "phi_slm_score",
        "qwen_slm_score",
        "smollm_slm_score",
        "category_match",
        "smoothed_ctr",
        "raw_ctr",
        "log_traffic",
        "log_views",
        "publication_count",
        "pagerank",
        "month_sin",
        "month_cos",
    ]
    numeric_meta = [
        "category_match",
        "smoothed_ctr",
        "raw_ctr",
        "log_traffic",
        "log_views",
        "publication_count",
        "pagerank",
        "month_sin",
        "month_cos",
    ]
    categorical_cols = ["pr_category", "outlet_category", "region", "media_outlet", "creator"]

    x_full, feature_spec = build_model_matrix(interactions, train_mask, numeric_full, categorical_cols, config)
    x_meta, _ = build_model_matrix(interactions, train_mask, numeric_meta, categorical_cols, config)
    y = interactions["clicked"].to_numpy(dtype=float)

    train_df = interactions.loc[train_mask].copy()
    val_df = interactions.loc[val_mask].copy()
    test_df = interactions.loc[test_mask].copy()
    interactions.loc[val_mask, "historical_prior_pred"] = score_from_endpoint_prior(train_df, val_df)
    interactions.loc[test_mask, "historical_prior_pred"] = score_from_endpoint_prior(train_df, test_df)
    global_train_rate = train_df["clicked"].mean()
    interactions["historical_prior_pred"] = interactions["historical_prior_pred"].fillna(global_train_rate)

    meta_lr = LogisticGD(lr=0.04, epochs=config.logistic_epochs, l2=1e-4, seed=config.seed)
    meta_lr.fit(x_meta[train_mask], y[train_mask])
    interactions["metadata_lr_pred"] = meta_lr.predict_proba(x_meta)

    deepfm = NumpyDeepFM(
        n_features=x_full.shape[1],
        factors=config.deepfm_factors,
        hidden=config.deepfm_hidden,
        epochs=config.deepfm_epochs,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    deepfm.fit(x_full[train_mask], y[train_mask], val=(x_full[val_mask], y[val_mask]))
    interactions["deepfm_pred_raw"] = deepfm.predict_proba(x_full)
    calibrated_val, calibration = platt_calibrate(
        interactions.loc[val_mask, "deepfm_pred_raw"].to_numpy(),
        y[val_mask],
        interactions.loc[val_mask, "deepfm_pred_raw"].to_numpy(),
    )
    calibrated_all, _ = platt_calibrate(
        interactions.loc[val_mask, "deepfm_pred_raw"].to_numpy(),
        y[val_mask],
        interactions["deepfm_pred_raw"].to_numpy(),
    )
    interactions["deepfm_pred"] = calibrated_all
    interactions.loc[val_mask, "deepfm_pred"] = calibrated_val

    interactions, hybrid_details = add_hybrid_scores(interactions, val_mask, best_semantic)
    test_df = interactions.loc[test_mask].copy()

    prediction_rows = [
        prediction_metrics("historical_prior", y[test_mask], interactions.loc[test_mask, "historical_prior_pred"].to_numpy()),
        prediction_metrics("metadata_logistic", y[test_mask], interactions.loc[test_mask, "metadata_lr_pred"].to_numpy()),
        prediction_metrics("deepfm_calibrated", y[test_mask], interactions.loc[test_mask, "deepfm_pred"].to_numpy()),
    ]
    prediction_table = pd.DataFrame(prediction_rows).sort_values("log_loss")

    rankers = [
        "static_tier_score",
        "keyword_score",
        "tfidf_similarity",
        "distilled_encoder_score",
        "llama_slm_score",
        "ministral_slm_score",
        "phi_slm_score",
        "qwen_slm_score",
        "smollm_slm_score",
        "historical_prior_pred",
        "metadata_lr_pred",
        "deepfm_pred",
        "hybrid_no_pagerank_score",
        "hybrid_pagerank_score",
        "hybrid_business_score",
    ]
    ranking_table = pd.DataFrame([ranking_metrics(test_df, col, config.top_ks) for col in rankers])
    ranking_table = ranking_table.sort_values("ndcg@10", ascending=False)

    encoder_test_rows = []
    for col in [
        "keyword_score",
        "tfidf_similarity",
        "distilled_encoder_score",
        "llama_slm_score",
        "ministral_slm_score",
        "phi_slm_score",
        "qwen_slm_score",
        "smollm_slm_score",
    ]:
        row = ranking_metrics(test_df, col, config.top_ks)
        row["encoder"] = col
        row["embedding_mode"] = embedding_modes.get(col.replace("_score", ""), "lexical_or_tfidf")
        encoder_test_rows.append(row)
    encoder_table = pd.DataFrame(encoder_test_rows).sort_values("ndcg@10", ascending=False)

    ablation_rows = []
    for col in ["semantic_best_score", "ctr_only_score", "hybrid_no_pagerank_score", "hybrid_pagerank_score"]:
        row = ranking_metrics(test_df, col, config.top_ks)
        row["ablation_family"] = "ranking"
        ablation_rows.append(row)
    for model_name, pred_col in [
        ("historical_prior", "historical_prior_pred"),
        ("metadata_only", "metadata_lr_pred"),
        ("slm_structured_deepfm", "deepfm_pred"),
    ]:
        row = prediction_metrics(model_name, y[test_mask], interactions.loc[test_mask, pred_col].to_numpy())
        row["ablation_family"] = "predictor"
        row["ranker"] = pred_col
        ablation_rows.append(row)
    ablation_table = pd.DataFrame(ablation_rows)

    robustness_rows = [
        bootstrap_ndcg_difference(test_df, "hybrid_pagerank_score", "static_tier_score", seed=config.seed, rounds=200)
    ]
    low_exposure_mask = test_df["estimated_views"] <= test_df["estimated_views"].quantile(0.25)
    if low_exposure_mask.any():
        cold = ranking_metrics(test_df.loc[low_exposure_mask].copy(), "hybrid_pagerank_score", config.top_ks)
        cold["comparison"] = "cold_start_low_exposure_slice"
        robustness_rows.append(cold)
    robustness_table = pd.DataFrame(robustness_rows)
    limited_table = limited_data_study(x_full[train_mask], y[train_mask], x_full[test_mask], y[test_mask], config.seed)

    sample_pr_id = str(test_df["pr_id"].iloc[0])
    recommendations = (
        test_df[test_df["pr_id"].astype(str).eq(sample_pr_id)]
        .sort_values("hybrid_pagerank_score", ascending=False)
        .head(20)[
            [
                "pr_id",
                "pr_title",
                "media_outlet",
                "region",
                "outlet_category",
                "hybrid_pagerank_score",
                "deepfm_pred",
                best_semantic,
                "pagerank",
                "clicked",
            ]
        ]
        .copy()
    )

    data_profile = {
        "posts_loaded": int(len(posts_all)),
        "prs_sampled": int(len(prs)),
        "outlets_loaded": int(len(outlets)),
        "interactions": int(len(interactions)),
        "positive_rate": float(interactions["clicked"].mean()),
        "label_mode": str(interactions["label_mode"].iloc[0]),
        "split_info": split_info,
        "config": asdict(config),
        "runtime_seconds": float(time.perf_counter() - start),
    }
    summary = {
        **data_profile,
        **hybrid_details,
        "calibration": calibration,
        "embedding_modes": embedding_modes,
    }

    prediction_table.to_csv(out_dir / "prediction_metrics.csv", index=False)
    ranking_table.to_csv(out_dir / "benchmark_matrix.csv", index=False)
    encoder_table.to_csv(out_dir / "encoder_comparison.csv", index=False)
    encoder_val.to_csv(out_dir / "encoder_validation_selection.csv", index=False)
    ablation_table.to_csv(out_dir / "ablation_matrix.csv", index=False)
    robustness_table.to_csv(out_dir / "robustness_checks.csv", index=False)
    limited_table.to_csv(out_dir / "limited_data_study.csv", index=False)
    recommendations.to_csv(out_dir / "recommendations_sample.csv", index=False)
    (out_dir / "data_profile.json").write_text(json.dumps(data_profile, indent=2), encoding="utf-8")
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_model_card(out_dir / "model_card.md", summary)

    gc.collect()
    return {
        "config": config,
        "data_profile": data_profile,
        "summary": summary,
        "prediction_metrics": prediction_table,
        "ranking_metrics": ranking_table,
        "encoder_comparison": encoder_table,
        "ablation_matrix": ablation_table,
        "robustness_checks": robustness_table,
        "limited_data_study": limited_table,
        "recommendations_sample": recommendations,
        "interactions": interactions,
        "prs": prs,
        "outlets": outlets,
        "output_dir": out_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PR distribution recommender research pipeline.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--fast", action="store_true", help="Use a smaller sample for a quick smoke test.")
    args = parser.parse_args()
    config = ResearchConfig.from_env()
    if args.fast:
        os.environ["PR_RESEARCH_FAST"] = "1"
        config = ResearchConfig.from_env()
    results = evaluate_pipeline(args.root, config)
    print("Research pipeline complete.")
    print(json.dumps(results["data_profile"], indent=2, default=str))
    print("\nTop benchmark rows:")
    print(results["ranking_metrics"].head(8).to_string(index=False))
    print("\nPrediction metrics:")
    print(results["prediction_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()

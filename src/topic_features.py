from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from .config import TOPIC_PARAMETERS


def topic_quality(model, frame: pd.DataFrame, embeddings: np.ndarray, train_mask) -> tuple[pd.DataFrame, dict]:
    info = model.get_topic_info().copy()
    rows, word_pool = [], []
    for topic_id in info.loc[info["Topic"] != -1, "Topic"].astype(int):
        words = [w for w, _ in model.get_topic(topic_id)[:TOPIC_PARAMETERS["top_n_words"]]]
        word_pool.extend(words)
        rows.append({"topic_id": topic_id, "topic_size": int((frame["topic_id"] == topic_id).sum()), "top_words": ", ".join(words)})
    summary = pd.DataFrame(rows)
    assigned = train_mask.to_numpy() & (frame["topic_id"].to_numpy() >= 0)
    labels = frame.loc[assigned, "topic_id"].to_numpy()
    silhouette = np.nan
    if assigned.sum() > 2 and 1 < len(np.unique(labels)) < assigned.sum():
        silhouette = float(silhouette_score(embeddings[assigned], labels, metric="cosine"))
    n_topics = len(summary)
    warnings = []
    if n_topics <= 3:
        warnings.append("Topic granularity terlalu rendah; eksperimen prediksi dapat menjadi tidak stabil.")
    outlier_ratio = float((frame["topic_id"] == -1).mean())
    if outlier_ratio > 0.50:
        warnings.append("Outlier ratio melebihi 50%; cakupan topic time series terbatas.")
    quality = {
        "number_of_topics": n_topics,
        "outlier_ratio": outlier_ratio,
        "topic_diversity": len(set(word_pool)) / max(1, len(word_pool)),
        "silhouette_score": silhouette,
        "coherence_score": None,
        "coherence_status": "not evaluated: gensim-compatible token corpus not configured",
        "warning": " ".join(warnings),
    }
    return summary, quality


def topic_time_series(frame: pd.DataFrame) -> pd.DataFrame:
    docs = frame.loc[frame["topic_id"] >= 0].copy()
    if docs.empty:
        raise ValueError("BERTopic tidak menghasilkan dokumen non-outlier; pipeline dihentikan.")
    dates = pd.date_range(docs["date"].min(), docs["date"].max(), freq="D")
    topics = sorted(docs["topic_id"].unique())
    grid = pd.MultiIndex.from_product([topics, dates], names=["topic_id", "date"]).to_frame(index=False)
    agg = docs.groupby(["topic_id", "date"]).agg(
        topic_volume=("article_key", "size"),
        topic_probability=("topic_probability", "mean"),
        topic_probability_std=("topic_probability", "std"),
    ).reset_index()
    out = grid.merge(agg, on=["topic_id", "date"], how="left")
    out["topic_volume"] = out["topic_volume"].fillna(0).astype(int)
    return out.sort_values(["topic_id", "date"]).reset_index(drop=True)


def semantic_features(frame: pd.DataFrame, embeddings: np.ndarray, topic_series: pd.DataFrame,
                      train_cutoff) -> pd.DataFrame:
    rows = []
    centroids = {}
    train = frame["date"].lt(pd.Timestamp(train_cutoff)) & frame["topic_id"].ge(0)
    for topic in sorted(frame.loc[train, "topic_id"].unique()):
        indices = frame.index[train & frame["topic_id"].eq(topic)].to_numpy()
        center = embeddings[indices].mean(axis=0)
        centroids[int(topic)] = center / max(np.linalg.norm(center), 1e-12)
    previous = {}
    valid_docs = frame.loc[frame["topic_id"] >= 0]
    for (topic, date), group in valid_docs.sort_values("date").groupby(["topic_id", "date"], sort=True):
        indices = group.index.to_numpy()
        vectors = embeddings[indices]
        center = vectors.mean(axis=0)
        center = center / max(np.linalg.norm(center), 1e-12)
        reference = centroids.get(int(topic))
        similarities = vectors @ reference if reference is not None else np.full(len(vectors), np.nan)
        prev = previous.get(int(topic))
        drift = np.nan if prev is None else 1.0 - float(center @ prev)
        rows.append({
            "topic_id": int(topic), "date": date,
            "semantic_similarity_mean": float(np.nanmean(similarities)),
            "semantic_similarity_std": float(np.nanstd(similarities)),
            "semantic_drift": drift,
            "topic_probability_mean": float(group["topic_probability"].mean()),
            "topic_probability_std": float(group["topic_probability"].std(ddof=0)),
        })
        previous[int(topic)] = center
    out = topic_series[["topic_id", "date"]].merge(pd.DataFrame(rows), on=["topic_id", "date"], how="left")
    out["topic_probability_change"] = out.groupby("topic_id")["topic_probability_mean"].diff()
    out["similarity_change"] = out.groupby("topic_id")["semantic_similarity_mean"].diff()
    return out

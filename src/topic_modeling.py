from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from .config import MODEL_DIR, RANDOM_SEED, TOPIC_PARAMETERS


INDONESIAN_STOPWORDS = ["yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "ini", "itu", "adalah", "dalam", "atau", "juga", "akan", "tidak", "sebagai", "oleh", "karena", "telah", "saat", "lebih", "antara", "setelah", "hingga", "kata"]


def fit_topic_model(frame: pd.DataFrame, embeddings: np.ndarray, train_cutoff):
    train_mask = frame["date"] < pd.Timestamp(train_cutoff)
    if train_mask.sum() < max(30, TOPIC_PARAMETERS["min_cluster_size"] * 3):
        raise ValueError(f"Dokumen training tidak cukup untuk BERTopic: {int(train_mask.sum())}.")
    umap_model = UMAP(n_neighbors=min(TOPIC_PARAMETERS["n_neighbors"], int(train_mask.sum()) - 1),
                      n_components=TOPIC_PARAMETERS["n_components"], metric="cosine",
                      min_dist=0.0, random_state=RANDOM_SEED)
    hdbscan_model = HDBSCAN(min_cluster_size=TOPIC_PARAMETERS["min_cluster_size"],
                            metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2, stop_words=INDONESIAN_STOPWORDS)
    model = BERTopic(embedding_model=None, umap_model=umap_model, hdbscan_model=hdbscan_model,
                     vectorizer_model=vectorizer, top_n_words=TOPIC_PARAMETERS["top_n_words"],
                     calculate_probabilities=False, verbose=True)
    train_idx = np.flatnonzero(train_mask.to_numpy())
    held_idx = np.flatnonzero((~train_mask).to_numpy())
    train_topics, train_probs = model.fit_transform(frame.loc[train_mask, "document"].tolist(), embeddings[train_idx])
    topics = np.full(len(frame), -1, dtype=int)
    probabilities = np.zeros(len(frame), dtype=float)
    topics[train_idx] = np.asarray(train_topics)
    probabilities[train_idx] = np.asarray(train_probs, dtype=float)
    if len(held_idx):
        future_topics, future_probs = model.transform(frame.loc[~train_mask, "document"].tolist(), embeddings[held_idx])
        topics[held_idx] = np.asarray(future_topics)
        probabilities[held_idx] = np.asarray(future_probs, dtype=float)
    model.save(MODEL_DIR / "bertopic_model", serialization="safetensors", save_ctfidf=True)
    metadata = {"fit_scope": "train_only", "train_cutoff": str(pd.Timestamp(train_cutoff)),
                "training_documents": int(train_mask.sum()), **TOPIC_PARAMETERS}
    return model, topics, probabilities, metadata, train_mask

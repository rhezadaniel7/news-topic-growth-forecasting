from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

from .step7_embeddings import MODEL_NAME, build_documents, file_sha256


RANDOM_SEED = 42
TRAIN_START = pd.Timestamp("2024-07-31")
TRAIN_END = pd.Timestamp("2025-04-26")
VALIDATION_START = pd.Timestamp("2025-04-27")
VALIDATION_END = pd.Timestamp("2025-07-25")
TEST_START = pd.Timestamp("2025-07-26")
TEST_END = pd.Timestamp("2025-10-23")
EXPECTED_SPLIT_COUNTS = {"train": 33899, "validation": 11243, "test": 22018}
UMAP_PARAMS = {
    "n_neighbors": 15,
    "n_components": 5,
    "metric": "cosine",
    "min_dist": 0.0,
    "random_state": RANDOM_SEED,
    "transform_seed": RANDOM_SEED,
    "low_memory": True,
    "n_jobs": 1,
}
CONFIGURATIONS = {
    "A": {"min_topic_size": 15, "min_cluster_size": 15},
    "B": {"min_topic_size": 25, "min_cluster_size": 25},
    "C": {"min_topic_size": 50, "min_cluster_size": 50},
}
VECTORIZER_PARAMS = {"ngram_range": (1, 2), "min_df": 5}
TOP_N_WORDS = 10
SILHOUETTE_SAMPLE_SIZE = 5000
MIN_TOPIC_HISTORY_ACTIVE_DAYS = 14
STOPWORD_SOURCE = "PySastrawi 1.0.1 StopWordRemoverFactory (https://github.com/har07/sastrawi)"


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def split_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(frame["date"]).dt.normalize()
    masks = {
        "train": dates.between(TRAIN_START, TRAIN_END, inclusive="both").to_numpy(),
        "validation": dates.between(VALIDATION_START, VALIDATION_END, inclusive="both").to_numpy(),
        "test": dates.between(TEST_START, TEST_END, inclusive="both").to_numpy(),
    }
    coverage = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(coverage == 1):
        raise RuntimeError("Setiap dokumen harus masuk tepat satu split temporal.")
    actual = {name: int(mask.sum()) for name, mask in masks.items()}
    if actual != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"Split berubah: actual={actual}, expected={EXPECTED_SPLIT_COUNTS}")
    return masks


def load_stopwords() -> list[str]:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    return sorted(set(word.casefold() for word in StopWordRemoverFactory().get_stop_words()))


def npmi_coherence(model: BERTopic, documents: list[str], top_n: int = TOP_N_WORDS) -> tuple[float, dict[int, float]]:
    topics = sorted(topic for topic in model.get_topics() if topic != -1)
    topic_words = {
        topic: [word for word, _ in model.get_topic(topic)[:top_n]] for topic in topics
    }
    feature_names = model.vectorizer_model.get_feature_names_out()
    feature_lookup = {word: idx for idx, word in enumerate(feature_names)}
    unique_words = sorted({word for words in topic_words.values() for word in words if word in feature_lookup})
    if len(unique_words) < 2:
        return float("nan"), {}

    matrix = model.vectorizer_model.transform(documents)
    selected_ids = np.array([feature_lookup[word] for word in unique_words], dtype=np.int64)
    binary = matrix[:, selected_ids].astype(bool).astype(np.int32)
    document_frequency = np.asarray(binary.sum(axis=0)).ravel().astype(np.float64)
    cooccurrence = (binary.T @ binary).toarray().astype(np.float64)
    word_position = {word: idx for idx, word in enumerate(unique_words)}
    n_docs = float(len(documents))
    epsilon_probability = 1.0 / n_docs

    per_topic = {}
    for topic, words in topic_words.items():
        positions = [word_position[word] for word in words if word in word_position]
        scores = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                left, right = positions[i], positions[j]
                p_left = max(document_frequency[left] / n_docs, epsilon_probability)
                p_right = max(document_frequency[right] / n_docs, epsilon_probability)
                p_pair = max(cooccurrence[left, right] / n_docs, epsilon_probability)
                pmi = math.log(p_pair / (p_left * p_right))
                denominator = -math.log(p_pair)
                scores.append(pmi / denominator if denominator > 0 else 0.0)
        per_topic[topic] = float(np.mean(scores)) if scores else float("nan")
    valid = [value for value in per_topic.values() if np.isfinite(value)]
    return (float(np.mean(valid)) if valid else float("nan")), per_topic


def topic_diversity(model: BERTopic, top_n: int = TOP_N_WORDS) -> float:
    topics = [topic for topic in model.get_topics() if topic != -1]
    words = [word for topic in topics for word, _ in model.get_topic(topic)[:top_n]]
    return len(set(words)) / len(words) if words else float("nan")


def topic_summary(model: BERTopic, labels: np.ndarray, config_name: str) -> pd.DataFrame:
    counts = pd.Series(labels[labels != -1]).value_counts().sort_index()
    representatives = model.get_representative_docs()
    rows = []
    for topic, size in counts.items():
        words = [word for word, _ in model.get_topic(int(topic))[:TOP_N_WORDS]]
        docs = representatives.get(int(topic), []) if isinstance(representatives, dict) else []
        rows.append(
            {
                "configuration": config_name,
                "topic_id": int(topic),
                "topic_size": int(size),
                "top_words": " | ".join(words),
                "representative_document_1": docs[0] if len(docs) > 0 else "",
                "representative_document_2": docs[1] if len(docs) > 1 else "",
                "representative_document_3": docs[2] if len(docs) > 2 else "",
            }
        )
    return pd.DataFrame(rows).sort_values("topic_size", ascending=False).reset_index(drop=True)


def calculate_metrics(
    model: BERTopic,
    labels: np.ndarray,
    reduced_embeddings: np.ndarray,
    documents: list[str],
    config_name: str,
    elapsed_seconds: float,
) -> tuple[dict, pd.DataFrame]:
    non_outlier = labels != -1
    topic_labels = labels[non_outlier]
    unique_topics = np.unique(topic_labels)
    outlier_count = int((~non_outlier).sum())
    counts = pd.Series(topic_labels).value_counts()

    if len(unique_topics) >= 2 and non_outlier.sum() > len(unique_topics):
        sample_size = min(SILHOUETTE_SAMPLE_SIZE, int(non_outlier.sum()))
        silhouette = float(
            silhouette_score(
                reduced_embeddings[non_outlier],
                topic_labels,
                metric="euclidean",
                sample_size=sample_size if sample_size < int(non_outlier.sum()) else None,
                random_state=RANDOM_SEED,
            )
        )
    else:
        sample_size = 0
        silhouette = float("nan")

    coherence, per_topic_coherence = npmi_coherence(model, documents)
    summary = topic_summary(model, labels, config_name)
    summary["coherence_npmi"] = summary["topic_id"].map(per_topic_coherence)
    metrics = {
        "configuration": config_name,
        "number_of_topics": int(len(unique_topics)),
        "outlier_count": outlier_count,
        "outlier_ratio": outlier_count / len(labels),
        "topic_diversity": topic_diversity(model),
        "coherence_npmi": coherence,
        "silhouette": silhouette,
        "silhouette_sample_size": sample_size,
        "largest_topic_size": int(counts.max()) if not counts.empty else 0,
        "largest_topic_ratio_non_outlier": float(counts.max() / counts.sum()) if not counts.empty else float("nan"),
        "median_topic_size": float(counts.median()) if not counts.empty else float("nan"),
        "smallest_topic_size": int(counts.min()) if not counts.empty else 0,
        "fit_seconds": elapsed_seconds,
    }
    return metrics, summary


def build_or_load_umap(
    root: Path,
    train_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    embedding_hash: str,
) -> tuple[UMAP, np.ndarray, np.ndarray, np.ndarray, dict]:
    intermediate = root / "data" / "intermediate"
    model_path = intermediate / "umap_train_only.joblib"
    train_path = intermediate / "umap_train.npy"
    validation_path = intermediate / "umap_validation.npy"
    test_path = intermediate / "umap_test.npy"
    metadata_path = intermediate / "umap_metadata.json"
    expected = {
        "embedding_hash": embedding_hash,
        "train_rows": int(train_embeddings.shape[0]),
        "validation_rows": int(validation_embeddings.shape[0]),
        "test_rows": int(test_embeddings.shape[0]),
        "parameters": UMAP_PARAMS,
    }
    if all(path.exists() for path in [model_path, train_path, validation_path, test_path, metadata_path]):
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in expected.items()):
            return (
                joblib.load(model_path),
                np.load(train_path, mmap_mode="r"),
                np.load(validation_path, mmap_mode="r"),
                np.load(test_path, mmap_mode="r"),
                existing | {"cache_hit": True},
            )

    from umap import UMAP

    started = time.perf_counter()
    reducer = UMAP(**UMAP_PARAMS)
    reduced_train = reducer.fit_transform(train_embeddings)
    reduced_validation = reducer.transform(validation_embeddings)
    reduced_test = reducer.transform(test_embeddings)
    np.save(train_path, reduced_train.astype(np.float32))
    np.save(validation_path, reduced_validation.astype(np.float32))
    np.save(test_path, reduced_test.astype(np.float32))
    joblib.dump(reducer, model_path, compress=3)
    metadata = expected | {
        "fit_scope": "train_only",
        "processing_seconds": time.perf_counter() - started,
        "cache_hit": False,
    }
    write_json(metadata_path, metadata)
    return reducer, reduced_train, reduced_validation, reduced_test, metadata


def run_candidates(root: Path) -> None:
    processed_path = root / "data" / "processed" / "news_cleaned.parquet"
    embedding_path = root / "data" / "intermediate" / "embeddings.npy"
    embedding_metadata_path = root / "data" / "intermediate" / "embedding_metadata.json"
    tables = root / "results" / "tables"
    models_dir = root / "models" / "bertopic_candidates"
    tables.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(processed_path, columns=["Judul", "Content", "date", "source", "Link"])
    masks = split_masks(frame)
    documents = build_documents(frame)
    embeddings = np.load(embedding_path, mmap_mode="r")
    embedding_metadata = json.loads(embedding_metadata_path.read_text(encoding="utf-8"))
    if tuple(embeddings.shape) != (len(frame), embedding_metadata["embedding_dimension"]):
        raise RuntimeError("Shape embedding tidak sesuai dataset/metadata.")
    if embedding_metadata["dataset_hash"] != file_sha256(processed_path):
        raise RuntimeError("Hash dataset embedding tidak sesuai news_cleaned.parquet.")

    train_idx = np.flatnonzero(masks["train"])
    validation_idx = np.flatnonzero(masks["validation"])
    test_idx = np.flatnonzero(masks["test"])
    train_embeddings = np.asarray(embeddings[train_idx], dtype=np.float32)
    validation_embeddings = np.asarray(embeddings[validation_idx], dtype=np.float32)
    test_embeddings = np.asarray(embeddings[test_idx], dtype=np.float32)
    embedding_hash = file_sha256(embedding_path)
    _, reduced_train, _, _, umap_metadata = build_or_load_umap(
        root, train_embeddings, validation_embeddings, test_embeddings, embedding_hash
    )
    del train_embeddings, validation_embeddings, test_embeddings

    train_documents = documents.iloc[train_idx].tolist()
    stopwords = load_stopwords()
    import psutil
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    from hdbscan import HDBSCAN

    comparison_rows = []
    process = psutil.Process()
    for name, params in CONFIGURATIONS.items():
        print(f"BERTopic configuration {name} mulai: {params}", flush=True)
        started = time.perf_counter()
        clusterer = HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        vectorizer = CountVectorizer(
            ngram_range=VECTORIZER_PARAMS["ngram_range"],
            min_df=VECTORIZER_PARAMS["min_df"],
            stop_words=stopwords,
        )
        model = BERTopic(
            embedding_model=None,
            umap_model=BaseDimensionalityReduction(),
            hdbscan_model=clusterer,
            vectorizer_model=vectorizer,
            min_topic_size=params["min_topic_size"],
            top_n_words=TOP_N_WORDS,
            calculate_probabilities=False,
            verbose=True,
        )
        labels, probabilities = model.fit_transform(train_documents, np.asarray(reduced_train))
        labels = np.asarray(labels, dtype=np.int32)
        probabilities = np.asarray(probabilities, dtype=np.float32)
        elapsed = time.perf_counter() - started
        metrics, summary = calculate_metrics(
            model,
            labels,
            np.asarray(reduced_train),
            train_documents,
            name,
            elapsed,
        )
        metrics.update(params)
        metrics["peak_process_rss_bytes"] = process.memory_info().rss
        comparison_rows.append(metrics)
        summary.to_csv(tables / f"topic_summary_config_{name}.csv", index=False, encoding="utf-8-sig")
        np.save(models_dir / f"train_topics_config_{name}.npy", labels)
        np.save(models_dir / f"train_probabilities_config_{name}.npy", probabilities)
        joblib.dump(model, models_dir / f"bertopic_config_{name}.joblib", compress=3)
        write_json(models_dir / f"metrics_config_{name}.json", metrics)
        print(f"BERTopic configuration {name} selesai: {metrics}", flush=True)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(tables / "topic_model_comparison.csv", index=False)
    run_metadata = {
        "status": "CANDIDATES_COMPLETE_SELECTION_PENDING",
        "random_seed": RANDOM_SEED,
        "dataset_path": str(processed_path.resolve()),
        "dataset_hash": embedding_metadata["dataset_hash"],
        "embedding_path": str(embedding_path.resolve()),
        "embedding_hash": embedding_hash,
        "embedding_model": MODEL_NAME,
        "fit_scope": "training_only",
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "train_period": [str(TRAIN_START.date()), str(TRAIN_END.date())],
        "validation_period": [str(VALIDATION_START.date()), str(VALIDATION_END.date())],
        "test_period": [str(TEST_START.date()), str(TEST_END.date())],
        "umap": umap_metadata,
        "hdbscan_common": {
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True,
        },
        "vectorizer": {
            "ngram_range": [1, 2],
            "min_df": 5,
            "stopword_count": len(stopwords),
            "stopword_source": STOPWORD_SOURCE,
        },
        "bertopic": {"top_n_words": TOP_N_WORDS, "calculate_probabilities": False},
        "configurations": CONFIGURATIONS,
        "coherence": "mean normalized pointwise mutual information (NPMI) over top-10 word pairs; all training documents; zero-pair smoothing=1/N",
        "silhouette": {
            "space": "5-dimensional train-only UMAP space",
            "metric": "euclidean",
            "maximum_reproducible_sample": SILHOUETTE_SAMPLE_SIZE,
        },
        "software": {
            "python": platform.python_version(),
            "bertopic": __import__("bertopic").__version__,
            "umap_learn": __import__("umap").__version__,
            "hdbscan": importlib.metadata.version("hdbscan"),
        },
    }
    write_json(root / "results" / "bertopic_run_metadata.json", run_metadata)
    print(comparison.to_string(index=False), flush=True)


def main() -> None:
    run_candidates(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()

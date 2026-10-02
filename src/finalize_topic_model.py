from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS

from .step7_embeddings import build_documents
from .step8_9_topic_modeling import (
    CONFIGURATIONS,
    EXPECTED_SPLIT_COUNTS,
    MIN_TOPIC_HISTORY_ACTIVE_DAYS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    file_sha256,
    split_masks,
    write_json,
)


def make_topic_distribution(assignments: pd.DataFrame) -> pd.DataFrame:
    counts = (
        assignments.groupby(["split", "topic_id"], observed=True)
        .size()
        .rename("article_count")
        .reset_index()
    )
    totals = counts.groupby("split")["article_count"].transform("sum")
    counts["percentage"] = 100.0 * counts["article_count"] / totals
    return counts.sort_values(["split", "article_count"], ascending=[True, False]).reset_index(drop=True)


def make_daily_volume(assignments: pd.DataFrame, topics: list[int]) -> pd.DataFrame:
    all_dates = pd.date_range(TRAIN_START, TEST_END, freq="D")
    grid = pd.MultiIndex.from_product([all_dates, topics], names=["date", "topic_id"])
    observed = (
        assignments.loc[assignments["topic_id"].ne(-1)]
        .groupby(["date", "topic_id"], observed=True)
        .size()
        .rename("volume")
    )
    return observed.reindex(grid, fill_value=0).reset_index()


def make_topic_history(assignments: pd.DataFrame, daily: pd.DataFrame, topics: list[int]) -> pd.DataFrame:
    non_outlier = assignments.loc[assignments["topic_id"].ne(-1)].copy()
    rows = []
    for topic in topics:
        topic_docs = non_outlier.loc[non_outlier["topic_id"].eq(topic)]
        topic_daily = daily.loc[daily["topic_id"].eq(topic)]
        training_docs = topic_docs.loc[topic_docs["split"].eq("train")]
        training_daily = topic_daily.loc[topic_daily["date"].between(TRAIN_START, TRAIN_END)]
        active = topic_daily.loc[topic_daily["volume"].gt(0)]
        training_active = training_daily.loc[training_daily["volume"].gt(0)]
        rows.append(
            {
                "topic_id": topic,
                "first_date": active["date"].min().date().isoformat(),
                "last_date": active["date"].max().date().isoformat(),
                "active_days": int(len(active)),
                "total_documents": int(len(topic_docs)),
                "median_daily_volume": float(topic_daily["volume"].median()),
                "median_volume_on_active_days": float(active["volume"].median()),
                "training_first_date": training_active["date"].min().date().isoformat(),
                "training_last_date": training_active["date"].max().date().isoformat(),
                "training_active_days": int(len(training_active)),
                "training_documents": int(len(training_docs)),
                "training_median_daily_volume": float(training_daily["volume"].median()),
                "history_short": bool(len(training_active) < MIN_TOPIC_HISTORY_ACTIVE_DAYS),
                "minimum_training_active_days_rule": MIN_TOPIC_HISTORY_ACTIVE_DAYS,
            }
        )
    return pd.DataFrame(rows).sort_values("total_documents", ascending=False).reset_index(drop=True)


def plot_topic_distribution(summary: pd.DataFrame, output: Path) -> None:
    top = summary.nlargest(30, "topic_size").sort_values("topic_size")
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    ax.barh(top["topic_id"].astype(str), top["topic_size"], color="#0052CC")
    ax.set_title("Distribusi 30 Topik Terbesar pada Training")
    ax.set_xlabel("Jumlah dokumen")
    ax.set_ylabel("Topic ID")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_topic_frequency(summary: pd.DataFrame, output: Path) -> None:
    ranked = summary.sort_values("topic_size", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(np.arange(1, len(ranked) + 1), ranked["topic_size"], color="#172B4D", linewidth=1.5)
    ax.scatter(np.arange(1, len(ranked) + 1), ranked["topic_size"], color="#36B37E", s=12)
    ax.set_title("Frekuensi Topik Training Berdasarkan Peringkat")
    ax.set_xlabel("Peringkat topik")
    ax.set_ylabel("Jumlah dokumen")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_topic_distance(model, summary: pd.DataFrame, output: Path) -> bool:
    topic_ids = sorted(summary["topic_id"].astype(int).tolist())
    if len(topic_ids) < 2:
        return False
    embeddings = np.asarray([model.topic_embeddings_[topic + model._outliers] for topic in topic_ids])
    distances = pairwise_distances(embeddings, metric="cosine")
    reducer = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
        n_init=4,
    )
    coordinates = reducer.fit_transform(distances)
    size_lookup = summary.set_index("topic_id")["topic_size"]
    sizes = np.array([size_lookup.loc[topic] for topic in topic_ids], dtype=float)
    marker_sizes = 20 + 480 * np.sqrt(sizes / sizes.max())
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=marker_sizes, alpha=0.65, color="#0052CC")
    largest = np.argsort(sizes)[-min(20, len(sizes)) :]
    for index in largest:
        ax.annotate(str(topic_ids[index]), coordinates[index], fontsize=8)
    ax.set_title("Jarak Antartopik Training (MDS dari Cosine Distance)")
    ax.set_xlabel("Dimensi MDS 1")
    ax.set_ylabel("Dimensi MDS 2")
    ax.grid(alpha=0.2)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def build_report(
    selected: str,
    status: str,
    selection_reason: str,
    comparison: pd.DataFrame,
    selected_metrics: pd.Series,
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
    history: pd.DataFrame,
    transform_seconds: dict,
    embedding_metadata: dict,
) -> str:
    split_outliers = distribution.loc[distribution["topic_id"].eq(-1)].set_index("split")
    validation_outlier = float(split_outliers.loc["validation", "percentage"])
    test_outlier = float(split_outliers.loc["test", "percentage"])
    train_outlier = float(split_outliers.loc["train", "percentage"])
    dominant_ratio = 100.0 * float(selected_metrics["largest_topic_ratio_non_outlier"])
    short_topics = int(history["history_short"].sum())
    return f"""# Topic quality report

## Ruang lingkup

BERTopic di-fit hanya pada 33.899 dokumen training ({TRAIN_START.date()} sampai {TRAIN_END.date()}). Validation dan test hanya dipetakan dengan `transform()` menggunakan UMAP, HDBSCAN, vectorizer, dan c-TF-IDF yang sudah di-fit pada training. Tidak ada label pertumbuhan atau classifier yang dibuat.

## Embedding

- Model: `{embedding_metadata['embedding_model']}`
- Shape: {embedding_metadata['row_count']:,} × {embedding_metadata['embedding_dimension']}
- Normalisasi L2: {embedding_metadata['normalize_embeddings']}
- Batch size final: {embedding_metadata['final_batch_size']}
- Batas input model: {embedding_metadata['model_max_sequence_length']} token
- Dokumen terkena right truncation: {embedding_metadata['documents_over_token_limit']:,} ({embedding_metadata['documents_over_token_limit_percent']:.2f}%)
- Judul yang melewati batas token: {embedding_metadata['titles_over_token_limit']}

Judul ditempatkan di awal input dan seluruh judul muat dalam batas model. Karena 99% lebih dokumen melebihi batas 128 token, embedding merepresentasikan judul dan bagian awal content, bukan seluruh isi artikel. Ini merupakan keterbatasan utama representasi semantik.

## Perbandingan konfigurasi training

{comparison.to_markdown(index=False)}

Coherence menggunakan mean NPMI pasangan top-10 kata pada seluruh corpus training. Pasangan tanpa ko-okurensi memakai smoothing 1/N. Silhouette dihitung pada ruang UMAP training lima dimensi, dokumen outlier dikeluarkan, dengan sample reproducible maksimum 5.000 dokumen (`random_state=42`).

## Konfigurasi terpilih

- Konfigurasi: **{selected}**
- Alasan: {selection_reason}
- Jumlah topik: {int(selected_metrics['number_of_topics'])}
- Training outlier: {train_outlier:.2f}%
- Validation outlier: {validation_outlier:.2f}%
- Test outlier: {test_outlier:.2f}%
- Topic diversity: {float(selected_metrics['topic_diversity']):.4f}
- Coherence NPMI: {float(selected_metrics['coherence_npmi']):.4f}
- Silhouette: {float(selected_metrics['silhouette']):.4f} (sample n={int(selected_metrics['silhouette_sample_size'])})
- Topik terbesar: {int(selected_metrics['largest_topic_size'])} dokumen atau {dominant_ratio:.2f}% dari dokumen training non-outlier
- Waktu transform validation: {transform_seconds['validation']:.2f} detik
- Waktu transform test: {transform_seconds['test']:.2f} detik

## Audit distribusi dan interpretabilitas

- Topik dengan history training <{MIN_TOPIC_HISTORY_ACTIVE_DAYS} hari aktif: {short_topics}/{len(history)}. Topik hanya ditandai dan tidak dibuang.
- `topic_summary.csv` menyimpan top words dan tiga representative documents untuk setiap topik.
- Topic diversity mengukur overlap kata topik; nilai tinggi berarti lebih sedikit kata yang berulang antartopik.
- Outlier validation/test mencerminkan dokumen future yang tidak cocok dengan ruang topik training. Tidak ada topik future baru yang di-fit.
- Median volume harian pada `topic_history.csv` memasukkan hari bernilai nol; median khusus hari aktif tersedia di kolom terpisah.

## Konsistensi temporal

Validation mencakup {EXPECTED_SPLIT_COUNTS['validation']:,} dokumen ({VALIDATION_START.date()} sampai {VALIDATION_END.date()}); test mencakup {EXPECTED_SPLIT_COUNTS['test']:,} dokumen ({TEST_START.date()} sampai {TEST_END.date()}). Lonjakan volume sumber pada September 2025 tetap merupakan potensi distribution shift dan tidak dipakai untuk memilih konfigurasi.

## Keputusan

**TOPIC MODEL STATUS: {status}**

Keputusan didasarkan pada metrik training, pemeriksaan top words/representative documents, distribusi ukuran topik, dan kemampuan model training menerima dokumen validation/test. Prediction belum dijalankan.
"""


def finalize(root: Path, selected: str, status: str, selection_reason: str) -> None:
    if selected not in CONFIGURATIONS:
        raise ValueError(f"Konfigurasi tidak dikenal: {selected}")
    if status not in {"READY", "LIMITED", "FAIL"}:
        raise ValueError(status)

    tables = root / "results" / "tables"
    figures = root / "results" / "figures"
    candidate_dir = root / "models" / "bertopic_candidates"
    figures.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(
        root / "data" / "processed" / "news_cleaned.parquet",
        columns=["Judul", "Content", "date", "source", "Link"],
    )
    masks = split_masks(frame)
    documents = build_documents(frame)
    comparison = pd.read_csv(tables / "topic_model_comparison.csv")
    selected_metrics = comparison.set_index("configuration").loc[selected]
    summary = pd.read_csv(tables / f"topic_summary_config_{selected}.csv")
    model = joblib.load(candidate_dir / f"bertopic_config_{selected}.joblib")

    train_labels = np.load(candidate_dir / f"train_topics_config_{selected}.npy")
    train_probabilities = np.load(candidate_dir / f"train_probabilities_config_{selected}.npy")
    reduced_validation = np.load(root / "data" / "intermediate" / "umap_validation.npy", mmap_mode="r")
    reduced_test = np.load(root / "data" / "intermediate" / "umap_test.npy", mmap_mode="r")
    validation_idx = np.flatnonzero(masks["validation"])
    test_idx = np.flatnonzero(masks["test"])
    train_idx = np.flatnonzero(masks["train"])

    started = time.perf_counter()
    validation_labels, validation_probabilities = model.transform(
        documents.iloc[validation_idx].tolist(), np.asarray(reduced_validation)
    )
    validation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    test_labels, test_probabilities = model.transform(
        documents.iloc[test_idx].tolist(), np.asarray(reduced_test)
    )
    test_seconds = time.perf_counter() - started

    all_labels = np.full(len(frame), -1, dtype=np.int32)
    all_probabilities = np.zeros(len(frame), dtype=np.float32)
    all_labels[train_idx] = train_labels
    all_probabilities[train_idx] = train_probabilities
    all_labels[validation_idx] = np.asarray(validation_labels, dtype=np.int32)
    all_probabilities[validation_idx] = np.asarray(validation_probabilities, dtype=np.float32)
    all_labels[test_idx] = np.asarray(test_labels, dtype=np.int32)
    all_probabilities[test_idx] = np.asarray(test_probabilities, dtype=np.float32)
    split_values = np.empty(len(frame), dtype=object)
    for split, mask in masks.items():
        split_values[mask] = split

    assignments = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"]).dt.normalize(),
            "source": frame["source"].to_numpy(),
            "url": frame["Link"].to_numpy(),
            "split": split_values,
            "topic_id": all_labels,
            "topic_probability": all_probabilities,
        }
    )
    assignment_path = root / "data" / "processed" / "topic_assignments_train_only.parquet"
    assignments.to_parquet(assignment_path, index=False, compression="zstd")

    distribution = make_topic_distribution(assignments)
    distribution.to_csv(tables / "topic_distribution_by_split.csv", index=False)
    topics = sorted(summary["topic_id"].astype(int).tolist())
    daily = make_daily_volume(assignments, topics)
    daily.to_csv(tables / "topic_daily_volume_preliminary.csv", index=False)
    history = make_topic_history(assignments, daily, topics)
    history.to_csv(tables / "topic_history.csv", index=False)
    summary.to_csv(tables / "topic_summary.csv", index=False, encoding="utf-8-sig")

    plot_topic_distribution(summary, figures / "topic_distribution.png")
    plot_topic_frequency(summary, figures / "topic_frequency.png")
    distance_created = plot_topic_distance(model, summary, figures / "topic_distance.png")

    selected_model_path = root / "models" / "bertopic_model_train_only.joblib"
    shutil.copy2(candidate_dir / f"bertopic_config_{selected}.joblib", selected_model_path)
    metadata_path = root / "results" / "bertopic_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embedding_metadata = json.loads(
        (root / "data" / "intermediate" / "embedding_metadata.json").read_text(encoding="utf-8")
    )
    split_outlier = {
        split: float((assignments.loc[assignments["split"].eq(split), "topic_id"] == -1).mean())
        for split in ["train", "validation", "test"]
    }
    metadata.update(
        {
            "status": status,
            "selected_configuration": selected,
            "selection_reason": selection_reason,
            "selected_model_path": str(selected_model_path.resolve()),
            "selected_model_hash": file_sha256(selected_model_path),
            "topic_assignments_path": str(assignment_path.resolve()),
            "topic_assignments_hash": file_sha256(assignment_path),
            "outlier_ratio_by_split": split_outlier,
            "transform_seconds": {"validation": validation_seconds, "test": test_seconds},
            "minimum_topic_history_active_days": MIN_TOPIC_HISTORY_ACTIVE_DAYS,
            "short_history_topics": int(history["history_short"].sum()),
            "topic_distance_figure_created": distance_created,
        }
    )
    write_json(metadata_path, metadata)

    report = build_report(
        selected,
        status,
        selection_reason,
        comparison,
        selected_metrics,
        summary,
        distribution,
        history,
        {"validation": validation_seconds, "test": test_seconds},
        embedding_metadata,
    )
    (root / "results" / "topic_quality_report.md").write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=sorted(CONFIGURATIONS))
    parser.add_argument("--status", required=True, choices=["READY", "LIMITED", "FAIL"])
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    finalize(Path(__file__).resolve().parents[1], args.config, args.status, args.reason)


if __name__ == "__main__":
    main()

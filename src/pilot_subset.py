"""Pemilihan subset pilot untuk perbandingan strategi embedding.

Encoding ulang seluruh 67.160 dokumen dengan chunking memakan waktu berjam-jam di
CPU. Modul ini memilih subset topik secara deterministik agar perbandingan antar
strategi embedding dapat dijalankan dengan anggaran waktu yang wajar, sementara
topic assignment dan label tetap dikunci dari run sebelumnya.

Subset dipilih pada level TOPIK, bukan dokumen, karena fitur semantik memerlukan
seluruh dokumen sebuah topik untuk membentuk centroid kumulatif. Memilih dokumen
secara acak akan membuat centroid tidak sebanding dengan run penuh.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PILOT_SEED = 42
PILOT_TAG = "pilot31"
# Anggaran dokumen non-outlier per kelas sparsity. Dipilih agar total sekitar
# 6.000 dokumen: cukup untuk tiga fold temporal dengan dua kelas, dan masih
# selesai dalam puluhan menit per strategi pada CPU ini.
DOCUMENT_BUDGET = {
    "LOW SPARSITY": 2600,
    "MEDIUM SPARSITY": 2000,
    "HIGH SPARSITY": 1600,
}
SPARSITY_ORDER = ["LOW SPARSITY", "MEDIUM SPARSITY", "HIGH SPARSITY"]


def select_pilot_topics(root: Path) -> tuple[list[int], pd.DataFrame]:
    history = pd.read_csv(root / "results" / "tables" / "topic_history_final.csv")
    assignments = pd.read_parquet(
        root / "data" / "processed" / "topic_assignments_train_only.parquet",
        columns=["topic_id"],
    )
    documents = (
        assignments.loc[assignments["topic_id"].ne(-1)]
        .groupby("topic_id")
        .size()
        .rename("documents")
    )
    eligible = history.loc[history["forecast_eligible"]].merge(
        documents, on="topic_id", how="left"
    )
    eligible["documents"] = eligible["documents"].fillna(0).astype(int)

    generator = np.random.default_rng(PILOT_SEED)
    selected: list[int] = []
    for category in SPARSITY_ORDER:
        pool = eligible.loc[eligible["sparsity_category"].eq(category)].sort_values("topic_id")
        budget = DOCUMENT_BUDGET[category]
        used = 0
        for position in generator.permutation(len(pool)):
            row = pool.iloc[int(position)]
            if used + int(row["documents"]) <= budget:
                selected.append(int(row["topic_id"]))
                used += int(row["documents"])
    selected = sorted(selected)
    summary = eligible.loc[eligible["topic_id"].isin(selected)][
        ["topic_id", "sparsity_category", "documents", "active_days", "training_active_days"]
    ].sort_values("topic_id").reset_index(drop=True)
    return selected, summary


def build_manifest(root: Path) -> dict:
    topics, summary = select_pilot_topics(root)
    assignments = pd.read_parquet(
        root / "data" / "processed" / "topic_assignments_train_only.parquet",
        columns=["topic_id", "split"],
    )
    # Indeks posisional ini valid karena step10_13 sudah memverifikasi bahwa urutan
    # baris topic_assignments identik dengan news_cleaned.parquet.
    row_index = np.flatnonzero(assignments["topic_id"].isin(topics).to_numpy())
    split_counts = (
        assignments.iloc[row_index].groupby("split", observed=True).size().to_dict()
    )
    manifest = {
        "pilot_tag": PILOT_TAG,
        "seed": PILOT_SEED,
        "document_budget": DOCUMENT_BUDGET,
        "selection_rule": (
            "stratified by sparsity_category over forecast_eligible topics; seeded "
            "permutation, greedy fill sampai anggaran dokumen per stratum terpakai"
        ),
        "topic_count": len(topics),
        "topics": topics,
        "document_count": int(len(row_index)),
        "documents_by_split": {key: int(value) for key, value in split_counts.items()},
        "documents_by_sparsity": summary.groupby("sparsity_category")["documents"].sum().to_dict(),
        "topics_by_sparsity": summary.groupby("sparsity_category").size().to_dict(),
        "note": (
            "topic assignment, topic_daily_volume, dan label tetap memakai artefak run "
            "sebelumnya; subset ini hanya menentukan dokumen mana yang di-encode ulang"
        ),
    }
    return manifest, row_index, summary


def manifest_path(root: Path) -> Path:
    return root / "data" / "processed" / f"pilot_subset_{PILOT_TAG}.json"


def row_index_path(root: Path) -> Path:
    return root / "data" / "processed" / f"pilot_subset_{PILOT_TAG}_rows.npy"


def load_manifest(root: Path) -> tuple[dict, np.ndarray]:
    manifest = json.loads(manifest_path(root).read_text(encoding="utf-8"))
    return manifest, np.load(row_index_path(root))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest, row_index, summary = build_manifest(root)
    manifest_path(root).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(row_index_path(root), row_index.astype(np.int64))
    summary.to_csv(root / "results" / "tables" / f"pilot_subset_{PILOT_TAG}_topics.csv", index=False)
    print(json.dumps({key: value for key, value in manifest.items() if key != "topics"}, indent=2))
    print(f"topics: {manifest['topics']}")


if __name__ == "__main__":
    main()

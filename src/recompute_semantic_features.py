"""Hitung ulang fitur semantik dari varian embedding, dengan topik dikunci.

Topic assignment, topic_daily_volume, dan label growth TIDAK dihitung ulang: semua
diambil dari artefak run sebelumnya. Yang berubah hanya vektor dokumen yang
dipakai untuk membentuk centroid kumulatif topik. Dengan begitu perbandingan
ablation antar varian embedding mengukur kualitas fitur semantik saja, bukan
perubahan topik atau perubahan label.

Konsekuensi yang harus dilaporkan: ruang topik tetap berasal dari embedding
ter-truncate, sehingga hasil di sini adalah batas bawah untuk manfaat mengatasi
truncation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .pilot_subset import PILOT_TAG, load_manifest
from .step7_embeddings import (
    LEGACY_VARIANT_KEY,
    VARIANTS,
    build_documents,
    file_sha256,
    get_variant,
    variant_paths,
)
from .step10_13_features_labels import SEMANTIC_FEATURE_COLUMNS, build_semantic_features


class SubsetEmbeddings:
    """Tampilan embedding subset yang dapat diindeks memakai nomor baris dataset.

    `build_semantic_features` mengindeks embedding dengan nomor baris
    topic_assignments. Adapter ini memetakan nomor baris tersebut ke posisi di
    dalam array subset, dan gagal keras bila ada baris yang diminta tetapi tidak
    ikut di-encode.
    """

    def __init__(self, vectors: np.ndarray, row_index: np.ndarray, dataset_rows: int):
        self._vectors = vectors
        lookup = np.full(dataset_rows, -1, dtype=np.int64)
        lookup[row_index] = np.arange(len(row_index), dtype=np.int64)
        self._lookup = lookup
        self.shape = (dataset_rows, int(vectors.shape[1]))

    def __getitem__(self, indices):
        positions = self._lookup[np.asarray(indices, dtype=np.int64)]
        if (positions < 0).any():
            missing = int((positions < 0).sum())
            raise RuntimeError(
                f"{missing} baris diminta tetapi tidak ada di subset embedding; "
                "subset pilot tidak mencakup seluruh dokumen topik ini."
            )
        return self._vectors[positions]


def semantic_output_path(root: Path, variant_key: str, pilot_tag: str) -> Path:
    return root / "data" / "processed" / f"semantic_features_{variant_key}__{pilot_tag}.parquet"


def load_fixed_artifacts(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    processed_path = root / "data" / "processed" / "news_cleaned.parquet"
    assignment_path = root / "data" / "processed" / "topic_assignments_train_only.parquet"
    assignments = pd.read_parquet(assignment_path)
    cleaned = pd.read_parquet(processed_path, columns=["date", "Link"])
    if len(cleaned) != len(assignments):
        raise RuntimeError("Jumlah baris cleaned dan topic assignment tidak sama.")
    if not (
        pd.to_datetime(cleaned["date"]).reset_index(drop=True).equals(
            pd.to_datetime(assignments["date"]).reset_index(drop=True)
        )
        and cleaned["Link"].reset_index(drop=True).equals(assignments["url"].reset_index(drop=True))
    ):
        raise RuntimeError("Urutan topic assignment tidak sama dengan news_cleaned.parquet.")
    daily = pd.read_csv(root / "results" / "tables" / "topic_daily_volume.csv", parse_dates=["date"])
    return assignments, daily, processed_path


def recompute(root: Path, variant_key: str, pilot_tag: str = PILOT_TAG) -> tuple[Path, dict]:
    manifest, row_index = load_manifest(root)
    if manifest["pilot_tag"] != pilot_tag:
        raise RuntimeError(f"Manifest pilot {manifest['pilot_tag']} != {pilot_tag}")
    topics = [int(topic) for topic in manifest["topics"]]
    output_path = semantic_output_path(root, variant_key, pilot_tag)
    started = time.perf_counter()

    if variant_key == LEGACY_VARIANT_KEY:
        # Pembanding historis: fitur semantik run sebelumnya, dibatasi ke topik pilot
        # supaya barisannya identik dengan varian lain.
        full = pd.read_parquet(root / "data" / "processed" / "semantic_features.parquet")
        semantic = full.loc[full["topic_id"].isin(topics)].reset_index(drop=True)
        provenance = {
            "variant": variant_key,
            "source": "data/processed/semantic_features.parquet",
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "note": (
                "embedding run sebelumnya (mesin lain, right truncation 128 token); "
                "tidak dihitung ulang"
            ),
        }
    else:
        variant = VARIANTS[variant_key]
        paths = variant_paths(root, variant, pilot_tag)
        if not paths["embeddings"].exists():
            raise FileNotFoundError(
                f"Embedding untuk {variant_key}/{pilot_tag} belum ada: {paths['embeddings']}"
            )
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        assignments, daily, processed_path = load_fixed_artifacts(root)

        # Verifikasi cache-hash: tolak embedding yang tidak cocok fingerprint varian
        # maupun dataset saat ini, supaya tidak ada embedding lama yang terpakai.
        expected_fingerprint = variant.fingerprint()
        if metadata.get("embedding_fingerprint") != expected_fingerprint:
            raise RuntimeError(
                f"Fingerprint embedding {variant_key} tidak cocok: "
                f"{metadata.get('embedding_fingerprint')} != {expected_fingerprint}"
            )
        if metadata.get("dataset_hash") != file_sha256(processed_path):
            raise RuntimeError(f"dataset_hash embedding {variant_key} tidak cocok dataset saat ini.")
        stored_rows = np.load(paths["rows"])
        if not np.array_equal(stored_rows, row_index):
            raise RuntimeError(
                f"Row index embedding {variant_key} tidak sama dengan manifest pilot."
            )

        vectors = np.load(paths["embeddings"], mmap_mode="r")
        if vectors.shape[0] != len(row_index):
            raise RuntimeError(
                f"Shape embedding {variant_key} {vectors.shape} tidak cocok subset {len(row_index)}"
            )
        view = SubsetEmbeddings(vectors, row_index, len(assignments))
        daily_subset = daily.loc[daily["topic_id"].isin(topics)].reset_index(drop=True)
        semantic = build_semantic_features(assignments, view, daily_subset, topics)
        provenance = {
            "variant": variant_key,
            "source": str(paths["embeddings"].relative_to(root)),
            "embedding_model": variant.model_name,
            "embedding_fingerprint": expected_fingerprint,
            "max_seq_length": variant.max_seq_length,
            "chunking": variant.chunking,
            "documents_over_token_limit_percent": metadata.get("documents_over_token_limit_percent"),
            "documents_still_truncated_after_chunking_percent": metadata.get(
                "documents_still_truncated_after_chunking_percent"
            ),
            "mean_chunks_per_document": metadata.get("mean_chunks_per_document"),
            "effective_token_coverage": metadata.get("effective_token_coverage"),
        }

    semantic.to_parquet(output_path, index=False, compression="zstd")
    stats = {
        **provenance,
        "pilot_tag": pilot_tag,
        "topics": len(topics),
        "rows": int(len(semantic)),
        "processing_seconds": round(time.perf_counter() - started, 3),
        "missing_rate": {
            column: round(float(semantic[column].isna().mean()), 6)
            for column in SEMANTIC_FEATURE_COLUMNS
            if column in semantic.columns
        },
        "similarity_mean": round(float(semantic["semantic_similarity_mean"].mean()), 6),
        "similarity_std_mean": round(float(semantic["semantic_similarity_std"].mean()), 6),
        "drift_mean": round(float(semantic["semantic_drift"].mean()), 6),
    }
    return output_path, stats


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    variant_key = sys.argv[1] if len(sys.argv) > 1 else get_variant().key
    path, stats = recompute(root, variant_key)
    print(json.dumps({"output": str(path.relative_to(root)), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

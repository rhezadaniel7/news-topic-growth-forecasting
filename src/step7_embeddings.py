from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import pandas as pd



MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LONG_CONTEXT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_BATCH_SIZE = 64
SPACE_RE = re.compile(r"\s+")

# Jumlah special token yang dipakai encoder XLM-R style (<s> ... </s>).
SPECIAL_TOKEN_BUDGET = 2


@dataclass(frozen=True)
class EmbeddingVariant:
    """Satu konfigurasi encoding yang menghasilkan satu file embedding.

    Setiap field di sini mempengaruhi isi embedding, sehingga seluruhnya masuk ke
    fingerprint cache. Menambah field baru tanpa memasukkannya ke fingerprint
    akan membuat cache lama dipakai ulang secara salah.
    """

    key: str
    model_name: str
    max_seq_length: int
    chunking: bool
    chunk_overlap_tokens: int
    max_chunks_per_document: int
    pooling: str
    text_prefix: str
    normalize_embeddings: bool
    description: str

    @property
    def chunk_window(self) -> int:
        """Token konten per chunk, setelah menyisakan ruang untuk special token."""
        return self.max_seq_length - SPECIAL_TOKEN_BUDGET

    def fingerprint_payload(self) -> dict:
        return {
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "chunking": self.chunking,
            "chunk_window": self.chunk_window if self.chunking else None,
            "chunk_overlap_tokens": self.chunk_overlap_tokens if self.chunking else None,
            "max_chunks_per_document": self.max_chunks_per_document if self.chunking else None,
            "pooling": self.pooling if self.chunking else "single_pass",
            "text_prefix": self.text_prefix,
            "normalize_embeddings": self.normalize_embeddings,
            "special_token_budget": SPECIAL_TOKEN_BUDGET,
            "document_construction": DOCUMENT_CONSTRUCTION,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.fingerprint_payload(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


DOCUMENT_CONSTRUCTION = (
    "whitespace-normalized Judul + '. ' + Content; Judul only if Content empty"
)

VARIANTS: dict[str, EmbeddingVariant] = {
    "minilm_trunc128": EmbeddingVariant(
        key="minilm_trunc128",
        model_name=MODEL_NAME,
        max_seq_length=128,
        chunking=False,
        chunk_overlap_tokens=0,
        max_chunks_per_document=1,
        pooling="single_pass",
        text_prefix="",
        normalize_embeddings=True,
        description=(
            "Kontrol: encoder dan batas token seperti sebelumnya (right truncation 128 "
            "token), tetapi dihitung ulang pada lingkungan saat ini sehingga perbandingan "
            "tidak tercampur perbedaan versi library."
        ),
    ),
    "minilm_chunk128": EmbeddingVariant(
        key="minilm_chunk128",
        model_name=MODEL_NAME,
        max_seq_length=128,
        chunking=True,
        chunk_overlap_tokens=16,
        max_chunks_per_document=64,
        pooling="mean",
        text_prefix="",
        normalize_embeddings=True,
        description=(
            "Strategi 1: dokumen dipotong menjadi chunk <=128 token dengan overlap 16 "
            "token, tiap chunk di-encode, lalu embedding antar-chunk di-mean-pool dan "
            "dinormalisasi ulang."
        ),
    ),
    "e5small_trunc512": EmbeddingVariant(
        key="e5small_trunc512",
        model_name=LONG_CONTEXT_MODEL_NAME,
        max_seq_length=512,
        chunking=False,
        chunk_overlap_tokens=0,
        max_chunks_per_document=1,
        pooling="single_pass",
        text_prefix="passage: ",
        normalize_embeddings=True,
        description=(
            "Strategi 2: encoder konteks-panjang multilingual (512 token, dimensi 384 "
            "sama dengan MiniLM). Prefix 'passage: ' wajib menurut model card E5."
        ),
    ),
    "e5small_chunk512": EmbeddingVariant(
        key="e5small_chunk512",
        model_name=LONG_CONTEXT_MODEL_NAME,
        max_seq_length=512,
        chunking=True,
        chunk_overlap_tokens=32,
        max_chunks_per_document=32,
        pooling="mean",
        text_prefix="passage: ",
        normalize_embeddings=True,
        description=(
            "Strategi 1 + 2: encoder 512 token dan chunking, sehingga tidak ada dokumen "
            "yang kehilangan token."
        ),
    ),
}

# Embedding dari run sebelumnya (mesin lain, right truncation 128 token). Dipakai
# hanya sebagai pembanding historis; tidak pernah ditulis ulang oleh modul ini.
LEGACY_VARIANT_KEY = "legacy_minilm_trunc128"
DEFAULT_VARIANT_KEY = "minilm_chunk128"


def get_variant(key: str | None = None) -> EmbeddingVariant:
    resolved = key or os.environ.get("EMBEDDING_VARIANT") or DEFAULT_VARIANT_KEY
    if resolved not in VARIANTS:
        raise ValueError(
            f"Varian embedding tidak dikenal: {resolved!r}. Pilihan: {sorted(VARIANTS)}"
        )
    return VARIANTS[resolved]


def variant_paths(project_root: Path, variant: EmbeddingVariant, subset_tag: str = "full") -> dict[str, Path]:
    intermediate = project_root / "data" / "intermediate"
    stem = f"{variant.key}__{subset_tag}"
    return {
        "embeddings": intermediate / f"embeddings_{stem}.npy",
        "rows": intermediate / f"embedding_rows_{stem}.npy",
        "metadata": intermediate / f"embedding_metadata_{stem}.json",
        "validation": intermediate / f"embedding_validation_{stem}.json",
    }


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_light(value: object) -> str:
    if pd.isna(value):
        return ""
    return SPACE_RE.sub(" ", str(value)).strip()


def build_documents(frame: pd.DataFrame) -> pd.Series:
    title = frame["Judul"].map(normalize_light)
    content = frame["Content"].map(normalize_light)
    return pd.Series(
        np.where(content.eq(""), title, title + ". " + content),
        index=frame.index,
        dtype="object",
        name="document",
    )


def token_lengths(model, documents: pd.Series, batch_size: int = 256) -> np.ndarray:
    """Panjang token tiap dokumen tanpa truncation."""
    lengths = np.empty(len(documents), dtype=np.int32)
    position = 0
    for start in range(0, len(documents), batch_size):
        batch = documents.iloc[start : start + batch_size].tolist()
        encoded = model.tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        batch_lengths = encoded.get("length")
        if batch_lengths is None:
            batch_lengths = [len(tokens) for tokens in encoded["input_ids"]]
        batch_lengths = np.asarray(batch_lengths, dtype=np.int32)
        lengths[position : position + len(batch_lengths)] = batch_lengths
        position += len(batch_lengths)
    return lengths[:position]


def token_length_audit(
    model,
    documents: pd.Series,
    variant: EmbeddingVariant,
    batch_size: int = 256,
) -> dict:
    max_tokens = int(variant.max_seq_length)
    lengths = token_lengths(model, documents, batch_size=batch_size)
    count = int(len(lengths))

    # Title prefix harus masuk batas token agar right truncation tetap menyimpannya.
    titles = documents.str.split(". ", n=1).str[0]
    title_lengths = token_lengths(model, titles, batch_size=batch_size)
    title_over_limit = int((title_lengths > max_tokens).sum())

    over_limit = int((lengths > max_tokens).sum())
    if variant.chunking:
        window = variant.chunk_window
        content_tokens = np.maximum(lengths - SPECIAL_TOKEN_BUDGET, 1)
        step = max(1, window - variant.chunk_overlap_tokens)
        chunks_needed = np.ceil((content_tokens - window) / step).astype(np.int64) + 1
        chunks_needed = np.maximum(chunks_needed, 1)
        chunks_used = np.minimum(chunks_needed, variant.max_chunks_per_document)
        still_truncated = int((chunks_needed > variant.max_chunks_per_document).sum())
        chunk_audit = {
            "chunking_enabled": True,
            "chunk_window_tokens": window,
            "chunk_overlap_tokens": variant.chunk_overlap_tokens,
            "max_chunks_per_document": variant.max_chunks_per_document,
            "mean_chunks_per_document": round(float(chunks_used.mean()), 6),
            "max_chunks_observed": int(chunks_used.max(initial=1)),
            "total_chunks": int(chunks_used.sum()),
            "documents_still_truncated_after_chunking": still_truncated,
            "documents_still_truncated_after_chunking_percent": round(
                100.0 * still_truncated / count, 6
            ),
            "effective_token_coverage": round(
                float(np.minimum(content_tokens, chunks_used * step + variant.chunk_overlap_tokens).sum())
                / float(content_tokens.sum()),
                6,
            ),
        }
        truncation_strategy = (
            f"token-window chunking {window}+{variant.chunk_overlap_tokens} overlap, "
            f"{variant.pooling}-pool antar chunk"
        )
    else:
        chunk_audit = {
            "chunking_enabled": False,
            "documents_still_truncated_after_chunking": over_limit,
            "documents_still_truncated_after_chunking_percent": round(
                100.0 * over_limit / count, 6
            ),
            "effective_token_coverage": round(
                float(np.minimum(lengths, max_tokens).sum()) / float(lengths.sum()), 6
            ),
        }
        truncation_strategy = (
            "right truncation by SentenceTransformer tokenizer; title is first"
        )

    return {
        "model_max_sequence_length": max_tokens,
        "documents_over_token_limit": over_limit,
        "documents_over_token_limit_percent": round(100.0 * over_limit / count, 6),
        "titles_over_token_limit": title_over_limit,
        "maximum_observed_tokens": int(lengths.max(initial=0)),
        "mean_observed_tokens": round(float(lengths.mean()), 6),
        "median_observed_tokens": float(np.median(lengths)),
        "truncation_strategy": truncation_strategy,
        **chunk_audit,
    }


def chunk_text_spans(tokenizer, text: str, variant: EmbeddingVariant) -> list[str]:
    """Potong `text` menjadi span teks asli yang masing-masing <= window token.

    Pemotongan dilakukan pada offset karakter hasil tokenizer, bukan dengan
    decode ulang, sehingga teks tiap chunk tetap identik dengan sumbernya.
    """
    window = variant.chunk_window
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_offsets_mapping=True,
    )
    offsets = [pair for pair in encoded["offset_mapping"] if pair[1] > pair[0]]
    if len(offsets) <= window:
        return [text]
    step = max(1, window - variant.chunk_overlap_tokens)
    spans = []
    for start in range(0, len(offsets), step):
        piece = offsets[start : start + window]
        if not piece:
            break
        spans.append(text[piece[0][0] : piece[-1][1]])
        if len(spans) >= variant.max_chunks_per_document:
            break
        if start + window >= len(offsets):
            break
    return spans


def encode_batch(model, documents: list[str], variant: EmbeddingVariant) -> np.ndarray:
    """Encode satu batch dokumen menurut varian; mean-pool jika chunking aktif."""
    if not variant.chunking:
        payload = [variant.text_prefix + document for document in documents]
        return np.asarray(
            model.encode(
                payload,
                batch_size=len(payload),
                show_progress_bar=False,
                normalize_embeddings=variant.normalize_embeddings,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    spans: list[str] = []
    owners: list[int] = []
    for index, document in enumerate(documents):
        for span in chunk_text_spans(model.tokenizer, document, variant):
            spans.append(variant.text_prefix + span)
            owners.append(index)
    chunk_vectors = np.asarray(
        model.encode(
            spans,
            batch_size=min(len(spans), 64),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    owners_array = np.asarray(owners, dtype=np.int64)
    dimension = chunk_vectors.shape[1]
    pooled = np.zeros((len(documents), dimension), dtype=np.float32)
    np.add.at(pooled, owners_array, chunk_vectors)
    counts = np.bincount(owners_array, minlength=len(documents)).astype(np.float32)
    pooled /= counts[:, None]
    if variant.normalize_embeddings:
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled /= np.maximum(norms, 1e-12)
    return pooled


def metadata_matches(
    metadata: dict,
    dataset_hash: str,
    rows: int,
    variant: EmbeddingVariant,
    subset_tag: str = "full",
) -> bool:
    """Cache hanya valid jika dataset, jumlah baris, subset, DAN fingerprint sama.

    Fingerprint mencakup model, batas token, konfigurasi chunking, pooling, dan
    prefix teks. Sebelumnya cek hanya memakai dataset_hash + nama model + jumlah
    baris, sehingga mengganti batas token atau mengaktifkan chunking pada model
    yang sama akan menghasilkan cache hit yang salah dan memakai embedding lama.
    """
    return (
        metadata.get("dataset_hash") == dataset_hash
        and metadata.get("embedding_model") == variant.model_name
        and metadata.get("row_count") == rows
        and metadata.get("embedding_variant") == variant.key
        and metadata.get("embedding_fingerprint") == variant.fingerprint()
        and metadata.get("subset_tag") == subset_tag
    )


def validate_embeddings(path: Path, rows: int, expected_dimension: int | None = None) -> dict:
    vectors = np.load(path, mmap_mode="r")
    if vectors.ndim != 2 or vectors.shape[0] != rows:
        raise ValueError(f"Shape embedding tidak valid: {vectors.shape}; expected rows={rows}")
    if expected_dimension is not None and vectors.shape[1] != expected_dimension:
        raise ValueError(f"Dimensi embedding tidak valid: {vectors.shape[1]} != {expected_dimension}")

    non_finite = 0
    norm_min = float("inf")
    norm_max = float("-inf")
    norm_sum = 0.0
    for start in range(0, rows, 4096):
        chunk = np.asarray(vectors[start : start + 4096])
        non_finite += int((~np.isfinite(chunk)).sum())
        norms = np.linalg.norm(chunk, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        norm_sum += float(norms.sum())
    if non_finite:
        raise ValueError(f"Embedding mengandung {non_finite} nilai non-finite")
    return {
        "status": "PASS",
        "shape": [int(v) for v in vectors.shape],
        "dtype": str(vectors.dtype),
        "non_finite_values": non_finite,
        "l2_norm_min": norm_min,
        "l2_norm_mean": norm_sum / rows,
        "l2_norm_max": norm_max,
        "file_size_bytes": path.stat().st_size,
    }


def build_embeddings(
    project_root: Path,
    force: bool = False,
    variant: EmbeddingVariant | None = None,
    row_subset: np.ndarray | None = None,
    subset_tag: str = "full",
) -> tuple[Path, dict, dict]:
    """Hitung embedding untuk satu varian encoding.

    `row_subset` adalah indeks baris news_cleaned.parquet yang ingin di-encode
    (dipakai untuk pilot). Urutan baris embedding mengikuti urutan `row_subset`,
    dan indeks itu ditulis berdampingan sehingga konsumen dapat memetakan
    kembali ke dataset asli tanpa menebak.
    """
    variant = variant or get_variant()
    processed_path = project_root / "data" / "processed" / "news_cleaned.parquet"
    paths = variant_paths(project_root, variant, subset_tag)
    output_path = paths["embeddings"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not processed_path.exists():
        raise FileNotFoundError(processed_path)
    dataset_hash = file_sha256(processed_path)
    frame = pd.read_parquet(processed_path, columns=["Judul", "Content"])
    all_documents = build_documents(frame)
    if row_subset is None:
        row_index = np.arange(len(all_documents), dtype=np.int64)
    else:
        row_index = np.asarray(row_subset, dtype=np.int64)
        if row_index.ndim != 1 or len(row_index) == 0:
            raise ValueError("row_subset harus array 1 dimensi dan tidak kosong")
        if len(np.unique(row_index)) != len(row_index):
            raise ValueError("row_subset berisi indeks duplikat")
        if row_index.min() < 0 or row_index.max() >= len(all_documents):
            raise ValueError("row_subset di luar rentang dataset")
    documents = all_documents.iloc[row_index].reset_index(drop=True)
    rows = len(documents)

    if output_path.exists() and paths["metadata"].exists() and not force:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if metadata_matches(metadata, dataset_hash, rows, variant, subset_tag):
            validation = validate_embeddings(output_path, rows, metadata.get("embedding_dimension"))
            validation["cache_hit"] = True
            paths["validation"].write_text(json.dumps(validation, indent=2), encoding="utf-8")
            return output_path, metadata | {"cache_hit": True}, validation
        print(
            f"cache ditolak untuk {variant.key}/{subset_tag}: fingerprint atau dataset berbeda; "
            "embedding dihitung ulang",
            flush=True,
        )

    from sentence_transformers import SentenceTransformer
    import sentence_transformers

    start_time = time.perf_counter()
    import psutil

    process = psutil.Process()
    initial_rss = process.memory_info().rss
    model = SentenceTransformer(variant.model_name, device="cpu")
    # Batas token diambil dari varian, bukan dari default model, agar konfigurasi
    # 512 token benar-benar dipakai dan tercatat pada fingerprint.
    model.max_seq_length = int(variant.max_seq_length)
    dimension = int(model.get_sentence_embedding_dimension())
    length_audit = token_length_audit(model, documents, variant)
    if length_audit["titles_over_token_limit"]:
        raise RuntimeError(
            f"Ada {length_audit['titles_over_token_limit']} title melebihi batas token model; "
            "strategi title-preserving perlu direvisi sebelum embedding."
        )

    resumed_from_rows = 0
    if output_path.exists():
        try:
            partial = np.load(output_path, mmap_mode="r+")
            if partial.shape == (rows, dimension) and partial.dtype == np.float32:
                for scan_start in range(0, rows, 4096):
                    scan_end = min(scan_start + 4096, rows)
                    norms = np.linalg.norm(np.asarray(partial[scan_start:scan_end]), axis=1)
                    zero_positions = np.flatnonzero(norms == 0.0)
                    if len(zero_positions):
                        resumed_from_rows = scan_start + int(zero_positions[0])
                        break
                else:
                    resumed_from_rows = rows
                vectors = partial
            else:
                del partial
                raise ValueError("partial shape/dtype mismatch")
        except (ValueError, OSError):
            vectors = np.lib.format.open_memmap(
                output_path, mode="w+", dtype=np.float32, shape=(rows, dimension)
            )
    else:
        vectors = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=(rows, dimension)
        )
    batch_size = DEFAULT_BATCH_SIZE
    processed = resumed_from_rows
    batch_count = int(math.ceil(processed / batch_size))
    peak_rss = initial_rss
    if resumed_from_rows:
        print(f"embedding resume rows={resumed_from_rows}/{rows}", flush=True)
    while processed < rows:
        # Satu call per batch menjaga latensi dan penggunaan RAM CPU tetap stabil.
        end_row = min(processed + batch_size, rows)
        try:
            batch_vectors = encode_batch(model, documents.iloc[processed:end_row].tolist(), variant)
            vectors[processed:end_row] = batch_vectors.astype(np.float32, copy=False)
            vectors.flush()
            encoded_now = end_row - processed
            processed = end_row
            batch_count += 1
            peak_rss = max(peak_rss, process.memory_info().rss)
            if batch_count <= 4 or batch_count % 10 == 0 or processed == rows:
                elapsed = time.perf_counter() - start_time
                rate = processed / elapsed if elapsed else 0.0
                remaining = (rows - processed) / rate if rate else float("nan")
                print(
                    f"[{variant.key}/{subset_tag}] rows={processed}/{rows} "
                    f"batch_size={batch_size} elapsed_sec={elapsed:.1f} "
                    f"docs_per_sec={rate:.2f} eta_sec={remaining:.0f}",
                    flush=True,
                )
        except (MemoryError, RuntimeError) as exc:
            message = str(exc).casefold()
            is_oom = isinstance(exc, MemoryError) or "out of memory" in message or "cannot allocate" in message
            if not is_oom or batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            gc.collect()
            print(f"memory retry: batch_size diturunkan menjadi {batch_size}", flush=True)

    del vectors
    np.save(paths["rows"], row_index.astype(np.int64))
    import psutil
    import torch

    elapsed = time.perf_counter() - start_time
    metadata = {
        "dataset_path": str(processed_path.resolve()),
        "dataset_hash": dataset_hash,
        "embedding_variant": variant.key,
        "embedding_variant_description": variant.description,
        "embedding_fingerprint": variant.fingerprint(),
        "embedding_fingerprint_payload": variant.fingerprint_payload(),
        "subset_tag": subset_tag,
        "subset_row_count": rows,
        "dataset_row_count": int(len(all_documents)),
        "row_index_path": str(paths["rows"].name),
        "embedding_model": variant.model_name,
        "row_count": rows,
        "embedding_dimension": dimension,
        "dtype": "float32",
        "normalize_embeddings": variant.normalize_embeddings,
        "device": "cpu",
        "initial_batch_size": DEFAULT_BATCH_SIZE,
        "final_batch_size": batch_size,
        "batch_count": batch_count,
        "resumed_from_rows": resumed_from_rows,
        "processing_seconds": elapsed,
        "processing_time_scope": (
            "current resumed process only; earlier interrupted partial runtime excluded"
            if resumed_from_rows
            else "complete uninterrupted embedding run"
        ),
        "peak_process_rss_bytes": peak_rss,
        "initial_process_rss_bytes": initial_rss,
        "available_ram_bytes_at_start": psutil.virtual_memory().available,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "document_construction": DOCUMENT_CONSTRUCTION,
        **length_audit,
        "cache_hit": False,
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = validate_embeddings(output_path, rows, dimension)
    validation["cache_hit"] = False
    paths["validation"].write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path, metadata, validation


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    variant = get_variant()
    path, metadata, validation = build_embeddings(root, variant=variant)
    print(json.dumps({"embedding_path": str(path), "metadata": metadata, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()

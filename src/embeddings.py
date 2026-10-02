from __future__ import annotations

import json
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np

from .config import BATCH_SIZE, EMBEDDING_MODEL, INTERMEDIATE_DIR
from .utils import file_sha256, write_json


def build_embeddings(cleaned_path, documents, force: bool = False):
    cache = INTERMEDIATE_DIR / "document_embeddings.npy"
    metadata_path = INTERMEDIATE_DIR / "document_embeddings.json"
    fingerprint = file_sha256(cleaned_path)
    expected = {"dataset_sha256": fingerprint, "model": EMBEDDING_MODEL, "rows": len(documents)}
    if cache.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(k) == v for k, v in expected.items()):
            return np.load(cache), metadata | {"cache_hit": True}
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode(documents.tolist(), batch_size=BATCH_SIZE, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True)
    np.save(cache, vectors)
    metadata = expected | {"shape": list(vectors.shape), "cache_hit": False}
    write_json(metadata_path, metadata)
    return vectors, metadata

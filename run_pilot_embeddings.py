"""Encode subset pilot untuk satu varian embedding.

Dijalankan dengan interpreter .venv-encoder (Python 3.11 + torch), bukan .venv,
karena torch tidak tersedia untuk Python 3.14 di platform ini.

    .venv-encoder/bin/python run_pilot_embeddings.py minilm_chunk128
"""

import json
import sys
from pathlib import Path

from src.pilot_subset import load_manifest
from src.step7_embeddings import build_embeddings, get_variant


def main() -> None:
    root = Path(__file__).resolve().parent
    variant = get_variant(sys.argv[1] if len(sys.argv) > 1 else None)
    force = "--force" in sys.argv
    manifest, row_index = load_manifest(root)
    print(
        f"encoding {variant.key} pada subset {manifest['pilot_tag']}: "
        f"{len(row_index)} dokumen dari {manifest['topic_count']} topik",
        flush=True,
    )
    path, metadata, validation = build_embeddings(
        root,
        force=force,
        variant=variant,
        row_subset=row_index,
        subset_tag=manifest["pilot_tag"],
    )
    report = {
        "embedding_path": str(path.relative_to(root)),
        "cache_hit": metadata.get("cache_hit"),
        "embedding_fingerprint": metadata.get("embedding_fingerprint"),
        "embedding_dimension": metadata.get("embedding_dimension"),
        "processing_seconds": metadata.get("processing_seconds"),
        "model_max_sequence_length": metadata.get("model_max_sequence_length"),
        "documents_over_token_limit_percent": metadata.get("documents_over_token_limit_percent"),
        "documents_still_truncated_after_chunking_percent": metadata.get(
            "documents_still_truncated_after_chunking_percent"
        ),
        "mean_chunks_per_document": metadata.get("mean_chunks_per_document"),
        "total_chunks": metadata.get("total_chunks"),
        "effective_token_coverage": metadata.get("effective_token_coverage"),
        "validation_status": validation.get("status"),
        "l2_norm_min": validation.get("l2_norm_min"),
        "l2_norm_max": validation.get("l2_norm_max"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temporal_boundaries(periods, validation_ratio: float, test_ratio: float):
    unique = np.array(sorted(pd.to_datetime(pd.Series(periods).dropna().unique())))
    if len(unique) < 5:
        raise ValueError(f"Minimal 5 periode diperlukan untuk split temporal; ditemukan {len(unique)}.")
    train_end = max(1, int(len(unique) * (1 - validation_ratio - test_ratio)))
    val_end = max(train_end + 1, int(len(unique) * (1 - test_ratio)))
    val_end = min(val_end, len(unique) - 1)
    return unique[train_end], unique[val_end]


def safe_metric(value):
    return None if value is None or not np.isfinite(value) else float(value)


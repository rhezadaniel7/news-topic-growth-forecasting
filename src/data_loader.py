from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import COLUMN_CANDIDATES, DATA_PATH


SUPPORTED = {".csv", ".json", ".jsonl", ".parquet", ".pq"}


def discover_dataset(data_dir: Path = DATA_PATH) -> Path:
    files = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"Tidak ada CSV/JSON/Parquet di {data_dir}.")
    if len(files) > 1:
        raise RuntimeError(f"Ditemukan lebih dari satu dataset mentah: {[p.name for p in files]}. Sisakan satu file atau tetapkan pemetaan eksplisit.")
    return files[0]


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    try:
        return pd.read_json(path)
    except ValueError:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return pd.json_normalize(payload)


def map_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    lookup = {str(col).strip().casefold(): col for col in frame.columns}
    mapping = {}
    for role, candidates in COLUMN_CANDIDATES.items():
        mapping[role] = next((lookup[c.casefold()] for c in candidates if c.casefold() in lookup), None)
    missing = [role for role in ("title", "text", "date") if mapping[role] is None]
    if missing:
        raise ValueError(f"Kolom penting tidak ditemukan: {missing}. Kolom tersedia: {list(frame.columns)}")
    return mapping


def inspect_dataset(path: Path, frame: pd.DataFrame, mapping: dict) -> dict:
    date_values = pd.to_datetime(frame[mapping["date"]], errors="coerce", utc=True)
    summary = {
        "file": str(path), "rows": int(len(frame)), "columns": int(frame.shape[1]),
        "column_names": list(frame.columns), "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing_values": frame.isna().sum().astype(int).to_dict(), "column_mapping": mapping,
        "date_min": date_values.min(), "date_max": date_values.max(),
        "sample": frame.head(5).where(pd.notna(frame.head(5)), None).to_dict("records"),
    }
    if mapping.get("source"):
        summary["articles_per_source"] = frame[mapping["source"]].value_counts(dropna=False).to_dict()
    return summary


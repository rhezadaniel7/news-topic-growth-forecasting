from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import COLUMN_CANDIDATES, DATA_PATH, RESULTS_DIR
from .data_loader import SUPPORTED, load_dataset


AUDIT_OUTPUT = RESULTS_DIR / "tables" / "dataset_audit.csv"


def discover_candidates(data_dir: Path = DATA_PATH) -> list[Path]:
    """Return every supported dataset below data/raw in deterministic order."""
    if not data_dir.exists():
        return []
    return sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED
    )


def detect_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    """Map known semantic roles without requiring any particular schema."""
    lookup = {str(column).strip().casefold(): str(column) for column in frame.columns}
    return {
        role: next(
            (lookup[name.casefold()] for name in candidates if name.casefold() in lookup),
            None,
        )
        for role, candidates in COLUMN_CANDIDATES.items()
    }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def _nonempty_rate(series: pd.Series | None) -> float:
    if series is None or len(series) == 0:
        return 0.0
    text = series.fillna("").astype(str).str.strip()
    return float(text.ne("").mean())


def _combined_text_rate(frame: pd.DataFrame, mapping: dict[str, str | None]) -> float:
    columns = [mapping.get("title"), mapping.get("text")]
    columns = [column for column in columns if column]
    if not columns:
        return 0.0
    available = pd.Series(False, index=frame.index)
    for column in columns:
        available |= frame[column].fillna("").astype(str).str.strip().ne("")
    return float(available.mean())


def check_dataset_suitability(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    """Assess temporal-forecasting suitability before any topic modeling.

    PASS means the descriptive, pre-model ideal gates are met. WARN means the
    minimum descriptive gates are met but at least one ideal gate is not. FAIL
    means the dataset should not proceed to cleaning/modeling. Positive growth
    events remain a later gate because they cannot be established before a
    leakage-safe topic time series and label audit exist.
    """
    reasons: list[str] = []
    rows = int(metrics.get("rows") or 0)
    span_days = int(metrics.get("date_span_days") or 0)
    unique_dates = int(metrics.get("unique_dates") or 0)
    sources = int(metrics.get("unique_sources") or 0)
    timestamp_rate = float(metrics.get("timestamp_completeness") or 0.0)
    text_rate = float(metrics.get("text_completeness") or 0.0)
    active_ratio = float(metrics.get("active_date_ratio") or 0.0)
    has_title = bool(metrics.get("title_column"))
    has_text = bool(metrics.get("text_column"))
    has_date = bool(metrics.get("date_column"))

    critical_failures = []
    if not has_date:
        critical_failures.append("kolom tanggal tidak ditemukan")
    if not (has_title or has_text):
        critical_failures.append("kolom judul/teks tidak ditemukan")
    if rows <= 2000:
        critical_failures.append(f"baris {rows:,} belum melebihi minimum 2.000")
    if span_days < 90:
        critical_failures.append(f"rentang {span_days} hari belum mencapai 3 bulan")
    if timestamp_rate < 0.80:
        critical_failures.append(f"kelengkapan timestamp {timestamp_rate:.1%} di bawah 80%")
    if text_rate < 0.90:
        critical_failures.append(f"kelengkapan teks {text_rate:.1%} di bawah 90%")

    dense_single_source = sources == 1 and unique_dates >= 90 and active_ratio >= 0.80
    if sources < 3 and not dense_single_source:
        critical_failures.append(
            f"hanya {sources} source dan tidak memenuhi syarat satu source dengan time series padat"
        )

    if critical_failures:
        return "FAIL", critical_failures + [
            "positive growth pada training/validation belum diuji karena dataset gagal pada audit awal"
        ]

    ideal_gaps = []
    if rows <= 5000:
        ideal_gaps.append(f"baris {rows:,} belum melebihi target ideal 5.000")
    if timestamp_rate < 0.95:
        ideal_gaps.append(f"kelengkapan timestamp {timestamp_rate:.1%} belum mencapai 95%")
    if active_ratio < 0.60:
        ideal_gaps.append(f"tanggal aktif hanya {active_ratio:.1%} dari rentang kalender")
    if ideal_gaps:
        return "WARN", ideal_gaps + [
            "lanjut hanya ke cleaning dan audit label; modeling final tetap menunggu dua kelas pada validation"
        ]

    return "PASS", [
        "memenuhi gerbang deskriptif awal",
        "kelayakan akhir tetap bergantung pada kualitas topik dan positive growth di training/validation",
    ]


def audit_candidate(path: Path) -> dict[str, Any]:
    base = {
        "dataset": path.name,
        "path": str(path.resolve()),
        "format": path.suffix.lower().lstrip("."),
        "file_size_bytes": path.stat().st_size,
    }
    try:
        frame = load_dataset(path)
    except Exception as exc:  # preserve a row for unreadable candidates
        return {
            **base,
            "status": "FAIL",
            "reasons": _json([f"file tidak dapat dibaca: {type(exc).__name__}: {exc}"]),
        }

    mapping = detect_columns(frame)
    normalized_columns = {str(column).strip().casefold(): str(column) for column in frame.columns}
    matched_candidates = {
        role: [normalized_columns[name.casefold()] for name in candidates if name.casefold() in normalized_columns]
        for role, candidates in COLUMN_CANDIDATES.items()
    }
    date_column = mapping.get("date")
    source_column = mapping.get("source")
    date_values = (
        pd.to_datetime(frame[date_column], errors="coerce", utc=True)
        if date_column else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    )
    valid_dates = date_values.dropna()
    normalized_dates = valid_dates.dt.floor("D")
    per_day = normalized_dates.value_counts().sort_index()
    date_min = valid_dates.min() if not valid_dates.empty else pd.NaT
    date_max = valid_dates.max() if not valid_dates.empty else pd.NaT
    span_days = int((date_max.floor("D") - date_min.floor("D")).days + 1) if not valid_dates.empty else 0
    unique_dates = int(normalized_dates.nunique())
    source_values = (
        frame[source_column].fillna("<MISSING>").astype(str).str.strip().replace("", "<EMPTY>")
        if source_column else pd.Series(dtype="object")
    )
    per_source = source_values.value_counts(dropna=False)
    missing = frame.isna().sum().astype(int).to_dict()

    row = {
        **base,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": _json([str(column) for column in frame.columns]),
        "missing_values": _json(missing),
        "matched_candidate_columns": _json(matched_candidates),
        "title_column": mapping.get("title"),
        "text_column": mapping.get("text"),
        "date_column": date_column,
        "source_column": source_column,
        "url_column": mapping.get("url"),
        "id_column": mapping.get("id"),
        "category_column": mapping.get("category"),
        "date_min": None if pd.isna(date_min) else date_min.isoformat(),
        "date_max": None if pd.isna(date_max) else date_max.isoformat(),
        "date_span_days": span_days,
        "unique_dates": unique_dates,
        "active_date_ratio": unique_dates / span_days if span_days else 0.0,
        "timestamp_completeness": float(date_values.notna().mean()) if len(frame) else 0.0,
        "title_completeness": _nonempty_rate(frame[mapping["title"]]) if mapping.get("title") else 0.0,
        "text_completeness": _combined_text_rate(frame, mapping),
        "source_completeness": _nonempty_rate(frame[source_column]) if source_column else 0.0,
        "unique_sources": int(source_values[source_values.ne("<MISSING>") & source_values.ne("<EMPTY>")].nunique()) if source_column else 0,
        "articles_per_day": _json({str(index.date()): int(value) for index, value in per_day.items()}),
        "daily_articles_min": int(per_day.min()) if not per_day.empty else 0,
        "daily_articles_median": float(per_day.median()) if not per_day.empty else 0.0,
        "daily_articles_mean": float(per_day.mean()) if not per_day.empty else 0.0,
        "daily_articles_max": int(per_day.max()) if not per_day.empty else 0,
        "articles_per_source": _json({str(index): int(value) for index, value in per_source.items()}),
    }
    status, reasons = check_dataset_suitability(row)
    row["status"] = status
    row["reasons"] = _json(reasons)
    return row


def audit_all_datasets(
    data_dir: Path = DATA_PATH,
    output_path: Path = AUDIT_OUTPUT,
) -> pd.DataFrame:
    candidates = discover_candidates(data_dir)
    if not candidates:
        raise FileNotFoundError(f"Tidak ada CSV/JSON/JSONL/Parquet di {data_dir}.")
    report = pd.DataFrame([audit_candidate(path) for path in candidates])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return report


def recommend_dataset(report: pd.DataFrame) -> tuple[str | None, str]:
    usable = report[report["status"].isin(["PASS", "WARN"])].copy()
    if usable.empty:
        return None, "Tidak ada dataset yang melewati minimum temporal forecasting; data acquisition adalah bottleneck."
    usable["rank_status"] = usable["status"].map({"PASS": 2, "WARN": 1})
    usable = usable.sort_values(
        ["rank_status", "date_span_days", "unique_dates", "timestamp_completeness", "text_completeness", "rows"],
        ascending=False,
    )
    best = usable.iloc[0]
    return str(best["dataset"]), (
        "Dipilih berdasarkan status kelayakan, rentang dan kepadatan waktu, kelengkapan timestamp/teks, "
        "baru kemudian jumlah baris; bukan semata-mata dataset terbesar."
    )

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


JAKARTA_TZ = "Asia/Jakarta"
EXPECTED_START = pd.Timestamp("2024-07-31")
EXPECTED_END = pd.Timestamp("2025-10-23")
KNOWN_SOURCES = {"kompas", "tempo", "detik"}
SHORT_TEXT_CHAR_THRESHOLD = 200

MONTHS = {
    "jan": 1,
    "januari": 1,
    "feb": 2,
    "februari": 2,
    "mar": 3,
    "maret": 3,
    "apr": 4,
    "april": 4,
    "mei": 5,
    "jun": 6,
    "juni": 6,
    "jul": 7,
    "juli": 7,
    "agu": 8,
    "agustus": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "okt": 10,
    "oktober": 10,
    "nov": 11,
    "november": 11,
    "des": 12,
    "desember": 12,
}

TEXTUAL_DATE_RE = re.compile(
    r"^(?:(?P<weekday>[A-Za-z]+)\s*,\s*)?"
    r"(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)\s+(?P<year>\d{4})"
    r"(?:\s*(?:\|\s*)?(?P<hour>\d{1,2})[\.:](?P<minute>\d{2})\s*(?P<zone>WIB))?$",
    re.IGNORECASE,
)
NUMERIC_DATE_RE = re.compile(r"^(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>\d{4})$")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"(?u)\b\w+\b")


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw" / "final_merge_dataset.csv"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed" / "news_cleaned.parquet"

    @property
    def tables(self) -> Path:
        return self.root / "results" / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "results" / "figures"

    @property
    def report(self) -> Path:
        return self.root / "results" / "phase_1_6_report.md"


def ensure_directories(paths: Paths) -> None:
    paths.processed.parent.mkdir(parents=True, exist_ok=True)
    paths.tables.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)


def _clean_scalar(value: object) -> str:
    if pd.isna(value):
        return ""
    return SPACE_RE.sub(" ", html.unescape(str(value))).strip()


def normalize_title(value: object) -> str:
    """Conservative normalization: Unicode, HTML entities, case, and whitespace only."""
    text = unicodedata.normalize("NFKC", _clean_scalar(value))
    return text.casefold()


def parse_timestamp(value: object) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, bool, str]:
    """Return timezone-aware datetime (only when time exists), date, time-known flag, format."""
    raw = _clean_scalar(value)
    if not raw:
        return pd.NaT, pd.NaT, False, "missing"

    match = TEXTUAL_DATE_RE.fullmatch(raw)
    if match:
        groups = match.groupdict()
        month = MONTHS.get(groups["month"].casefold())
        if month is None:
            return pd.NaT, pd.NaT, False, "unknown_month"
        try:
            parsed_date = pd.Timestamp(int(groups["year"]), month, int(groups["day"]))
            if groups["hour"] is None:
                return pd.NaT, parsed_date, False, "text_date_only"
            parsed_datetime = pd.Timestamp(
                year=int(groups["year"]),
                month=month,
                day=int(groups["day"]),
                hour=int(groups["hour"]),
                minute=int(groups["minute"]),
                tz=JAKARTA_TZ,
            )
            style = "weekday_time_wib" if groups["weekday"] else "pipe_time_wib"
            return parsed_datetime, parsed_date, True, style
        except (ValueError, TypeError):
            return pd.NaT, pd.NaT, False, "invalid_text_date"

    match = NUMERIC_DATE_RE.fullmatch(raw)
    if match:
        groups = match.groupdict()
        try:
            parsed_date = pd.Timestamp(int(groups["year"]), int(groups["month"]), int(groups["day"]))
            return pd.NaT, parsed_date, False, "numeric_date_only"
        except (ValueError, TypeError):
            return pd.NaT, pd.NaT, False, "invalid_numeric_date"

    return pd.NaT, pd.NaT, False, "unrecognized"


def is_valid_url(value: object) -> bool:
    raw = _clean_scalar(value)
    if not raw or any(ch.isspace() for ch in raw):
        return False
    try:
        parsed = urlparse(raw)
        return parsed.scheme.casefold() in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


def safe_percentage(count: int, total: int) -> float:
    return round((100.0 * count / total), 6) if total else 0.0


def serializable(value: object) -> object:
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_inspection(frame: pd.DataFrame, path: Path) -> dict:
    per_column_duplicates = {
        col: int(frame[col].duplicated(keep="first").sum()) for col in frame.columns
    }
    sample = frame.head(10).where(frame.head(10).notna(), None).to_dict("records")
    return {
        "file": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": [str(col) for col in frame.columns],
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing_values": frame.isna().sum().astype(int).to_dict(),
        "exact_duplicate_rows": int(frame.duplicated(keep="first").sum()),
        "duplicate_values_by_column": per_column_duplicates,
        "articles_per_source": frame["source"].value_counts(dropna=False).astype(int).to_dict(),
        "sample_10": sample,
    }


def parse_timestamps(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parsed = frame["Waktu"].map(parse_timestamp)
    output = frame.copy()
    output["original_timestamp"] = output["Waktu"]
    output["parsed_datetime"] = pd.Series(
        [item[0] for item in parsed], index=output.index, dtype=f"datetime64[ns, {JAKARTA_TZ}]"
    )
    output["date"] = pd.to_datetime([item[1] for item in parsed], errors="coerce")
    output["time_known"] = [item[2] for item in parsed]
    output["timestamp_format"] = [item[3] for item in parsed]
    success_mask = output["date"].notna()

    summary = pd.DataFrame(
        [
            {"metric": "total_rows", "value": int(len(output))},
            {"metric": "parsed_success", "value": int(success_mask.sum())},
            {"metric": "parsed_failure", "value": int((~success_mask).sum())},
            {"metric": "parse_success_percent", "value": safe_percentage(int(success_mask.sum()), len(output))},
            {"metric": "time_known", "value": int(output["time_known"].sum())},
            {"metric": "date_only_time_missing", "value": int((success_mask & ~output["time_known"]).sum())},
        ]
    )
    failures = output.loc[~success_mask, ["original_timestamp", "Judul", "Link", "source"]].copy()
    return output, summary, failures


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.copy()
    working["_original_row"] = np.arange(len(working), dtype=np.int64)
    working["_normalized_title"] = working["Judul"].map(normalize_title)
    # Keep the earliest publication; original row order breaks same-date ties reproducibly.
    working = working.sort_values(["date", "_original_row"], na_position="last", kind="stable")

    rows_initial = len(working)
    valid_url = working["Link"].map(_clean_scalar).ne("")
    duplicate_url_mask = valid_url & working["Link"].duplicated(keep="first")
    removed_url = int(duplicate_url_mask.sum())
    working = working.loc[~duplicate_url_mask].copy()

    valid_title = working["Judul"].map(_clean_scalar).ne("")
    duplicate_title_mask = valid_title & working["Judul"].duplicated(keep="first")
    removed_title = int(duplicate_title_mask.sum())
    working = working.loc[~duplicate_title_mask].copy()

    valid_normalized_title = working["_normalized_title"].ne("")
    duplicate_normalized_mask = valid_normalized_title & working["_normalized_title"].duplicated(keep="first")
    removed_normalized_title = int(duplicate_normalized_mask.sum())
    working = working.loc[~duplicate_normalized_mask].copy()

    working = working.sort_values(["date", "_original_row"], na_position="last", kind="stable")
    working = working.drop(columns=["_original_row", "_normalized_title"]).reset_index(drop=True)
    report = pd.DataFrame(
        [
            {"stage": "raw", "rows_before": rows_initial, "rows_removed": 0, "rows_after": rows_initial},
            {
                "stage": "exact_url_duplicate",
                "rows_before": rows_initial,
                "rows_removed": removed_url,
                "rows_after": rows_initial - removed_url,
            },
            {
                "stage": "exact_title_duplicate",
                "rows_before": rows_initial - removed_url,
                "rows_removed": removed_title,
                "rows_after": rows_initial - removed_url - removed_title,
            },
            {
                "stage": "normalized_title_duplicate",
                "rows_before": rows_initial - removed_url - removed_title,
                "rows_removed": removed_normalized_title,
                "rows_after": len(working),
            },
        ]
    )
    return working, report


def describe_numeric(series: pd.Series, prefix: str) -> list[dict]:
    valid = series.dropna()
    if valid.empty:
        return []
    return [
        {"metric": f"{prefix}_min", "count": float(valid.min()), "percentage": np.nan, "definition": "minimum"},
        {"metric": f"{prefix}_p25", "count": float(valid.quantile(0.25)), "percentage": np.nan, "definition": "25th percentile"},
        {"metric": f"{prefix}_median", "count": float(valid.median()), "percentage": np.nan, "definition": "median"},
        {"metric": f"{prefix}_mean", "count": float(valid.mean()), "percentage": np.nan, "definition": "mean"},
        {"metric": f"{prefix}_p75", "count": float(valid.quantile(0.75)), "percentage": np.nan, "definition": "75th percentile"},
        {"metric": f"{prefix}_max", "count": float(valid.max()), "percentage": np.nan, "definition": "maximum"},
    ]


def audit_quality(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    title = frame["Judul"].map(_clean_scalar)
    content = frame["Content"].map(_clean_scalar)
    source = frame["source"].map(_clean_scalar).str.casefold()
    url = frame["Link"].map(_clean_scalar)
    normalized_title = frame["Judul"].map(normalize_title)

    content_char_count = content.str.len().astype("int64")
    content_token_count = content.map(lambda value: len(TOKEN_RE.findall(value))).astype("int64")
    frame["content_char_count"] = content_char_count
    frame["content_token_count"] = content_token_count

    duplicate_url = url.ne("") & url.duplicated(keep=False)
    duplicate_title = title.ne("") & title.duplicated(keep=False)
    duplicate_normalized = normalized_title.ne("") & normalized_title.duplicated(keep=False)
    duplicate_any = duplicate_url | duplicate_title | duplicate_normalized
    malformed_url = ~url.map(is_valid_url)
    outside_expected = frame["date"].notna() & (
        (frame["date"] < EXPECTED_START) | (frame["date"] > EXPECTED_END)
    )
    unknown_source = ~source.isin(KNOWN_SOURCES)
    short_text = content_char_count.lt(SHORT_TEXT_CHAR_THRESHOLD)

    checks = [
        ("missing_title", int(title.eq("").sum()), "Judul kosong setelah trim"),
        ("missing_content", int(content.eq("").sum()), "Content kosong setelah trim"),
        ("missing_timestamp", int(frame["date"].isna().sum()), "Tanggal gagal diparse atau kosong"),
        ("missing_source", int(source.eq("").sum()), "Source kosong setelah trim"),
        ("duplicate_article", int(duplicate_any.sum()), "Duplikat URL/title/title-normalized setelah deduplikasi"),
        ("malformed_url", int(malformed_url.sum()), "URL bukan HTTP/HTTPS valid dengan hostname"),
        (
            "text_too_short",
            int(short_text.sum()),
            f"Content < {SHORT_TEXT_CHAR_THRESHOLD} karakter; hanya flag, tidak dihapus",
        ),
        (
            "date_outside_declared_range",
            int(outside_expected.sum()),
            f"Tanggal di luar {EXPECTED_START.date()} sampai {EXPECTED_END.date()}",
        ),
        ("unknown_source", int(unknown_source.sum()), "Source bukan Kompas/Tempo/Detik"),
    ]
    report_rows = [
        {"metric": metric, "count": count, "percentage": safe_percentage(count, n), "definition": definition}
        for metric, count, definition in checks
    ]
    report_rows.extend(describe_numeric(content_char_count, "content_characters"))
    report_rows.extend(describe_numeric(content_token_count, "content_tokens_regex"))
    report = pd.DataFrame(report_rows)

    issue_flags = pd.DataFrame(
        {
            "row_index": frame.index,
            "missing_title": title.eq(""),
            "missing_content": content.eq(""),
            "missing_timestamp": frame["date"].isna(),
            "missing_source": source.eq(""),
            "duplicate_article": duplicate_any,
            "malformed_url": malformed_url,
            "text_too_short": short_text,
            "date_outside_declared_range": outside_expected,
            "unknown_source": unknown_source,
        }
    )
    return report, issue_flags


def temporal_density(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = frame.dropna(subset=["date"]).copy()
    start = valid["date"].min().normalize()
    end = valid["date"].max().normalize()
    calendar = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    counts = valid.groupby(["date", valid["source"].str.casefold()]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=sorted(KNOWN_SOURCES), fill_value=0)
    counts["total_articles"] = counts.sum(axis=1)
    daily = calendar.merge(counts.reset_index(), on="date", how="left").fillna(0)
    for col in ["total_articles", "kompas", "tempo", "detik"]:
        daily[col] = daily[col].astype("int64")
    daily = daily[["date", "total_articles", "kompas", "tempo", "detik"]]
    daily = daily.rename(columns={"kompas": "Kompas", "tempo": "Tempo", "detik": "Detik"})

    total_days = len(daily)
    active_dates = int(daily["total_articles"].gt(0).sum())
    density = pd.DataFrame(
        [
            {"metric": "date_start", "value": start.date().isoformat()},
            {"metric": "date_end", "value": end.date().isoformat()},
            {"metric": "calendar_days", "value": total_days},
            {"metric": "active_dates", "value": active_dates},
            {"metric": "zero_volume_dates", "value": int(total_days - active_dates)},
            {"metric": "active_dates_percent", "value": safe_percentage(active_dates, total_days)},
            {"metric": "median_daily_volume", "value": float(daily["total_articles"].median())},
            {"metric": "mean_daily_volume", "value": float(daily["total_articles"].mean())},
            {"metric": "std_daily_volume", "value": float(daily["total_articles"].std(ddof=1))},
            {"metric": "min_daily_volume", "value": int(daily["total_articles"].min())},
            {"metric": "max_daily_volume", "value": int(daily["total_articles"].max())},
        ]
    )
    for source_name in ["Kompas", "Tempo", "Detik"]:
        coverage = int(daily[source_name].gt(0).sum())
        density.loc[len(density)] = {
            "metric": f"{source_name.casefold()}_coverage_dates",
            "value": coverage,
        }
        density.loc[len(density)] = {
            "metric": f"{source_name.casefold()}_coverage_percent",
            "value": safe_percentage(coverage, total_days),
        }

    source_counts = frame["source"].str.casefold().value_counts().rename_axis("source").reset_index(name="articles")
    source_counts["percentage"] = source_counts["articles"].map(lambda x: safe_percentage(int(x), len(frame)))
    source_counts["active_dates"] = source_counts["source"].map(
        frame.assign(_source=frame["source"].str.casefold()).groupby("_source")["date"].nunique()
    )
    return daily, density, source_counts


def propose_temporal_split(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.dropna(subset=["date"]).copy()
    unique_dates = pd.DatetimeIndex(sorted(valid["date"].dt.normalize().unique()))
    n_dates = len(unique_dates)
    train_n = int(np.floor(n_dates * 0.60))
    validation_n = int(np.floor(n_dates * 0.20))
    test_n = n_dates - train_n - validation_n
    partitions = [
        ("train", unique_dates[:train_n]),
        ("validation", unique_dates[train_n : train_n + validation_n]),
        ("test", unique_dates[train_n + validation_n :]),
    ]
    rows = []
    for name, split_dates in partitions:
        split_mask = valid["date"].dt.normalize().isin(split_dates)
        rows.append(
            {
                "split": name,
                "start_date": split_dates.min().date().isoformat(),
                "end_date": split_dates.max().date().isoformat(),
                "active_dates": int(len(split_dates)),
                "percentage_of_active_dates": safe_percentage(len(split_dates), n_dates),
                "rows": int(split_mask.sum()),
                "percentage_of_rows": safe_percentage(int(split_mask.sum()), len(valid)),
            }
        )
    assert train_n + validation_n + test_n == n_dates
    assert partitions[0][1].max() < partitions[1][1].min() < partitions[2][1].min()
    return pd.DataFrame(rows)


def plot_daily_volume(daily: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    ax.plot(daily["date"], daily["total_articles"], label="Total", color="#172B4D", linewidth=1.8)
    ax.plot(daily["date"], daily["Kompas"], label="Kompas", color="#0052CC", linewidth=0.9, alpha=0.85)
    ax.plot(daily["date"], daily["Tempo"], label="Tempo", color="#FF8B00", linewidth=0.9, alpha=0.85)
    ax.plot(daily["date"], daily["Detik"], label="Detik", color="#36B37E", linewidth=0.9, alpha=0.85)
    ax.set_title("Volume Artikel Harian Setelah Deduplikasi")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Jumlah artikel")
    ax.legend(ncol=4, frameon=False)
    ax.margins(x=0)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_report(
    inspection: dict,
    timestamp_summary: pd.DataFrame,
    dedup: pd.DataFrame,
    quality: pd.DataFrame,
    daily: pd.DataFrame,
    density: pd.DataFrame,
    sources: pd.DataFrame,
    split: pd.DataFrame,
) -> str:
    get_density = density.set_index("metric")["value"].to_dict()
    get_timestamp = timestamp_summary.set_index("metric")["value"].to_dict()
    final_rows = int(dedup.iloc[-1]["rows_after"])
    removed_total = int(inspection["rows"] - final_rows)
    critical_metrics = quality.set_index("metric")["count"].to_dict()
    bertopic_ready = (
        final_rows >= 2000
        and int(get_density["active_dates"]) >= 90
        and int(get_density["zero_volume_dates"]) == 0
        and float(get_timestamp["parse_success_percent"]) >= 99.0
        and int(critical_metrics.get("missing_title", 0)) == 0
    )
    readiness = "READY" if bertopic_ready else "LIMITED"
    monthly_mean = daily.set_index("date")["total_articles"].resample("MS").mean()
    monthly_ratio = monthly_mean.div(monthly_mean.shift(1))
    largest_jump_month = monthly_ratio.idxmax()
    previous_month = largest_jump_month - pd.offsets.MonthBegin(1)
    previous_mean = float(monthly_mean.loc[previous_month])
    jump_mean = float(monthly_mean.loc[largest_jump_month])
    jump_ratio = float(monthly_ratio.loc[largest_jump_month])
    return f"""# Laporan STEP 1–6: inspeksi dan audit temporal

## Ringkasan hasil

- Baris mentah: {inspection['rows']:,}
- Baris final setelah deduplikasi: {final_rows:,}
- Total baris dihapus sebagai duplikat: {removed_total:,}
- Rentang tanggal: {get_density['date_start']} sampai {get_density['date_end']}
- Tanggal aktif: {int(get_density['active_dates']):,}/{int(get_density['calendar_days']):,} ({float(get_density['active_dates_percent']):.2f}%)
- Volume harian: median {float(get_density['median_daily_volume']):.2f}, mean {float(get_density['mean_daily_volume']):.2f}, minimum {int(get_density['min_daily_volume'])}, maksimum {int(get_density['max_daily_volume'])}, SD {float(get_density['std_daily_volume']):.2f}
- Timestamp berhasil diparse: {int(get_timestamp['parsed_success']):,}/{int(get_timestamp['total_rows']):,} ({float(get_timestamp['parse_success_percent']):.4f}%)
- Kesiapan masuk BERTopic: **{readiness}**

Jam tidak dibuat untuk timestamp date-only. `parsed_datetime` hanya berisi nilai untuk record yang memiliki jam; `date` berisi tanggal untuk seluruh record yang berhasil diparse. Timestamp berlabel WIB disimpan dalam zona waktu Asia/Jakarta.

## Deduplikasi

{dedup.to_markdown(index=False)}

Aturan `title normalized` bersifat konservatif: Unicode NFKC, decode HTML entity, case-fold, trim, dan collapse whitespace. Tanda baca tidak dihapus. Pada duplikat, artikel dengan tanggal publikasi paling awal dipertahankan; urutan asli memecahkan tie.

## Distribusi sumber setelah deduplikasi

{sources.to_markdown(index=False)}

## Audit kualitas

{quality.to_markdown(index=False)}

`text_too_short` menggunakan ambang audit Content < {SHORT_TEXT_CHAR_THRESHOLD} karakter. Record hanya ditandai dan tidak dihapus. Token dihitung dengan regex kata Unicode, bukan tokenizer model embedding.

## Audit kepadatan temporal

{density.to_markdown(index=False)}

## Usulan split temporal

{split.to_markdown(index=False)}

Split dibuat dari urutan tanggal aktif aktual dengan rasio 60%/20%/20%. Tidak ada hasil model atau label yang digunakan untuk memilih cutoff. Usulan ini harus dibekukan sebelum tuning model; test tetap tidak disentuh.

## Catatan nonstasioneritas

Kenaikan mean volume bulanan terbesar terjadi dari {previous_month.strftime('%Y-%m')} ({previous_mean:.2f} artikel/hari) ke {largest_jump_month.strftime('%Y-%m')} ({jump_mean:.2f} artikel/hari), atau {jump_ratio:.2f}×. Audit ini tidak menyimpulkan penyebabnya. Perubahan tersebut perlu diperlakukan sebagai potensi perubahan cakupan sumber/regime pengumpulan pada tahap forecasting, terutama karena periode test mencakup lonjakan itu.

## Keputusan

**DATA READY FOR BERTOPIC: {readiness}.** Tahap embedding dan BERTopic belum dijalankan.
"""


def main(project_root: Path) -> None:
    paths = Paths(project_root.resolve())
    ensure_directories(paths)
    if not paths.raw.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {paths.raw}")

    frame = pd.read_csv(paths.raw, low_memory=False)
    required = {"Judul", "Waktu", "Link", "Content", "source"}
    missing_required = sorted(required.difference(frame.columns))
    if missing_required:
        raise ValueError(f"Kolom wajib aktual tidak ditemukan: {missing_required}. Kolom: {list(frame.columns)}")

    inspection = make_inspection(frame, paths.raw)
    with (paths.tables / "dataset_inspection.json").open("w", encoding="utf-8") as handle:
        json.dump(inspection, handle, ensure_ascii=False, indent=2, default=serializable)
    frame.head(10).to_csv(paths.tables / "dataset_sample_10.csv", index=False, encoding="utf-8-sig")

    parsed, timestamp_summary, failures = parse_timestamps(frame)
    timestamp_summary.to_csv(paths.tables / "timestamp_parsing_report.csv", index=False)
    failures.to_csv(paths.tables / "timestamp_parse_failures.csv", index=False, encoding="utf-8-sig")
    failure_rate = len(failures) / len(parsed) if len(parsed) else 1.0
    if failure_rate > 0.01:
        raise RuntimeError(
            f"Timestamp parsing gagal {failure_rate:.2%} (>1%). Tahap berikutnya dihentikan. "
            f"Lihat {paths.tables / 'timestamp_parse_failures.csv'}"
        )

    cleaned, dedup_report = deduplicate(parsed)
    dedup_report.to_csv(paths.tables / "deduplication_report.csv", index=False)

    quality_report, issue_flags = audit_quality(cleaned)
    quality_report.to_csv(paths.tables / "data_quality_report.csv", index=False)
    issue_flags.to_csv(paths.tables / "data_quality_flags.csv", index=False)
    cleaned.to_parquet(paths.processed, index=False, compression="zstd")

    daily, density, source_distribution = temporal_density(cleaned)
    daily.to_csv(paths.tables / "daily_article_volume.csv", index=False)
    density.to_csv(paths.tables / "temporal_density_report.csv", index=False)
    source_distribution.to_csv(paths.tables / "source_distribution.csv", index=False)
    plot_daily_volume(daily, paths.figures / "daily_article_volume.png")

    split = propose_temporal_split(cleaned)
    split.to_csv(paths.tables / "temporal_split.csv", index=False)

    report = build_report(
        inspection,
        timestamp_summary,
        dedup_report,
        quality_report,
        daily,
        density,
        source_distribution,
        split,
    )
    paths.report.write_text(report, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])

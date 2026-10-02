from __future__ import annotations

import json

from src.config import DATA_PATH
from src.dataset_audit import AUDIT_OUTPUT, audit_all_datasets, recommend_dataset


REPORT_OUTPUT = AUDIT_OUTPUT.parents[1] / "dataset_audit_report.md"


def write_report(report, selected: str | None, recommendation: str) -> None:
    lines = [
        "# Dataset Audit Report", "",
        "Audit ini dijalankan sebelum cleaning, embedding, BERTopic, dan machine learning.", "",
        "| Dataset | Rows | Sources | Date range | Active dates | Timestamp | Text | Status |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in report.itertuples(index=False):
        date_range = f"{row.date_min or '-'} — {row.date_max or '-'}"
        lines.append(
            f"| {row.dataset} | {int(row.rows):,} | {int(row.unique_sources)} | {date_range} | "
            f"{int(row.unique_dates)} | {row.timestamp_completeness:.1%} | "
            f"{row.text_completeness:.1%} | {row.status} |"
        )
        lines.extend(["", f"## {row.dataset}: {row.status}", ""])
        lines.extend(f"- {reason}" for reason in json.loads(row.reasons))
    lines.extend([
        "", "## Recommendation", "",
        f"Selected dataset: **{selected or 'NONE'}**.", "",
        recommendation, "",
    ])
    if selected is None:
        lines.append(
            "Pipeline berhenti pada STEP 1. Diperlukan dataset berita nyata dengan lebih dari 2.000 artikel, "
            "rentang minimal tiga bulan, timestamp dan teks yang lengkap, serta sedikitnya tiga sumber atau "
            "satu sumber dengan deret waktu harian yang padat."
        )
    REPORT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    report = audit_all_datasets()
    selected, recommendation = recommend_dataset(report)
    write_report(report, selected, recommendation)
    columns = [
        "dataset", "rows", "unique_sources", "date_min", "date_max",
        "date_span_days", "unique_dates", "timestamp_completeness",
        "text_completeness", "status",
    ]
    printable = report.reindex(columns=columns).copy()
    for column in ["timestamp_completeness", "text_completeness"]:
        printable[column] = printable[column].map(lambda value: f"{value:.1%}" if value == value else "-")
    print(f"Direktori audit: {DATA_PATH.resolve()}")
    print(printable.to_string(index=False))
    print(f"\nLaporan: {AUDIT_OUTPUT.resolve()}")
    print(f"Ringkasan: {REPORT_OUTPUT.resolve()}")
    for row in report.itertuples(index=False):
        print(f"\n{row.dataset}: {row.status}")
        for reason in json.loads(row.reasons):
            print(f"- {reason}")
    print(f"\nRekomendasi: {selected or 'TIDAK ADA'}")
    print(recommendation)
    if selected is None:
        print("Pipeline dihentikan sebelum cleaning, embedding, BERTopic, dan machine learning.")
        return 2
    print("Audit awal lolos. Tahap berikutnya tetap harus dijalankan secara eksplisit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

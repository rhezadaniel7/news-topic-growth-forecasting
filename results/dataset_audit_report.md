# Dataset Audit Report

Audit ini dijalankan sebelum cleaning, embedding, BERTopic, dan machine learning.

| Dataset | Rows | Sources | Date range | Active dates | Timestamp | Text | Status |
|---|---:|---:|---|---:|---:|---:|---|
| news_raw.csv | 455 | 4 | 2026-07-27T02:54:30+00:00 — 2026-09-20T09:45:00+00:00 | 40 | 100.0% | 100.0% | FAIL |

## news_raw.csv: FAIL

- baris 455 belum melebihi minimum 2.000
- rentang 56 hari belum mencapai 3 bulan
- positive growth pada training/validation belum diuji karena dataset gagal pada audit awal

## Recommendation

Selected dataset: **NONE**.

Tidak ada dataset yang melewati minimum temporal forecasting; data acquisition adalah bottleneck.

Pipeline berhenti pada STEP 1. Diperlukan dataset berita nyata dengan lebih dari 2.000 artikel, rentang minimal tiga bulan, timestamp dan teks yang lengkap, serta sedikitnya tiga sumber atau satu sumber dengan deret waktu harian yang padat.

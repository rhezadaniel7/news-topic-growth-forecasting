# Pilot embedding readout

Subset pilot: `pilot31` — 31 topik, 6047 dokumen non-outlier (train 3288, validation 871, test 1888).

Topic assignment, `topic_daily_volume`, label growth, fitur temporal, dan split TIDAK diubah dan TIDAK dihitung ulang. Tidak ada tuning di luar grid kandidat yang sudah ada pada pipeline.

**Surface metrik:** seluruh angka F1/PR-AUC di laporan ini adalah **official validation** (train 2024-07-31..2025-04-26, validation 2025-04-27..2025-07-25). Test tidak diakses di sini karena pipeline hanya mengevaluasi test untuk satu konfigurasi terpilih setelah seleksi dibekukan, dan kontrol itu tidak diubah.

Prosedur per (horizon, feature set, model): kandidat hyperparameter dipilih lewat temporal CV 3 fold, lalu kandidat terpilih dievaluasi pada official validation — identik dengan prosedur ablation pipeline. Nilai yang dilaporkan adalah yang terbaik antar learner per feature set (Majority Class dikecualikan): dipilih satu model dengan F1 official validation tertinggi, dan PR-AUC yang dilaporkan adalah PR-AUC model yang sama itu — bukan maksimum PR-AUC yang diambil independen. Verifikasi: prosedur ini mereproduksi F1 ablation pipeline pada varian legacy sampai 8,3e-17 di sembilan sel (3 horizon x 3 feature set asli).

## 0. Ketersediaan varian

| variant                | embedding_ready   | semantic_features_ready   | status   |
|:-----------------------|:------------------|:--------------------------|:---------|
| legacy_minilm_trunc128 | True              | True                      | READY    |
| minilm_trunc128        | True              | True                      | READY    |
| minilm_chunk128        | True              | True                      | READY    |
| e5small_trunc512       | True              | True                      | READY    |
| e5small_chunk512       | True              | True                      | READY    |

## 1. GATE VALIDITAS — Temporal Only lintas varian

`Temporal Only` tidak memakai embedding, sehingga nilainya wajib identik di semua varian. Toleransi: |F1_varian − F1_legacy| < 1e-06.

| variant                |   horizon | model               |     f1 |   pr_auc |
|:-----------------------|----------:|:--------------------|-------:|---------:|
| e5small_chunk512       |         1 | Random Forest       | 0.1951 |   0.0959 |
| e5small_chunk512       |         2 | XGBoost             | 0.2090 |   0.1111 |
| e5small_chunk512       |         3 | Logistic Regression | 0.2235 |   0.1038 |
| e5small_trunc512       |         1 | Random Forest       | 0.1951 |   0.0959 |
| e5small_trunc512       |         2 | XGBoost             | 0.2090 |   0.1111 |
| e5small_trunc512       |         3 | Logistic Regression | 0.2235 |   0.1038 |
| legacy_minilm_trunc128 |         1 | Random Forest       | 0.1951 |   0.0959 |
| legacy_minilm_trunc128 |         2 | XGBoost             | 0.2090 |   0.1111 |
| legacy_minilm_trunc128 |         3 | Logistic Regression | 0.2235 |   0.1038 |
| minilm_chunk128        |         1 | Random Forest       | 0.1951 |   0.0959 |
| minilm_chunk128        |         2 | XGBoost             | 0.2090 |   0.1111 |
| minilm_chunk128        |         3 | Logistic Regression | 0.2235 |   0.1038 |
| minilm_trunc128        |         1 | Random Forest       | 0.1951 |   0.0959 |
| minilm_trunc128        |         2 | XGBoost             | 0.2090 |   0.1111 |
| minilm_trunc128        |         3 | Logistic Regression | 0.2235 |   0.1038 |

- Deviasi absolut maksimum F1 (nilai terbaik per feature set): **0.000e+00**
- Deviasi absolut maksimum F1 (seluruh sel horizon × model): **0.000e+00**
- **GATE 1: PASS**

## 2. GATE DELTA EMBEDDING — cosine similarity terhadap legacy

Per dokumen, subset identik. Leading indicator: median > 0.9 berarti embedding nyaris tidak berubah.

| variant          |   documents |   cosine_min |   cosine_p25 |   cosine_median |   cosine_p75 |   cosine_max |   share_above_0_9 |
|:-----------------|------------:|-------------:|-------------:|----------------:|-------------:|-------------:|------------------:|
| minilm_trunc128  |        6047 |       1.0000 |       1.0000 |          1.0000 |       1.0000 |       1.0000 |            1.0000 |
| minilm_chunk128  |        6047 |       0.4786 |       0.8066 |          0.8508 |       0.8872 |       1.0000 |            0.1718 |
| e5small_trunc512 |        6047 |      -0.1061 |       0.0017 |          0.0310 |       0.0615 |       0.1963 |            0.0000 |
| e5small_chunk512 |        6047 |      -0.1095 |       0.0024 |          0.0313 |       0.0621 |       0.2143 |            0.0000 |

Caveat: cosine similarity terhadap legacy hanya bermakna sebagai leading indicator untuk varian dengan encoder YANG SAMA. Untuk `e5small_*`, ruang embedding berbeda sehingga cosine mendekati nol menandakan "ruang lain", bukan "embedding terdegradasi".

## 3. EFEK SEBENARNYA

ΔF1 dihitung terhadap `legacy_minilm_trunc128` pada sel yang sama.

### 3a. Semantic Only penuh (7 fitur)

|   horizon | variant                |     f1 |   pr_auc |   delta_f1_vs_legacy |   delta_pr_auc_vs_legacy |
|----------:|:-----------------------|-------:|---------:|---------------------:|-------------------------:|
|         1 | e5small_chunk512       | 0.1856 |   0.0738 |               0.0419 |                   0.0206 |
|         1 | e5small_trunc512       | 0.1346 |   0.0431 |              -0.0091 |                  -0.0101 |
|         1 | legacy_minilm_trunc128 | 0.1437 |   0.0532 |               0.0000 |                   0.0000 |
|         1 | minilm_chunk128        | 0.1892 |   0.0731 |               0.0455 |                   0.0199 |
|         1 | minilm_trunc128        | 0.1437 |   0.0532 |               0.0000 |                   0.0000 |
|         2 | e5small_chunk512       | 0.1290 |   0.0450 |               0.0016 |                   0.0021 |
|         2 | e5small_trunc512       | 0.1348 |   0.0492 |               0.0074 |                   0.0063 |
|         2 | legacy_minilm_trunc128 | 0.1275 |   0.0429 |               0.0000 |                   0.0000 |
|         2 | minilm_chunk128        | 0.1419 |   0.0546 |               0.0145 |                   0.0118 |
|         2 | minilm_trunc128        | 0.1275 |   0.0429 |               0.0000 |                   0.0000 |
|         3 | e5small_chunk512       | 0.1463 |   0.0634 |               0.0004 |                  -0.0099 |
|         3 | e5small_trunc512       | 0.1667 |   0.0656 |               0.0207 |                  -0.0077 |
|         3 | legacy_minilm_trunc128 | 0.1460 |   0.0733 |               0.0000 |                   0.0000 |
|         3 | minilm_chunk128        | 0.1750 |   0.0754 |               0.0290 |                   0.0021 |
|         3 | minilm_trunc128        | 0.1460 |   0.0733 |               0.0000 |                   0.0000 |

### 3b. Semantic Only similarity-only (4 fitur berbasis embedding)

Fitur: `semantic_similarity_mean`, `semantic_similarity_std`, `semantic_drift`, `similarity_change`. Tiga fitur `topic_probability_mean`, `topic_probability_std`, `topic_probability_change` dibuang karena berasal dari `topic_assignments` dan invariant ketika topik dikunci.

|   horizon | variant                |     f1 |   pr_auc |   delta_f1_vs_legacy |   delta_pr_auc_vs_legacy |
|----------:|:-----------------------|-------:|---------:|---------------------:|-------------------------:|
|         1 | e5small_chunk512       | 0.1818 |   0.0895 |               0.0247 |                   0.0424 |
|         1 | e5small_trunc512       | 0.1569 |   0.0676 |              -0.0003 |                   0.0205 |
|         1 | legacy_minilm_trunc128 | 0.1571 |   0.0470 |               0.0000 |                   0.0000 |
|         1 | minilm_chunk128        | 0.1351 |   0.0547 |              -0.0220 |                   0.0077 |
|         1 | minilm_trunc128        | 0.1571 |   0.0470 |               0.0000 |                   0.0000 |
|         2 | e5small_chunk512       | 0.1389 |   0.0515 |               0.0119 |                   0.0097 |
|         2 | e5small_trunc512       | 0.1633 |   0.0565 |               0.0363 |                   0.0146 |
|         2 | legacy_minilm_trunc128 | 0.1270 |   0.0419 |               0.0000 |                   0.0000 |
|         2 | minilm_chunk128        | 0.1538 |   0.0643 |               0.0269 |                   0.0225 |
|         2 | minilm_trunc128        | 0.1270 |   0.0419 |               0.0000 |                   0.0000 |
|         3 | e5small_chunk512       | 0.1504 |   0.0510 |               0.0170 |                  -0.0208 |
|         3 | e5small_trunc512       | 0.1509 |   0.0534 |               0.0176 |                  -0.0184 |
|         3 | legacy_minilm_trunc128 | 0.1333 |   0.0718 |               0.0000 |                   0.0000 |
|         3 | minilm_chunk128        | 0.1609 |   0.0758 |               0.0276 |                   0.0040 |
|         3 | minilm_trunc128        | 0.1333 |   0.0718 |               0.0000 |                   0.0000 |

### 3c. Fusion (Semantic + Temporal), penuh dan similarity-only

|   horizon | variant                |     f1 |   pr_auc |   delta_f1_vs_legacy |   delta_pr_auc_vs_legacy |
|----------:|:-----------------------|-------:|---------:|---------------------:|-------------------------:|
|         1 | e5small_chunk512       | 0.2121 |   0.1620 |               0.0239 |                   0.0412 |
|         1 | e5small_trunc512       | 0.2353 |   0.1408 |               0.0471 |                   0.0200 |
|         1 | legacy_minilm_trunc128 | 0.1882 |   0.1208 |               0.0000 |                   0.0000 |
|         1 | minilm_chunk128        | 0.1887 |   0.1256 |               0.0004 |                   0.0049 |
|         1 | minilm_trunc128        | 0.1882 |   0.1208 |               0.0000 |                   0.0000 |
|         2 | e5small_chunk512       | 0.1975 |   0.1159 |               0.0113 |                   0.0035 |
|         2 | e5small_trunc512       | 0.1988 |   0.1161 |               0.0125 |                   0.0036 |
|         2 | legacy_minilm_trunc128 | 0.1863 |   0.1124 |               0.0000 |                   0.0000 |
|         2 | minilm_chunk128        | 0.1988 |   0.1162 |               0.0126 |                   0.0038 |
|         2 | minilm_trunc128        | 0.1863 |   0.1124 |               0.0000 |                   0.0000 |
|         3 | e5small_chunk512       | 0.2236 |   0.1012 |              -0.0164 |                  -0.0035 |
|         3 | e5small_trunc512       | 0.2206 |   0.1020 |              -0.0194 |                  -0.0027 |
|         3 | legacy_minilm_trunc128 | 0.2400 |   0.1047 |               0.0000 |                   0.0000 |
|         3 | minilm_chunk128        | 0.2388 |   0.1065 |              -0.0012 |                   0.0018 |
|         3 | minilm_trunc128        | 0.2400 |   0.1047 |               0.0000 |                   0.0000 |

|   horizon | variant                |     f1 |   pr_auc |   delta_f1_vs_legacy |   delta_pr_auc_vs_legacy |
|----------:|:-----------------------|-------:|---------:|---------------------:|-------------------------:|
|         1 | e5small_chunk512       | 0.1895 |   0.1428 |               0.0104 |                  -0.0039 |
|         1 | e5small_trunc512       | 0.2391 |   0.1519 |               0.0600 |                   0.0052 |
|         1 | legacy_minilm_trunc128 | 0.1791 |   0.1467 |               0.0000 |                   0.0000 |
|         1 | minilm_chunk128        | 0.2022 |   0.1264 |               0.0231 |                  -0.0202 |
|         1 | minilm_trunc128        | 0.1791 |   0.1467 |               0.0000 |                   0.0000 |
|         2 | e5small_chunk512       | 0.2222 |   0.1170 |               0.0310 |                   0.0042 |
|         2 | e5small_trunc512       | 0.2105 |   0.1177 |               0.0193 |                   0.0049 |
|         2 | legacy_minilm_trunc128 | 0.1912 |   0.1128 |               0.0000 |                   0.0000 |
|         2 | minilm_chunk128        | 0.2036 |   0.1181 |               0.0124 |                   0.0053 |
|         2 | minilm_trunc128        | 0.1912 |   0.1128 |               0.0000 |                   0.0000 |
|         3 | e5small_chunk512       | 0.2299 |   0.1027 |              -0.0129 |                  -0.0073 |
|         3 | e5small_trunc512       | 0.2312 |   0.1036 |              -0.0116 |                  -0.0064 |
|         3 | legacy_minilm_trunc128 | 0.2428 |   0.1100 |               0.0000 |                   0.0000 |
|         3 | minilm_chunk128        | 0.2428 |   0.1064 |               0.0000 |                  -0.0036 |
|         3 | minilm_trunc128        | 0.2428 |   0.1100 |               0.0000 |                   0.0000 |

Gain fusion terhadap Temporal Only pada sel yang sama:

| variant                |   horizon |   temporal_only_f1 |   fusion_full_f1 |   fusion_similarity_f1 |   fusion_full_gain |   fusion_similarity_gain |
|:-----------------------|----------:|-------------------:|-----------------:|-----------------------:|-------------------:|-------------------------:|
| e5small_chunk512       |         1 |             0.1951 |           0.2121 |                 0.1895 |             0.0170 |                  -0.0056 |
| e5small_chunk512       |         2 |             0.2090 |           0.1975 |                 0.2222 |            -0.0114 |                   0.0133 |
| e5small_chunk512       |         3 |             0.2235 |           0.2236 |                 0.2299 |             0.0001 |                   0.0064 |
| e5small_trunc512       |         1 |             0.1951 |           0.2353 |                 0.2391 |             0.0402 |                   0.0440 |
| e5small_trunc512       |         2 |             0.2090 |           0.1988 |                 0.2105 |            -0.0102 |                   0.0016 |
| e5small_trunc512       |         3 |             0.2235 |           0.2206 |                 0.2312 |            -0.0029 |                   0.0077 |
| legacy_minilm_trunc128 |         1 |             0.1951 |           0.1882 |                 0.1791 |            -0.0069 |                  -0.0160 |
| legacy_minilm_trunc128 |         2 |             0.2090 |           0.1863 |                 0.1912 |            -0.0227 |                  -0.0178 |
| legacy_minilm_trunc128 |         3 |             0.2235 |           0.2400 |                 0.2428 |             0.0165 |                   0.0192 |
| minilm_chunk128        |         1 |             0.1951 |           0.1887 |                 0.2022 |            -0.0064 |                   0.0071 |
| minilm_chunk128        |         2 |             0.2090 |           0.1988 |                 0.2036 |            -0.0101 |                  -0.0054 |
| minilm_chunk128        |         3 |             0.2235 |           0.2388 |                 0.2428 |             0.0153 |                   0.0192 |
| minilm_trunc128        |         1 |             0.1951 |           0.1882 |                 0.1791 |            -0.0069 |                  -0.0160 |
| minilm_trunc128        |         2 |             0.2090 |           0.1863 |                 0.1912 |            -0.0227 |                  -0.0178 |
| minilm_trunc128        |         3 |             0.2235 |           0.2400 |                 0.2428 |             0.0165 |                   0.0192 |

## 4. CEK e5 — prefix 'passage:'

| variant          | prefix_in_metadata   | verdict   | reason                                                                       |
|:-----------------|:---------------------|:----------|:-----------------------------------------------------------------------------|
| e5small_trunc512 | passage:             | OK        | prefix 'passage: ' tercatat pada fingerprint payload dan dipakai saat encode |
| e5small_chunk512 | passage:             | OK        | prefix 'passage: ' tercatat pada fingerprint payload dan dipakai saat encode |

## 5. VERDICT

Ambang yang dipakai (ditetapkan sebelum angka dilihat):

- NO-GO jika gate 1 FAIL.
- Condong NO-GO jika median cosine similarity > 0.9 pada semua varian.
- GO hanya jika ΔF1 Semantic Only (similarity-only) ≥ +0.02 vs legacy DAN Fusion similarity-only − Temporal Only ≥ +0.01, konsisten pada ≥2 dari 3 horizon.
- Selain itu NO-GO.

**Verdict akhir dibaca dari `minilm_chunk128` SAJA.** Varian itu mengubah hanya cara truncation ditangani, dengan encoder identik, sehingga efeknya dapat diatribusikan ke truncation. Varian `e5*` mengganti encoder sekaligus sehingga confounded: dilaporkan sebagai sekunder dan tidak menentukan verdict, serta ditandai SUSPECT bila prefix 'passage:' tidak terkonfirmasi.

Kondisi Fusion pada ambang GO memakai **fusion similarity-only**. Kolom `horizons_meeting_full_fusion_diagnostic` dan `max_fusion_full_gain` dicetak sebagai diagnostik dilusi (efek yang terlarut oleh 3 fitur `topic_probability_*` yang invariant), bukan sebagai veto.

| variant          | role                                    | verdict   | suspect   | suspect_reason   |   cosine_median_vs_legacy |   horizons_meeting_go_condition |   horizons_meeting_full_fusion_diagnostic |   max_delta_semantic_similarity_f1 |   max_fusion_similarity_gain |   max_fusion_full_gain |
|:-----------------|:----------------------------------------|:----------|:----------|:-----------------|--------------------------:|--------------------------------:|------------------------------------------:|-----------------------------------:|-----------------------------:|-----------------------:|
| minilm_trunc128  | SECONDARY                               | NO-GO     | False     |                  |                    1.0000 |                               0 |                                         0 |                             0.0000 |                       0.0192 |                 0.0165 |
| minilm_chunk128  | PRIMARY                                 | NO-GO     | False     |                  |                    0.8508 |                               1 |                                         1 |                             0.0276 |                       0.0192 |                 0.0153 |
| e5small_trunc512 | SECONDARY (confounded: encoder berubah) | NO-GO     | False     |                  |                    0.0310 |                               0 |                                         0 |                             0.0363 |                       0.0440 |                 0.0402 |
| e5small_chunk512 | SECONDARY (confounded: encoder berubah) | NO-GO     | False     |                  |                    0.0313 |                               0 |                                         1 |                             0.0247 |                       0.0133 |                 0.0170 |

**VERDICT AKHIR: NO-GO — varian primer minilm_chunk128 memenuhi ambang GO hanya pada 1/3 horizon; efek pilot datar; pilot ber-CI lebar, Δ kecil = noise.**

## 6. Keterbatasan

Readout ini adalah pengukuran pilot, bukan bukti. Yang membatasi kekuatan
kesimpulannya, dari yang paling menentukan:

1. **Daya statistik pilot sangat kecil.** Subset hanya 31 dari 170 topik (18%) dan
   6.047 dari 37.933 dokumen non-outlier (16%). Official validation pilot hanya
   memuat 42 observasi positif pada t+1 (43 pada t+2, 44 pada t+3). Pada ukuran
   ini, selisih F1 sebesar 0,02-0,05 berada di dalam lebar interval kepercayaan
   yang wajar, sehingga verdict NO-GO harus dibaca sebagai "efek tidak terdeteksi
   pada daya yang tersedia", BUKAN sebagai "efek tidak ada". Tidak ada interval
   kepercayaan maupun uji signifikansi yang dihitung di sini; satu run, satu seed.

2. **Ruang topik tetap berasal dari embedding ter-truncate.** Sesuai keputusan
   mengunci topik, `topic_assignments`, `topic_daily_volume`, dan label tidak
   dihitung ulang. Artinya UMAP dan BERTopic masih di-fit pada embedding 128
   token, dan hanya centroid semantik yang memakai embedding baru. Angka di sini
   karena itu adalah BATAS BAWAH untuk manfaat mengatasi truncation: jika ruang
   topik ikut diperbaiki, efeknya bisa lebih besar. Sebaliknya, ini juga berarti
   readout tidak dapat menjawab apakah topik yang lebih baik akan menolong.

3. **Tiga dari tujuh fitur semantik invariant.** `topic_probability_mean`,
   `topic_probability_std`, dan `topic_probability_change` berasal dari
   `topic_assignments`, bukan dari embedding, sehingga identik di semua varian.
   Itulah sebabnya keputusan dibaca dari feature set similarity-only; angka
   Semantic Only penuh dan Fusion penuh mengandung dilusi yang tidak berhubungan
   dengan kualitas embedding.

4. **Hanya official validation yang diukur.** Test tidak diakses karena pipeline
   hanya mengevaluasi test untuk satu konfigurasi yang sudah dibekukan, dan
   kontrol itu tidak diubah. Tidak ada konfirmasi out-of-sample untuk selisih
   antar varian.

5. **Probability threshold dipilih pada set yang dilaporkan.** `choose_probability_threshold`
   memaksimalkan F1 pada validation yang sama yang kemudian dilaporkan. Ini
   perilaku pipeline yang sudah berlaku dan sama untuk semua varian, sehingga
   perbandingan antar varian tetap adil, tetapi level absolut F1 optimistis dan
   tidak boleh dibaca sebagai performa yang diharapkan.

6. **Varian e5 confounded secara desain.** `e5small_trunc512` dan
   `e5small_chunk512` mengubah encoder, batas token, dan data pralatih sekaligus,
   sehingga efeknya tidak dapat diatribusikan ke penanganan truncation. Keduanya
   dilaporkan sebagai sekunder dan tidak menentukan verdict, apa pun angkanya.
   Cosine similarity terhadap legacy juga tidak dapat ditafsirkan untuk keduanya
   karena ruang embedding berbeda.

7. **Validitas pilot bergantung pada threshold label yang tetap.** Skema label
   `burst_mad_k2` memakai threshold tetap, sehingga label tidak berubah ketika
   populasi topik dibatasi. Dengan skema label berbasis kuantil, label pada subset
   pilot TIDAK akan identik dengan run penuh dan perbandingan ini tidak sah.

8. **Distribution shift yang sudah terdokumentasi tetap ada.** Volume artikel per
   hari naik dari 125,6 (train) ke 244,6 (test). Readout ini tidak menyentuh dan
   tidak memperbaiki hal itu.

Implikasi praktis: NO-GO di sini adalah keputusan tentang alokasi waktu komputasi
(tidak menjalankan encoding penuh 6-8 jam sekarang), bukan kesimpulan ilmiah
bahwa truncation tidak penting. Truncation terbukti nyata dan terselesaikan oleh
chunking (cakupan token 23,8% menjadi 100%), tetapi pada daya pilot yang tersedia
perbaikan itu belum terbaca sebagai kenaikan kontribusi semantik yang konsisten.

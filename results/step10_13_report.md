# STEP 10–13 report

## 1. Topic time series

- Topik non-outlier: 182
- Tanggal: 450
- Topic-day: 81,900
- Total volume non-outlier: 37,933
- Semua kombinasi topic-date tersedia; tanggal tanpa dokumen diberi volume nol.
- Dokumen outlier tidak dihitung sebagai volume topik dan diaudit secara terpisah.

## 2. Topic sparsity

Threshold ditetapkan sebelum evaluasi label: LOW <50% hari nol; MEDIUM 50–<80%; HIGH ≥80%.

- Kategori: {'HIGH SPARSITY': 115, 'MEDIUM SPARSITY': 51, 'LOW SPARSITY': 16}
- Active days seluruh periode ≥14/30/60/90: {14: 176, 30: 147, 60: 98, 90: 68}
- Active days training ≥14/30/60/90: {14: 170, 30: 106, 60: 44, 90: 22}
- Topic layak berdasarkan ≥14 active days training: 170/182

## 3. Semantic features

Centroid topik bersifat kumulatif dan hanya memakai dokumen hingga origin t. Similarity mengukur dokumen hari t terhadap centroid kumulatif t. Drift adalah `1 − cosine(centroid_t, centroid_t−1)`. Hari tanpa dokumen menghasilkan semantic feature missing, bukan nilai buatan.

| feature_family   | feature                  |   count |   missing |   finite |         min |        mean |       median |       std |      max |
|:-----------------|:-------------------------|--------:|----------:|---------:|------------:|------------:|-------------:|----------:|---------:|
| semantic         | semantic_similarity_mean |   16565 |     65335 |    16565 |  0.0613074  |  0.751064   |  0.7607      | 0.100363  | 1        |
| semantic         | semantic_similarity_std  |   16565 |     65335 |    16565 |  0          |  0.0207046  |  0           | 0.0331828 | 0.251251 |
| semantic         | semantic_drift           |   16383 |     65517 |    16383 |  9.2747e-08 |  0.00265641 |  6.58253e-05 | 0.0149518 | 0.426946 |
| semantic         | topic_probability_mean   |   16565 |     65335 |    16565 |  0.0183422  |  0.793269   |  0.886492    | 0.237298  | 1        |
| semantic         | topic_probability_std    |   16565 |     65335 |    16565 |  0          |  0.0387431  |  0           | 0.072647  | 0.473378 |
| semantic         | topic_probability_change |   16383 |     65517 |    16383 | -0.946441   | -0.00025794 |  0           | 0.219736  | 0.948117 |
| semantic         | similarity_change        |   16383 |     65517 |    16383 | -0.615256   | -0.00279509 | -0.00166237  | 0.10238   | 0.726041 |

## 4. Temporal features

`current_volume` menggunakan volume origin t. Lag dan rolling memakai masa lalu; rolling menerapkan `shift(1)`. Growth rate memakai denominator `abs(lag_1)+1`; percentage change memakai `max(abs(lag_1),1)`. `zero_baseline_flag` mempertahankan kasus baseline nol tanpa menghasilkan infinity.

| metric                             |   value |
|:-----------------------------------|--------:|
| total_topic_days                   |   81900 |
| zero_baseline_count                |   65180 |
| raw_percentage_infinite_candidates |    8580 |
| growth_rate_nan                    |     182 |
| growth_rate_infinite               |       0 |
| growth_rate_finite                 |   81718 |
| percentage_change_nan              |     182 |
| percentage_change_infinite         |       0 |
| percentage_change_finite           |   81718 |

| feature_family   | feature            |   count |   missing |   finite |         min |        mean |   median |      std |     max |
|:-----------------|:-------------------|--------:|----------:|---------:|------------:|------------:|---------:|---------:|--------:|
| temporal         | current_volume     |   81900 |         0 |    81900 |    0        |  0.463162   | 0        | 1.61366  | 79      |
| temporal         | lag_1              |   81718 |       182 |    81718 |    0        |  0.463521   | 0        | 1.6144   | 79      |
| temporal         | lag_2              |   81536 |       364 |    81536 |    0        |  0.462397   | 0        | 1.61351  | 79      |
| temporal         | lag_3              |   81354 |       546 |    81354 |    0        |  0.461403   | 0        | 1.61317  | 79      |
| temporal         | lag_4              |   81172 |       728 |    81172 |    0        |  0.459888   | 0        | 1.61001  | 79      |
| temporal         | rolling_mean_3     |   81354 |       546 |    81354 |    0        |  0.463059   | 0        | 1.36085  | 40.3333 |
| temporal         | rolling_mean_7     |   80626 |      1274 |    80626 |    0        |  0.461749   | 0.142857 | 1.22078  | 27.1429 |
| temporal         | rolling_std_7      |   80626 |      1274 |    80626 |    0        |  0.504099   | 0.349927 | 0.929326 | 24.3218 |
| temporal         | growth_rate        |   81718 |       182 |    81718 |   -0.958333 |  0.122662   | 0        | 0.770607 | 38      |
| temporal         | percentage_change  |   81718 |       182 |    81718 |   -1        |  0.0965286  | 0        | 0.913075 | 45      |
| temporal         | momentum           |   81536 |       364 |    81536 | -123        | -0.00177836 | 0        | 2.42383  | 71      |
| temporal         | burst_intensity    |   80626 |      1274 |    80626 |   -5        |  0.0365452  | 0        | 0.595234 | 32      |
| temporal         | zero_baseline_flag |   81900 |         0 |    81900 |    0        |  0.795849   | 1        | 0.403083 |  1      |

## 5. Growth score distribution training

Growth score = `(future_volume − historical_median) / (1 + 1.4826 × MAD)`. Histori dimulai pada kemunculan pertama topik, minimal 14 hari kalender, dan origin-target wajib berada dalam split yang sama.

|   horizon |   count |   min |   max |   median |     mean |   p50 |   p75 |   p80 |   p85 |   p90 |   p95 |   count_greater_than_zero |
|----------:|--------:|------:|------:|---------:|---------:|------:|------:|------:|------:|------:|------:|--------------------------:|
|         1 |   40295 |    -5 |    60 |        0 | 0.298703 |     0 |     0 |     0 |     1 |     1 |     2 |                      7314 |
|         2 |   40125 |    -5 |    60 |        0 | 0.299473 |     0 |     0 |     0 |     1 |     1 |     2 |                      7284 |
|         3 |   39955 |    -5 |    60 |        0 | 0.299779 |     0 |     0 |     0 |     1 |     1 |     2 |                      7253 |

## 6. Threshold sensitivity

Threshold p75/p80/p85/p90 seluruhnya dihitung dari training.

WARNING: p80 bernilai nol pada horizon [1, 2, 3]. Konsekuensinya diperiksa pada bagian 6b: `growth_score > 0` identik dengan `future_volume > median historis`, dan karena median historis nol pada mayoritas observasi eligible, label kolaps menjadi "ada artikel pada target date". Kuantil p80 karena itu tidak lagi dipakai sebagai definisi label utama.

|   horizon | threshold_quantile   |   threshold_value | threshold_fit_split   | split   |   total_observations |   growth_count |   normal_count |   growth_rate |   active_topics |   temporal_origins |
|----------:|:---------------------|------------------:|:----------------------|:--------|---------------------:|---------------:|---------------:|--------------:|----------------:|-------------------:|
|         1 | p75                  |                 0 | train                 | train   |                40295 |           7314 |          32981 |     0.181511  |             170 |                256 |
|         1 | p80                  |                 0 | train                 | train   |                40295 |           7314 |          32981 |     0.181511  |             170 |                256 |
|         1 | p85                  |                 1 | train                 | train   |                40295 |           2459 |          37836 |     0.0610249 |             170 |                256 |
|         1 | p90                  |                 1 | train                 | train   |                40295 |           2459 |          37836 |     0.0610249 |             170 |                256 |
|         2 | p75                  |                 0 | train                 | train   |                40125 |           7284 |          32841 |     0.181533  |             170 |                255 |
|         2 | p80                  |                 0 | train                 | train   |                40125 |           7284 |          32841 |     0.181533  |             170 |                255 |
|         2 | p85                  |                 1 | train                 | train   |                40125 |           2452 |          37673 |     0.061109  |             170 |                255 |
|         2 | p90                  |                 1 | train                 | train   |                40125 |           2452 |          37673 |     0.061109  |             170 |                255 |
|         3 | p75                  |                 0 | train                 | train   |                39955 |           7253 |          32702 |     0.181529  |             170 |                254 |
|         3 | p80                  |                 0 | train                 | train   |                39955 |           7253 |          32702 |     0.181529  |             170 |                254 |
|         3 | p85                  |                 1 | train                 | train   |                39955 |           2443 |          37512 |     0.0611438 |             170 |                254 |
|         3 | p90                  |                 1 | train                 | train   |                39955 |           2443 |          37512 |     0.0611438 |             170 |                254 |

## 6b. Zero-inflation growth_score menurut sparsity

Distribusi growth_score training (t+1) dipisah kelas sparsity topik. Kolom `share_exactly_zero` adalah proporsi observasi dengan growth_score tepat nol; `share_median_zero` dan `share_mad_zero` adalah proporsi observasi dengan median dan MAD historis nol.

| sparsity_category   |   topics |   observations |   share_negative |   share_exactly_zero |   share_positive |   share_median_zero |   share_mad_zero |   p80 |   p90 |     max |
|:--------------------|---------:|---------------:|-----------------:|---------------------:|-----------------:|--------------------:|-----------------:|------:|------:|--------:|
| LOW SPARSITY        |       16 |           4060 |       0.325369   |             0.290148 |         0.384483 |            0.265025 |         0.265517 |     1 |     2 | 31.4187 |
| MEDIUM SPARSITY     |       51 |          12659 |       0.0155621  |             0.738921 |         0.245517 |            0.974564 |         0.97559  |     1 |     1 | 32      |
| HIGH SPARSITY       |      103 |          23576 |       0.00339328 |             0.884416 |         0.11219  |            0.995037 |         0.995801 |     0 |     1 | 60      |

Figure: `results/figures/growth_score_histogram_by_sparsity.png`. Tabel bin penuh: `results/tables/growth_score_histogram_by_sparsity.csv`.

## 6c. Perbandingan skema label

Empat skema dibandingkan pada populasi eligible yang sama. Semua threshold yang di-fit hanya memakai baris split training; skema burst tidak melakukan fitting sama sekali.

| scheme                | population      | threshold_kind          |   threshold_value | split      |   observations |   growth_count |   normal_count |   growth_rate |
|:----------------------|:----------------|:------------------------|------------------:|:-----------|---------------:|---------------:|---------------:|--------------:|
| baseline_quantile_p80 | all             | train_quantile          |                 0 | train      |          40295 |           7314 |          32981 |     0.181511  |
| baseline_quantile_p80 | all             | train_quantile          |                 0 | validation |          15130 |           2579 |          12551 |     0.170456  |
| baseline_quantile_p80 | all             | train_quantile          |                 0 | test       |          15130 |           4123 |          11007 |     0.272505  |
| active_baseline_p80   | active_baseline | train_quantile          |                 1 | train      |           8322 |           1189 |           7133 |     0.142874  |
| active_baseline_p80   | active_baseline | train_quantile          |                 1 | validation |           2927 |            396 |           2531 |     0.135292  |
| active_baseline_p80   | active_baseline | train_quantile          |                 1 | test       |           4373 |           1232 |           3141 |     0.281729  |
| positive_only_p80     | all             | train_quantile_positive |                 2 | train      |          40295 |           1181 |          39114 |     0.0293088 |
| positive_only_p80     | all             | train_quantile_positive |                 2 | validation |          15130 |            329 |          14801 |     0.0217449 |
| positive_only_p80     | all             | train_quantile_positive |                 2 | test       |          15130 |            909 |          14221 |     0.0600793 |
| burst_mad_k2          | all             | fixed                   |                 2 | train      |          40295 |           1374 |          38921 |     0.0340985 |
| burst_mad_k2          | all             | fixed                   |                 2 | validation |          15130 |            406 |          14724 |     0.0268341 |
| burst_mad_k2          | all             | fixed                   |                 2 | test       |          15130 |           1045 |          14085 |     0.0690681 |

Skema label yang dipakai sebagai label utama: **burst_mad_k2** — future_volume > median historis + 2 * max(1,4826 * MAD historis, 1,0).

Alasan: Threshold tetap, tidak di-fit dari distribusi, sehingga arti label tidak berubah ketika distribusi volume bergeser antar split. Floor skala 1,0 artikel dinyatakan eksplisit untuk MAD = 0.

Tabel lengkap tiga horizon: `results/tables/label_scheme_class_balance.csv`.

## 7. Positive-event distribution dan 8. horizon feasibility

|   horizon | threshold_quantile   |   threshold_value | threshold_fit_split   | split      |   total_observations |   growth_count |   normal_count |   growth_rate |   active_topics |   temporal_origins | status   |
|----------:|:---------------------|------------------:|:----------------------|:-----------|---------------------:|---------------:|---------------:|--------------:|----------------:|-------------------:|:---------|
|         1 | p80_primary          |                 0 | train                 | train      |                40295 |           7314 |          32981 |      0.181511 |             170 |                256 | READY    |
|         1 | p80_primary          |                 0 | train                 | validation |                15130 |           2579 |          12551 |      0.170456 |             170 |                 89 | READY    |
|         1 | p80_primary          |                 0 | train                 | test       |                15130 |           4123 |          11007 |      0.272505 |             170 |                 89 | READY    |
|         2 | p80_primary          |                 0 | train                 | train      |                40125 |           7284 |          32841 |      0.181533 |             170 |                255 | READY    |
|         2 | p80_primary          |                 0 | train                 | validation |                14960 |           2542 |          12418 |      0.16992  |             170 |                 88 | READY    |
|         2 | p80_primary          |                 0 | train                 | test       |                14960 |           4097 |          10863 |      0.273864 |             170 |                 88 | READY    |
|         3 | p80_primary          |                 0 | train                 | train      |                39955 |           7253 |          32702 |      0.181529 |             170 |                254 | READY    |
|         3 | p80_primary          |                 0 | train                 | validation |                14790 |           2516 |          12274 |      0.170115 |             170 |                 87 | READY    |
|         3 | p80_primary          |                 0 | train                 | test       |                14790 |           4073 |          10717 |      0.275389 |             170 |                 87 | READY    |

## 9. Temporal distribution shift

| split      | dimension                  | metric            |        value |
|:-----------|:---------------------------|:------------------|-------------:|
| train      | total_daily_article_volume | total             | 33899        |
| train      | total_daily_article_volume | median            |   132        |
| train      | total_daily_article_volume | mean              |   125.552    |
| train      | total_daily_article_volume | std               |    14.9264   |
| train      | topic_volume_distribution  | median            |     0        |
| train      | topic_volume_distribution  | mean              |     0.41805  |
| train      | topic_volume_distribution  | std               |     1.58892  |
| train      | topic_volume_distribution  | zero_ratio        |     0.81512  |
| train      | outlier_daily_volume       | total             | 13356        |
| train      | outlier_daily_volume       | mean              |    49.4667   |
| train      | source_distribution        | detik_percentage  |    11.2304   |
| train      | source_distribution        | kompas_percentage |    61.9251   |
| train      | source_distribution        | tempo_percentage  |    26.8444   |
| validation | total_daily_article_volume | total             | 11243        |
| validation | total_daily_article_volume | median            |   127        |
| validation | total_daily_article_volume | mean              |   124.922    |
| validation | total_daily_article_volume | std               |    10.3677   |
| validation | topic_volume_distribution  | median            |     0        |
| validation | topic_volume_distribution  | mean              |     0.356349 |
| validation | topic_volume_distribution  | std               |     1.14141  |
| validation | topic_volume_distribution  | zero_ratio        |     0.816789 |
| validation | outlier_daily_volume       | total             |  5406        |
| validation | outlier_daily_volume       | mean              |    60.0667   |
| validation | source_distribution        | detik_percentage  |    11.6428   |
| validation | source_distribution        | kompas_percentage |    63.4528   |
| validation | source_distribution        | tempo_percentage  |    24.9044   |
| test       | total_daily_article_volume | total             | 22018        |
| test       | total_daily_article_volume | median            |   265.5      |
| test       | total_daily_article_volume | mean              |   244.644    |
| test       | total_daily_article_volume | std               |   114.277    |
| test       | topic_volume_distribution  | median            |     0        |
| test       | topic_volume_distribution  | mean              |     0.705311 |
| test       | topic_volume_distribution  | std               |     2.01664  |
| test       | topic_volume_distribution  | zero_ratio        |     0.726557 |
| test       | outlier_daily_volume       | total             | 10465        |
| test       | outlier_daily_volume       | mean              |   116.278    |
| test       | source_distribution        | detik_percentage  |     6.36752  |
| test       | source_distribution        | kompas_percentage |    32.4371   |
| test       | source_distribution        | tempo_percentage  |    61.1954   |

Perubahan distribusi hanya dilaporkan; audit ini tidak menyimpulkan penyebabnya. Source digunakan sebagai audit dan tidak dimasukkan sebagai predictor.

## 10. Leakage audit

| check                              | passed   | detail                                                             |
|:-----------------------------------|:---------|:-------------------------------------------------------------------|
| target_not_in_features             | True     | []                                                                 |
| future_volume_not_in_features      | True     | future_volume hanya ada pada label audit                           |
| future_timestamp_not_in_features   | True     | target_date hanya ada pada label audit                             |
| feature_timestamp_not_after_origin | True     | semantic dan temporal feature_timestamp <= origin date             |
| semantic_centroid_not_future       | True     | centroid kumulatif berhenti pada origin date                       |
| rolling_features_past_only         | True     | rolling windows memakai shift(1)                                   |
| growth_score_not_predictor         | True     | growth_score hanya label audit                                     |
| threshold_training_only            | True     | max origin=2025-04-25 00:00:00; max target=2025-04-26 00:00:00     |
| embedding_fixed_pretrained_encoder | True     | pretrained SentenceTransformer; tidak ada fit pada corpus future   |
| topic_model_train_only             | True     | fit_scope=training_only; train_period=['2024-07-31', '2025-04-26'] |
| forbidden_columns_absent           | True     | []                                                                 |

## 11. Recommendation

- FEATURE ENGINEERING STATUS: **READY**
- LABEL STATUS: **READY**
- FORECASTING READINESS: **LIMITED**

`FORECASTING READINESS` tetap LIMITED walaupun label memiliki dua kelas, karena topic model berstatus LIMITED, truncation embedding sangat tinggi, outlier future sekitar 48%, dan terdapat distribution shift volume/source. Classifier belum dijalankan.

# Modeling report

## 0. Definisi label

Label growth memakai skema **burst_mad_k2** (S3 burst MAD k=2 (threshold tetap)).

- Definisi: `future_volume > median historis + 2 * max(1,4826 * MAD historis, 1,0)`
- Jenis threshold: `fixed`
- Populasi: `all`
- Alasan pemilihan: Threshold tetap, tidak di-fit dari distribusi, sehingga arti label tidak berubah ketika distribusi volume bergeser antar split. Floor skala 1,0 artikel dinyatakan eksplisit untuk MAD = 0.

Positive rate label per split (dari partisi resmi):

|   horizon |   test |   train |   validation |
|----------:|-------:|--------:|-------------:|
|         1 | 0.0691 |  0.0341 |       0.0268 |
|         2 | 0.0698 |  0.0343 |       0.0268 |
|         3 | 0.0707 |  0.0343 |       0.0268 |

## 1. Feature dataset

Final dataset menggabungkan semantic, temporal, dan label audit dengan key `(topic_id, date, forecast_horizon)`. Tidak ada duplicate key. Target, future volume, target date, growth score, burst score, dan source ID tidak digunakan sebagai predictor.

| record_type   | label_scheme   |   label_threshold |   horizon | split      | feature   |   rows |   growth |   normal |   missing |   missing_rate |
|:--------------|:---------------|------------------:|----------:|:-----------|:----------|-------:|---------:|---------:|----------:|---------------:|
| partition     | burst_mad_k2   |                 2 |         1 | train      |           |  40295 |     1374 |    38921 |    223874 |       0.292414 |
| partition     | burst_mad_k2   |                 2 |         1 | validation |           |  15130 |      406 |    14724 |     85414 |       0.297123 |
| partition     | burst_mad_k2   |                 2 |         1 | test       |           |  15130 |     1045 |    14085 |     75061 |       0.261109 |
| partition     | burst_mad_k2   |                 2 |         2 | train      |           |  40125 |     1376 |    38749 |    222887 |       0.292359 |
| partition     | burst_mad_k2   |                 2 |         2 | validation |           |  14960 |      401 |    14559 |     84434 |       0.297052 |
| partition     | burst_mad_k2   |                 2 |         2 | test       |           |  14960 |     1044 |    13916 |     74347 |       0.261564 |
| partition     | burst_mad_k2   |                 2 |         3 | train      |           |  39955 |     1371 |    38584 |    221956 |       0.292376 |
| partition     | burst_mad_k2   |                 2 |         3 | validation |           |  14790 |      396 |    14394 |     83482 |       0.297078 |
| partition     | burst_mad_k2   |                 2 |         3 | test       |           |  14790 |     1046 |    13744 |     73654 |       0.262105 |

## 2. Temporal folds

Digunakan tiga expanding-window fold dengan validation block yang tidak tumpang tindih. Jumlah fold valid: **3/3**. Test tidak digunakan untuk tuning.

|   fold |   horizon | train_start   | train_end   | validation_start   | validation_end   |   n_train |   n_validation |   growth_train |   normal_train |   growth_validation |   normal_validation | status   |
|-------:|----------:|:--------------|:------------|:-------------------|:-----------------|----------:|---------------:|---------------:|---------------:|--------------------:|--------------------:|:---------|
|      1 |         1 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |     25282 |          10083 |            876 |          24406 |                 342 |                9741 | VALID    |
|      2 |         1 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |     35365 |          10030 |           1218 |          34147 |                 307 |                9723 | VALID    |
|      3 |         1 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |     45395 |          10030 |           1525 |          43870 |                 255 |                9775 | VALID    |
|      1 |         2 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |     25282 |          10083 |            881 |          24401 |                 343 |                9740 | VALID    |
|      2 |         2 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |     35365 |           9860 |           1224 |          34141 |                 300 |                9560 | VALID    |
|      3 |         2 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |     45225 |           9860 |           1524 |          43701 |                 253 |                9607 | VALID    |
|      1 |         3 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |     25282 |          10083 |            880 |          24402 |                 343 |                9740 | VALID    |
|      2 |         3 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |     35365 |           9690 |           1223 |          34142 |                 297 |                9393 | VALID    |
|      3 |         3 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |     45055 |           9690 |           1520 |          43535 |                 247 |                9443 | VALID    |

## 3. Model configuration dan 4. hyperparameters

Model: Majority Class, Logistic Regression, Random Forest, dan XGBoost. Setiap learner diuji dengan Temporal Only, Semantic Only, dan Semantic + Temporal. Logistic memakai median imputation dan StandardScaler; tree model memakai median imputation. Class weighting diterapkan hanya saat fitting training. Tidak ada SMOTE.

Konfigurasi final per horizon dibekukan dari official validation:

```json
{
  "1": {
    "model": "XGBoost",
    "feature_set": "Semantic + Temporal",
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 3,
      "learning_rate": 0.05,
      "subsample": 0.85,
      "colsample_bytree": 0.85
    },
    "probability_threshold": 0.8,
    "growth_label_threshold": 2.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.3030973451327434,
    "validation_pr_auc": 0.2224317844038609
  },
  "2": {
    "model": "Logistic Regression",
    "feature_set": "Temporal Only",
    "hyperparameters": {
      "C": 1.0,
      "max_iter": 2000,
      "class_weight": "balanced",
      "solver": "lbfgs"
    },
    "probability_threshold": 0.79,
    "growth_label_threshold": 2.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.2807017543859649,
    "validation_pr_auc": 0.18915256585051643
  },
  "3": {
    "model": "XGBoost",
    "feature_set": "Temporal Only",
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 3,
      "learning_rate": 0.05,
      "subsample": 0.85,
      "colsample_bytree": 0.85
    },
    "probability_threshold": 0.78,
    "growth_label_threshold": 2.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.2689721421709894,
    "validation_pr_auc": 0.17252274229187656
  }
}
```

## 5. Validation performance

|   horizon | feature_set         | model               |   precision |   recall |       f1 |   macro_f1 |   roc_auc |    pr_auc |   probability_threshold |
|----------:|:--------------------|:--------------------|------------:|---------:|---------:|-----------:|----------:|----------:|------------------------:|
|         1 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.4932   |  0.5      | 0.0268341 |                    0.5  |
|         1 | Temporal Only       | Logistic Regression |    0.230548 | 0.394089 | 0.290909 |   0.63208  |  0.866496 | 0.22003   |                    0.73 |
|         1 | Temporal Only       | Random Forest       |    0.257796 | 0.305419 | 0.279594 |   0.62892  |  0.850523 | 0.200595  |                    0.75 |
|         1 | Temporal Only       | XGBoost             |    0.255692 | 0.359606 | 0.298874 |   0.637741 |  0.863625 | 0.22592   |                    0.82 |
|         1 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.4932   |  0.5      | 0.0268341 |                    0.5  |
|         1 | Semantic Only       | Logistic Regression |    0.185961 | 0.371921 | 0.247947 |   0.608203 |  0.716707 | 0.109074  |                    0.74 |
|         1 | Semantic Only       | Random Forest       |    0.22037  | 0.293103 | 0.251586 |   0.613717 |  0.7407   | 0.138639  |                    0.76 |
|         1 | Semantic Only       | XGBoost             |    0.202096 | 0.332512 | 0.251397 |   0.611925 |  0.735824 | 0.137512  |                    0.82 |
|         1 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.4932   |  0.5      | 0.0268341 |                    0.5  |
|         1 | Semantic + Temporal | Logistic Regression |    0.269307 | 0.334975 | 0.298573 |   0.6384   |  0.863622 | 0.213987  |                    0.83 |
|         1 | Semantic + Temporal | Random Forest       |    0.241026 | 0.347291 | 0.284561 |   0.630169 |  0.851105 | 0.218446  |                    0.65 |
|         1 | Semantic + Temporal | XGBoost             |    0.2751   | 0.337438 | 0.303097 |   0.640818 |  0.859892 | 0.222432  |                    0.8  |
|         2 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.493208 |  0.5      | 0.0268048 |                    0.5  |
|         2 | Temporal Only       | Logistic Regression |    0.264317 | 0.299252 | 0.280702 |   0.629771 |  0.844754 | 0.189153  |                    0.79 |
|         2 | Temporal Only       | Random Forest       |    0.26969  | 0.281796 | 0.27561  |   0.627599 |  0.836904 | 0.191627  |                    0.74 |
|         2 | Temporal Only       | XGBoost             |    0.226917 | 0.361596 | 0.278846 |   0.626438 |  0.839875 | 0.196906  |                    0.78 |
|         2 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.493208 |  0.5      | 0.0268048 |                    0.5  |
|         2 | Semantic Only       | Logistic Regression |    0.163095 | 0.341646 | 0.22079  |   0.593536 |  0.693377 | 0.089794  |                    0.69 |
|         2 | Semantic Only       | Random Forest       |    0.157576 | 0.32419  | 0.212072 |   0.589203 |  0.718316 | 0.107746  |                    0.63 |
|         2 | Semantic Only       | XGBoost             |    0.164835 | 0.336658 | 0.221311 |   0.594105 |  0.717365 | 0.11226   |                    0.75 |
|         2 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.493208 |  0.5      | 0.0268048 |                    0.5  |
|         2 | Semantic + Temporal | Logistic Regression |    0.211864 | 0.374065 | 0.270514 |   0.621217 |  0.842761 | 0.179715  |                    0.73 |
|         2 | Semantic + Temporal | Random Forest       |    0.20887  | 0.36409  | 0.265455 |   0.618709 |  0.835473 | 0.195405  |                    0.63 |
|         2 | Semantic + Temporal | XGBoost             |    0.246032 | 0.309227 | 0.274033 |   0.625695 |  0.832976 | 0.202256  |                    0.77 |
|         3 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.493215 |  0.5      | 0.0267748 |                    0.5  |
|         3 | Temporal Only       | Logistic Regression |    0.248899 | 0.285354 | 0.265882 |   0.622081 |  0.836738 | 0.170439  |                    0.78 |
|         3 | Temporal Only       | Random Forest       |    0.233068 | 0.295455 | 0.260579 |   0.618714 |  0.824441 | 0.169937  |                    0.72 |
|         3 | Temporal Only       | XGBoost             |    0.217054 | 0.353535 | 0.268972 |   0.621153 |  0.831855 | 0.172523  |                    0.78 |
|         3 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.493215 |  0.5      | 0.0267748 |                    0.5  |
|         3 | Semantic Only       | Logistic Regression |    0.161248 | 0.313131 | 0.212876 |   0.590302 |  0.692131 | 0.0929329 |                    0.72 |
|         3 | Semantic Only       | Random Forest       |    0.153763 | 0.361111 | 0.215686 |   0.589439 |  0.700465 | 0.109289  |                    0.6  |
|         3 | Semantic Only       | XGBoost             |    0.186139 | 0.237374 | 0.208657 |   0.591898 |  0.716008 | 0.12088   |                    0.81 |
|         3 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.493215 |  0.5      | 0.0267748 |                    0.5  |
|         3 | Semantic + Temporal | Logistic Regression |    0.207055 | 0.340909 | 0.257634 |   0.615183 |  0.835198 | 0.163999  |                    0.74 |
|         3 | Semantic + Temporal | Random Forest       |    0.23494  | 0.295455 | 0.261745 |   0.619369 |  0.823208 | 0.181066  |                    0.68 |
|         3 | Semantic + Temporal | XGBoost             |    0.190639 | 0.421717 | 0.262579 |   0.614722 |  0.818828 | 0.19237   |                    0.72 |

## 6. Ablation

Perbandingan t+1 terbaik per feature set pada official validation (test tidak dipakai untuk memilih ablation):

| feature_set         | model         |       f1 |   pr_auc |
|:--------------------|:--------------|---------:|---------:|
| Semantic + Temporal | XGBoost       | 0.303097 | 0.222432 |
| Semantic Only       | Random Forest | 0.251586 | 0.138639 |
| Temporal Only       | XGBoost       | 0.298874 | 0.22592  |

## 7. Multi-horizon dan 8. final test

|   horizon | feature_set         | model               |   precision |   recall |       f1 |   macro_f1 |   roc_auc |   pr_auc |
|----------:|:--------------------|:--------------------|------------:|---------:|---------:|-----------:|----------:|---------:|
|         1 | Semantic + Temporal | XGBoost             |    0.405386 | 0.547368 | 0.465798 |   0.709305 |  0.893216 | 0.49364  |
|         2 | Temporal Only       | Logistic Regression |    0.447644 | 0.491379 | 0.468493 |   0.713258 |  0.882989 | 0.455034 |
|         3 | Temporal Only       | XGBoost             |    0.344478 | 0.551625 | 0.424109 |   0.682884 |  0.872758 | 0.413675 |

## 9. Distribution shift

Mean article/day berubah dari 125,55 pada training menjadi 244,64 pada test. Perubahan validation ke test untuk model terpilih:

|   horizon |   validation_f1 |   test_f1 |   f1_change |   validation_pr_auc |   test_pr_auc |   pr_auc_change |
|----------:|----------------:|----------:|------------:|--------------------:|--------------:|----------------:|
|         1 |        0.303097 |  0.465798 |    0.162701 |            0.222432 |      0.49364  |        0.271208 |
|         2 |        0.280702 |  0.468493 |    0.187791 |            0.189153 |      0.455034 |        0.265881 |
|         3 |        0.268972 |  0.424109 |    0.155137 |            0.172523 |      0.413675 |        0.241153 |

Perubahan ini dilaporkan sebagai temporal distribution shift; penyebabnya tidak disimpulkan.

## 10. Feature importance

Feature importance menyatakan predictive importance, bukan hubungan sebab-akibat. Tabel lengkap memuat native importance dan permutation importance.

|   horizon | model   | feature_set         | feature                  | feature_group   |   native_importance | native_method   |   permutation_importance_mean_f1 |   permutation_importance_std_f1 |
|----------:|:--------|:--------------------|:-------------------------|:----------------|--------------------:|:----------------|---------------------------------:|--------------------------------:|
|         1 | XGBoost | Semantic + Temporal | rolling_mean_7           | Temporal        |          0.370628   | tree_importance |                      0.0125993   |                     0.0032717   |
|         1 | XGBoost | Semantic + Temporal | current_volume           | Temporal        |          0.10853    | tree_importance |                      0.00631874  |                     0.000507581 |
|         1 | XGBoost | Semantic + Temporal | lag_1                    | Temporal        |          0.0141565  | tree_importance |                      0.00208098  |                     0.000544671 |
|         1 | XGBoost | Semantic + Temporal | semantic_similarity_std  | Semantic        |          0.121263   | tree_importance |                      0.00192338  |                     0.000241172 |
|         1 | XGBoost | Semantic + Temporal | topic_probability_mean   | Semantic        |          0.00944094 | tree_importance |                      0.000457281 |                     0.000676164 |
|         1 | XGBoost | Semantic + Temporal | lag_2                    | Temporal        |          0.00896173 | tree_importance |                      0.000170325 |                     0.00053243  |
|         1 | XGBoost | Semantic + Temporal | lag_4                    | Temporal        |          0.00748872 | tree_importance |                      1.30135e-05 |                     0.000401033 |
|         1 | XGBoost | Semantic + Temporal | growth_rate              | Temporal        |          0.0108215  | tree_importance |                     -8.56323e-05 |                     0.000176353 |
|         1 | XGBoost | Semantic + Temporal | semantic_similarity_mean | Semantic        |          0.0115681  | tree_importance |                     -0.000171109 |                     0.000527224 |
|         1 | XGBoost | Semantic + Temporal | topic_probability_std    | Semantic        |          0.011566   | tree_importance |                     -0.000245355 |                     0.00015054  |
|         1 | XGBoost | Semantic + Temporal | topic_probability_change | Semantic        |          0.00847834 | tree_importance |                     -0.000491054 |                     0.000312849 |
|         1 | XGBoost | Semantic + Temporal | rolling_mean_3           | Temporal        |          0.0361363  | tree_importance |                     -0.000552998 |                     0.000243766 |
|         1 | XGBoost | Semantic + Temporal | lag_3                    | Temporal        |          0.00853591 | tree_importance |                     -0.000602093 |                     0.000362144 |
|         1 | XGBoost | Semantic + Temporal | percentage_change        | Temporal        |          0.0240519  | tree_importance |                     -0.00160106  |                     0.000363461 |
|         1 | XGBoost | Semantic + Temporal | semantic_drift           | Semantic        |          0.0106695  | tree_importance |                     -0.00163103  |                     0.000236117 |

## 11. Statistical comparison

Statistical significance was not reliably assessed because the number of independent temporal evaluation origins was limited.

## 12. Limitations

- Embedding menggunakan batas 128 token; 99,41% dokumen terkena right truncation.
- BERTopic dan forecasting readiness sebelumnya berstatus LIMITED.
- Outlier BERTopic sekitar 39% pada training dan 48% pada validation/test.
- Test mengalami perubahan volume dan komposisi sumber yang besar.
- Semantic feature missing pada hari ketika suatu topik tidak memiliki dokumen dan diimputasi berdasarkan training.
- Hanya tiga temporal folds tersedia; hasil antar-feature-set belum mendukung klaim signifikansi statistik.

## 13. RQ1 — Semantic features

F1 validation terbaik Semantic Only untuk t+1/t+2/t+3: **0.2516 / 0.2213 / 0.2157**; majority baseline pada horizon yang sama: **0.0000 / 0.0000 / 0.0000**. Semantic Only berada di atas majority baseline pada **3/3** horizon. Angka ini menyatakan predictive signal, bukan klaim kausal.

## 14. RQ2 — Temporal features

F1 validation terbaik Temporal Only untuk t+1/t+2/t+3: **0.2989 / 0.2807 / 0.2690**. Dibanding Semantic Only pada split dan kandidat learner yang sama, Temporal Only lebih tinggi pada **3/3** horizon.

## 15. RQ3 — Fusion

F1 validation terbaik Semantic + Temporal untuk t+1/t+2/t+3: **0.3031 / 0.2740 / 0.2626**. Fusion melampaui Temporal Only pada **1/3** horizon. Feature set yang terpilih per horizon: {1: 'Semantic + Temporal', 2: 'Temporal Only', 3: 'Temporal Only'}. Karena hanya tiga fold, perbedaan antar feature set tidak dinyatakan signifikan.

## 16. RQ4 — Forecast horizon

Pada final test, F1 model terpilih adalah **0.4658** (t+1), **0.4685** (t+2), dan **0.4241** (t+3) — tidak monoton terhadap horizon. PR-AUC adalah **0.4936**, **0.4550**, dan **0.4137** — menurun monoton. Tidak ada horizon yang dicampur saat fitting.

## Status penelitian

**LIMITED** — eksperimen dapat dievaluasi secara temporal dan bebas leakage yang terdeteksi, tetapi distribution shift, truncation, topic outlier, serta jumlah fold membatasi generalisasi hasil.

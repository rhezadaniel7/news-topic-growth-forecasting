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
|         1 | 0.0739 |  0.0326 |       0.0152 |
|         2 | 0.0748 |  0.0332 |       0.0158 |
|         3 | 0.0756 |  0.0332 |       0.0163 |

## 1. Feature dataset

Final dataset menggabungkan semantic, temporal, dan label audit dengan key `(topic_id, date, forecast_horizon)`. Tidak ada duplicate key. Target, future volume, target date, growth score, burst score, dan source ID tidak digunakan sebagai predictor.

| record_type   | label_scheme   |   label_threshold |   horizon | split      | feature   |   rows |   growth |   normal |   missing |   missing_rate |
|:--------------|:---------------|------------------:|----------:|:-----------|:----------|-------:|---------:|---------:|----------:|---------------:|
| partition     | burst_mad_k2   |                 2 |         1 | train      |           |   7389 |      241 |     7148 |     41069 |       0.292533 |
| partition     | burst_mad_k2   |                 2 |         1 | validation |           |   2759 |       42 |     2717 |     15498 |       0.295645 |
| partition     | burst_mad_k2   |                 2 |         1 | test       |           |   2759 |      204 |     2555 |     13440 |       0.256386 |
| partition     | burst_mad_k2   |                 2 |         2 | train      |           |   7358 |      244 |     7114 |     40887 |       0.292464 |
| partition     | burst_mad_k2   |                 2 |         2 | validation |           |   2728 |       43 |     2685 |     15309 |       0.295358 |
| partition     | burst_mad_k2   |                 2 |         2 | test       |           |   2728 |      204 |     2524 |     13300 |       0.256598 |
| partition     | burst_mad_k2   |                 2 |         3 | train      |           |   7327 |      243 |     7084 |     40726 |       0.292545 |
| partition     | burst_mad_k2   |                 2 |         3 | validation |           |   2697 |       44 |     2653 |     15127 |       0.295201 |
| partition     | burst_mad_k2   |                 2 |         3 | test       |           |   2697 |      204 |     2493 |     13181 |       0.257225 |

## 2. Temporal folds

Digunakan tiga expanding-window fold dengan validation block yang tidak tumpang tindih. Jumlah fold valid: **3/3**. Test tidak digunakan untuk tuning.

|   fold |   horizon | train_start   | train_end   | validation_start   | validation_end   |   n_train |   n_validation |   growth_train |   normal_train |   growth_validation |   normal_validation | status   |
|-------:|----------:|:--------------|:------------|:-------------------|:-----------------|----------:|---------------:|---------------:|---------------:|--------------------:|--------------------:|:---------|
|      1 |         1 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |      4630 |           1860 |            150 |           4480 |                  49 |                1811 | VALID    |
|      2 |         1 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |      6490 |           1829 |            199 |           6291 |                  54 |                1775 | VALID    |
|      3 |         1 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |      8319 |           1829 |            253 |           8066 |                  30 |                1799 | VALID    |
|      1 |         2 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |      4630 |           1860 |            155 |           4475 |                  48 |                1812 | VALID    |
|      2 |         2 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |      6490 |           1798 |            203 |           6287 |                  54 |                1744 | VALID    |
|      3 |         2 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |      8288 |           1798 |            257 |           8031 |                  30 |                1768 | VALID    |
|      1 |         3 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |      4630 |           1860 |            155 |           4475 |                  47 |                1813 | VALID    |
|      2 |         3 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |      6490 |           1767 |            202 |           6288 |                  56 |                1711 | VALID    |
|      3 |         3 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |      8257 |           1767 |            258 |           7999 |                  29 |                1738 | VALID    |

## 3. Model configuration dan 4. hyperparameters

Model: Majority Class, Logistic Regression, Random Forest, dan XGBoost. Setiap learner diuji dengan Temporal Only, Semantic Only, dan Semantic + Temporal. Logistic memakai median imputation dan StandardScaler; tree model memakai median imputation. Class weighting diterapkan hanya saat fitting training. Tidak ada SMOTE.

Konfigurasi final per horizon dibekukan dari official validation:

```json
{
  "1": {
    "model": "Random Forest",
    "feature_set": "Temporal Only",
    "hyperparameters": {
      "n_estimators": 250,
      "max_depth": 14,
      "min_samples_leaf": 4,
      "max_features": "sqrt",
      "class_weight": "balanced_subsample"
    },
    "probability_threshold": 0.63,
    "growth_label_threshold": 2.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.1951219512195122,
    "validation_pr_auc": 0.09585088188473968
  },
  "2": {
    "model": "XGBoost",
    "feature_set": "Temporal Only",
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 3,
      "learning_rate": 0.05,
      "subsample": 0.85,
      "colsample_bytree": 0.85
    },
    "probability_threshold": 0.76,
    "growth_label_threshold": 2.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.208955223880597,
    "validation_pr_auc": 0.11113312124199189
  },
  "3": {
    "model": "Logistic Regression",
    "feature_set": "Semantic + Temporal",
    "hyperparameters": {
      "C": 0.1,
      "max_iter": 2000,
      "class_weight": "balanced",
      "solver": "lbfgs"
    },
    "probability_threshold": 0.77,
    "growth_label_threshold": 2.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.24,
    "validation_pr_auc": 0.10468632059081473
  }
}
```

## 5. Validation performance

|   horizon | feature_set         | model               |   precision |    recall |        f1 |   macro_f1 |   roc_auc |    pr_auc |   probability_threshold |
|----------:|:--------------------|:--------------------|------------:|----------:|----------:|-----------:|----------:|----------:|------------------------:|
|         1 | Temporal Only       | Majority Class      |   0         | 0         | 0         |   0.496165 |  0.5      | 0.0152229 |                    0.5  |
|         1 | Temporal Only       | Logistic Regression |   0.119658  | 0.333333  | 0.176101  |   0.575828 |  0.838241 | 0.145045  |                    0.74 |
|         1 | Temporal Only       | Random Forest       |   0.2       | 0.190476  | 0.195122  |   0.59149  |  0.688842 | 0.0958509 |                    0.63 |
|         1 | Temporal Only       | XGBoost             |   0.150685  | 0.261905  | 0.191304  |   0.587046 |  0.754706 | 0.147698  |                    0.82 |
|         1 | Semantic Only       | Majority Class      |   0         | 0         | 0         |   0.496165 |  0.5      | 0.0152229 |                    0.5  |
|         1 | Semantic Only       | Logistic Regression |   0.096     | 0.285714  | 0.143713  |   0.558494 |  0.613413 | 0.0531951 |                    0.69 |
|         1 | Semantic Only       | Random Forest       |   0.0461957 | 0.404762  | 0.0829268 |   0.504658 |  0.578334 | 0.0287207 |                    0.25 |
|         1 | Semantic Only       | XGBoost             |   0.0542986 | 0.285714  | 0.0912548 |   0.522887 |  0.557416 | 0.0333529 |                    0.5  |
|         1 | Semantic + Temporal | Majority Class      |   0         | 0         | 0         |   0.496165 |  0.5      | 0.0152229 |                    0.5  |
|         1 | Semantic + Temporal | Logistic Regression |   0.117647  | 0.380952  | 0.179775  |   0.576217 |  0.830845 | 0.143509  |                    0.71 |
|         1 | Semantic + Temporal | Random Forest       |   0.6       | 0.0714286 | 0.12766   |   0.560083 |  0.640487 | 0.0971742 |                    0.61 |
|         1 | Semantic + Temporal | XGBoost             |   0.186047  | 0.190476  | 0.188235  |   0.587768 |  0.725218 | 0.120788  |                    0.75 |
|         2 | Temporal Only       | Majority Class      |   0         | 0         | 0         |   0.496028 |  0.5      | 0.0157625 |                    0.5  |
|         2 | Temporal Only       | Logistic Regression |   0.130435  | 0.488372  | 0.205882  |   0.587518 |  0.814707 | 0.120973  |                    0.66 |
|         2 | Temporal Only       | Random Forest       |   0.138889  | 0.232558  | 0.173913  |   0.578063 |  0.77926  | 0.0940902 |                    0.6  |
|         2 | Temporal Only       | XGBoost             |   0.153846  | 0.325581  | 0.208955  |   0.594519 |  0.804075 | 0.111133  |                    0.76 |
|         2 | Semantic Only       | Majority Class      |   0         | 0         | 0         |   0.496028 |  0.5      | 0.0157625 |                    0.5  |
|         2 | Semantic Only       | Logistic Regression |   0.0807453 | 0.302326  | 0.127451  |   0.54678  |  0.635867 | 0.0428653 |                    0.54 |
|         2 | Semantic Only       | Random Forest       |   0.0754717 | 0.0930233 | 0.0833333 |   0.533458 |  0.356477 | 0.0186691 |                    0.37 |
|         2 | Semantic Only       | XGBoost             |   0.142857  | 0.0930233 | 0.112676  |   0.550488 |  0.496211 | 0.0466344 |                    0.81 |
|         2 | Semantic + Temporal | Majority Class      |   0         | 0         | 0         |   0.496028 |  0.5      | 0.0157625 |                    0.5  |
|         2 | Semantic + Temporal | Logistic Regression |   0.118012  | 0.44186   | 0.186275  |   0.577334 |  0.810978 | 0.112408  |                    0.68 |
|         2 | Semantic + Temporal | Random Forest       |   0.159091  | 0.162791  | 0.16092   |   0.573661 |  0.742861 | 0.074857  |                    0.56 |
|         2 | Semantic + Temporal | XGBoost             |   0.106667  | 0.186047  | 0.135593  |   0.558242 |  0.731298 | 0.0796727 |                    0.68 |
|         3 | Temporal Only       | Majority Class      |   0         | 0         | 0         |   0.495888 |  0.5      | 0.0163144 |                    0.5  |
|         3 | Temporal Only       | Logistic Regression |   0.150794  | 0.431818  | 0.223529  |   0.599131 |  0.826688 | 0.103796  |                    0.7  |
|         3 | Temporal Only       | Random Forest       |   0.0511111 | 0.522727  | 0.0931174 |   0.500844 |  0.741519 | 0.0392648 |                    0.4  |
|         3 | Temporal Only       | XGBoost             |   0.0833333 | 0.340909  | 0.133929  |   0.548202 |  0.755114 | 0.0556274 |                    0.67 |
|         3 | Semantic Only       | Majority Class      |   0         | 0         | 0         |   0.495888 |  0.5      | 0.0163144 |                    0.5  |
|         3 | Semantic Only       | Logistic Regression |   0.107527  | 0.227273  | 0.145985  |   0.561865 |  0.654199 | 0.0733327 |                    0.73 |
|         3 | Semantic Only       | Random Forest       |   0.0952381 | 0.136364  | 0.11215   |   0.54709  |  0.419097 | 0.0237103 |                    0.44 |
|         3 | Semantic Only       | XGBoost             |   0.0440252 | 0.159091  | 0.0689655 |   0.516278 |  0.438329 | 0.017858  |                    0.55 |
|         3 | Semantic + Temporal | Majority Class      |   0         | 0         | 0         |   0.495888 |  0.5      | 0.0163144 |                    0.5  |
|         3 | Semantic + Temporal | Logistic Regression |   0.185185  | 0.340909  | 0.24      |   0.610985 |  0.824564 | 0.104686  |                    0.77 |
|         3 | Semantic + Temporal | Random Forest       |   0.0625    | 0.0681818 | 0.0652174 |   0.524499 |  0.501045 | 0.0247397 |                    0.53 |
|         3 | Semantic + Temporal | XGBoost             |   0.0909091 | 0.272727  | 0.136364  |   0.553617 |  0.618451 | 0.0467304 |                    0.6  |

## 6. Ablation

Perbandingan t+1 terbaik per feature set pada official validation (test tidak dipakai untuk memilih ablation):

| feature_set         | model               |       f1 |    pr_auc |
|:--------------------|:--------------------|---------:|----------:|
| Semantic + Temporal | XGBoost             | 0.188235 | 0.120788  |
| Semantic Only       | Logistic Regression | 0.143713 | 0.0531951 |
| Temporal Only       | Random Forest       | 0.195122 | 0.0958509 |

## 7. Multi-horizon dan 8. final test

|   horizon | feature_set         | model               |   precision |   recall |       f1 |   macro_f1 |   roc_auc |   pr_auc |
|----------:|:--------------------|:--------------------|------------:|---------:|---------:|-----------:|----------:|---------:|
|         1 | Temporal Only       | Random Forest       |    0.369565 | 0.166667 | 0.22973  |   0.593034 |  0.784609 | 0.260652 |
|         2 | Temporal Only       | XGBoost             |    0.295139 | 0.416667 | 0.345528 |   0.640331 |  0.828652 | 0.279442 |
|         3 | Semantic + Temporal | Logistic Regression |    0.313984 | 0.583333 | 0.408233 |   0.668261 |  0.845319 | 0.3287   |

## 9. Distribution shift

Mean article/day berubah dari 125,55 pada training menjadi 244,64 pada test. Perubahan validation ke test untuk model terpilih:

|   horizon |   validation_f1 |   test_f1 |   f1_change |   validation_pr_auc |   test_pr_auc |   pr_auc_change |
|----------:|----------------:|----------:|------------:|--------------------:|--------------:|----------------:|
|         1 |        0.195122 |  0.22973  |   0.0346078 |           0.0958509 |      0.260652 |        0.164802 |
|         2 |        0.208955 |  0.345528 |   0.136573  |           0.111133  |      0.279442 |        0.168309 |
|         3 |        0.24     |  0.408233 |   0.168233  |           0.104686  |      0.3287   |        0.224014 |

Perubahan ini dilaporkan sebagai temporal distribution shift; penyebabnya tidak disimpulkan.

## 10. Feature importance

Feature importance menyatakan predictive importance, bukan hubungan sebab-akibat. Tabel lengkap memuat native importance dan permutation importance.

|   horizon | model         | feature_set   | feature           | feature_group   |   native_importance | native_method   |   permutation_importance_mean_f1 |   permutation_importance_std_f1 |
|----------:|:--------------|:--------------|:------------------|:----------------|--------------------:|:----------------|---------------------------------:|--------------------------------:|
|         1 | Random Forest | Temporal Only | current_volume    | Temporal        |           0.165951  | tree_importance |                      0.204628    |                      0.0103347  |
|         1 | Random Forest | Temporal Only | rolling_mean_3    | Temporal        |           0.105286  | tree_importance |                      0.0890374   |                      0.0085907  |
|         1 | Random Forest | Temporal Only | rolling_std_7     | Temporal        |           0.156573  | tree_importance |                      0.0864968   |                      0.0116534  |
|         1 | Random Forest | Temporal Only | rolling_mean_7    | Temporal        |           0.146583  | tree_importance |                      0.0721663   |                      0.0144048  |
|         1 | Random Forest | Temporal Only | lag_1             | Temporal        |           0.0401644 | tree_importance |                      0.0643615   |                      0.00662118 |
|         1 | Random Forest | Temporal Only | growth_rate       | Temporal        |           0.0509577 | tree_importance |                      0.0378214   |                      0.00953697 |
|         1 | Random Forest | Temporal Only | burst_intensity   | Temporal        |           0.130452  | tree_importance |                      0.0283762   |                      0.00788785 |
|         1 | Random Forest | Temporal Only | percentage_change | Temporal        |           0.0468561 | tree_importance |                      0.0151517   |                      0.00309027 |
|         1 | Random Forest | Temporal Only | lag_4             | Temporal        |           0.0496679 | tree_importance |                      9.21274e-05 |                      0.0056929  |
|         1 | Random Forest | Temporal Only | lag_2             | Temporal        |           0.0252667 | tree_importance |                     -0.00775323  |                      0.00285199 |
|         1 | Random Forest | Temporal Only | momentum          | Temporal        |           0.0521651 | tree_importance |                     -0.011209    |                      0.00838339 |
|         1 | Random Forest | Temporal Only | lag_3             | Temporal        |           0.0300774 | tree_importance |                     -0.0223585   |                      0.00487666 |

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

F1 validation terbaik Semantic Only untuk t+1/t+2/t+3: **0.1437 / 0.1275 / 0.1460**; majority baseline pada horizon yang sama: **0.0000 / 0.0000 / 0.0000**. Semantic Only berada di atas majority baseline pada **3/3** horizon. Angka ini menyatakan predictive signal, bukan klaim kausal.

## 14. RQ2 — Temporal features

F1 validation terbaik Temporal Only untuk t+1/t+2/t+3: **0.1951 / 0.2090 / 0.2235**. Dibanding Semantic Only pada split dan kandidat learner yang sama, Temporal Only lebih tinggi pada **3/3** horizon.

## 15. RQ3 — Fusion

F1 validation terbaik Semantic + Temporal untuk t+1/t+2/t+3: **0.1882 / 0.1863 / 0.2400**. Fusion melampaui Temporal Only pada **1/3** horizon. Feature set yang terpilih per horizon: {1: 'Temporal Only', 2: 'Temporal Only', 3: 'Semantic + Temporal'}. Karena hanya tiga fold, perbedaan antar feature set tidak dinyatakan signifikan.

## 16. RQ4 — Forecast horizon

Pada final test, F1 model terpilih adalah **0.2297** (t+1), **0.3455** (t+2), dan **0.4082** (t+3) — meningkat monoton terhadap horizon. PR-AUC adalah **0.2607**, **0.2794**, dan **0.3287** — meningkat monoton. Tidak ada horizon yang dicampur saat fitting.

## Status penelitian

**LIMITED** — eksperimen dapat dievaluasi secara temporal dan bebas leakage yang terdeteksi, tetapi distribution shift, truncation, topic outlier, serta jumlah fold membatasi generalisasi hasil.

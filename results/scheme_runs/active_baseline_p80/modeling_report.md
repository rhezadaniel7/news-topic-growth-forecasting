# Modeling report

## 0. Definisi label

Label growth memakai skema **active_baseline_p80** (S1 baseline aktif (lag_1 > 0), kuantil p80).

- Definisi: `hanya topic-day dengan lag_1 > 0; growth_score > kuantil p80 growth_score training pada populasi tersebut`
- Jenis threshold: `train_quantile`
- Populasi: `active_baseline`
- Alasan pemilihan: Memperbaiki populasi, bukan threshold: growth hanya didefinisikan ketika ada baseline aktif untuk tumbuh dari. lag_1 diketahui pada origin date sehingga seleksi ini tidak melihat masa depan.

Positive rate label per split (dari partisi resmi):

|   horizon |   test |   train |   validation |
|----------:|-------:|--------:|-------------:|
|         1 | 0.2817 |  0.1429 |       0.1353 |
|         2 | 0.2736 |  0.1377 |       0.1307 |
|         3 | 0.2735 |  0.1307 |       0.1245 |

## 1. Feature dataset

Final dataset menggabungkan semantic, temporal, dan label audit dengan key `(topic_id, date, forecast_horizon)`. Tidak ada duplicate key. Target, future volume, target date, growth score, burst score, dan source ID tidak digunakan sebagai predictor.

| record_type   | label_scheme        |   label_threshold |   horizon | split      | feature   |   rows |   growth |   normal |   missing |   missing_rate |
|:--------------|:--------------------|------------------:|----------:|:-----------|:----------|-------:|---------:|---------:|----------:|---------------:|
| partition     | active_baseline_p80 |                 1 |         1 | train      |           |   8322 |     1189 |     7133 |     32025 |       0.202539 |
| partition     | active_baseline_p80 |                 1 |         1 | validation |           |   2927 |      396 |     2531 |     11256 |       0.202399 |
| partition     | active_baseline_p80 |                 1 |         1 | test       |           |   4373 |     1232 |     3141 |     12656 |       0.152322 |
| partition     | active_baseline_p80 |                 1 |         2 | train      |           |   8285 |     1141 |     7144 |     31857 |       0.202376 |
| partition     | active_baseline_p80 |                 1 |         2 | validation |           |   2893 |      378 |     2515 |     11116 |       0.20223  |
| partition     | active_baseline_p80 |                 1 |         2 | test       |           |   4302 |     1177 |     3125 |     12474 |       0.15261  |
| partition     | active_baseline_p80 |                 1 |         3 | train      |           |   8247 |     1078 |     7169 |     31731 |       0.202504 |
| partition     | active_baseline_p80 |                 1 |         3 | validation |           |   2859 |      356 |     2503 |     10983 |       0.202187 |
| partition     | active_baseline_p80 |                 1 |         3 | test       |           |   4238 |     1159 |     3079 |     12327 |       0.153089 |

## 2. Temporal folds

Digunakan tiga expanding-window fold dengan validation block yang tidak tumpang tindih. Jumlah fold valid: **3/3**. Test tidak digunakan untuk tuning.

|   fold |   horizon | train_start   | train_end   | validation_start   | validation_end   |   n_train |   n_validation |   growth_train |   normal_train |   growth_validation |   normal_validation | status   |
|-------:|----------:|:--------------|:------------|:-------------------|:-----------------|----------:|---------------:|---------------:|---------------:|--------------------:|--------------------:|:---------|
|      1 |         1 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |      5394 |           2053 |            769 |           4625 |                 301 |                1752 | VALID    |
|      2 |         1 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |      7447 |           1880 |           1070 |           6377 |                 276 |                1604 | VALID    |
|      3 |         1 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |      9327 |           1922 |           1346 |           7981 |                 239 |                1683 | VALID    |
|      1 |         2 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |      5394 |           2053 |            744 |           4650 |                 288 |                1765 | VALID    |
|      2 |         2 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |      7447 |           1843 |           1032 |           6415 |                 263 |                1580 | VALID    |
|      3 |         2 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |      9290 |           1888 |           1295 |           7995 |                 224 |                1664 | VALID    |
|      1 |         3 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |      5394 |           2053 |            691 |           4703 |                 277 |                1776 | VALID    |
|      2 |         3 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |      7447 |           1805 |            968 |           6479 |                 247 |                1558 | VALID    |
|      3 |         3 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |      9252 |           1854 |           1215 |           8037 |                 219 |                1635 | VALID    |

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
    "probability_threshold": 0.46,
    "growth_label_threshold": 1.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.411214953271028,
    "validation_pr_auc": 0.36908188673890807
  },
  "2": {
    "model": "Random Forest",
    "feature_set": "Semantic + Temporal",
    "hyperparameters": {
      "n_estimators": 150,
      "max_depth": 8,
      "min_samples_leaf": 2,
      "max_features": "sqrt",
      "class_weight": "balanced_subsample"
    },
    "probability_threshold": 0.47,
    "growth_label_threshold": 1.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.38596491228070173,
    "validation_pr_auc": 0.3121942009516687
  },
  "3": {
    "model": "Random Forest",
    "feature_set": "Semantic + Temporal",
    "hyperparameters": {
      "n_estimators": 150,
      "max_depth": 8,
      "min_samples_leaf": 2,
      "max_features": "sqrt",
      "class_weight": "balanced_subsample"
    },
    "probability_threshold": 0.44,
    "growth_label_threshold": 1.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.37058823529411766,
    "validation_pr_auc": 0.29542034138426765
  }
}
```

## 5. Validation performance

|   horizon | feature_set         | model               |   precision |   recall |       f1 |   macro_f1 |   roc_auc |   pr_auc |   probability_threshold |
|----------:|:--------------------|:--------------------|------------:|---------:|---------:|-----------:|----------:|---------:|------------------------:|
|         1 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.463723 |  0.5      | 0.135292 |                    0.5  |
|         1 | Temporal Only       | Logistic Regression |    0.378973 | 0.391414 | 0.385093 |   0.643527 |  0.757227 | 0.355007 |                    0.57 |
|         1 | Temporal Only       | Random Forest       |    0.326376 | 0.434343 | 0.372698 |   0.627639 |  0.754699 | 0.344544 |                    0.5  |
|         1 | Temporal Only       | XGBoost             |    0.342466 | 0.441919 | 0.385888 |   0.636647 |  0.759765 | 0.352085 |                    0.57 |
|         1 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.463723 |  0.5      | 0.135292 |                    0.5  |
|         1 | Semantic Only       | Logistic Regression |    0.311189 | 0.449495 | 0.367769 |   0.621256 |  0.641061 | 0.228482 |                    0.52 |
|         1 | Semantic Only       | Random Forest       |    0.31323  | 0.406566 | 0.353846 |   0.617457 |  0.677896 | 0.269782 |                    0.48 |
|         1 | Semantic Only       | XGBoost             |    0.269382 | 0.517677 | 0.354365 |   0.597664 |  0.681834 | 0.280216 |                    0.49 |
|         1 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.463723 |  0.5      | 0.135292 |                    0.5  |
|         1 | Semantic + Temporal | Logistic Regression |    0.358491 | 0.431818 | 0.391753 |   0.642574 |  0.757123 | 0.350394 |                    0.57 |
|         1 | Semantic + Temporal | Random Forest       |    0.324047 | 0.558081 | 0.410019 |   0.638426 |  0.761608 | 0.35837  |                    0.41 |
|         1 | Semantic + Temporal | XGBoost             |    0.326409 | 0.555556 | 0.411215 |   0.639763 |  0.76289  | 0.369082 |                    0.46 |
|         2 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.465052 |  0.5      | 0.13066  |                    0.5  |
|         2 | Temporal Only       | Logistic Regression |    0.280505 | 0.529101 | 0.366636 |   0.609729 |  0.73258  | 0.307427 |                    0.48 |
|         2 | Temporal Only       | Random Forest       |    0.266846 | 0.52381  | 0.353571 |   0.599203 |  0.724524 | 0.301551 |                    0.47 |
|         2 | Temporal Only       | XGBoost             |    0.282086 | 0.558201 | 0.374778 |   0.611853 |  0.733341 | 0.30815  |                    0.51 |
|         2 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.465052 |  0.5      | 0.13066  |                    0.5  |
|         2 | Semantic Only       | Logistic Regression |    0.25182  | 0.457672 | 0.324883 |   0.586292 |  0.59347  | 0.19458  |                    0.43 |
|         2 | Semantic Only       | Random Forest       |    0.260174 | 0.473545 | 0.335835 |   0.592917 |  0.641474 | 0.220514 |                    0.42 |
|         2 | Semantic Only       | XGBoost             |    0.270096 | 0.444444 | 0.336    |   0.598631 |  0.633664 | 0.219116 |                    0.51 |
|         2 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.465052 |  0.5      | 0.13066  |                    0.5  |
|         2 | Semantic + Temporal | Logistic Regression |    0.261538 | 0.584656 | 0.361406 |   0.595124 |  0.728445 | 0.303451 |                    0.45 |
|         2 | Semantic + Temporal | Random Forest       |    0.329588 | 0.465608 | 0.385965 |   0.635535 |  0.727416 | 0.312194 |                    0.47 |
|         2 | Semantic + Temporal | XGBoost             |    0.305732 | 0.507937 | 0.38171  |   0.625792 |  0.716956 | 0.322896 |                    0.49 |
|         3 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.466803 |  0.5      | 0.124519 |                    0.5  |
|         3 | Temporal Only       | Logistic Regression |    0.257707 | 0.587079 | 0.358183 |   0.596802 |  0.727173 | 0.277168 |                    0.47 |
|         3 | Temporal Only       | Random Forest       |    0.258412 | 0.539326 | 0.349409 |   0.597307 |  0.727812 | 0.27356  |                    0.47 |
|         3 | Temporal Only       | XGBoost             |    0.268056 | 0.542135 | 0.358736 |   0.605047 |  0.733788 | 0.277628 |                    0.53 |
|         3 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.466803 |  0.5      | 0.124519 |                    0.5  |
|         3 | Semantic Only       | Logistic Regression |    0.228614 | 0.435393 | 0.299807 |   0.572619 |  0.593004 | 0.173998 |                    0.43 |
|         3 | Semantic Only       | Random Forest       |    0.232227 | 0.412921 | 0.29727  |   0.575152 |  0.629894 | 0.209746 |                    0.44 |
|         3 | Semantic Only       | XGBoost             |    0.226686 | 0.44382  | 0.300095 |   0.571055 |  0.632458 | 0.203885 |                    0.51 |
|         3 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.466803 |  0.5      | 0.124519 |                    0.5  |
|         3 | Semantic + Temporal | Logistic Regression |    0.232558 | 0.617978 | 0.337942 |   0.571371 |  0.72208  | 0.265545 |                    0.45 |
|         3 | Semantic + Temporal | Random Forest       |    0.284639 | 0.530899 | 0.370588 |   0.616967 |  0.732049 | 0.29542  |                    0.44 |
|         3 | Semantic + Temporal | XGBoost             |    0.272984 | 0.542135 | 0.363123 |   0.608844 |  0.726043 | 0.296773 |                    0.49 |

## 6. Ablation

Perbandingan t+1 terbaik per feature set pada official validation (test tidak dipakai untuk memilih ablation):

| feature_set         | model               |       f1 |   pr_auc |
|:--------------------|:--------------------|---------:|---------:|
| Semantic + Temporal | XGBoost             | 0.411215 | 0.369082 |
| Semantic Only       | Logistic Regression | 0.367769 | 0.228482 |
| Temporal Only       | XGBoost             | 0.385888 | 0.352085 |

## 7. Multi-horizon dan 8. final test

|   horizon | feature_set         | model         |   precision |   recall |       f1 |   macro_f1 |   roc_auc |   pr_auc |
|----------:|:--------------------|:--------------|------------:|---------:|---------:|-----------:|----------:|---------:|
|         1 | Semantic + Temporal | XGBoost       |    0.443649 | 0.754058 | 0.558629 |   0.64389  |  0.77057  | 0.608919 |
|         2 | Semantic + Temporal | Random Forest |    0.469588 | 0.610025 | 0.530673 |   0.657673 |  0.749639 | 0.554474 |
|         3 | Semantic + Temporal | Random Forest |    0.45072  | 0.70233  | 0.549073 |   0.653233 |  0.752988 | 0.554099 |

## 9. Distribution shift

Mean article/day berubah dari 125,55 pada training menjadi 244,64 pada test. Perubahan validation ke test untuk model terpilih:

|   horizon |   validation_f1 |   test_f1 |   f1_change |   validation_pr_auc |   test_pr_auc |   pr_auc_change |
|----------:|----------------:|----------:|------------:|--------------------:|--------------:|----------------:|
|         1 |        0.411215 |  0.558629 |    0.147414 |            0.369082 |      0.608919 |        0.239838 |
|         2 |        0.385965 |  0.530673 |    0.144708 |            0.312194 |      0.554474 |        0.24228  |
|         3 |        0.370588 |  0.549073 |    0.178484 |            0.29542  |      0.554099 |        0.258678 |

Perubahan ini dilaporkan sebagai temporal distribution shift; penyebabnya tidak disimpulkan.

## 10. Feature importance

Feature importance menyatakan predictive importance, bukan hubungan sebab-akibat. Tabel lengkap memuat native importance dan permutation importance.

|   horizon | model   | feature_set         | feature                  | feature_group   |   native_importance | native_method   |   permutation_importance_mean_f1 |   permutation_importance_std_f1 |
|----------:|:--------|:--------------------|:-------------------------|:----------------|--------------------:|:----------------|---------------------------------:|--------------------------------:|
|         1 | XGBoost | Semantic + Temporal | rolling_mean_7           | Temporal        |           0.216308  | tree_importance |                      0.0225775   |                     0.00450694  |
|         1 | XGBoost | Semantic + Temporal | current_volume           | Temporal        |           0.180305  | tree_importance |                      0.0145368   |                     0.0025944   |
|         1 | XGBoost | Semantic + Temporal | semantic_drift           | Semantic        |           0.0246076 | tree_importance |                      0.0125117   |                     0.00340458  |
|         1 | XGBoost | Semantic + Temporal | rolling_mean_3           | Temporal        |           0.0500846 | tree_importance |                      0.00517292  |                     0.0033601   |
|         1 | XGBoost | Semantic + Temporal | topic_probability_mean   | Semantic        |           0.0250079 | tree_importance |                      0.00506883  |                     0.0073113   |
|         1 | XGBoost | Semantic + Temporal | rolling_std_7            | Temporal        |           0.0866868 | tree_importance |                      0.00395412  |                     0.00315573  |
|         1 | XGBoost | Semantic + Temporal | topic_probability_std    | Semantic        |           0.0544198 | tree_importance |                      0.00254223  |                     0.00245637  |
|         1 | XGBoost | Semantic + Temporal | similarity_change        | Semantic        |           0.0159612 | tree_importance |                      0.000539818 |                     0.00190877  |
|         1 | XGBoost | Semantic + Temporal | lag_3                    | Temporal        |           0.0147452 | tree_importance |                      0.000497555 |                     0.00385069  |
|         1 | XGBoost | Semantic + Temporal | topic_probability_change | Semantic        |           0.0172976 | tree_importance |                      0.000467104 |                     0.00167696  |
|         1 | XGBoost | Semantic + Temporal | semantic_similarity_std  | Semantic        |           0.0290701 | tree_importance |                      0.000463574 |                     0.000819044 |
|         1 | XGBoost | Semantic + Temporal | percentage_change        | Temporal        |           0.0814726 | tree_importance |                      0.000252106 |                     0.00174869  |
|         1 | XGBoost | Semantic + Temporal | lag_4                    | Temporal        |           0.0128651 | tree_importance |                      0.000189901 |                     0.000556386 |
|         1 | XGBoost | Semantic + Temporal | lag_2                    | Temporal        |           0.022042  | tree_importance |                     -0.00122029  |                     0.00115626  |
|         1 | XGBoost | Semantic + Temporal | growth_rate              | Temporal        |           0.0269468 | tree_importance |                     -0.00128115  |                     0.00127401  |

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

F1 validation terbaik Semantic Only untuk t+1/t+2/t+3: **0.3678 / 0.3360 / 0.3001**; majority baseline pada horizon yang sama: **0.0000 / 0.0000 / 0.0000**. Semantic Only berada di atas majority baseline pada **3/3** horizon. Angka ini menyatakan predictive signal, bukan klaim kausal.

## 14. RQ2 — Temporal features

F1 validation terbaik Temporal Only untuk t+1/t+2/t+3: **0.3859 / 0.3748 / 0.3587**. Dibanding Semantic Only pada split dan kandidat learner yang sama, Temporal Only lebih tinggi pada **3/3** horizon.

## 15. RQ3 — Fusion

F1 validation terbaik Semantic + Temporal untuk t+1/t+2/t+3: **0.4112 / 0.3860 / 0.3706**. Fusion melampaui Temporal Only pada **3/3** horizon. Feature set yang terpilih per horizon: {1: 'Semantic + Temporal', 2: 'Semantic + Temporal', 3: 'Semantic + Temporal'}. Karena hanya tiga fold, perbedaan antar feature set tidak dinyatakan signifikan.

## 16. RQ4 — Forecast horizon

Pada final test, F1 model terpilih adalah **0.5586** (t+1), **0.5307** (t+2), dan **0.5491** (t+3) — tidak monoton terhadap horizon. PR-AUC adalah **0.6089**, **0.5545**, dan **0.5541** — menurun monoton. Tidak ada horizon yang dicampur saat fitting.

## Status penelitian

**LIMITED** — eksperimen dapat dievaluasi secara temporal dan bebas leakage yang terdeteksi, tetapi distribution shift, truncation, topic outlier, serta jumlah fold membatasi generalisasi hasil.

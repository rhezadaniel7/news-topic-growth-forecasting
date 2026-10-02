# Modeling report

## 0. Definisi label

Label growth memakai skema **positive_only_p80** (S2 kuantil p80 pada growth_score > 0 saja).

- Definisi: `growth_score > kuantil p80 dari growth_score training yang > 0`
- Jenis threshold: `train_quantile_positive`
- Populasi: `all`
- Alasan pemilihan: Memperbaiki threshold, bukan populasi: kuantil di-fit pada ekor positif sehingga tidak lagi jatuh di 0. Referensi kuantilnya dipilih berdasarkan outcome, dan untuk baris bermedian 0 aturannya tetap menjadi pernyataan volume absolut.

Positive rate label per split (dari partisi resmi):

|   horizon |   test |   train |   validation |
|----------:|-------:|--------:|-------------:|
|         1 | 0.0601 |  0.0293 |       0.0217 |
|         2 | 0.0608 |  0.0295 |       0.0217 |
|         3 | 0.0617 |  0.0296 |       0.0216 |

## 1. Feature dataset

Final dataset menggabungkan semantic, temporal, dan label audit dengan key `(topic_id, date, forecast_horizon)`. Tidak ada duplicate key. Target, future volume, target date, growth score, burst score, dan source ID tidak digunakan sebagai predictor.

| record_type   | label_scheme      |   label_threshold |   horizon | split      | feature   |   rows |   growth |   normal |   missing |   missing_rate |
|:--------------|:------------------|------------------:|----------:|:-----------|:----------|-------:|---------:|---------:|----------:|---------------:|
| partition     | positive_only_p80 |                 2 |         1 | train      |           |  40295 |     1181 |    39114 |    223874 |       0.292414 |
| partition     | positive_only_p80 |                 2 |         1 | validation |           |  15130 |      329 |    14801 |     85414 |       0.297123 |
| partition     | positive_only_p80 |                 2 |         1 | test       |           |  15130 |      909 |    14221 |     75061 |       0.261109 |
| partition     | positive_only_p80 |                 2 |         2 | train      |           |  40125 |     1184 |    38941 |    222887 |       0.292359 |
| partition     | positive_only_p80 |                 2 |         2 | validation |           |  14960 |      324 |    14636 |     84434 |       0.297052 |
| partition     | positive_only_p80 |                 2 |         2 | test       |           |  14960 |      910 |    14050 |     74347 |       0.261564 |
| partition     | positive_only_p80 |                 2 |         3 | train      |           |  39955 |     1181 |    38774 |    221956 |       0.292376 |
| partition     | positive_only_p80 |                 2 |         3 | validation |           |  14790 |      320 |    14470 |     83482 |       0.297078 |
| partition     | positive_only_p80 |                 2 |         3 | test       |           |  14790 |      912 |    13878 |     73654 |       0.262105 |

## 2. Temporal folds

Digunakan tiga expanding-window fold dengan validation block yang tidak tumpang tindih. Jumlah fold valid: **3/3**. Test tidak digunakan untuk tuning.

|   fold |   horizon | train_start   | train_end   | validation_start   | validation_end   |   n_train |   n_validation |   growth_train |   normal_train |   growth_validation |   normal_validation | status   |
|-------:|----------:|:--------------|:------------|:-------------------|:-----------------|----------:|---------------:|---------------:|---------------:|--------------------:|--------------------:|:---------|
|      1 |         1 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |     25282 |          10083 |            734 |          24548 |                 298 |                9785 | VALID    |
|      2 |         1 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |     35365 |          10030 |           1032 |          34333 |                 273 |                9757 | VALID    |
|      3 |         1 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |     45395 |          10030 |           1305 |          44090 |                 205 |                9825 | VALID    |
|      1 |         2 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |     25282 |          10083 |            741 |          24541 |                 299 |                9784 | VALID    |
|      2 |         2 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |     35365 |           9860 |           1040 |          34325 |                 267 |                9593 | VALID    |
|      3 |         2 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |     45225 |           9860 |           1307 |          43918 |                 201 |                9659 | VALID    |
|      1 |         3 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |     25282 |          10083 |            742 |          24540 |                 299 |                9784 | VALID    |
|      2 |         3 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |     35365 |           9690 |           1041 |          34324 |                 265 |                9425 | VALID    |
|      3 |         3 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |     45055 |           9690 |           1306 |          43749 |                 195 |                9495 | VALID    |

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
    "probability_threshold": 0.74,
    "growth_label_threshold": 2.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.23943661971830985,
    "validation_pr_auc": 0.14748845996320975
  },
  "2": {
    "model": "XGBoost",
    "feature_set": "Semantic + Temporal",
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 3,
      "learning_rate": 0.05,
      "subsample": 0.85,
      "colsample_bytree": 0.85
    },
    "probability_threshold": 0.68,
    "growth_label_threshold": 2.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.2256509161041466,
    "validation_pr_auc": 0.13858722220618377
  },
  "3": {
    "model": "Logistic Regression",
    "feature_set": "Temporal Only",
    "hyperparameters": {
      "C": 0.1,
      "max_iter": 2000,
      "class_weight": "balanced",
      "solver": "lbfgs"
    },
    "probability_threshold": 0.63,
    "growth_label_threshold": 2.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.19410745233968804,
    "validation_pr_auc": 0.10518874816752165
  }
}
```

## 5. Validation performance

|   horizon | feature_set         | model               |   precision |   recall |       f1 |   macro_f1 |   roc_auc |    pr_auc |   probability_threshold |
|----------:|:--------------------|:--------------------|------------:|---------:|---------:|-----------:|----------:|----------:|------------------------:|
|         1 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.494504 |  0.5      | 0.0217449 |                    0.5  |
|         1 | Temporal Only       | Logistic Regression |    0.169687 | 0.31307  | 0.220085 |   0.597596 |  0.841331 | 0.139773  |                    0.74 |
|         1 | Temporal Only       | Random Forest       |    0.175439 | 0.273556 | 0.213777 |   0.595637 |  0.822076 | 0.120955  |                    0.69 |
|         1 | Temporal Only       | XGBoost             |    0.147208 | 0.440729 | 0.2207   |   0.592662 |  0.839703 | 0.134625  |                    0.7  |
|         1 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.494504 |  0.5      | 0.0217449 |                    0.5  |
|         1 | Semantic Only       | Logistic Regression |    0.143906 | 0.297872 | 0.194059 |   0.583115 |  0.694167 | 0.0750283 |                    0.79 |
|         1 | Semantic Only       | Random Forest       |    0.122596 | 0.31003  | 0.175711 |   0.571411 |  0.714477 | 0.089038  |                    0.61 |
|         1 | Semantic Only       | XGBoost             |    0.166359 | 0.273556 | 0.206897 |   0.59171  |  0.705782 | 0.0977462 |                    0.8  |
|         1 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.494504 |  0.5      | 0.0217449 |                    0.5  |
|         1 | Semantic + Temporal | Logistic Regression |    0.174061 | 0.31003  | 0.222951 |   0.599361 |  0.839362 | 0.13834   |                    0.77 |
|         1 | Semantic + Temporal | Random Forest       |    0.183824 | 0.303951 | 0.229095 |   0.603097 |  0.817787 | 0.136133  |                    0.59 |
|         1 | Semantic + Temporal | XGBoost             |    0.195029 | 0.31003  | 0.239437 |   0.608701 |  0.822511 | 0.147488  |                    0.74 |
|         2 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.494526 |  0.5      | 0.0216578 |                    0.5  |
|         2 | Temporal Only       | Logistic Regression |    0.158784 | 0.290123 | 0.20524  |   0.59007  |  0.813926 | 0.11654   |                    0.71 |
|         2 | Temporal Only       | Random Forest       |    0.159176 | 0.262346 | 0.198135 |   0.587231 |  0.803949 | 0.10521   |                    0.66 |
|         2 | Temporal Only       | XGBoost             |    0.149441 | 0.330247 | 0.205769 |   0.588584 |  0.811544 | 0.11923   |                    0.73 |
|         2 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.494526 |  0.5      | 0.0216578 |                    0.5  |
|         2 | Semantic Only       | Logistic Regression |    0.110254 | 0.308642 | 0.16247  |   0.563266 |  0.65723  | 0.0598933 |                    0.64 |
|         2 | Semantic Only       | Random Forest       |    0.122963 | 0.256173 | 0.166166 |   0.568682 |  0.67034  | 0.0705475 |                    0.59 |
|         2 | Semantic Only       | XGBoost             |    0.133333 | 0.246914 | 0.17316  |   0.573406 |  0.692353 | 0.0819364 |                    0.75 |
|         2 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.494526 |  0.5      | 0.0216578 |                    0.5  |
|         2 | Semantic + Temporal | Logistic Regression |    0.139169 | 0.382716 | 0.204115 |   0.585214 |  0.812872 | 0.111123  |                    0.66 |
|         2 | Semantic + Temporal | Random Forest       |    0.144531 | 0.342593 | 0.203297 |   0.586559 |  0.801178 | 0.114837  |                    0.55 |
|         2 | Semantic + Temporal | XGBoost             |    0.164095 | 0.361111 | 0.225651 |   0.598925 |  0.811133 | 0.138587  |                    0.68 |
|         3 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.494532 |  0.5      | 0.0216362 |                    0.5  |
|         3 | Temporal Only       | Logistic Regression |    0.134293 | 0.35     | 0.194107 |   0.580695 |  0.804741 | 0.105189  |                    0.63 |
|         3 | Temporal Only       | Random Forest       |    0.124031 | 0.25     | 0.165803 |   0.568836 |  0.779243 | 0.0904301 |                    0.64 |
|         3 | Temporal Only       | XGBoost             |    0.157563 | 0.234375 | 0.188442 |   0.583    |  0.79765  | 0.0994428 |                    0.77 |
|         3 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.494532 |  0.5      | 0.0216362 |                    0.5  |
|         3 | Semantic Only       | Logistic Regression |    0.11186  | 0.259375 | 0.156309 |   0.562445 |  0.645177 | 0.061606  |                    0.71 |
|         3 | Semantic Only       | Random Forest       |    0.135021 | 0.2      | 0.161209 |   0.569036 |  0.613103 | 0.0629763 |                    0.62 |
|         3 | Semantic Only       | XGBoost             |    0.136771 | 0.190625 | 0.159269 |   0.568459 |  0.648451 | 0.067697  |                    0.76 |
|         3 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.494532 |  0.5      | 0.0216362 |                    0.5  |
|         3 | Semantic + Temporal | Logistic Regression |    0.133166 | 0.33125  | 0.189964 |   0.579102 |  0.803879 | 0.101762  |                    0.67 |
|         3 | Semantic + Temporal | Random Forest       |    0.123116 | 0.30625  | 0.175627 |   0.571653 |  0.772409 | 0.0972343 |                    0.57 |
|         3 | Semantic + Temporal | XGBoost             |    0.125967 | 0.35625  | 0.186122 |   0.575481 |  0.786529 | 0.113565  |                    0.65 |

## 6. Ablation

Perbandingan t+1 terbaik per feature set pada official validation (test tidak dipakai untuk memilih ablation):

| feature_set         | model   |       f1 |    pr_auc |
|:--------------------|:--------|---------:|----------:|
| Semantic + Temporal | XGBoost | 0.239437 | 0.147488  |
| Semantic Only       | XGBoost | 0.206897 | 0.0977462 |
| Temporal Only       | XGBoost | 0.2207   | 0.134625  |

## 7. Multi-horizon dan 8. final test

|   horizon | feature_set         | model               |   precision |   recall |       f1 |   macro_f1 |   roc_auc |   pr_auc |
|----------:|:--------------------|:--------------------|------------:|---------:|---------:|-----------:|----------:|---------:|
|         1 | Semantic + Temporal | XGBoost             |    0.318304 | 0.520352 | 0.39499  |   0.671494 |  0.872525 | 0.413424 |
|         2 | Semantic + Temporal | XGBoost             |    0.267109 | 0.574725 | 0.364714 |   0.648681 |  0.846761 | 0.359397 |
|         3 | Temporal Only       | Logistic Regression |    0.276387 | 0.557018 | 0.369455 |   0.652413 |  0.858793 | 0.350649 |

## 9. Distribution shift

Mean article/day berubah dari 125,55 pada training menjadi 244,64 pada test. Perubahan validation ke test untuk model terpilih:

|   horizon |   validation_f1 |   test_f1 |   f1_change |   validation_pr_auc |   test_pr_auc |   pr_auc_change |
|----------:|----------------:|----------:|------------:|--------------------:|--------------:|----------------:|
|         1 |        0.239437 |  0.39499  |    0.155553 |            0.147488 |      0.413424 |        0.265935 |
|         2 |        0.225651 |  0.364714 |    0.139063 |            0.138587 |      0.359397 |        0.22081  |
|         3 |        0.194107 |  0.369455 |    0.175347 |            0.105189 |      0.350649 |        0.24546  |

Perubahan ini dilaporkan sebagai temporal distribution shift; penyebabnya tidak disimpulkan.

## 10. Feature importance

Feature importance menyatakan predictive importance, bukan hubungan sebab-akibat. Tabel lengkap memuat native importance dan permutation importance.

|   horizon | model   | feature_set         | feature                  | feature_group   |   native_importance | native_method   |   permutation_importance_mean_f1 |   permutation_importance_std_f1 |
|----------:|:--------|:--------------------|:-------------------------|:----------------|--------------------:|:----------------|---------------------------------:|--------------------------------:|
|         1 | XGBoost | Semantic + Temporal | current_volume           | Temporal        |          0.290809   | tree_importance |                      0.0236296   |                     0.000591893 |
|         1 | XGBoost | Semantic + Temporal | rolling_std_7            | Temporal        |          0.183686   | tree_importance |                      0.0051011   |                     0.00263831  |
|         1 | XGBoost | Semantic + Temporal | semantic_drift           | Semantic        |          0.0125881  | tree_importance |                      0.00459944  |                     0.000477693 |
|         1 | XGBoost | Semantic + Temporal | semantic_similarity_std  | Semantic        |          0.108004   | tree_importance |                      0.0029781   |                     0.000979366 |
|         1 | XGBoost | Semantic + Temporal | lag_1                    | Temporal        |          0.0218966  | tree_importance |                      0.00254869  |                     0.0002962   |
|         1 | XGBoost | Semantic + Temporal | rolling_mean_3           | Temporal        |          0.0260775  | tree_importance |                      0.00225104  |                     0.00106719  |
|         1 | XGBoost | Semantic + Temporal | topic_probability_change | Semantic        |          0.0090795  | tree_importance |                      0.00173664  |                     0.000715302 |
|         1 | XGBoost | Semantic + Temporal | topic_probability_mean   | Semantic        |          0.0119125  | tree_importance |                      0.00050614  |                     0.0011019   |
|         1 | XGBoost | Semantic + Temporal | lag_2                    | Temporal        |          0.0106276  | tree_importance |                      0.000377268 |                     0.000485384 |
|         1 | XGBoost | Semantic + Temporal | growth_rate              | Temporal        |          0.0162033  | tree_importance |                      0.000117453 |                     0.000404736 |
|         1 | XGBoost | Semantic + Temporal | momentum                 | Temporal        |          0.0164835  | tree_importance |                     -8.63289e-05 |                     0.000543854 |
|         1 | XGBoost | Semantic + Temporal | topic_probability_std    | Semantic        |          0.0211221  | tree_importance |                     -0.000126373 |                     0.000518164 |
|         1 | XGBoost | Semantic + Temporal | lag_3                    | Temporal        |          0.0100571  | tree_importance |                     -0.000372548 |                     0.000439486 |
|         1 | XGBoost | Semantic + Temporal | lag_4                    | Temporal        |          0.00992939 | tree_importance |                     -0.000619758 |                     0.000308622 |
|         1 | XGBoost | Semantic + Temporal | percentage_change        | Temporal        |          0.0186055  | tree_importance |                     -0.0010367   |                     0.000466388 |

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

F1 validation terbaik Semantic Only untuk t+1/t+2/t+3: **0.2069 / 0.1732 / 0.1612**; majority baseline pada horizon yang sama: **0.0000 / 0.0000 / 0.0000**. Semantic Only berada di atas majority baseline pada **3/3** horizon. Angka ini menyatakan predictive signal, bukan klaim kausal.

## 14. RQ2 — Temporal features

F1 validation terbaik Temporal Only untuk t+1/t+2/t+3: **0.2207 / 0.2058 / 0.1941**. Dibanding Semantic Only pada split dan kandidat learner yang sama, Temporal Only lebih tinggi pada **3/3** horizon.

## 15. RQ3 — Fusion

F1 validation terbaik Semantic + Temporal untuk t+1/t+2/t+3: **0.2394 / 0.2257 / 0.1900**. Fusion melampaui Temporal Only pada **2/3** horizon. Feature set yang terpilih per horizon: {1: 'Semantic + Temporal', 2: 'Semantic + Temporal', 3: 'Temporal Only'}. Karena hanya tiga fold, perbedaan antar feature set tidak dinyatakan signifikan.

## 16. RQ4 — Forecast horizon

Pada final test, F1 model terpilih adalah **0.3950** (t+1), **0.3647** (t+2), dan **0.3695** (t+3) — tidak monoton terhadap horizon. PR-AUC adalah **0.4134**, **0.3594**, dan **0.3506** — menurun monoton. Tidak ada horizon yang dicampur saat fitting.

## Status penelitian

**LIMITED** — eksperimen dapat dievaluasi secara temporal dan bebas leakage yang terdeteksi, tetapi distribution shift, truncation, topic outlier, serta jumlah fold membatasi generalisasi hasil.

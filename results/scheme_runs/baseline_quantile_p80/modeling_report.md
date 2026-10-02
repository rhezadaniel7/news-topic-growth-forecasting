# Modeling report

## 0. Definisi label

Label growth memakai skema **baseline_quantile_p80** (S0 baseline kuantil p80).

- Definisi: `growth_score > kuantil p80 growth_score training`
- Jenis threshold: `train_quantile`
- Populasi: `all`
- Alasan pemilihan: Skema yang berlaku sebelumnya. Karena zero-inflation, p80 training bernilai 0,0 sehingga aturan kolaps menjadi future_volume > median historis, dan untuk 91,5% observasi eligible median historis = 0. Label 'growth' praktis berarti 'besok ada artikel'.

Positive rate label per split (dari partisi resmi):

|   horizon |   test |   train |   validation |
|----------:|-------:|--------:|-------------:|
|         1 | 0.2725 |  0.1815 |       0.1705 |
|         2 | 0.2739 |  0.1815 |       0.1699 |
|         3 | 0.2754 |  0.1815 |       0.1701 |

## 1. Feature dataset

Final dataset menggabungkan semantic, temporal, dan label audit dengan key `(topic_id, date, forecast_horizon)`. Tidak ada duplicate key. Target, future volume, target date, growth score, burst score, dan source ID tidak digunakan sebagai predictor.

| record_type   | label_scheme          |   label_threshold |   horizon | split      | feature   |   rows |   growth |   normal |   missing |   missing_rate |
|:--------------|:----------------------|------------------:|----------:|:-----------|:----------|-------:|---------:|---------:|----------:|---------------:|
| partition     | baseline_quantile_p80 |                 0 |         1 | train      |           |  40295 |     7314 |    32981 |    223874 |       0.292414 |
| partition     | baseline_quantile_p80 |                 0 |         1 | validation |           |  15130 |     2579 |    12551 |     85414 |       0.297123 |
| partition     | baseline_quantile_p80 |                 0 |         1 | test       |           |  15130 |     4123 |    11007 |     75061 |       0.261109 |
| partition     | baseline_quantile_p80 |                 0 |         2 | train      |           |  40125 |     7284 |    32841 |    222887 |       0.292359 |
| partition     | baseline_quantile_p80 |                 0 |         2 | validation |           |  14960 |     2542 |    12418 |     84434 |       0.297052 |
| partition     | baseline_quantile_p80 |                 0 |         2 | test       |           |  14960 |     4097 |    10863 |     74347 |       0.261564 |
| partition     | baseline_quantile_p80 |                 0 |         3 | train      |           |  39955 |     7253 |    32702 |    221956 |       0.292376 |
| partition     | baseline_quantile_p80 |                 0 |         3 | validation |           |  14790 |     2516 |    12274 |     83482 |       0.297078 |
| partition     | baseline_quantile_p80 |                 0 |         3 | test       |           |  14790 |     4073 |    10717 |     73654 |       0.262105 |

## 2. Temporal folds

Digunakan tiga expanding-window fold dengan validation block yang tidak tumpang tindih. Jumlah fold valid: **3/3**. Test tidak digunakan untuk tuning.

|   fold |   horizon | train_start   | train_end   | validation_start   | validation_end   |   n_train |   n_validation |   growth_train |   normal_train |   growth_validation |   normal_validation | status   |
|-------:|----------:|:--------------|:------------|:-------------------|:-----------------|----------:|---------------:|---------------:|---------------:|--------------------:|--------------------:|:---------|
|      1 |         1 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |     25282 |          10083 |           4708 |          20574 |                1835 |                8248 | VALID    |
|      2 |         1 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |     35365 |          10030 |           6543 |          28822 |                1667 |                8363 | VALID    |
|      3 |         1 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |     45395 |          10030 |           8210 |          37185 |                1683 |                8347 | VALID    |
|      1 |         2 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |     25282 |          10083 |           4710 |          20572 |                1821 |                8262 | VALID    |
|      2 |         2 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |     35365 |           9860 |           6531 |          28834 |                1642 |                8218 | VALID    |
|      3 |         2 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |     45225 |           9860 |           8173 |          37052 |                1653 |                8207 | VALID    |
|      1 |         3 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |     25282 |          10083 |           4710 |          20572 |                1819 |                8264 | VALID    |
|      2 |         3 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |     35365 |           9690 |           6529 |          28836 |                1615 |                8075 | VALID    |
|      3 |         3 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |     45055 |           9690 |           8144 |          36911 |                1625 |                8065 | VALID    |

## 3. Model configuration dan 4. hyperparameters

Model: Majority Class, Logistic Regression, Random Forest, dan XGBoost. Setiap learner diuji dengan Temporal Only, Semantic Only, dan Semantic + Temporal. Logistic memakai median imputation dan StandardScaler; tree model memakai median imputation. Class weighting diterapkan hanya saat fitting training. Tidak ada SMOTE.

Konfigurasi final per horizon dibekukan dari official validation:

```json
{
  "1": {
    "model": "XGBoost",
    "feature_set": "Temporal Only",
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 3,
      "learning_rate": 0.05,
      "subsample": 0.85,
      "colsample_bytree": 0.85
    },
    "probability_threshold": 0.54,
    "growth_label_threshold": 0.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.4436437013548573,
    "validation_pr_auc": 0.4067896391609971
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
    "probability_threshold": 0.55,
    "growth_label_threshold": 0.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.4306235201262826,
    "validation_pr_auc": 0.3946475130548408
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
    "probability_threshold": 0.5,
    "growth_label_threshold": 0.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.41892639734366355,
    "validation_pr_auc": 0.3714588145060409
  }
}
```

## 5. Validation performance

|   horizon | feature_set         | model               |   precision |   recall |       f1 |   macro_f1 |   roc_auc |   pr_auc |   probability_threshold |
|----------:|:--------------------|:--------------------|------------:|---------:|---------:|-----------:|----------:|---------:|------------------------:|
|         1 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.453416 |  0.5      | 0.170456 |                    0.5  |
|         1 | Temporal Only       | Logistic Regression |    0.360092 | 0.547887 | 0.434569 |   0.639897 |  0.737094 | 0.397777 |                    0.48 |
|         1 | Temporal Only       | Random Forest       |    0.354283 | 0.590151 | 0.442764 |   0.63947  |  0.740986 | 0.401783 |                    0.54 |
|         1 | Temporal Only       | XGBoost             |    0.353063 | 0.596743 | 0.443644 |   0.639067 |  0.742274 | 0.40679  |                    0.54 |
|         1 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.453416 |  0.5      | 0.170456 |                    0.5  |
|         1 | Semantic Only       | Logistic Regression |    0.401179 | 0.343156 | 0.369906 |   0.625777 |  0.607177 | 0.279656 |                    0.46 |
|         1 | Semantic Only       | Random Forest       |    0.371438 | 0.414502 | 0.39179  |   0.628988 |  0.640695 | 0.306823 |                    0.43 |
|         1 | Semantic Only       | XGBoost             |    0.372322 | 0.411012 | 0.390711 |   0.628794 |  0.639486 | 0.308506 |                    0.41 |
|         1 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.453416 |  0.5      | 0.170456 |                    0.5  |
|         1 | Semantic + Temporal | Logistic Regression |    0.359859 | 0.554091 | 0.436336 |   0.64031  |  0.737721 | 0.398283 |                    0.48 |
|         1 | Semantic + Temporal | Random Forest       |    0.351983 | 0.578131 | 0.437564 |   0.637038 |  0.740023 | 0.407533 |                    0.53 |
|         1 | Semantic + Temporal | XGBoost             |    0.350805 | 0.591702 | 0.440468 |   0.637147 |  0.73948  | 0.405781 |                    0.53 |
|         2 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.453576 |  0.5      | 0.16992  |                    0.5  |
|         2 | Temporal Only       | Logistic Regression |    0.371834 | 0.485051 | 0.420963 |   0.639997 |  0.722572 | 0.375641 |                    0.5  |
|         2 | Temporal Only       | Random Forest       |    0.359226 | 0.533045 | 0.429205 |   0.638266 |  0.726162 | 0.38214  |                    0.56 |
|         2 | Temporal Only       | XGBoost             |    0.354731 | 0.545633 | 0.429944 |   0.63661  |  0.727487 | 0.385419 |                    0.55 |
|         2 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.453576 |  0.5      | 0.16992  |                    0.5  |
|         2 | Semantic Only       | Logistic Regression |    0.374399 | 0.337136 | 0.354792 |   0.615277 |  0.608634 | 0.264907 |                    0.46 |
|         2 | Semantic Only       | Random Forest       |    0.342384 | 0.381983 | 0.361101 |   0.610548 |  0.622379 | 0.283636 |                    0.43 |
|         2 | Semantic Only       | XGBoost             |    0.340107 | 0.375295 | 0.356836 |   0.608443 |  0.617547 | 0.279368 |                    0.43 |
|         2 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.453576 |  0.5      | 0.16992  |                    0.5  |
|         2 | Semantic + Temporal | Logistic Regression |    0.34454  | 0.536192 | 0.419514 |   0.629234 |  0.722122 | 0.373328 |                    0.48 |
|         2 | Semantic + Temporal | Random Forest       |    0.35961  | 0.536585 | 0.430624 |   0.638844 |  0.728118 | 0.394648 |                    0.55 |
|         2 | Semantic + Temporal | XGBoost             |    0.34974  | 0.556255 | 0.429461 |   0.634229 |  0.727074 | 0.389035 |                    0.54 |
|         3 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.453518 |  0.5      | 0.170115 |                    0.5  |
|         3 | Temporal Only       | Logistic Regression |    0.322559 | 0.575119 | 0.41331  |   0.615682 |  0.717854 | 0.36532  |                    0.47 |
|         3 | Temporal Only       | Random Forest       |    0.335613 | 0.546105 | 0.415734 |   0.623801 |  0.718949 | 0.370513 |                    0.54 |
|         3 | Temporal Only       | XGBoost             |    0.347068 | 0.522258 | 0.41701  |   0.62959  |  0.720507 | 0.37401  |                    0.55 |
|         3 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.453518 |  0.5      | 0.170115 |                    0.5  |
|         3 | Semantic Only       | Logistic Regression |    0.363923 | 0.321542 | 0.341422 |   0.607892 |  0.597817 | 0.256229 |                    0.47 |
|         3 | Semantic Only       | Random Forest       |    0.337251 | 0.376391 | 0.355748 |   0.60717  |  0.616718 | 0.273094 |                    0.44 |
|         3 | Semantic Only       | XGBoost             |    0.332007 | 0.364865 | 0.347661 |   0.602943 |  0.60825  | 0.267132 |                    0.44 |
|         3 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.453518 |  0.5      | 0.170115 |                    0.5  |
|         3 | Semantic + Temporal | Logistic Regression |    0.323956 | 0.576709 | 0.414868 |   0.616821 |  0.717872 | 0.362925 |                    0.47 |
|         3 | Semantic + Temporal | Random Forest       |    0.321307 | 0.601749 | 0.418926 |   0.615512 |  0.720228 | 0.371459 |                    0.5  |
|         3 | Semantic + Temporal | XGBoost             |    0.324439 | 0.579889 | 0.416084 |   0.617312 |  0.717335 | 0.366998 |                    0.51 |

## 6. Ablation

Perbandingan t+1 terbaik per feature set pada official validation (test tidak dipakai untuk memilih ablation):

| feature_set         | model         |       f1 |   pr_auc |
|:--------------------|:--------------|---------:|---------:|
| Semantic + Temporal | XGBoost       | 0.440468 | 0.405781 |
| Semantic Only       | Random Forest | 0.39179  | 0.306823 |
| Temporal Only       | XGBoost       | 0.443644 | 0.40679  |

## 7. Multi-horizon dan 8. final test

|   horizon | feature_set         | model         |   precision |   recall |       f1 |   macro_f1 |   roc_auc |   pr_auc |
|----------:|:--------------------|:--------------|------------:|---------:|---------:|-----------:|----------:|---------:|
|         1 | Temporal Only       | XGBoost       |    0.491296 | 0.766675 | 0.598844 |   0.691951 |  0.804097 | 0.636226 |
|         2 | Semantic + Temporal | Random Forest |    0.5083   | 0.710032 | 0.592464 |   0.69668  |  0.79241  | 0.614387 |
|         3 | Semantic + Temporal | Random Forest |    0.46214  | 0.768721 | 0.577249 |   0.666214 |  0.785681 | 0.60739  |

## 9. Distribution shift

Mean article/day berubah dari 125,55 pada training menjadi 244,64 pada test. Perubahan validation ke test untuk model terpilih:

|   horizon |   validation_f1 |   test_f1 |   f1_change |   validation_pr_auc |   test_pr_auc |   pr_auc_change |
|----------:|----------------:|----------:|------------:|--------------------:|--------------:|----------------:|
|         1 |        0.443644 |  0.598844 |    0.155201 |            0.40679  |      0.636226 |        0.229436 |
|         2 |        0.430624 |  0.592464 |    0.161841 |            0.394648 |      0.614387 |        0.21974  |
|         3 |        0.418926 |  0.577249 |    0.158323 |            0.371459 |      0.60739  |        0.235932 |

Perubahan ini dilaporkan sebagai temporal distribution shift; penyebabnya tidak disimpulkan.

## 10. Feature importance

Feature importance menyatakan predictive importance, bukan hubungan sebab-akibat. Tabel lengkap memuat native importance dan permutation importance.

|   horizon | model   | feature_set   | feature           | feature_group   |   native_importance | native_method   |   permutation_importance_mean_f1 |   permutation_importance_std_f1 |
|----------:|:--------|:--------------|:------------------|:----------------|--------------------:|:----------------|---------------------------------:|--------------------------------:|
|         1 | XGBoost | Temporal Only | rolling_mean_7    | Temporal        |          0.460737   | tree_importance |                      0.0575886   |                     0.00289431  |
|         1 | XGBoost | Temporal Only | current_volume    | Temporal        |          0.173917   | tree_importance |                      0.0440424   |                     0.0021002   |
|         1 | XGBoost | Temporal Only | rolling_std_7     | Temporal        |          0.159287   | tree_importance |                      0.00244476  |                     0.00168712  |
|         1 | XGBoost | Temporal Only | percentage_change | Temporal        |          0.0188434  | tree_importance |                     -7.02236e-05 |                     9.01403e-05 |
|         1 | XGBoost | Temporal Only | rolling_mean_3    | Temporal        |          0.0732157  | tree_importance |                     -0.000145443 |                     0.000251219 |
|         1 | XGBoost | Temporal Only | lag_4             | Temporal        |          0.00842531 | tree_importance |                     -0.000470351 |                     0.000283156 |
|         1 | XGBoost | Temporal Only | momentum          | Temporal        |          0.0143738  | tree_importance |                     -0.000729099 |                     0.000461953 |
|         1 | XGBoost | Temporal Only | lag_2             | Temporal        |          0.0129521  | tree_importance |                     -0.00162517  |                     0.000230749 |
|         1 | XGBoost | Temporal Only | lag_3             | Temporal        |          0.00630182 | tree_importance |                     -0.00204086  |                     0.000811039 |
|         1 | XGBoost | Temporal Only | lag_1             | Temporal        |          0.0325546  | tree_importance |                     -0.00297393  |                     0.00026032  |
|         1 | XGBoost | Temporal Only | growth_rate       | Temporal        |          0.00985741 | tree_importance |                     -0.00461571  |                     0.000776358 |
|         1 | XGBoost | Temporal Only | burst_intensity   | Temporal        |          0.0295346  | tree_importance |                     -0.00600096  |                     0.000712725 |

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

F1 validation terbaik Semantic Only untuk t+1/t+2/t+3: **0.3918 / 0.3611 / 0.3557**; majority baseline pada horizon yang sama: **0.0000 / 0.0000 / 0.0000**. Semantic Only berada di atas majority baseline pada **3/3** horizon. Angka ini menyatakan predictive signal, bukan klaim kausal.

## 14. RQ2 — Temporal features

F1 validation terbaik Temporal Only untuk t+1/t+2/t+3: **0.4436 / 0.4299 / 0.4170**. Dibanding Semantic Only pada split dan kandidat learner yang sama, Temporal Only lebih tinggi pada **3/3** horizon.

## 15. RQ3 — Fusion

F1 validation terbaik Semantic + Temporal untuk t+1/t+2/t+3: **0.4405 / 0.4306 / 0.4189**. Fusion melampaui Temporal Only pada **2/3** horizon. Feature set yang terpilih per horizon: {1: 'Temporal Only', 2: 'Semantic + Temporal', 3: 'Semantic + Temporal'}. Karena hanya tiga fold, perbedaan antar feature set tidak dinyatakan signifikan.

## 16. RQ4 — Forecast horizon

Pada final test, F1 model terpilih adalah **0.5988** (t+1), **0.5925** (t+2), dan **0.5772** (t+3) — menurun monoton terhadap horizon. PR-AUC adalah **0.6362**, **0.6144**, dan **0.6074** — menurun monoton. Tidak ada horizon yang dicampur saat fitting.

## Status penelitian

**LIMITED** — eksperimen dapat dievaluasi secara temporal dan bebas leakage yang terdeteksi, tetapi distribution shift, truncation, topic outlier, serta jumlah fold membatasi generalisasi hasil.

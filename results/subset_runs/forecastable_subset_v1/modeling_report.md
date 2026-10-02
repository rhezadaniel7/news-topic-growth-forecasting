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
|         1 | 0.1837 |  0.0722 |       0.0731 |
|         2 | 0.1858 |  0.0728 |       0.0733 |
|         3 | 0.1884 |  0.0732 |       0.0734 |

## 1. Feature dataset

Final dataset menggabungkan semantic, temporal, dan label audit dengan key `(topic_id, date, forecast_horizon)`. Tidak ada duplicate key. Target, future volume, target date, growth score, burst score, dan source ID tidak digunakan sebagai predictor.

| record_type   | label_scheme   |   label_threshold |   horizon | split      | feature   |   rows |   growth |   normal |   missing |   missing_rate |
|:--------------|:---------------|------------------:|----------:|:-----------|:----------|-------:|---------:|---------:|----------:|---------------:|
| partition     | burst_mad_k2   |                 2 |         1 | train      |           |  12344 |      891 |    11453 |     52143 |       0.222324 |
| partition     | burst_mad_k2   |                 2 |         1 | validation |           |   4361 |      319 |     4042 |     17626 |       0.212723 |
| partition     | burst_mad_k2   |                 2 |         1 | test       |           |   4361 |      801 |     3560 |     13118 |       0.158317 |
| partition     | burst_mad_k2   |                 2 |         2 | train      |           |  12295 |      895 |    11400 |     51933 |       0.222311 |
| partition     | burst_mad_k2   |                 2 |         2 | validation |           |   4312 |      316 |     3996 |     17423 |       0.212662 |
| partition     | burst_mad_k2   |                 2 |         2 | test       |           |   4312 |      801 |     3511 |     13020 |       0.15892  |
| partition     | burst_mad_k2   |                 2 |         3 | train      |           |  12246 |      896 |    11350 |     51758 |       0.222449 |
| partition     | burst_mad_k2   |                 2 |         3 | validation |           |   4263 |      313 |     3950 |     17248 |       0.212946 |
| partition     | burst_mad_k2   |                 2 |         3 | test       |           |   4263 |      803 |     3460 |     12929 |       0.159623 |

## 2. Temporal folds

Digunakan tiga expanding-window fold dengan validation block yang tidak tumpang tindih. Jumlah fold valid: **3/3**. Test tidak digunakan untuk tuning.

|   fold |   horizon | train_start   | train_end   | validation_start   | validation_end   |   n_train |   n_validation |   growth_train |   normal_train |   growth_validation |   normal_validation | status   |
|-------:|----------:|:--------------|:------------|:-------------------|:-----------------|----------:|---------------:|---------------:|---------------:|--------------------:|--------------------:|:---------|
|      1 |         1 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |      7983 |           2940 |            566 |           7417 |                 206 |                2734 | VALID    |
|      2 |         1 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |     10923 |           2891 |            772 |          10151 |                 233 |                2658 | VALID    |
|      3 |         1 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |     13814 |           2891 |           1005 |          12809 |                 205 |                2686 | VALID    |
|      1 |         2 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |      7983 |           2940 |            571 |           7412 |                 206 |                2734 | VALID    |
|      2 |         2 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |     10923 |           2842 |            777 |          10146 |                 231 |                2611 | VALID    |
|      3 |         2 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |     13765 |           2842 |           1008 |          12757 |                 203 |                2639 | VALID    |
|      1 |         3 | 2024-07-31    | 2025-01-26  | 2025-01-27         | 2025-03-27       |      7983 |           2940 |            573 |           7410 |                 207 |                2733 | VALID    |
|      2 |         3 | 2024-07-31    | 2025-03-27  | 2025-03-28         | 2025-05-26       |     10923 |           2793 |            780 |          10143 |                 230 |                2563 | VALID    |
|      3 |         3 | 2024-07-31    | 2025-05-26  | 2025-05-27         | 2025-07-25       |     13716 |           2793 |           1010 |          12706 |                 199 |                2594 | VALID    |

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
    "probability_threshold": 0.64,
    "growth_label_threshold": 2.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.3490304709141274,
    "validation_pr_auc": 0.2877423009308882
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
    "probability_threshold": 0.65,
    "growth_label_threshold": 2.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.3261538461538461,
    "validation_pr_auc": 0.270949096544647
  },
  "3": {
    "model": "XGBoost",
    "feature_set": "Semantic + Temporal",
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 3,
      "learning_rate": 0.05,
      "subsample": 0.85,
      "colsample_bytree": 0.85
    },
    "probability_threshold": 0.7,
    "growth_label_threshold": 2.0,
    "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
    "validation_f1": 0.30550621669627,
    "validation_pr_auc": 0.2411226453147473
  }
}
```

## 5. Validation performance

|   horizon | feature_set         | model               |   precision |   recall |       f1 |   macro_f1 |   roc_auc |    pr_auc |   probability_threshold |
|----------:|:--------------------|:--------------------|------------:|---------:|---------:|-----------:|----------:|----------:|------------------------:|
|         1 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.481019 |  0.5      | 0.0731484 |                    0.5  |
|         1 | Temporal Only       | Logistic Regression |    0.306701 | 0.373041 | 0.336634 |   0.639059 |  0.790933 | 0.270886  |                    0.66 |
|         1 | Temporal Only       | Random Forest       |    0.265393 | 0.39185  | 0.316456 |   0.624189 |  0.782016 | 0.253346  |                    0.57 |
|         1 | Temporal Only       | XGBoost             |    0.266791 | 0.448276 | 0.334503 |   0.631088 |  0.789102 | 0.269166  |                    0.65 |
|         1 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.481019 |  0.5      | 0.0731484 |                    0.5  |
|         1 | Semantic Only       | Logistic Regression |    0.215504 | 0.435737 | 0.288382 |   0.599978 |  0.681431 | 0.152799  |                    0.63 |
|         1 | Semantic Only       | Random Forest       |    0.244161 | 0.360502 | 0.291139 |   0.61027  |  0.702547 | 0.206319  |                    0.55 |
|         1 | Semantic Only       | XGBoost             |    0.261176 | 0.347962 | 0.298387 |   0.616479 |  0.692736 | 0.192154  |                    0.71 |
|         1 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.481019 |  0.5      | 0.0731484 |                    0.5  |
|         1 | Semantic + Temporal | Logistic Regression |    0.26465  | 0.438871 | 0.330189 |   0.629026 |  0.784102 | 0.26034   |                    0.64 |
|         1 | Semantic + Temporal | Random Forest       |    0.38189  | 0.304075 | 0.338569 |   0.64603  |  0.792058 | 0.291141  |                    0.57 |
|         1 | Semantic + Temporal | XGBoost             |    0.312655 | 0.394984 | 0.34903  |   0.64514  |  0.78159  | 0.287742  |                    0.64 |
|         2 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.480982 |  0.5      | 0.0732839 |                    0.5  |
|         2 | Temporal Only       | Logistic Regression |    0.258696 | 0.376582 | 0.306701 |   0.619074 |  0.772028 | 0.241193  |                    0.59 |
|         2 | Temporal Only       | Random Forest       |    0.285714 | 0.322785 | 0.30312  |   0.622067 |  0.749865 | 0.224009  |                    0.6  |
|         2 | Temporal Only       | XGBoost             |    0.310769 | 0.31962  | 0.315133 |   0.63007  |  0.764738 | 0.242636  |                    0.72 |
|         2 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.480982 |  0.5      | 0.0732839 |                    0.5  |
|         2 | Semantic Only       | Logistic Regression |    0.176822 | 0.468354 | 0.256722 |   0.571006 |  0.645924 | 0.137113  |                    0.46 |
|         2 | Semantic Only       | Random Forest       |    0.189831 | 0.35443  | 0.247241 |   0.579438 |  0.641903 | 0.159431  |                    0.48 |
|         2 | Semantic Only       | XGBoost             |    0.181586 | 0.449367 | 0.258652 |   0.575247 |  0.682416 | 0.168511  |                    0.56 |
|         2 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.480982 |  0.5      | 0.0732839 |                    0.5  |
|         2 | Semantic + Temporal | Logistic Regression |    0.22807  | 0.452532 | 0.303287 |   0.608876 |  0.762917 | 0.228535  |                    0.58 |
|         2 | Semantic + Temporal | Random Forest       |    0.257143 | 0.370253 | 0.303502 |   0.61756  |  0.749138 | 0.250404  |                    0.52 |
|         2 | Semantic + Temporal | XGBoost             |    0.317365 | 0.335443 | 0.326154 |   0.635613 |  0.765076 | 0.270949  |                    0.65 |
|         3 | Temporal Only       | Majority Class      |    0        | 0        | 0        |   0.480945 |  0.5      | 0.0734225 |                    0.5  |
|         3 | Temporal Only       | Logistic Regression |    0.238921 | 0.396166 | 0.298077 |   0.611087 |  0.75746  | 0.21077   |                    0.57 |
|         3 | Temporal Only       | Random Forest       |    0.192757 | 0.527157 | 0.282293 |   0.584126 |  0.720323 | 0.197025  |                    0.49 |
|         3 | Temporal Only       | XGBoost             |    0.204261 | 0.520767 | 0.293429 |   0.593781 |  0.739803 | 0.214784  |                    0.58 |
|         3 | Semantic Only       | Majority Class      |    0        | 0        | 0        |   0.480945 |  0.5      | 0.0734225 |                    0.5  |
|         3 | Semantic Only       | Logistic Regression |    0.192308 | 0.399361 | 0.259605 |   0.582665 |  0.629643 | 0.138998  |                    0.6  |
|         3 | Semantic Only       | Random Forest       |    0.262799 | 0.246006 | 0.254125 |   0.598527 |  0.62671  | 0.156297  |                    0.55 |
|         3 | Semantic Only       | XGBoost             |    0.193548 | 0.325879 | 0.242857 |   0.580055 |  0.650233 | 0.163444  |                    0.61 |
|         3 | Semantic + Temporal | Majority Class      |    0        | 0        | 0        |   0.480945 |  0.5      | 0.0734225 |                    0.5  |
|         3 | Semantic + Temporal | Logistic Regression |    0.226212 | 0.402556 | 0.289655 |   0.604467 |  0.751769 | 0.201553  |                    0.6  |
|         3 | Semantic + Temporal | Random Forest       |    0.241055 | 0.408946 | 0.303318 |   0.613387 |  0.715822 | 0.21567   |                    0.5  |
|         3 | Semantic + Temporal | XGBoost             |    0.344    | 0.27476  | 0.305506 |   0.628202 |  0.717387 | 0.241123  |                    0.7  |

## 6. Ablation

Perbandingan t+1 terbaik per feature set pada official validation (test tidak dipakai untuk memilih ablation):

| feature_set         | model               |       f1 |   pr_auc |
|:--------------------|:--------------------|---------:|---------:|
| Semantic + Temporal | XGBoost             | 0.34903  | 0.287742 |
| Semantic Only       | XGBoost             | 0.298387 | 0.192154 |
| Temporal Only       | Logistic Regression | 0.336634 | 0.270886 |

## 7. Multi-horizon dan 8. final test

|   horizon | feature_set         | model   |   precision |   recall |       f1 |   macro_f1 |   roc_auc |   pr_auc |
|----------:|:--------------------|:--------|------------:|---------:|---------:|-----------:|----------:|---------:|
|         1 | Semantic + Temporal | XGBoost |    0.459064 | 0.588015 | 0.515599 |   0.693623 |  0.806281 | 0.538774 |
|         2 | Semantic + Temporal | XGBoost |    0.44469  | 0.501873 | 0.471554 |   0.670667 |  0.764254 | 0.47651  |
|         3 | Semantic + Temporal | XGBoost |    0.469717 | 0.43462  | 0.451488 |   0.664999 |  0.767791 | 0.473998 |

## 9. Distribution shift

Mean article/day berubah dari 125,55 pada training menjadi 244,64 pada test. Perubahan validation ke test untuk model terpilih:

|   horizon |   validation_f1 |   test_f1 |   f1_change |   validation_pr_auc |   test_pr_auc |   pr_auc_change |
|----------:|----------------:|----------:|------------:|--------------------:|--------------:|----------------:|
|         1 |        0.34903  |  0.515599 |    0.166569 |            0.287742 |      0.538774 |        0.251031 |
|         2 |        0.326154 |  0.471554 |    0.1454   |            0.270949 |      0.47651  |        0.205561 |
|         3 |        0.305506 |  0.451488 |    0.145981 |            0.241123 |      0.473998 |        0.232875 |

Perubahan ini dilaporkan sebagai temporal distribution shift; penyebabnya tidak disimpulkan.

## 10. Feature importance

Feature importance menyatakan predictive importance, bukan hubungan sebab-akibat. Tabel lengkap memuat native importance dan permutation importance.

|   horizon | model   | feature_set         | feature                  | feature_group   |   native_importance | native_method   |   permutation_importance_mean_f1 |   permutation_importance_std_f1 |
|----------:|:--------|:--------------------|:-------------------------|:----------------|--------------------:|:----------------|---------------------------------:|--------------------------------:|
|         1 | XGBoost | Semantic + Temporal | rolling_mean_7           | Temporal        |           0.289965  | tree_importance |                      0.0533876   |                     0.00176584  |
|         1 | XGBoost | Semantic + Temporal | current_volume           | Temporal        |           0.0946699 | tree_importance |                      0.021357    |                     0.00308132  |
|         1 | XGBoost | Semantic + Temporal | rolling_mean_3           | Temporal        |           0.0506926 | tree_importance |                      0.0106859   |                     0.00133319  |
|         1 | XGBoost | Semantic + Temporal | semantic_drift           | Semantic        |           0.0201723 | tree_importance |                      0.0105931   |                     0.00277798  |
|         1 | XGBoost | Semantic + Temporal | topic_probability_mean   | Semantic        |           0.0167999 | tree_importance |                      0.00647278  |                     0.0018647   |
|         1 | XGBoost | Semantic + Temporal | lag_1                    | Temporal        |           0.0336227 | tree_importance |                      0.00608776  |                     0.00197271  |
|         1 | XGBoost | Semantic + Temporal | semantic_similarity_std  | Semantic        |           0.201928  | tree_importance |                      0.00527677  |                     0.00216782  |
|         1 | XGBoost | Semantic + Temporal | lag_2                    | Temporal        |           0.0223997 | tree_importance |                      0.00476532  |                     0.000598992 |
|         1 | XGBoost | Semantic + Temporal | similarity_change        | Semantic        |           0.016579  | tree_importance |                      0.00266514  |                     0.00228507  |
|         1 | XGBoost | Semantic + Temporal | lag_4                    | Temporal        |           0.0210711 | tree_importance |                      0.00241939  |                     0.00154423  |
|         1 | XGBoost | Semantic + Temporal | topic_probability_change | Semantic        |           0.0187328 | tree_importance |                      0.0014037   |                     0.00389453  |
|         1 | XGBoost | Semantic + Temporal | rolling_std_7            | Temporal        |           0.0542622 | tree_importance |                      0.00111854  |                     0.00316642  |
|         1 | XGBoost | Semantic + Temporal | topic_probability_std    | Semantic        |           0.0219352 | tree_importance |                      0.000955655 |                     0.00156877  |
|         1 | XGBoost | Semantic + Temporal | semantic_similarity_mean | Semantic        |           0.0214693 | tree_importance |                      0.000516887 |                     0.00289522  |
|         1 | XGBoost | Semantic + Temporal | percentage_change        | Temporal        |           0.0193311 | tree_importance |                      0.000217949 |                     0.00216992  |

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

F1 validation terbaik Semantic Only untuk t+1/t+2/t+3: **0.2984 / 0.2587 / 0.2596**; majority baseline pada horizon yang sama: **0.0000 / 0.0000 / 0.0000**. Semantic Only berada di atas majority baseline pada **3/3** horizon. Angka ini menyatakan predictive signal, bukan klaim kausal.

## 14. RQ2 — Temporal features

F1 validation terbaik Temporal Only untuk t+1/t+2/t+3: **0.3366 / 0.3151 / 0.2981**. Dibanding Semantic Only pada split dan kandidat learner yang sama, Temporal Only lebih tinggi pada **3/3** horizon.

## 15. RQ3 — Fusion

F1 validation terbaik Semantic + Temporal untuk t+1/t+2/t+3: **0.3490 / 0.3262 / 0.3055**. Fusion melampaui Temporal Only pada **3/3** horizon. Feature set yang terpilih per horizon: {1: 'Semantic + Temporal', 2: 'Semantic + Temporal', 3: 'Semantic + Temporal'}. Karena hanya tiga fold, perbedaan antar feature set tidak dinyatakan signifikan.

## 16. RQ4 — Forecast horizon

Pada final test, F1 model terpilih adalah **0.5156** (t+1), **0.4716** (t+2), dan **0.4515** (t+3) — menurun monoton terhadap horizon. PR-AUC adalah **0.5388**, **0.4765**, dan **0.4740** — menurun monoton. Tidak ada horizon yang dicampur saat fitting.

## Status penelitian

**LIMITED** — eksperimen dapat dievaluasi secara temporal dan bebas leakage yang terdeteksi, tetapi distribution shift, truncation, topic outlier, serta jumlah fold membatasi generalisasi hasil.

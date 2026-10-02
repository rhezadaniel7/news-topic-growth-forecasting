# Subset forecastable — laporan

Skema label **tetap beku**: `burst_mad_k2`, `threshold label tetap (k=2) sehingga label TIDAK berubah akibat pembatasan topik`.
Tidak ada definisi label yang diubah, tidak ada threshold yang dipilih demi angka,
dan test dievaluasi sekali di akhir untuk satu konfigurasi yang sudah dibekukan
dari official validation.

## STEP 1 — Diagnosa sumber F1 rendah

F1 test t+1 pada model dan label beku, dipecah per kelompok sparsity yang dihitung
dari **training window saja** (270 hari). Catatan: `sparsity_category` yang tersimpan
di `topic_history_final.csv` dihitung atas seluruh periode termasuk test, sehingga
TIDAK dipakai untuk mengelompokkan maupun menyeleksi di sini.

| sparsity_group   |   topics |   observations |   positives |   prevalence |   precision |   recall |     f1 |   pr_auc |   roc_auc |
|:-----------------|---------:|---------------:|------------:|-------------:|------------:|---------:|-------:|---------:|----------:|
| LOW              |       12 |           1068 |         312 |       0.2921 |      0.4895 |   0.7468 | 0.5914 |   0.6448 |    0.7892 |
| MEDIUM           |       37 |           3293 |         489 |       0.1485 |      0.3904 |   0.5460 | 0.4552 |   0.4882 |    0.8017 |
| HIGH             |      121 |          10769 |         244 |       0.0227 |      0.2869 |   0.2951 | 0.2909 |   0.2884 |    0.8645 |
| ALL              |      170 |          15130 |        1045 |       0.0691 |      0.4054 |   0.5474 | 0.4658 |   0.4936 |    0.8932 |

Spread F1 antar kelompok: **0.3005**. Monoton LOW > MEDIUM > HIGH:
**True**. Hipotesis bahwa F1 rendah terkonsentrasi di topik
high-sparsity: **TERKONFIRMASI**.

Peringatan interpretasi yang harus dibaca bersama tabel di atas: ROC-AUC justru
tertinggi pada kelompok HIGH. Model me-ranking topik high-sparsity dengan baik;
yang kolaps adalah precision karena prevalensi positif di kelompok itu sangat
rendah. Artinya F1 di sini sebagian besar didorong base rate, bukan kualitas
diskriminasi.

## STEP 2 — Kriteria subset, dibekukan dari training saja

```json
{
  "train_zero_ratio_strictly_below": 0.8,
  "train_active_days_min": 30,
  "require_forecast_eligible": true
}
```

Sumber kriteria: SPARSITY_MEDIUM_MAX=0.80 pre-registered di step10_13 dengan pertidaksamaan ketat identik sparsity_category(); 30 = 4x rolling window 7 dan >2x MIN_CALENDAR_HISTORY=14; forecast_eligible = gate MIN_ACTIVE_TRAINING_DAYS=14

Catatan: N=30 non-binding: M<0.80 pada 270 hari training sudah menyiratkan train_active_days>=55

Kriteria dibekukan ke `data/processed/forecastable_subset_v1.json` sebelum validation dan
test subset dilihat. Pipeline menolak berjalan jika berkas itu tidak menandai
dirinya beku dan berbasis training window saja, dan menolak berjalan jika skema
label memakai threshold yang di-fit (karena pembatasan topik akan mengubah label).

Topik lolos: **49 dari 182** (26.9%).
Distribusi kategori training pada subset:
{'MEDIUM': 37, 'LOW': 12}

## STEP 3 — Prosedur run subset

Identik run utama: temporal CV 3 fold expanding untuk memilih hyperparameter dan
feature set, official validation untuk memilih model final dan probability
threshold, refit train+validation, lalu test satu kali. Tiga horizon, tiga feature
set, empat learner. Gate validitas: Temporal Only berjalan di ketiga horizon dan
Majority Class tetap dilaporkan (F1 growth 0,0000 di semua horizon).

## STEP 4 — Perbandingan full-topic-set vs forecastable-subset

| topic_set           |   horizon | model               | feature_set         |   n_test |   positives_test |   test_prevalence |   validation_precision |   validation_recall |   validation_f1 |   validation_pr_auc |   validation_roc_auc |   test_precision |   test_recall |   test_f1 |   test_pr_auc |   test_roc_auc |   test_pr_auc_lift_over_prevalence |
|:--------------------|----------:|:--------------------|:--------------------|---------:|-----------------:|------------------:|-----------------------:|--------------------:|----------------:|--------------------:|---------------------:|-----------------:|--------------:|----------:|--------------:|---------------:|-----------------------------------:|
| full_topic_set      |         1 | XGBoost             | Semantic + Temporal |    15130 |             1045 |            0.0691 |                 0.2751 |              0.3374 |          0.3031 |              0.2224 |               0.8599 |           0.4054 |        0.5474 |    0.4658 |        0.4936 |         0.8932 |                             7.1471 |
| forecastable_subset |         1 | XGBoost             | Semantic + Temporal |     4361 |              801 |            0.1837 |                 0.3127 |              0.3950 |          0.3490 |              0.2877 |               0.7816 |           0.4591 |        0.5880 |    0.5156 |        0.5388 |         0.8063 |                             2.9333 |
| full_topic_set      |         2 | Logistic Regression | Temporal Only       |    14960 |             1044 |            0.0698 |                 0.2643 |              0.2993 |          0.2807 |              0.1892 |               0.8448 |           0.4476 |        0.4914 |    0.4685 |        0.4550 |         0.8830 |                             6.5204 |
| forecastable_subset |         2 | XGBoost             | Semantic + Temporal |     4312 |              801 |            0.1858 |                 0.3174 |              0.3354 |          0.3262 |              0.2709 |               0.7651 |           0.4447 |        0.5019 |    0.4716 |        0.4765 |         0.7643 |                             2.5652 |
| full_topic_set      |         3 | XGBoost             | Temporal Only       |    14790 |             1046 |            0.0707 |                 0.2171 |              0.3535 |          0.2690 |              0.1725 |               0.8319 |           0.3445 |        0.5516 |    0.4241 |        0.4137 |         0.8728 |                             5.8492 |
| forecastable_subset |         3 | XGBoost             | Semantic + Temporal |     4263 |              803 |            0.1884 |                 0.3440 |              0.2748 |          0.3055 |              0.2411 |               0.7174 |           0.4697 |        0.4346 |    0.4515 |        0.4740 |         0.7678 |                             2.5164 |

Selisih:

|   horizon |   delta_validation_f1 |   delta_test_f1 |   delta_test_pr_auc |   delta_test_roc_auc |   delta_test_pr_auc_lift |   test_prevalence_ratio |
|----------:|----------------------:|----------------:|--------------------:|---------------------:|-------------------------:|------------------------:|
|   +1.0000 |               +0.0459 |         +0.0498 |             +0.0451 |              -0.0869 |                  -4.2138 |                 +2.6593 |
|   +2.0000 |               +0.0455 |         +0.0031 |             +0.0215 |              -0.1187 |                  -3.9552 |                 +2.6619 |
|   +3.0000 |               +0.0365 |         +0.0274 |             +0.0603 |              -0.1050 |                  -3.3328 |                 +2.6634 |

### Biaya cakupan

|   horizon | split      |   observations_full |   observations_subset |   observations_dropped |   observations_dropped_percent |   positives_full |   positives_subset |   positives_dropped_percent |   prevalence_full |   prevalence_subset |
|----------:|:-----------|--------------------:|----------------------:|-----------------------:|-------------------------------:|-----------------:|-------------------:|----------------------------:|------------------:|--------------------:|
|         1 | train      |               40295 |                 12344 |                  27951 |                        69.3659 |             1374 |                891 |                     35.1528 |            0.0341 |              0.0722 |
|         1 | validation |               15130 |                  4361 |                  10769 |                        71.1765 |              406 |                319 |                     21.4286 |            0.0268 |              0.0731 |
|         1 | test       |               15130 |                  4361 |                  10769 |                        71.1765 |             1045 |                801 |                     23.3493 |            0.0691 |              0.1837 |
|         2 | train      |               40125 |                 12295 |                  27830 |                        69.3583 |             1376 |                895 |                     34.9564 |            0.0343 |              0.0728 |
|         2 | validation |               14960 |                  4312 |                  10648 |                        71.1765 |              401 |                316 |                     21.1970 |            0.0268 |              0.0733 |
|         2 | test       |               14960 |                  4312 |                  10648 |                        71.1765 |             1044 |                801 |                     23.2759 |            0.0698 |              0.1858 |
|         3 | train      |               39955 |                 12246 |                  27709 |                        69.3505 |             1371 |                896 |                     34.6462 |            0.0343 |              0.0732 |
|         3 | validation |               14790 |                  4263 |                  10527 |                        71.1765 |              396 |                313 |                     20.9596 |            0.0268 |              0.0734 |
|         3 | test       |               14790 |                  4263 |                  10527 |                        71.1765 |             1046 |                803 |                     23.2314 |            0.0707 |              0.1884 |

Subset membuang **133 dari 182 topik**
(73.1%) dan
**71.2% observasi test**
(10769 dari 15130).
Positif test yang dibuang: 23.3%.
Prevalensi test naik dari 0.0691 menjadi
0.1837.

Konsistensi kenaikan F1 test: naik pada **3/3 horizon** — konsisten (>=2 dari 3 horizon).

Horizon dengan kenaikan di bawah 0,01 (praktis datar): t+2.

### Yang tidak boleh disembunyikan dari reviewer

1. **Cakupan turun drastis.** Task menyempit dari 182 menjadi 49 topik;
   71.2% observasi test tidak lagi diprediksi.
   Model subset tidak menjawab pertanyaan yang sama dengan model penuh.
2. **Kenaikan F1 sebagian besar efek prevalensi.** Prevalensi test naik
   2.66x. F1 dan PR-AUC keduanya sensitif base rate.
3. **Diskriminasi justru menurun.** ROC-AUC test turun di ketiga horizon
   (-0.0869, -0.1187, -0.1050), dan PR-AUC
   relatif prevalensi turun tajam
   (-4.21x, -3.96x, -3.33x).
   Pada subset, model lebih dekat ke baseline acaknya sendiri daripada model penuh.
4. Angka lama dan angka baru berasal dari label yang identik. Itu sah hanya karena
   threshold label `burst_mad_k2` tetap; dengan skema kuantil, perbandingan ini batal.

## STEP 5 — Verdict

**(a) F1 naik sah via penyempitan task** — F1 test naik pada 3/3 horizon.

F1 test: 0.4658 / 0.4685 / 0.4241 (full)
menjadi 0.5156 / 0.4716 / 0.4515 (subset)
untuk t+1/t+2/t+3.

Biaya cakupan: 73.1% topik dan
71.2% observasi test dibuang.

Kualifikasi wajib: kenaikan ini adalah penyempitan task yang sah secara prosedur,
BUKAN bukti model menjadi lebih baik. ROC-AUC dan lift atas prevalensi turun di
ketiga horizon. Angka 0.5156
pada t+1 hanya boleh dilaporkan bersama cakupan yang dikorbankan dan bersama
ROC-AUC-nya.

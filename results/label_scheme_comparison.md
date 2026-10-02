# Investigasi dan perbaikan definisi label growth

Ruang lingkup: `build_growth_labels()` di `src/step10_13_features_labels.py` dan
penerapan label di `src/step14_21_modeling.py`. Kontrol leakage tidak diubah;
audit leakage `results/tables/pre_feature_leakage_audit.csv` tetap lolos 11/11.

## 1. Konfirmasi masalah

`growth_score = (future_volume − median historis) / (1 + 1,4826 × MAD historis)`.
Kuantil p80 pada training bernilai **0,0** untuk ketiga horizon, sehingga
`growth_score > p80` identik dengan `growth_score > 0`, yaitu
`future_volume > median historis`.

Penyebabnya bukan sekadar threshold, melainkan median historis itu sendiri.
Distribusi `growth_score` training (t+1) dipisah kelas sparsity topik
(`results/tables/growth_score_summary_by_sparsity.csv`):

| sparsity | topik | observasi | share median = 0 | share MAD = 0 | share skor tepat 0 | p80 |
|:---|---:|---:|---:|---:|---:|---:|
| LOW SPARSITY | 16 | 4.060 | 0,265 | 0,266 | 0,290 | 1,0 |
| MEDIUM SPARSITY | 51 | 12.659 | 0,975 | 0,976 | 0,739 | 1,0 |
| HIGH SPARSITY | 103 | 23.576 | 0,995 | 0,996 | 0,884 | 0,0 |

Untuk 154 dari 170 topik (MEDIUM + HIGH), median dan MAD historis nol pada
hampir semua observasi eligible, sehingga `growth_score` menyusut menjadi
`future_volume` dan label "growth" berarti "ada artikel pada target date".

Histogram bin penuh: `results/tables/growth_score_histogram_by_sparsity.csv`,
figure: `results/figures/growth_score_histogram_by_sparsity.png`. Massa di bin
`== 0` adalah 0,290 / 0,739 / 0,884 untuk LOW / MEDIUM / HIGH.

Isi label positif skema lama membenarkan keluhan secara kuantitatif
(`results/tables/label_scheme_positive_composition.csv`, t+1 training):
**57,4%** observasi positif adalah kasus `future_volume` tepat **satu artikel**,
75,7% adalah ≤2 artikel, 61,7% berasal dari hari yang kemarin tidak aktif, dan
kenaikan median terhadap median historis hanya **1 artikel**.

## 2. Tiga skema alternatif

Definisi terpusat di `src/label_schemes.py`, dipakai bersama oleh audit label dan
modeling sehingga tidak bisa berbeda antar tahap.

| key | definisi | sifat threshold |
|:---|:---|:---|
| `baseline_quantile_p80` | `growth_score > p80 training` | di-fit, jatuh di titik massa nol |
| `active_baseline_p80` (a) | hanya topic-day `lag_1 > 0`; `growth_score > p80 training` pada populasi tersebut | di-fit, populasi dibatasi |
| `positive_only_p80` (b) | `growth_score > p80` dari `growth_score > 0` training | di-fit pada ekor positif |
| `burst_mad_k2` (c) | `future_volume > median + 2 × max(1,4826 × MAD, 1,0)` | tetap, tidak di-fit |

Catatan pada (c): floor skala 1,0 artikel diperlukan karena MAD = 0 pada 91,6%
observasi eligible; tanpa floor, `median + k·MAD` kembali kolaps menjadi
`future > median`. Floor menyatakan konvensi eksplisit bahwa perubahan terkecil
yang bermakna pada volume harian adalah satu artikel, dan ditetapkan sebelum
metrik model dilihat. `k = 2` adalah dua deviasi robust, bukan kuantil.

Kontrol leakage yang dipertahankan: threshold yang di-fit hanya memakai baris
training (audit label) atau training window fold (temporal CV, sesuai aturan yang
sudah berlaku); seleksi populasi (a) hanya memakai `lag_1`, yaitu volume t−1 yang
sudah diketahui pada origin date; `growth_score` dan `burst_score` tetap kolom
sisi label dan ditolak sebagai predictor.

## 3. Class balance train/validation/test

Sumber: `results/tables/label_scheme_class_balance.csv` (lengkap 3 horizon).
Ringkasan t+1:

| skema | threshold | split | observasi | growth | normal | positive rate |
|:---|---:|:---|---:|---:|---:|---:|
| `baseline_quantile_p80` | 0,0 | train | 40.295 | 7.314 | 32.981 | 0,1815 |
| | | validation | 15.130 | 2.579 | 12.551 | 0,1705 |
| | | test | 15.130 | 4.123 | 11.007 | 0,2725 |
| `active_baseline_p80` | 1,0 | train | 8.322 | 1.189 | 7.133 | 0,1429 |
| | | validation | 2.927 | 396 | 2.531 | 0,1353 |
| | | test | 4.373 | 1.232 | 3.141 | 0,2817 |
| `positive_only_p80` | 2,0 | train | 40.295 | 1.181 | 39.114 | 0,0293 |
| | | validation | 15.130 | 329 | 14.801 | 0,0217 |
| | | test | 15.130 | 909 | 14.221 | 0,0601 |
| `burst_mad_k2` | 2,0 | train | 40.295 | 1.374 | 38.921 | 0,0341 |
| | | validation | 15.130 | 406 | 14.724 | 0,0269 |
| | | test | 15.130 | 1.045 | 14.085 | 0,0691 |

Keempat skema menghasilkan dua kelas pada semua split dan horizon; horizon 2 dan
3 hampir identik dengan horizon 1.

## 4. Isi substantif label positif

`results/tables/label_scheme_positive_composition.csv`, t+1:

| skema | split | positif | dari hari tidak aktif | `future = 1` artikel | `future ≤ 2` | median `future` | median kenaikan atas median historis |
|:---|:---|---:|---:|---:|---:|---:|---:|
| `baseline_quantile_p80` | train | 7.314 | 0,617 | **0,574** | 0,757 | 1 | 1 |
| | test | 4.123 | 0,444 | **0,504** | 0,706 | 1 | 1 |
| `active_baseline_p80` | train | 1.189 | 0,000 | 0,000 | 0,317 | 4 | 3 |
| | test | 1.232 | 0,000 | 0,000 | 0,323 | 4 | 3 |
| `positive_only_p80` | train | 1.181 | 0,461 | 0,000 | 0,000 | 4 | 4 |
| | test | 909 | 0,234 | 0,000 | 0,000 | 4 | 4 |
| `burst_mad_k2` | train | 1.374 | 0,433 | 0,000 | 0,000 | 5 | 4 |
| | test | 1.045 | 0,219 | 0,000 | 0,000 | 5 | 4 |

Ketiga skema alternatif menghilangkan kasus "besok ada satu artikel" dari kelas
positif sepenuhnya.

## 5. Hasil model — dilaporkan apa adanya

Satu run `run_step14_21.py` penuh per skema (3 horizon × 3 feature set × 4 model
× 3 fold, official validation, lalu test untuk konfigurasi terpilih).
Sumber: `results/tables/label_scheme_model_comparison.csv` dan
`..._summary.csv`. Rata-rata tiga horizon:

| skema | positive rate train / val / test | mean validation F1 | mean test F1 | mean test PR-AUC | PR-AUC ÷ prevalence test |
|:---|:---|---:|---:|---:|---:|
| `baseline_quantile_p80` | 0,182 / 0,170 / 0,274 | 0,4311 | **0,5895** | **0,6193** | 2,26 |
| `active_baseline_p80` | 0,137 / 0,130 / 0,276 | 0,3893 | 0,5461 | 0,5725 | 2,07 |
| `positive_only_p80` | 0,030 / 0,022 / 0,061 | 0,2197 | 0,3764 | 0,3745 | 6,16 |
| `burst_mad_k2` | 0,034 / 0,027 / 0,070 | 0,2843 | 0,4528 | 0,4541 | **6,51** |

Per horizon:

| skema | h | model terpilih | feature set | validation F1 | test precision | test recall | test F1 | test PR-AUC |
|:---|--:|:---|:---|---:|---:|---:|---:|---:|
| `baseline_quantile_p80` | 1 | XGBoost | Temporal Only | 0,4436 | 0,4913 | 0,7667 | 0,5988 | 0,6362 |
| `baseline_quantile_p80` | 2 | Random Forest | Semantic + Temporal | 0,4306 | 0,5083 | 0,7100 | 0,5925 | 0,6144 |
| `baseline_quantile_p80` | 3 | Random Forest | Semantic + Temporal | 0,4189 | 0,4621 | 0,7687 | 0,5772 | 0,6074 |
| `active_baseline_p80` | 1 | XGBoost | Semantic + Temporal | 0,4112 | 0,4436 | 0,7541 | 0,5586 | 0,6089 |
| `active_baseline_p80` | 2 | Random Forest | Semantic + Temporal | 0,3860 | 0,4696 | 0,6100 | 0,5307 | 0,5545 |
| `active_baseline_p80` | 3 | Random Forest | Semantic + Temporal | 0,3706 | 0,4507 | 0,7023 | 0,5491 | 0,5541 |
| `positive_only_p80` | 1 | XGBoost | Semantic + Temporal | 0,2394 | 0,3183 | 0,5204 | 0,3950 | 0,4134 |
| `positive_only_p80` | 2 | XGBoost | Semantic + Temporal | 0,2257 | 0,2671 | 0,5747 | 0,3647 | 0,3594 |
| `positive_only_p80` | 3 | Logistic Regression | Temporal Only | 0,1941 | 0,2764 | 0,5570 | 0,3695 | 0,3506 |
| `burst_mad_k2` | 1 | XGBoost | Semantic + Temporal | 0,3031 | 0,4054 | 0,5474 | 0,4658 | 0,4936 |
| `burst_mad_k2` | 2 | Logistic Regression | Temporal Only | 0,2807 | 0,4476 | 0,4914 | 0,4685 | 0,4550 |
| `burst_mad_k2` | 3 | XGBoost | Temporal Only | 0,2690 | 0,3445 | 0,5516 | 0,4241 | 0,4137 |

Majority-class baseline memberi F1 growth 0,0000 pada semua horizon dan semua
skema, sehingga keempat skema berada di atas baseline trivial.

F1 tertinggi dipegang skema lama. Angka itu tidak dipakai sebagai alasan memilih:
targetnya 5× lebih sering dan secara substansi adalah "apakah topik akan
disebut besok", bukan pertumbuhan. Kolom PR-AUC ÷ prevalence adalah rasio bergaya
lift, bukan metrik standar, dan disertakan hanya untuk menunjukkan bahwa
penurunan F1 absolut sejalan dengan target yang jauh lebih jarang, bukan dengan
hilangnya signal.

## 6. Rekomendasi

**Pakai `burst_mad_k2` sebagai definisi label utama.** Dasarnya validitas
definisi, bukan metrik:

1. Menghapus degenerasi pada sumbernya. Tidak ada kuantil yang di-fit, sehingga
   threshold tidak mungkin jatuh di titik massa distribusi zero-inflated. Skema
   (b) hanya menggeser kuantil ke ekor positif; titik massa nol masih menentukan
   di mana ekor itu dimulai.
2. Arti label tetap sama antar split dan horizon. Ini penting karena volume
   artikel/hari naik dari 125,6 (train) ke 244,6 (test). Threshold yang di-fit
   dari training berarti satu hal pada training dan hal lain pada test; threshold
   tetap tidak punya masalah itu. Prevalensi tetap bergeser (0,034 → 0,070),
   tetapi itu adalah pergeseran nyata pada data, bukan pada definisi.
3. Kelas positif memang burst: nol kasus satu artikel, median `future_volume` 5,
   median kenaikan 4 artikel di atas median historis.
4. Mempertahankan seluruh populasi eligible (40.295 baris training), sehingga
   pertanyaan penelitian tetap berlaku untuk semua topik, bukan hanya 20,6%
   topic-day yang kemarin aktif seperti pada skema (a).
5. Satu-satunya konvensi tambahan — floor skala 1 artikel untuk MAD = 0 —
   dinyatakan eksplisit dan dapat ditinjau, bukan artefak yang tersembunyi.

Skema (a) adalah pembatasan **populasi** terbaik, bukan definisi threshold
terbaik: label positifnya nol persen berasal dari hari tidak aktif, tetapi
thresholdnya masih kuantil yang di-fit dan mendarat di 1,0, sehingga untuk 69,1%
positif yang bermedian nol aturannya tetap "besok ≥2 artikel". Skema (a) dan (c)
bersifat saling melengkapi. Sebagai catatan sensitivitas (belum dimodelkan),
`burst_score > 2` dibatasi pada `lag_1 > 0` memberi positive rate t+1
0,0936 / 0,0817 / 0,1866 pada train/validation/test dari 8.322 baris training —
kandidat penyempurnaan berikutnya jika pertanyaan penelitian dipersempit menjadi
pertumbuhan topik yang sudah aktif.

Skema (b) tidak direkomendasikan: populasi referensi kuantilnya dipilih
berdasarkan outcome, prevalensinya secara mekanis menjadi ±20% dari observasi
berskor positif, dan definisinya tetap ikut bergeser bersama distribusi.

`baseline_quantile_p80` tidak boleh dipakai untuk klaim tentang pertumbuhan
topik. Hasilnya tetap disimpan di `results/scheme_runs/baseline_quantile_p80/`
sebagai pembanding.

## 7. Catatan reproduksi

Run pembanding dijalankan pada lingkungan Python 3.14 dengan pandas 3.0.1,
scikit-learn 1.8.0, dan xgboost 3.4.1 — lebih baru daripada pin di
`requirements.txt` yang dipakai run 21 September. Label untuk
`baseline_quantile_p80` bereproduksi tepat (baris dan jumlah growth/normal
identik pada sembilan partisi), tetapi pilihan model pada dua horizon berbeda
dari `results/frozen_model_selection_legacy_baseline_quantile_p80.json` karena
selisih F1 antar kandidat kecil dan versi library berbeda. Keempat skema
dijalankan pada lingkungan yang sama, sehingga perbandingan antar skema
konsisten secara internal.

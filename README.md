# Prediksi Pertumbuhan Topik Berita Daring Indonesia

Pipeline penelitian end-to-end untuk menguji apakah fusi fitur semantik dan temporal meningkatkan prediksi pertumbuhan topik dibandingkan setiap kelompok fitur secara terpisah. Pipeline tidak melakukan web scraping dan tidak membuat data sintetis.

## Pemetaan bidang

- **Web Mining:** dataset berita daring lokal, URL, source, dan timestamp.
- **Text Mining:** cleaning ringan, embedding multilingual, BERTopic, dan representasi semantik.
- **Data Mining:** fitur temporal/semantik, label robust berbasis median–MAD, klasifikasi, validasi temporal, ablation, evaluasi, serta explainability.

## Struktur

`data/raw` menyimpan input; `data/processed` dan `data/intermediate` menyimpan artefak turunan; `models` menyimpan BERTopic dan classifier; `results` menyimpan tabel, metrik, prediksi, gambar, log eksperimen, dan laporan akhir; `src` berisi modul pipeline.

## Instalasi

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Alternatif Conda: `conda env create -f environment.yml`.

## Dataset dan pemetaan kolom

Taruh tepat satu file CSV, JSON/JSONL, atau Parquet di `data/raw/`. Program menginspeksi nama kolom dan memetakan kandidat judul, isi, tanggal, source, URL, dan ID. Kolom judul, isi, dan tanggal wajib ada. Jika ambigu atau hilang, proses berhenti dengan error; pipeline tidak menebak nilai.

## Menjalankan

Untuk dataset utama `data/raw/final_merge_dataset.csv`, jalankan hanya STEP 1–6
(inspeksi, parsing timestamp, deduplikasi, audit kualitas, audit kepadatan temporal,
dan usulan split) dengan:

```bash
python run_phase1_data_preparation.py
```

Perintah ini menulis `data/processed/news_cleaned.parquet`, tabel audit di
`results/tables/`, grafik `results/figures/daily_article_volume.png`, dan
`results/phase_1_6_report.md`. Perintah tersebut tidak menjalankan embedding,
BERTopic, atau machine learning.

Untuk STEP 7, jalankan embedding dengan cache berbasis hash:

```bash
python run_step7_embeddings.py
```

Untuk STEP 8–9, fit tiga kandidat BERTopic pada training saja:

```bash
python run_step8_9_candidates.py
```

Setelah meninjau metrik training dan interpretabilitas, finalisasi kandidat
terpilih. Eksekusi aktual memilih konfigurasi B dan berstatus `LIMITED`:

```bash
python run_finalize_topic_model.py --config B --status LIMITED --reason "alasan pemilihan berbasis training"
```

Perintah final hanya mentransformasi validation/test ke ruang topik training.
Perintah ini tidak membuat label pertumbuhan atau menjalankan classifier.

Untuk STEP 10–13 (topic time series, semantic features, temporal features,
dan audit label pertumbuhan), jalankan:

```bash
python run_step10_13.py
```

Runner memakai embedding dan topic assignment yang sudah ada. Output fitur penuh
disimpan sebagai Parquet, sedangkan preview dan seluruh audit disimpan di
`results/tables/`. Tahap ini tidak melatih classifier.

Untuk STEP 14–21 (final feature dataset, expanding-window validation, empat
learner, tiga ablation feature set, tiga forecast horizon, final test, feature
importance, dan laporan modeling), jalankan:

```bash
python run_step14_21.py
```

Pemilihan hyperparameter, feature set, model, growth threshold, dan probability
threshold hanya menggunakan training/validation. Pilihan final disimpan di
`results/frozen_model_selection.json` sebelum test dievaluasi. Runner menolak
melanjutkan jika pilihan yang sudah dibekukan berubah atau dibuat untuk skema
label berbeda. Test hanya dievaluasi untuk satu konfigurasi terpilih pada setiap
horizon; ablation dilaporkan dari official validation.

### Varian embedding dan truncation

Encoder lama (`paraphrase-multilingual-MiniLM-L12-v2`) membatasi input pada 128
token, sedangkan mean panjang dokumen 536 token, sehingga 99,41% dokumen terkena
right truncation dan hanya sekitar 24% token yang benar-benar masuk encoder.
Varian encoding didefinisikan di `src/step7_embeddings.py`:

| key | encoder | batas token | strategi |
|:----|:--------|------------:|:---------|
| `minilm_trunc128` | MiniLM-L12-v2 | 128 | kontrol, right truncation |
| `minilm_chunk128` | MiniLM-L12-v2 | 128 | chunk 126 token + overlap 16, mean-pool antar chunk |
| `e5small_trunc512` | multilingual-e5-small | 512 | encoder konteks-panjang, right truncation |
| `e5small_chunk512` | multilingual-e5-small | 512 | chunk 510 token + overlap 32, mean-pool antar chunk |

Cache embedding divalidasi dengan `embedding_fingerprint`: SHA-256 atas nama
model, batas token, konfigurasi chunking, pooling, prefix teks, dan cara dokumen
dibentuk. Sebelumnya cache hanya dicek dengan dataset hash + nama model + jumlah
baris, sehingga mengganti batas token atau mengaktifkan chunking pada model yang
sama akan menghasilkan cache hit yang salah dan tetap memakai embedding lama.
Setiap varian juga menulis ke berkas sendiri (`embeddings_<varian>__<subset>.npy`),
sehingga `embeddings.npy` dari run sebelumnya tidak pernah ditimpa.

torch tidak tersedia untuk Python 3.14 pada macOS x86_64, sehingga encoding
memakai interpreter terpisah:

```bash
python -m venv .venv-encoder                       # Python 3.11
.venv-encoder/bin/pip install "numpy<2" "torch==2.2.2" "sentence-transformers==3.0.1" pandas pyarrow psutil

python -m src.pilot_subset                         # pilih subset pilot (deterministik)
.venv-encoder/bin/python run_pilot_embeddings.py minilm_trunc128
.venv-encoder/bin/python run_pilot_embeddings.py minilm_chunk128
.venv-encoder/bin/python run_pilot_embeddings.py e5small_trunc512
.venv-encoder/bin/python run_pilot_embeddings.py e5small_chunk512
```

Lalu hitung ulang fitur semantik dan jalankan ablation per varian. Topic
assignment, `topic_daily_volume`, dan label growth tetap dikunci dari run
sebelumnya, sehingga perbedaan hasil hanya berasal dari kualitas embedding:

```bash
for v in legacy_minilm_trunc128 minilm_trunc128 minilm_chunk128 e5small_trunc512 e5small_chunk512; do
  python run_semantic_variant.py $v
  EMBEDDING_VARIANT=$v LABEL_SCHEME=burst_mad_k2 python run_step14_21.py
done
python run_embedding_variant_comparison.py
```

Output per varian ada di `results/embedding_runs/<varian>__<subset>/`. Karena
`Temporal Only` tidak memakai embedding, metriknya wajib identik di semua varian
dan dipakai sebagai kontrol internal (`embedding_variant_temporal_control.csv`).

### Skema label growth

Definisi label berada di `src/label_schemes.py` dan dipakai bersama oleh audit
label (STEP 10–13) dan modeling (STEP 14–21). Skema aktif dipilih lewat variabel
lingkungan `LABEL_SCHEME` atau argumen pertama runner; default adalah
`burst_mad_k2`.

| key | definisi |
|:----|:---------|
| `baseline_quantile_p80` | `growth_score > kuantil p80 training`; kuantil ini bernilai 0,0 sehingga label kolaps menjadi "ada artikel pada target date" |
| `active_baseline_p80` | sama, tetapi hanya pada topic-day dengan `lag_1 > 0` |
| `positive_only_p80` | kuantil p80 di-fit hanya pada `growth_score > 0` training |
| `burst_mad_k2` | `future_volume > median historis + 2 × max(1,4826 × MAD historis, 1,0)`, threshold tetap |

Untuk membandingkan skema, jalankan modeling per skema lalu kumpulkan hasilnya:

```bash
LABEL_SCHEME=baseline_quantile_p80 python run_step14_21.py
LABEL_SCHEME=active_baseline_p80   python run_step14_21.py
LABEL_SCHEME=positive_only_p80     python run_step14_21.py
LABEL_SCHEME=burst_mad_k2          python run_step14_21.py
python run_label_scheme_comparison.py
```

Skema default menulis ke lokasi output canonical; skema lain menulis ke
`results/scheme_runs/<key>/`. Perbandingan lengkap ada di
`results/label_scheme_comparison.md`.

Audit semua kandidat terlebih dahulu. Perintah ini hanya memindai `data/raw`, menulis
`results/tables/dataset_audit.csv` dan `results/dataset_audit_report.md`, menilai kelayakan, lalu berhenti tanpa menjalankan
cleaning, embedding, BERTopic, atau machine learning:

```bash
python audit_datasets.py
```

Eksperimen final:

```bash
python run_pipeline.py --mode full
```

Debug cepat (bukan hasil jurnal):

```bash
python run_pipeline.py --mode sample
```

Gunakan `--force-embeddings` hanya jika cache embedding perlu dibuat ulang. Cache divalidasi memakai SHA-256 cleaned dataset, nama model, dan jumlah baris.

## Strategi anti-leakage

BERTopic dan ruang topik di-fit hanya pada 60% periode paling awal; validation dan test ditransformasi ke ruang tersebut. Centroid semantik global memakai dokumen training. Rolling temporal memakai `shift(1)`. Threshold label yang di-fit hanya dipelajari dari origin training (skema `burst_mad_k2` tidak melakukan fitting sama sekali); seleksi populasi `active_baseline` hanya memakai `lag_1`, yang sudah diketahui pada origin date. Split bersifat temporal 60/20/20, bukan acak. Audit gagal secara keras jika syarat utama leakage dilanggar.

## Output dan interpretasi

- `results/tables/`: ringkasan dataset/topik, hasil model, ablation, horizon, importance.
- `results/predictions/test_predictions.csv`: prediksi test t+1 dari model yang dipilih berdasarkan validation.
- `results/figures/`: 12 visualisasi wajib.
- `results/experiment_log.csv`: parameter, periode, dataset hash, dan metrik setiap run.
- `results/final_report.md`: ringkasan empiris untuk penulisan jurnal.

Fokus interpretasi adalah F1 kelas positif dan PR-AUC. Feature importance disebut kontribusi prediktif, bukan hubungan kausal. Jika data/horizon tidak memadai, output menyatakan `not evaluated`.

## Reproducibility

Parameter utama terpusat di `src/config.py`, termasuk seed, model embedding, horizon, rasio split, BERTopic, dan model klasifikasi. Input dilacak dengan SHA-256. Mode `sample` tidak boleh dipakai sebagai hasil final jurnal. 

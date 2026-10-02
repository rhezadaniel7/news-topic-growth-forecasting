# Topic quality report

## Ruang lingkup

BERTopic di-fit hanya pada 33.899 dokumen training (2024-07-31 sampai 2025-04-26). Validation dan test hanya dipetakan dengan `transform()` menggunakan UMAP, HDBSCAN, vectorizer, dan c-TF-IDF yang sudah di-fit pada training. Tidak ada label pertumbuhan atau classifier yang dibuat.

## Embedding

- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Shape: 67,160 × 384
- Normalisasi L2: True
- Batch size final: 64
- Batas input model: 128 token
- Dokumen terkena right truncation: 66,764 (99.41%)
- Judul yang melewati batas token: 0

Judul ditempatkan di awal input dan seluruh judul muat dalam batas model. Karena 99% lebih dokumen melebihi batas 128 token, embedding merepresentasikan judul dan bagian awal content, bukan seluruh isi artikel. Ini merupakan keterbatasan utama representasi semantik.

## Perbandingan konfigurasi training

| configuration   |   number_of_topics |   outlier_count |   outlier_ratio |   topic_diversity |   coherence_npmi |   silhouette |   silhouette_sample_size |   largest_topic_size |   largest_topic_ratio_non_outlier |   median_topic_size |   smallest_topic_size |   fit_seconds |   min_topic_size |   min_cluster_size |   peak_process_rss_bytes |
|:----------------|-------------------:|----------------:|----------------:|------------------:|-----------------:|-------------:|-------------------------:|---------------------:|----------------------------------:|--------------------:|----------------------:|--------------:|-----------------:|-------------------:|-------------------------:|
| A               |                325 |           14278 |        0.421192 |          0.744923 |         0.464703 |     0.612854 |                     5000 |                  552 |                         0.0281331 |                36   |                    15 |       25.1064 |               15 |                 15 |               3103514624 |
| B               |                182 |           13356 |        0.393994 |          0.784066 |         0.447403 |     0.592767 |                     5000 |                 1336 |                         0.0650343 |                57.5 |                    25 |       23.7292 |               25 |                 25 |               3095891968 |
| C               |                 81 |           10406 |        0.306971 |          0.853086 |         0.406665 |     0.405881 |                     5000 |                 6118 |                         0.260418  |               123   |                    51 |       20.8131 |               50 |                 50 |               3056267264 |

Coherence menggunakan mean NPMI pasangan top-10 kata pada seluruh corpus training. Pasangan tanpa ko-okurensi memakai smoothing 1/N. Silhouette dihitung pada ruang UMAP training lima dimensi, dokumen outlier dikeluarkan, dengan sample reproducible maksimum 5.000 dokumen (`random_state=42`).

## Konfigurasi terpilih

- Konfigurasi: **B**
- Alasan: Konfigurasi B dipilih hanya dari training karena memberi kompromi terbaik: 182 topik, largest-topic share 6,50%, NPMI 0,4474, silhouette 0,5928, diversity 0,7841, dan outlier 39,40%. A lebih terfragmentasi (325 topik; median size 36; outlier 42,12%), sedangkan C membentuk topik dominan 26,04% dan silhouette 0,4059.
- Jumlah topik: 182
- Training outlier: 39.40%
- Validation outlier: 48.08%
- Test outlier: 47.53%
- Topic diversity: 0.7841
- Coherence NPMI: 0.4474
- Silhouette: 0.5928 (sample n=5000)
- Topik terbesar: 1336 dokumen atau 6.50% dari dokumen training non-outlier
- Waktu transform validation: 0.59 detik
- Waktu transform test: 1.15 detik

## Audit distribusi dan interpretabilitas

- Topik dengan history training <14 hari aktif: 12/182. Topik hanya ditandai dan tidak dibuang.
- `topic_summary.csv` menyimpan top words dan tiga representative documents untuk setiap topik.
- Topic diversity mengukur overlap kata topik; nilai tinggi berarti lebih sedikit kata yang berulang antartopik.
- Outlier validation/test mencerminkan dokumen future yang tidak cocok dengan ruang topik training. Tidak ada topik future baru yang di-fit.
- Median volume harian pada `topic_history.csv` memasukkan hari bernilai nol; median khusus hari aktif tersedia di kolom terpisah.

## Konsistensi temporal

Validation mencakup 11,243 dokumen (2025-04-27 sampai 2025-07-25); test mencakup 22,018 dokumen (2025-07-26 sampai 2025-10-23). Lonjakan volume sumber pada September 2025 tetap merupakan potensi distribution shift dan tidak dipakai untuk memilih konfigurasi.

## Keputusan

**TOPIC MODEL STATUS: LIMITED**

Keputusan didasarkan pada metrik training, pemeriksaan top words/representative documents, distribusi ukuran topik, dan kemampuan model training menerima dokumen validation/test. Prediction belum dijalankan.

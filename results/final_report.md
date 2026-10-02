# Laporan Akhir Penelitian

## 1. Dataset
Dataset nyata lokal berisi 455 baris dan 8 kolom sebelum cleaning, dengan rentang 2026-07-27 02:54:30+00:00 hingga 2026-09-20 09:45:00+00:00.

## 2. Preprocessing
Data setelah cleaning: 447 baris; duplikat URL dibuang: 0; duplikat judul dibuang: 8.

## 3. Topic modeling
BERTopic di-fit hanya pada dokumen sebelum cutoff training 2026-08-30 00:00:00; dokumen validation/test hanya ditransformasi. Model embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

## 4. Topic quality
|   number_of_topics |   outlier_ratio |   topic_diversity |   silhouette_score | coherence_score   | coherence_status                                             | warning                                                         |
|-------------------:|----------------:|------------------:|-------------------:|:------------------|:-------------------------------------------------------------|:----------------------------------------------------------------|
|                  4 |        0.760626 |               0.9 |          0.0654604 |                   | not evaluated: gensim-compatible token corpus not configured | Outlier ratio melebihi 50%; cakupan topic time series terbatas. |

## 5. Feature engineering
Fitur semantik memakai centroid topik yang dibentuk dari training dan sinyal pada origin t. Fitur temporal rolling memakai `shift(1)` sehingga tidak membaca periode masa depan.

## 6. Label distribution
Label memakai skor robust `(volume_future - median_history)/(1 + 1.4826*MAD)`; threshold kuantil training per horizon: {1: 0.0, 2: 0.0, 3: 0.0}.

## 7. Temporal split
Train: sebelum 2026-08-30 00:00:00; validation: [2026-08-30 00:00:00, 2026-09-13 00:00:00); test: mulai 2026-09-13 00:00:00.

## 8. Model performance
| model   |   forecast_horizon |
|:--------|-------------------:|
| all     |                  1 |
| all     |                  2 |
| all     |                  3 |

## 9. Ablation
| status                                       |
|:---------------------------------------------|
| not evaluated: insufficient temporal history |

## 10. Multi-horizon result
|   forecast_horizon | status                                                                                                  |
|-------------------:|:--------------------------------------------------------------------------------------------------------|
|                  1 | not evaluated: validation/test split must contain two classes (validation={0: 56}, test={0: 22, 1: 10}) |
|                  2 | not evaluated: validation/test split must contain two classes (validation={0: 56}, test={0: 22, 1: 10}) |
|                  3 | not evaluated: validation/test split must contain two classes (validation={0: 56}, test={0: 22, 1: 10}) |

## 11. Feature importance
Permutation importance mengukur predictive contribution dan tidak membuktikan sebab-akibat.
Tidak tersedia.

## 12. Leakage audit
| check                             | passed   | detail                                                                                                                                                                                             |
|:----------------------------------|:---------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| target_not_in_features            | True     | []                                                                                                                                                                                                 |
| feature_timestamp_not_future      | True     | feature_timestamp <= origin date                                                                                                                                                                   |
| target_after_origin               | True     | target_date > date                                                                                                                                                                                 |
| rolling_windows_shifted           | True     | rolling statistics use shift(1)                                                                                                                                                                    |
| threshold_train_only              | True     | latest origin=2026-08-28 00:00:00; latest target=2026-08-29 00:00:00; cutoff=2026-08-30 00:00:00                                                                                                   |
| topic_model_train_only            | True     | {'fit_scope': 'train_only', 'train_cutoff': '2026-08-30 00:00:00', 'training_documents': 121, 'min_topic_size': 8, 'n_neighbors': 15, 'n_components': 5, 'min_cluster_size': 8, 'top_n_words': 10} |
| no_duplicate_url                  | True     | 0                                                                                                                                                                                                  |
| no_duplicate_article_across_split | True     | cleaned article keys unique                                                                                                                                                                        |

## 13. Statistical comparison
Statistical significance testing was not performed because the number of independent evaluation origins was insufficient.

## 14. Pemetaan bidang
- **Web Mining:** dataset berita daring, URL, source, dan timestamp lokal yang sudah tersedia; pipeline ini tidak melakukan scraping.
- **Text Mining:** cleaning ringan, sentence embedding multilingual, BERTopic, dan fitur semantik.
- **Data Mining:** rekayasa fitur, target pertumbuhan, validasi temporal, klasifikasi, ablation, dan evaluasi.

## 15. Limitations
Korpus berukuran terbatas dan mencakup rentang waktu pendek. Outlier BERTopic mencapai 76.06%, sehingga cakupan topic time series terbatas. Coherence berbasis korpus eksternal tidak dihitung; kualitas topik dinilai melalui diversity, silhouette, jumlah topik, dan outlier ratio. Hasil tidak boleh digeneralisasi melampaui sumber serta periode observasi.

## 16. Kesimpulan empiris
Data tidak cukup untuk kesimpulan empiris.

# Laporan STEP 1–6: inspeksi dan audit temporal

## Ringkasan hasil

- Baris mentah: 80,472
- Baris final setelah deduplikasi: 67,160
- Total baris dihapus sebagai duplikat: 13,312
- Rentang tanggal: 2024-07-31 sampai 2025-10-23
- Tanggal aktif: 450/450 (100.00%)
- Volume harian: median 133.00, mean 149.24, minimum 67, maksimum 414, SD 70.88
- Timestamp berhasil diparse: 80,472/80,472 (100.0000%)
- Kesiapan masuk BERTopic: **READY**

Jam tidak dibuat untuk timestamp date-only. `parsed_datetime` hanya berisi nilai untuk record yang memiliki jam; `date` berisi tanggal untuk seluruh record yang berhasil diparse. Timestamp berlabel WIB disimpan dalam zona waktu Asia/Jakarta.

## Deduplikasi

| stage                      |   rows_before |   rows_removed |   rows_after |
|:---------------------------|--------------:|---------------:|-------------:|
| raw                        |         80472 |              0 |        80472 |
| exact_url_duplicate        |         80472 |          13091 |        67381 |
| exact_title_duplicate      |         67381 |            213 |        67168 |
| normalized_title_duplicate |         67168 |              8 |        67160 |

Aturan `title normalized` bersifat konservatif: Unicode NFKC, decode HTML entity, case-fold, trim, dan collapse whitespace. Tanda baca tidak dihapus. Pada duplikat, artikel dengan tanggal publikasi paling awal dipertahankan; urutan asli memecahkan tie.

## Distribusi sumber setelah deduplikasi

| source   |   articles |   percentage |   active_dates |
|:---------|-----------:|-------------:|---------------:|
| kompas   |      35268 |     52.5134  |            450 |
| tempo    |      25374 |     37.7814  |            441 |
| detik    |       6518 |      9.70518 |            449 |

## Audit kualitas

| metric                      |     count |   percentage | definition                                              |
|:----------------------------|----------:|-------------:|:--------------------------------------------------------|
| missing_title               |     0     |     0        | Judul kosong setelah trim                               |
| missing_content             |     3     |     0.004467 | Content kosong setelah trim                             |
| missing_timestamp           |     0     |     0        | Tanggal gagal diparse atau kosong                       |
| missing_source              |     0     |     0        | Source kosong setelah trim                              |
| duplicate_article           |     0     |     0        | Duplikat URL/title/title-normalized setelah deduplikasi |
| malformed_url               |     0     |     0        | URL bukan HTTP/HTTPS valid dengan hostname              |
| text_too_short              |    13     |     0.019357 | Content < 200 karakter; hanya flag, tidak dihapus       |
| date_outside_declared_range |     0     |     0        | Tanggal di luar 2024-07-31 sampai 2025-10-23            |
| unknown_source              |     0     |     0        | Source bukan Kompas/Tempo/Detik                         |
| content_characters_min      |     0     |   nan        | minimum                                                 |
| content_characters_p25      |  1744     |   nan        | 25th percentile                                         |
| content_characters_median   |  2247     |   nan        | median                                                  |
| content_characters_mean     |  2402.57  |   nan        | mean                                                    |
| content_characters_p75      |  2822     |   nan        | 75th percentile                                         |
| content_characters_max      | 37279     |   nan        | maximum                                                 |
| content_tokens_regex_min    |     0     |   nan        | minimum                                                 |
| content_tokens_regex_p25    |   249     |   nan        | 25th percentile                                         |
| content_tokens_regex_median |   319     |   nan        | median                                                  |
| content_tokens_regex_mean   |   340.454 |   nan        | mean                                                    |
| content_tokens_regex_p75    |   399     |   nan        | 75th percentile                                         |
| content_tokens_regex_max    |  6258     |   nan        | maximum                                                 |

`text_too_short` menggunakan ambang audit Content < 200 karakter. Record hanya ditandai dan tidak dihapus. Token dihitung dengan regex kata Unicode, bukan tokenizer model embedding.

## Audit kepadatan temporal

| metric                  | value              |
|:------------------------|:-------------------|
| date_start              | 2024-07-31         |
| date_end                | 2025-10-23         |
| calendar_days           | 450                |
| active_dates            | 450                |
| zero_volume_dates       | 0                  |
| active_dates_percent    | 100.0              |
| median_daily_volume     | 133.0              |
| mean_daily_volume       | 149.24444444444444 |
| std_daily_volume        | 70.87878405560222  |
| min_daily_volume        | 67                 |
| max_daily_volume        | 414                |
| kompas_coverage_dates   | 450                |
| kompas_coverage_percent | 100.0              |
| tempo_coverage_dates    | 441                |
| tempo_coverage_percent  | 98.0               |
| detik_coverage_dates    | 449                |
| detik_coverage_percent  | 99.777778          |

## Usulan split temporal

| split      | start_date   | end_date   |   active_dates |   percentage_of_active_dates |   rows |   percentage_of_rows |
|:-----------|:-------------|:-----------|---------------:|-----------------------------:|-------:|---------------------:|
| train      | 2024-07-31   | 2025-04-26 |            270 |                           60 |  33899 |              50.475  |
| validation | 2025-04-27   | 2025-07-25 |             90 |                           20 |  11243 |              16.7406 |
| test       | 2025-07-26   | 2025-10-23 |             90 |                           20 |  22018 |              32.7844 |

Split dibuat dari urutan tanggal aktif aktual dengan rasio 60%/20%/20%. Tidak ada hasil model atau label yang digunakan untuk memilih cutoff. Usulan ini harus dibekukan sebelum tuning model; test tetap tidak disentuh.

## Catatan nonstasioneritas

Kenaikan mean volume bulanan terbesar terjadi dari 2025-08 (125.52 artikel/hari) ke 2025-09 (320.40 artikel/hari), atau 2.55×. Audit ini tidak menyimpulkan penyebabnya. Perubahan tersebut perlu diperlakukan sebagai potensi perubahan cakupan sumber/regime pengumpulan pada tahap forecasting, terutama karena periode test mencakup lonjakan itu.

## Keputusan

**DATA READY FOR BERTOPIC: READY.** Tahap embedding dan BERTopic belum dijalankan.

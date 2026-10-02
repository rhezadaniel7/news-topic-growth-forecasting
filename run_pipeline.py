from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.ablation import FEATURE_SETS, run_ablation
from src.config import (EMBEDDING_MODEL, FORECAST_HORIZONS, MODEL_DIR, RANDOM_SEED, RESULTS_DIR,
                        ROOT, TEST_RATIO, TOPIC_PARAMETERS, VALIDATION_RATIO,
                        ensure_directories)
from src.data_loader import discover_dataset, inspect_dataset, load_dataset, map_columns
from src.data_validation import audit_leakage, validate_raw
from src.dataset_builder import SEMANTIC_COLUMNS, assemble_dataset, temporal_split
from src.embeddings import build_embeddings
from src.evaluation import choose_and_explain, train_models
from src.preprocessing import clean_data
from src.temporal_features import TEMPORAL_COLUMNS, build_temporal_features
from src.topic_features import semantic_features, topic_quality, topic_time_series
from src.topic_modeling import fit_topic_model
from src.label_generation import build_labels
from src.utils import file_sha256, set_seed, temporal_boundaries, write_json
from src.visualization import make_figures


LOG = logging.getLogger("research_pipeline")


def csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def report_markdown(dataset_report, cleaning_report, quality, split_info, model_results,
                    ablation_results, horizon_results, importance, thresholds, audit):
    def table(frame, columns=None):
        if frame is None or frame.empty:
            return "Tidak tersedia."
        view = frame[columns] if columns else frame
        return view.to_markdown(index=False)
    evaluated = ablation_results[ablation_results.get("status", "") == "evaluated"] if not ablation_results.empty else ablation_results
    conclusion = "Data tidak cukup untuk kesimpulan empiris."
    if not evaluated.empty and "pr_auc" in evaluated and evaluated["pr_auc"].notna().any():
        best = evaluated.sort_values(["pr_auc", "f1_positive"], ascending=False).iloc[0]
        fusion = evaluated[evaluated.feature_set == "semantic_temporal"]
        semantic = evaluated[evaluated.feature_set == "semantic_only"]
        temporal_only = evaluated[evaluated.feature_set == "temporal_only"]
        if (not fusion.empty and not semantic.empty and not temporal_only.empty
                and pd.notna(fusion.iloc[0].pr_auc) and pd.notna(semantic.iloc[0].pr_auc)
                and pd.notna(temporal_only.iloc[0].pr_auc)):
            f_pr, s_pr, t_pr = fusion.iloc[0].pr_auc, semantic.iloc[0].pr_auc, temporal_only.iloc[0].pr_auc
            if f_pr > s_pr and f_pr > t_pr:
                verdict = "Fusion mengungguli kedua feature set tunggal pada PR-AUC."
            else:
                verdict = "Fusion tidak mengungguli kedua feature set tunggal; hipotesis keunggulan fusion belum didukung."
            conclusion = (f"Pada test temporal t+1, PR-AUC Temporal Only={t_pr:.4f}, Semantic Only={s_pr:.4f}, "
                          f"dan Semantic + Temporal={f_pr:.4f}. {verdict} "
                          "Temuan ini bersifat prediktif, bukan kausal.")
    stats_note = "Statistical significance testing was not performed because the number of independent evaluation origins was insufficient."
    lines = [
        "# Laporan Akhir Penelitian", "", "## 1. Dataset",
        f"Dataset nyata lokal berisi {dataset_report['rows']} baris dan {dataset_report['columns']} kolom sebelum cleaning, "
        f"dengan rentang {dataset_report['date_min']} hingga {dataset_report['date_max']}.", "",
        "## 2. Preprocessing", f"Data setelah cleaning: {cleaning_report['cleaned_rows']} baris; duplikat URL dibuang: {cleaning_report['duplicate_url_removed']}; duplikat judul dibuang: {cleaning_report['duplicate_title_removed']}.", "",
        "## 3. Topic modeling", f"BERTopic di-fit hanya pada dokumen sebelum cutoff training {split_info['train_cutoff']}; dokumen validation/test hanya ditransformasi. Model embedding: `{EMBEDDING_MODEL}`.", "",
        "## 4. Topic quality", table(pd.DataFrame([quality])), "",
        "## 5. Feature engineering", "Fitur semantik memakai centroid topik yang dibentuk dari training dan sinyal pada origin t. Fitur temporal rolling memakai `shift(1)` sehingga tidak membaca periode masa depan.", "",
        "## 6. Label distribution", f"Label memakai skor robust `(volume_future - median_history)/(1 + 1.4826*MAD)`; threshold kuantil training per horizon: {thresholds}.", "",
        "## 7. Temporal split", f"Train: sebelum {split_info['train_cutoff']}; validation: [{split_info['train_cutoff']}, {split_info['test_cutoff']}); test: mulai {split_info['test_cutoff']}.", "",
        "## 8. Model performance", table(model_results, [c for c in ["model","forecast_horizon","precision_positive","recall_positive","f1_positive","macro_f1","roc_auc","pr_auc"] if c in model_results]), "",
        "## 9. Ablation", table(ablation_results, [c for c in ["feature_set","model","f1_positive","macro_f1","roc_auc","pr_auc"] if c in ablation_results]), "",
        "## 10. Multi-horizon result", table(horizon_results), "",
        "## 11. Feature importance", "Permutation importance mengukur predictive contribution dan tidak membuktikan sebab-akibat.", table(importance.head(15) if importance is not None else pd.DataFrame()), "",
        "## 12. Leakage audit", table(audit), "",
        "## 13. Statistical comparison", stats_note, "",
        "## 14. Pemetaan bidang", "- **Web Mining:** dataset berita daring, URL, source, dan timestamp lokal yang sudah tersedia; pipeline ini tidak melakukan scraping.\n- **Text Mining:** cleaning ringan, sentence embedding multilingual, BERTopic, dan fitur semantik.\n- **Data Mining:** rekayasa fitur, target pertumbuhan, validasi temporal, klasifikasi, ablation, dan evaluasi.", "",
        "## 15. Limitations", f"Korpus berukuran terbatas dan mencakup rentang waktu pendek. Outlier BERTopic mencapai {quality['outlier_ratio']:.2%}, sehingga cakupan topic time series terbatas. Coherence berbasis korpus eksternal tidak dihitung; kualitas topik dinilai melalui diversity, silhouette, jumlah topik, dan outlier ratio. Hasil tidak boleh digeneralisasi melampaui sumber serta periode observasi.", "",
        "## 16. Kesimpulan empiris", conclusion, "",
    ]
    (RESULTS_DIR / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "full"], default="full")
    parser.add_argument("--force-embeddings", action="store_true")
    args = parser.parse_args()
    ensure_directories(); set_seed()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(RESULTS_DIR / "pipeline.log", encoding="utf-8")])

    LOG.info("STEP 01 dataset inspection")
    dataset_path = discover_dataset()
    raw = load_dataset(dataset_path)
    mapping = map_columns(raw); validate_raw(raw, mapping)
    inspection = inspect_dataset(dataset_path, raw, mapping)
    write_json(RESULTS_DIR / "tables" / "dataset_inspection.json", inspection)
    print(f"Rows={len(raw)}, columns={list(raw.columns)}")
    print(raw.head(5).to_string(index=False)); print(raw.dtypes); print(raw.isna().sum())

    LOG.info("STEP 02-03 cleaning and preprocessing")
    if args.mode == "sample" and len(raw) > 300:
        raw = raw.sort_values(mapping["date"]).iloc[:300].copy()
    cleaned, cleaning = clean_data(raw, mapping)
    cleaned_path = ROOT / "data" / "processed" / "cleaned_news.parquet"
    cleaned.to_parquet(cleaned_path, index=False)
    dataset_summary = pd.DataFrame([{
        "dataset_file": dataset_path.name, "dataset_sha256": file_sha256(dataset_path), "mode": args.mode,
        "initial_rows": inspection["rows"], "initial_columns": inspection["columns"],
        "cleaned_rows": cleaning["cleaned_rows"], "duplicate_url_removed": cleaning["duplicate_url_removed"],
        "duplicate_title_removed": cleaning["duplicate_title_removed"], "date_min": cleaning["date_min"],
        "date_max": cleaning["date_max"], "sources": cleaned.source.nunique(), "column_mapping": json.dumps(mapping),
    }])
    csv(RESULTS_DIR / "tables" / "dataset_summary.csv", dataset_summary)
    periods = sorted(cleaned.date.unique())
    train_cutoff, test_cutoff = temporal_boundaries(periods, VALIDATION_RATIO, TEST_RATIO)

    LOG.info("STEP 04 embeddings")
    embeddings, embedding_metadata = build_embeddings(cleaned_path, cleaned["document"], args.force_embeddings)
    LOG.info("STEP 05 BERTopic (fit train only, transform future)")
    model, topics, probabilities, topic_metadata, train_mask = fit_topic_model(cleaned, embeddings, train_cutoff)
    cleaned["topic_id"] = topics; cleaned["topic_probability"] = probabilities
    cleaned.to_parquet(ROOT / "data" / "processed" / "topic_assignments.parquet", index=False)
    write_json(MODEL_DIR / "topic_model_metadata.json", topic_metadata)

    LOG.info("STEP 06 topic evaluation")
    topic_summary, quality = topic_quality(model, cleaned, embeddings, train_mask)
    csv(RESULTS_DIR / "tables" / "topic_summary.csv", topic_summary)
    csv(RESULTS_DIR / "metrics" / "topic_quality.csv", pd.DataFrame([quality]))
    if quality["warning"]: LOG.warning(quality["warning"])

    LOG.info("STEP 07-09 topic time series and features")
    series = topic_time_series(cleaned)
    series.to_parquet(ROOT / "data" / "intermediate" / "topic_time_series.parquet", index=False)
    semantic = semantic_features(cleaned, embeddings, series, train_cutoff)
    temporal = build_temporal_features(series)
    semantic.to_parquet(ROOT / "data" / "processed" / "semantic_features.parquet", index=False)
    temporal.to_parquet(ROOT / "data" / "processed" / "temporal_features.parquet", index=False)

    all_model_results, all_ablation, horizon_rows, all_predictions = [], [], [], []
    main_importance = None; main_predictions = None; main_audit = None; thresholds = {}
    full_features = TEMPORAL_COLUMNS + SEMANTIC_COLUMNS
    for horizon in FORECAST_HORIZONS:
        LOG.info("STEP 10-16 horizon t+%s", horizon)
        labels, threshold = build_labels(series, horizon, train_cutoff)
        thresholds[horizon] = threshold
        dataset = assemble_dataset(temporal, semantic, labels)
        audit = audit_leakage(temporal, labels, cleaned, full_features, train_cutoff, topic_metadata)
        if horizon == 1:
            main_audit = audit
            csv(RESULTS_DIR / "metrics" / "leakage_audit.csv", audit)
        try:
            train, validation, test = temporal_split(dataset, train_cutoff, test_cutoff)
            results, fitted, _ = train_models(train, validation, test, full_features, "semantic_temporal", horizon)
            if "status" in results and (results.status == "evaluated").any():
                ablation = run_ablation(train, validation, test, horizon)
                all_ablation.append(ablation)
                evaluated = results[results.status == "evaluated"]
                horizon_rows.append({"forecast_horizon": horizon, "status": "evaluated", "train_rows": len(train),
                                     "validation_rows": len(validation), "test_rows": len(test),
                                     "best_validation_model": evaluated.sort_values(["validation_pr_auc", "validation_f1_positive"], ascending=False).iloc[0].model,
                                     "best_test_pr_auc": evaluated.pr_auc.max(), "best_test_f1_positive": evaluated.f1_positive.max()})
                if horizon == 1:
                    _, _, main_importance, main_predictions = choose_and_explain(results, fitted, test, full_features)
                    main_importance["feature_group"] = main_importance.feature.map(lambda x: "SEMANTIC" if x in SEMANTIC_COLUMNS else "TEMPORAL")
                    all_predictions.append(main_predictions)
            else:
                horizon_rows.append({"forecast_horizon": horizon, "status": results.iloc[0].status})
            all_model_results.append(results)
        except ValueError as exc:
            status = f"not evaluated: insufficient temporal history ({exc})"
            horizon_rows.append({"forecast_horizon": horizon, "status": status})
            LOG.warning(status)

    model_results = pd.concat(all_model_results, ignore_index=True) if all_model_results else pd.DataFrame()
    ablation_results = pd.concat(all_ablation, ignore_index=True) if all_ablation else pd.DataFrame([{"status": "not evaluated: insufficient temporal history"}])
    horizon_results = pd.DataFrame(horizon_rows)
    csv(RESULTS_DIR / "tables" / "model_results.csv", model_results)
    csv(RESULTS_DIR / "tables" / "ablation_results.csv", ablation_results)
    csv(RESULTS_DIR / "tables" / "horizon_results.csv", horizon_results)
    csv(RESULTS_DIR / "tables" / "feature_importance.csv", main_importance if main_importance is not None else pd.DataFrame([{"status": "not evaluated"}]))
    csv(RESULTS_DIR / "predictions" / "test_predictions.csv", main_predictions if main_predictions is not None else pd.DataFrame(columns=["date","topic_id","actual_label","predicted_label","prediction_probability","forecast_horizon"]))

    LOG.info("STEP 17-18 explainability and visualizations")
    make_figures(cleaned, series, model_results[model_results.forecast_horizon.eq(1)] if "forecast_horizon" in model_results else model_results,
                 ablation_results[ablation_results.forecast_horizon.eq(1)] if "forecast_horizon" in ablation_results else ablation_results,
                 main_importance, main_predictions)

    LOG.info("STEP 19 experiment tracking")
    experiment = model_results.copy()
    if not experiment.empty:
        experiment.insert(0, "run_at_utc", datetime.now(timezone.utc).isoformat())
        experiment["dataset_version"] = file_sha256(dataset_path)
        experiment["random_seed"] = RANDOM_SEED; experiment["embedding_model"] = EMBEDDING_MODEL
        experiment["bertopic_parameters"] = json.dumps(TOPIC_PARAMETERS)
        experiment["train_period"] = f"{cleaned.date.min()}..< {train_cutoff}"
        experiment["validation_period"] = f"{train_cutoff}..< {test_cutoff}"
        experiment["test_period"] = f"{test_cutoff}..{cleaned.date.max()}"
    csv(RESULTS_DIR / "experiment_log.csv", experiment)

    LOG.info("STEP 20 final report")
    report_markdown(inspection, cleaning, quality, {"train_cutoff": train_cutoff, "test_cutoff": test_cutoff},
                    model_results, ablation_results, horizon_results, main_importance, thresholds, main_audit)
    print(f"Pipeline selesai. Hasil: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

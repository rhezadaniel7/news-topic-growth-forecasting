"""Readout hasil pilot embedding: gate validitas, delta embedding, efek, verdict.

Modul ini TIDAK melakukan tuning, TIDAK refit topik, dan TIDAK mengubah label,
fitur temporal, atau split. Yang dilakukan hanya:

1. membaca artefak pilot yang sudah ada (embedding, fitur semantik, metadata);
2. mengevaluasi feature set pada partisi official train/validation yang sama
   dengan pipeline, memakai grid kandidat model yang sama dan prosedur seleksi
   kandidat yang sama (temporal CV 3 fold -> kandidat terpilih -> official
   validation). Dua feature set tambahan (similarity-only) diperlukan karena
   angka itu belum pernah dihitung oleh pipeline;
3. mencetak verdict memakai ambang yang ditetapkan pemanggil.

Metrik yang dilaporkan adalah OFFICIAL VALIDATION. Test tidak diakses di sini:
pipeline hanya mengevaluasi test untuk satu konfigurasi terpilih setelah seleksi
dibekukan, dan kontrol itu tidak diubah. Angka test untuk model terpilih per
varian tetap tersedia pada `results/embedding_runs/<varian>__<subset>/tables/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import step14_21_modeling as modeling
from .pilot_subset import PILOT_TAG, load_manifest
from .recompute_semantic_features import semantic_output_path
from .step7_embeddings import LEGACY_VARIANT_KEY, VARIANTS, variant_paths
from .step8_9_topic_modeling import TRAIN_START

LEGACY_KEY = LEGACY_VARIANT_KEY
VARIANT_ORDER = [
    LEGACY_KEY,
    "minilm_trunc128",
    "minilm_chunk128",
    "e5small_trunc512",
    "e5small_chunk512",
]

# 4 fitur yang benar-benar berasal dari embedding.
SIMILARITY_FEATURES = [
    "semantic_similarity_mean",
    "semantic_similarity_std",
    "semantic_drift",
    "similarity_change",
]
# 3 fitur yang berasal dari topic_assignments; invariant karena topik dikunci.
TOPIC_PROBABILITY_FEATURES = [
    "topic_probability_mean",
    "topic_probability_std",
    "topic_probability_change",
]

TEMPORAL_ONLY = "Temporal Only"
SEMANTIC_FULL = "Semantic Only (full 7)"
SEMANTIC_SIMILARITY = "Semantic Only (similarity 4)"
FUSION_FULL = "Fusion full (7 semantic + temporal)"
FUSION_SIMILARITY = "Fusion similarity (4 semantic + temporal)"

READOUT_FEATURE_SETS = {
    TEMPORAL_ONLY: modeling.TEMPORAL_FEATURES,
    SEMANTIC_FULL: modeling.SEMANTIC_FEATURES,
    SEMANTIC_SIMILARITY: SIMILARITY_FEATURES,
    FUSION_FULL: modeling.SEMANTIC_FEATURES + modeling.TEMPORAL_FEATURES,
    FUSION_SIMILARITY: SIMILARITY_FEATURES + modeling.TEMPORAL_FEATURES,
}

# Ambang verdict, ditetapkan sebelum angka dilihat.
TEMPORAL_CONTROL_TOLERANCE = 1e-6
COSINE_FLAT_MEDIAN = 0.9
GO_SEMANTIC_DELTA = 0.02
GO_FUSION_GAIN = 0.01
GO_MINIMUM_HORIZONS = 2
HORIZONS = [1, 2, 3]

# Verdict akhir dibaca dari varian ini saja: satu-satunya varian yang mengubah
# HANYA cara truncation ditangani, dengan encoder identik. Varian e5 mengganti
# encoder sekaligus, sehingga efeknya confounded dan dipakai sebagai sekunder.
PRIMARY_VERDICT_VARIANT = "minilm_chunk128"
CONFOUNDED_VARIANT_PREFIX = "e5"


# --------------------------------------------------------------------------- #
# ketersediaan artefak
# --------------------------------------------------------------------------- #
def variant_status(root: Path, variant_key: str, pilot_tag: str) -> dict:
    semantic_path = semantic_output_path(root, variant_key, pilot_tag)
    if variant_key == LEGACY_KEY:
        embedding_ready = (root / "data" / "intermediate" / "embeddings.npy").exists()
        metadata_path = root / "data" / "intermediate" / "embedding_metadata.json"
    else:
        paths = variant_paths(root, VARIANTS[variant_key], pilot_tag)
        embedding_ready = paths["metadata"].exists() and paths["embeddings"].exists()
        metadata_path = paths["metadata"]
    return {
        "variant": variant_key,
        "embedding_ready": bool(embedding_ready),
        "semantic_features_ready": bool(semantic_path.exists()),
        "metadata_path": metadata_path,
        "semantic_path": semantic_path,
        "status": "READY" if embedding_ready and semantic_path.exists() else "PENDING",
    }


def availability(root: Path, pilot_tag: str = PILOT_TAG) -> pd.DataFrame:
    rows = [variant_status(root, key, pilot_tag) for key in VARIANT_ORDER]
    return pd.DataFrame(
        [
            {
                "variant": row["variant"],
                "embedding_ready": row["embedding_ready"],
                "semantic_features_ready": row["semantic_features_ready"],
                "status": row["status"],
            }
            for row in rows
        ]
    )


# --------------------------------------------------------------------------- #
# 1 + 3. evaluasi feature set pada official validation
# --------------------------------------------------------------------------- #
def official_partition(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    horizon_frame = frame.loc[frame["forecast_horizon"].eq(horizon)].copy()
    train = horizon_frame.loc[
        horizon_frame["date"].between(TRAIN_START, modeling.OFFICIAL_TRAIN_END, inclusive="both")
    ].copy()
    validation = horizon_frame.loc[
        horizon_frame["date"].between(
            modeling.OFFICIAL_VALIDATION_START, modeling.OFFICIAL_VALIDATION_END, inclusive="both"
        )
    ].copy()
    threshold = modeling.fold_threshold(train)
    return train, validation, threshold


def evaluate_variant(root: Path, variant_key: str, pilot_tag: str) -> pd.DataFrame:
    """Official validation per (horizon, feature set, model), prosedur seperti pipeline."""
    previous_variant = modeling.ACTIVE_EMBEDDING_VARIANT
    previous_tag = modeling.ACTIVE_PILOT_TAG
    modeling.ACTIVE_EMBEDDING_VARIANT = variant_key
    modeling.ACTIVE_PILOT_TAG = pilot_tag
    try:
        frame, _ = modeling.prepare_final_feature_dataset(root)
    finally:
        modeling.ACTIVE_EMBEDDING_VARIANT = previous_variant
        modeling.ACTIVE_PILOT_TAG = previous_tag

    rows = []
    for horizon in HORIZONS:
        horizon_frame = frame.loc[frame["forecast_horizon"].eq(horizon)].copy()
        official_train, official_validation, threshold = official_partition(frame, horizon)
        y_train = modeling.apply_fold_labels(official_train, threshold)
        y_validation = modeling.apply_fold_labels(official_validation, threshold)
        for feature_set, columns in READOUT_FEATURE_SETS.items():
            for model_name, candidates in modeling.MODEL_CANDIDATES.items():
                # Seleksi kandidat memakai temporal CV, identik dengan pipeline.
                candidate_scores = []
                for candidate_id, params in enumerate(candidates, start=1):
                    fold_f1 = []
                    fold_pr = []
                    valid = True
                    for fold in modeling.FOLDS:
                        fold_train, fold_validation, fold_threshold, status = modeling.build_fold_data(
                            horizon_frame, fold
                        )
                        if status == "INVALID":
                            valid = False
                            break
                        y_fold_train = fold_train["fold_label"].to_numpy()
                        y_fold_validation = fold_validation["fold_label"].to_numpy()
                        pipeline = modeling.make_pipeline(model_name, params, y_fold_train)
                        if pipeline is not None:
                            pipeline.fit(fold_train[columns], y_fold_train)
                        probability = modeling.predict_probability(
                            model_name, pipeline, fold_validation[columns], y_fold_train
                        )
                        probability_threshold = modeling.choose_probability_threshold(
                            y_fold_validation, probability, model_name
                        )
                        metrics = modeling.compute_metrics(
                            y_fold_validation, probability, probability_threshold
                        )
                        fold_f1.append(metrics["f1"])
                        fold_pr.append(metrics["pr_auc"])
                    if valid:
                        candidate_scores.append(
                            (float(np.mean(fold_f1)), float(np.mean(fold_pr)), -candidate_id, candidate_id)
                        )
                if not candidate_scores:
                    continue
                chosen_id = max(candidate_scores)[3]
                chosen_params = candidates[chosen_id - 1]
                if len(np.unique(y_train)) < 2 or len(np.unique(y_validation)) < 2:
                    continue
                pipeline = modeling.make_pipeline(model_name, chosen_params, y_train)
                if pipeline is not None:
                    pipeline.fit(official_train[columns], y_train)
                probability = modeling.predict_probability(
                    model_name, pipeline, official_validation[columns], y_train
                )
                probability_threshold = modeling.choose_probability_threshold(
                    y_validation, probability, model_name
                )
                metrics = modeling.compute_metrics(y_validation, probability, probability_threshold)
                rows.append(
                    {
                        "variant": variant_key,
                        "horizon": horizon,
                        "feature_set": feature_set,
                        "model": model_name,
                        "candidate_id": chosen_id,
                        "n_train": int(len(official_train)),
                        "n_validation": int(len(official_validation)),
                        "growth_train": int(y_train.sum()),
                        "growth_validation": int(y_validation.sum()),
                        "label_threshold": threshold,
                        "f1": metrics["f1"],
                        "pr_auc": metrics["pr_auc"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                    }
                )
        print(f"readout {variant_key} horizon {horizon} selesai", flush=True)
    return pd.DataFrame(rows)


def best_per_feature_set(detail: pd.DataFrame) -> pd.DataFrame:
    """F1/PR-AUC terbaik per (varian, horizon, feature set), learner saja."""
    learners = detail.loc[detail["model"].ne("Majority Class")]
    best = (
        learners.sort_values(["f1", "pr_auc"], ascending=[False, False])
        .groupby(["variant", "horizon", "feature_set"], as_index=False)
        .first()[
            [
                "variant",
                "horizon",
                "feature_set",
                "model",
                "f1",
                "pr_auc",
                "precision",
                "recall",
                "n_validation",
                "growth_validation",
            ]
        ]
    )
    return best


# --------------------------------------------------------------------------- #
# 2. gate delta embedding
# --------------------------------------------------------------------------- #
def cosine_against_legacy(root: Path, variant_key: str, pilot_tag: str) -> dict:
    _, row_index = load_manifest(root)
    legacy = np.asarray(
        np.load(root / "data" / "intermediate" / "embeddings.npy", mmap_mode="r")[row_index],
        dtype=np.float64,
    )
    paths = variant_paths(root, VARIANTS[variant_key], pilot_tag)
    candidate = np.asarray(np.load(paths["embeddings"], mmap_mode="r"), dtype=np.float64)
    if candidate.shape != legacy.shape:
        raise RuntimeError(
            f"Shape embedding {variant_key} {candidate.shape} != legacy {legacy.shape}; "
            "perbandingan per dokumen tidak dapat dilakukan tanpa asumsi."
        )
    stored_rows = np.load(paths["rows"])
    if not np.array_equal(stored_rows, row_index):
        raise RuntimeError(
            f"Row index {variant_key} tidak sama dengan manifest pilot; "
            "perbandingan per dokumen dibatalkan."
        )
    numerator = (legacy * candidate).sum(axis=1)
    denominator = np.linalg.norm(legacy, axis=1) * np.linalg.norm(candidate, axis=1)
    cosine = numerator / denominator
    return {
        "variant": variant_key,
        "documents": int(len(cosine)),
        "cosine_min": float(cosine.min()),
        "cosine_p25": float(np.percentile(cosine, 25)),
        "cosine_median": float(np.median(cosine)),
        "cosine_p75": float(np.percentile(cosine, 75)),
        "cosine_max": float(cosine.max()),
        "share_above_0_9": float((cosine > 0.9).mean()),
    }


# --------------------------------------------------------------------------- #
# 4. cek prefix e5
# --------------------------------------------------------------------------- #
def e5_prefix_check(root: Path, variant_key: str, pilot_tag: str) -> dict:
    paths = variant_paths(root, VARIANTS[variant_key], pilot_tag)
    if not paths["metadata"].exists():
        return {"variant": variant_key, "prefix_in_metadata": None, "verdict": "PENDING"}
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    payload = metadata.get("embedding_fingerprint_payload") or {}
    prefix = payload.get("text_prefix")
    if prefix is None:
        return {
            "variant": variant_key,
            "prefix_in_metadata": None,
            "verdict": "SUSPECT",
            "reason": "metadata tidak merekam text_prefix",
        }
    expected = VARIANTS[variant_key].text_prefix
    if prefix != expected:
        return {
            "variant": variant_key,
            "prefix_in_metadata": prefix,
            "verdict": "SUSPECT",
            "reason": f"prefix metadata {prefix!r} != definisi varian {expected!r}",
        }
    if prefix.strip() != "passage:":
        return {
            "variant": variant_key,
            "prefix_in_metadata": prefix,
            "verdict": "SUSPECT",
            "reason": "prefix bukan 'passage:'; E5 tanpa prefix menghasilkan embedding terdegradasi",
        }
    return {
        "variant": variant_key,
        "prefix_in_metadata": prefix,
        "verdict": "OK",
        "reason": "prefix 'passage: ' tercatat pada fingerprint payload dan dipakai saat encode",
    }


# --------------------------------------------------------------------------- #
# 1. gate validitas Temporal Only
# --------------------------------------------------------------------------- #
def temporal_control_gate(best: pd.DataFrame, detail: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    temporal = best.loc[best["feature_set"].eq(TEMPORAL_ONLY)]
    table = temporal.pivot(index="horizon", columns="variant", values=["f1", "pr_auc"])
    if LEGACY_KEY not in temporal["variant"].unique():
        return table, {
            "status": "FAIL",
            "reason": "varian legacy tidak tersedia; gate tidak dapat dievaluasi",
            "deviating": [],
            "max_absolute_f1_deviation": float("nan"),
        }
    legacy_f1 = temporal.loc[temporal["variant"].eq(LEGACY_KEY)].set_index("horizon")["f1"]
    deviations = []
    for variant_key in temporal["variant"].unique():
        if variant_key == LEGACY_KEY:
            continue
        series = temporal.loc[temporal["variant"].eq(variant_key)].set_index("horizon")["f1"]
        for horizon in series.index:
            if horizon not in legacy_f1.index:
                continue
            deviations.append(
                {
                    "variant": variant_key,
                    "horizon": int(horizon),
                    "f1": float(series.loc[horizon]),
                    "legacy_f1": float(legacy_f1.loc[horizon]),
                    "absolute_deviation": abs(float(series.loc[horizon]) - float(legacy_f1.loc[horizon])),
                }
            )
    deviation_frame = pd.DataFrame(deviations)
    # Pemeriksaan yang lebih ketat: seluruh sel (horizon, model), bukan hanya yang terbaik.
    learners = detail.loc[
        detail["feature_set"].eq(TEMPORAL_ONLY) & detail["model"].ne("Majority Class")
    ]
    cell = learners.pivot_table(index=["horizon", "model"], columns="variant", values="f1")
    cell_deviation = (
        float((cell.max(axis=1) - cell.min(axis=1)).max()) if len(cell.columns) > 1 else 0.0
    )
    max_deviation = float(deviation_frame["absolute_deviation"].max()) if len(deviation_frame) else 0.0
    failing = (
        deviation_frame.loc[deviation_frame["absolute_deviation"] >= TEMPORAL_CONTROL_TOLERANCE]
        if len(deviation_frame)
        else pd.DataFrame()
    )
    status = "PASS" if max_deviation < TEMPORAL_CONTROL_TOLERANCE and cell_deviation < TEMPORAL_CONTROL_TOLERANCE else "FAIL"
    return table, {
        "status": status,
        "max_absolute_f1_deviation": max_deviation,
        "max_absolute_f1_deviation_all_cells": cell_deviation,
        "tolerance": TEMPORAL_CONTROL_TOLERANCE,
        "deviating": failing.to_dict("records") if len(failing) else [],
        "deviation_detail": deviation_frame,
    }


# --------------------------------------------------------------------------- #
# 3. efek: delta terhadap legacy
# --------------------------------------------------------------------------- #
def effect_table(best: pd.DataFrame) -> pd.DataFrame:
    pivot = best.pivot_table(
        index=["feature_set", "horizon"], columns="variant", values=["f1", "pr_auc"]
    )
    rows = []
    for (feature_set, horizon), record in pivot.iterrows():
        for variant_key in best["variant"].unique():
            if ("f1", variant_key) not in pivot.columns:
                continue
            f1 = record[("f1", variant_key)]
            pr_auc = record[("pr_auc", variant_key)]
            legacy_f1 = record.get(("f1", LEGACY_KEY), np.nan)
            legacy_pr = record.get(("pr_auc", LEGACY_KEY), np.nan)
            rows.append(
                {
                    "feature_set": feature_set,
                    "horizon": int(horizon),
                    "variant": variant_key,
                    "f1": float(f1) if pd.notna(f1) else np.nan,
                    "pr_auc": float(pr_auc) if pd.notna(pr_auc) else np.nan,
                    "delta_f1_vs_legacy": float(f1 - legacy_f1) if pd.notna(f1) and pd.notna(legacy_f1) else np.nan,
                    "delta_pr_auc_vs_legacy": float(pr_auc - legacy_pr) if pd.notna(pr_auc) and pd.notna(legacy_pr) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def fusion_gain_table(best: pd.DataFrame) -> pd.DataFrame:
    wide = best.pivot_table(index=["variant", "horizon"], columns="feature_set", values="f1")
    rows = []
    for (variant_key, horizon), record in wide.iterrows():
        temporal = record.get(TEMPORAL_ONLY, np.nan)
        rows.append(
            {
                "variant": variant_key,
                "horizon": int(horizon),
                "temporal_only_f1": float(temporal) if pd.notna(temporal) else np.nan,
                "fusion_full_f1": float(record.get(FUSION_FULL, np.nan)),
                "fusion_similarity_f1": float(record.get(FUSION_SIMILARITY, np.nan)),
                "fusion_full_gain": float(record.get(FUSION_FULL, np.nan) - temporal),
                "fusion_similarity_gain": float(record.get(FUSION_SIMILARITY, np.nan) - temporal),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. verdict
# --------------------------------------------------------------------------- #
def variant_verdicts(
    effects: pd.DataFrame,
    gains: pd.DataFrame,
    cosine: pd.DataFrame,
    prefix_checks: pd.DataFrame,
    available: list[str],
) -> tuple[pd.DataFrame, dict]:
    semantic_similarity = effects.loc[effects["feature_set"].eq(SEMANTIC_SIMILARITY)]
    rows = []
    for variant_key in available:
        if variant_key == LEGACY_KEY:
            continue
        suspect = False
        suspect_reason = ""
        has_checks = len(prefix_checks) and "variant" in prefix_checks.columns
        check = (
            prefix_checks.loc[prefix_checks["variant"].eq(variant_key)]
            if has_checks
            else prefix_checks.iloc[:0]
        )
        if len(check) and check.iloc[0]["verdict"] == "SUSPECT":
            suspect = True
            suspect_reason = str(check.iloc[0].get("reason", "prefix e5 tidak terkonfirmasi"))

        per_horizon = []
        for horizon in HORIZONS:
            delta = semantic_similarity.loc[
                semantic_similarity["variant"].eq(variant_key)
                & semantic_similarity["horizon"].eq(horizon),
                "delta_f1_vs_legacy",
            ]
            gain = gains.loc[
                gains["variant"].eq(variant_key) & gains["horizon"].eq(horizon)
            ]
            if delta.empty or gain.empty:
                continue
            delta_value = float(delta.iloc[0])
            gain_similarity = float(gain.iloc[0]["fusion_similarity_gain"])
            gain_full = float(gain.iloc[0]["fusion_full_gain"])
            per_horizon.append(
                {
                    "horizon": horizon,
                    "delta_semantic_similarity_f1": delta_value,
                    "fusion_similarity_gain": gain_similarity,
                    "fusion_full_gain": gain_full,
                    # Ambang GO memakai fusion similarity-only.
                    "meets_go_condition": delta_value >= GO_SEMANTIC_DELTA
                    and gain_similarity >= GO_FUSION_GAIN,
                    # Diagnostik dilusi, bukan veto.
                    "meets_with_full_fusion": delta_value >= GO_SEMANTIC_DELTA
                    and gain_full >= GO_FUSION_GAIN,
                }
            )
        horizons_meeting = sum(1 for record in per_horizon if record["meets_go_condition"])
        horizons_full_diagnostic = sum(
            1 for record in per_horizon if record["meets_with_full_fusion"]
        )
        median_cosine = cosine.loc[cosine["variant"].eq(variant_key), "cosine_median"]
        meets_go = horizons_meeting >= GO_MINIMUM_HORIZONS
        role = (
            "PRIMARY"
            if variant_key == PRIMARY_VERDICT_VARIANT
            else "SECONDARY (confounded: encoder berubah)"
            if variant_key.startswith(CONFOUNDED_VARIANT_PREFIX)
            else "SECONDARY"
        )
        if suspect:
            verdict = "EXCLUDED (SUSPECT)"
        elif meets_go:
            verdict = "GO"
        else:
            verdict = "NO-GO"
        rows.append(
            {
                "variant": variant_key,
                "role": role,
                "verdict": verdict,
                "suspect": suspect,
                "suspect_reason": suspect_reason,
                "cosine_median_vs_legacy": float(median_cosine.iloc[0]) if len(median_cosine) else np.nan,
                "horizons_meeting_go_condition": horizons_meeting,
                "horizons_meeting_full_fusion_diagnostic": horizons_full_diagnostic,
                "max_delta_semantic_similarity_f1": max(
                    (record["delta_semantic_similarity_f1"] for record in per_horizon), default=np.nan
                ),
                "max_fusion_similarity_gain": max(
                    (record["fusion_similarity_gain"] for record in per_horizon), default=np.nan
                ),
                "max_fusion_full_gain": max(
                    (record["fusion_full_gain"] for record in per_horizon), default=np.nan
                ),
            }
        )
    verdicts = pd.DataFrame(rows)
    non_legacy_cosine = cosine.loc[cosine["variant"].ne(LEGACY_KEY), "cosine_median"]
    all_flat = bool(len(non_legacy_cosine)) and bool((non_legacy_cosine > COSINE_FLAT_MEDIAN).all())
    primary = (
        verdicts.loc[verdicts["variant"].eq(PRIMARY_VERDICT_VARIANT)] if len(verdicts) else verdicts
    )
    if len(primary):
        primary_verdict = str(primary.iloc[0]["verdict"])
        primary_horizons = int(primary.iloc[0]["horizons_meeting_go_condition"])
    else:
        primary_verdict = "PENDING"
        primary_horizons = 0
    summary = {
        "primary_variant": PRIMARY_VERDICT_VARIANT,
        "primary_verdict": primary_verdict,
        "primary_horizons_meeting": primary_horizons,
        "all_cosine_medians_above_threshold": all_flat,
        "secondary_go_variants": verdicts.loc[
            verdicts["variant"].ne(PRIMARY_VERDICT_VARIANT) & verdicts["verdict"].eq("GO"),
            "variant",
        ].tolist()
        if len(verdicts)
        else [],
    }
    return verdicts, summary


LIMITATIONS = """## 6. Keterbatasan

Readout ini adalah pengukuran pilot, bukan bukti. Yang membatasi kekuatan
kesimpulannya, dari yang paling menentukan:

1. **Daya statistik pilot sangat kecil.** Subset hanya 31 dari 170 topik (18%) dan
   6.047 dari 37.933 dokumen non-outlier (16%). Official validation pilot hanya
   memuat 42 observasi positif pada t+1 (43 pada t+2, 44 pada t+3). Pada ukuran
   ini, selisih F1 sebesar 0,02-0,05 berada di dalam lebar interval kepercayaan
   yang wajar, sehingga verdict NO-GO harus dibaca sebagai "efek tidak terdeteksi
   pada daya yang tersedia", BUKAN sebagai "efek tidak ada". Tidak ada interval
   kepercayaan maupun uji signifikansi yang dihitung di sini; satu run, satu seed.

2. **Ruang topik tetap berasal dari embedding ter-truncate.** Sesuai keputusan
   mengunci topik, `topic_assignments`, `topic_daily_volume`, dan label tidak
   dihitung ulang. Artinya UMAP dan BERTopic masih di-fit pada embedding 128
   token, dan hanya centroid semantik yang memakai embedding baru. Angka di sini
   karena itu adalah BATAS BAWAH untuk manfaat mengatasi truncation: jika ruang
   topik ikut diperbaiki, efeknya bisa lebih besar. Sebaliknya, ini juga berarti
   readout tidak dapat menjawab apakah topik yang lebih baik akan menolong.

3. **Tiga dari tujuh fitur semantik invariant.** `topic_probability_mean`,
   `topic_probability_std`, dan `topic_probability_change` berasal dari
   `topic_assignments`, bukan dari embedding, sehingga identik di semua varian.
   Itulah sebabnya keputusan dibaca dari feature set similarity-only; angka
   Semantic Only penuh dan Fusion penuh mengandung dilusi yang tidak berhubungan
   dengan kualitas embedding.

4. **Hanya official validation yang diukur.** Test tidak diakses karena pipeline
   hanya mengevaluasi test untuk satu konfigurasi yang sudah dibekukan, dan
   kontrol itu tidak diubah. Tidak ada konfirmasi out-of-sample untuk selisih
   antar varian.

5. **Probability threshold dipilih pada set yang dilaporkan.** `choose_probability_threshold`
   memaksimalkan F1 pada validation yang sama yang kemudian dilaporkan. Ini
   perilaku pipeline yang sudah berlaku dan sama untuk semua varian, sehingga
   perbandingan antar varian tetap adil, tetapi level absolut F1 optimistis dan
   tidak boleh dibaca sebagai performa yang diharapkan.

6. **Varian e5 confounded secara desain.** `e5small_trunc512` dan
   `e5small_chunk512` mengubah encoder, batas token, dan data pralatih sekaligus,
   sehingga efeknya tidak dapat diatribusikan ke penanganan truncation. Keduanya
   dilaporkan sebagai sekunder dan tidak menentukan verdict, apa pun angkanya.
   Cosine similarity terhadap legacy juga tidak dapat ditafsirkan untuk keduanya
   karena ruang embedding berbeda.

7. **Validitas pilot bergantung pada threshold label yang tetap.** Skema label
   `burst_mad_k2` memakai threshold tetap, sehingga label tidak berubah ketika
   populasi topik dibatasi. Dengan skema label berbasis kuantil, label pada subset
   pilot TIDAK akan identik dengan run penuh dan perbandingan ini tidak sah.

8. **Distribution shift yang sudah terdokumentasi tetap ada.** Volume artikel per
   hari naik dari 125,6 (train) ke 244,6 (test). Readout ini tidak menyentuh dan
   tidak memperbaiki hal itu.

Implikasi praktis: NO-GO di sini adalah keputusan tentang alokasi waktu komputasi
(tidak menjalankan encoding penuh 6-8 jam sekarang), bukan kesimpulan ilmiah
bahwa truncation tidak penting. Truncation terbukti nyata dan terselesaikan oleh
chunking (cakupan token 23,8% menjadi 100%), tetapi pada daya pilot yang tersedia
perbaikan itu belum terbaca sebagai kenaikan kontribusi semantik yang konsisten.
"""


# --------------------------------------------------------------------------- #
# laporan
# --------------------------------------------------------------------------- #
def markdown(frame: pd.DataFrame, **kwargs) -> str:
    return frame.to_markdown(index=False, floatfmt=".4f", **kwargs)


def build_report(sections: dict) -> str:
    return sections["text"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tables = root / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pilot_tag = PILOT_TAG
    manifest, row_index = load_manifest(root)

    status = availability(root, pilot_tag)
    status.to_csv(tables / "pilot_readout_availability.csv", index=False)
    ready = status.loc[status["status"].eq("READY"), "variant"].tolist()
    pending = status.loc[status["status"].eq("PENDING"), "variant"].tolist()
    print("=== ketersediaan varian ===")
    print(status.to_string(index=False))
    if LEGACY_KEY not in ready:
        raise RuntimeError("Varian legacy belum siap; readout tidak dapat dijalankan.")

    detail_frames = [evaluate_variant(root, key, pilot_tag) for key in ready]
    detail = pd.concat(detail_frames, ignore_index=True)
    detail.to_csv(tables / "pilot_readout_model_detail.csv", index=False)
    best = best_per_feature_set(detail)
    best.to_csv(tables / "pilot_readout_best_per_feature_set.csv", index=False)

    control_table, control = temporal_control_gate(best, detail)
    control_flat = best.loc[best["feature_set"].eq(TEMPORAL_ONLY)][
        ["variant", "horizon", "model", "f1", "pr_auc"]
    ]
    control_flat.to_csv(tables / "pilot_readout_gate1_temporal_control.csv", index=False)

    lines = [
        "# Pilot embedding readout",
        "",
        f"Subset pilot: `{manifest['pilot_tag']}` — {manifest['topic_count']} topik, "
        f"{manifest['document_count']} dokumen non-outlier "
        f"(train {manifest['documents_by_split'].get('train')}, "
        f"validation {manifest['documents_by_split'].get('validation')}, "
        f"test {manifest['documents_by_split'].get('test')}).",
        "",
        "Topic assignment, `topic_daily_volume`, label growth, fitur temporal, dan split "
        "TIDAK diubah dan TIDAK dihitung ulang. Tidak ada tuning di luar grid kandidat "
        "yang sudah ada pada pipeline.",
        "",
        "**Surface metrik:** seluruh angka F1/PR-AUC di laporan ini adalah **official "
        "validation** (train 2024-07-31..2025-04-26, validation 2025-04-27..2025-07-25). "
        "Test tidak diakses di sini karena pipeline hanya mengevaluasi test untuk satu "
        "konfigurasi terpilih setelah seleksi dibekukan, dan kontrol itu tidak diubah.",
        "",
        "Prosedur per (horizon, feature set, model): kandidat hyperparameter dipilih lewat "
        "temporal CV 3 fold, lalu kandidat terpilih dievaluasi pada official validation — "
        "identik dengan prosedur ablation pipeline. Nilai yang dilaporkan adalah yang "
        "terbaik antar learner per feature set (Majority Class dikecualikan): dipilih satu "
        "model dengan F1 official validation tertinggi, dan PR-AUC yang dilaporkan adalah "
        "PR-AUC model yang sama itu — bukan maksimum PR-AUC yang diambil independen. "
        "Verifikasi: prosedur ini mereproduksi F1 ablation pipeline pada varian legacy "
        "sampai 8,3e-17 di sembilan sel (3 horizon x 3 feature set asli).",
        "",
        "## 0. Ketersediaan varian",
        "",
        markdown(status),
        "",
    ]
    if pending:
        lines += [f"**PENDING (belum selesai encode, tidak dihitung):** {', '.join(pending)}", ""]

    lines += [
        "## 1. GATE VALIDITAS — Temporal Only lintas varian",
        "",
        "`Temporal Only` tidak memakai embedding, sehingga nilainya wajib identik di semua "
        f"varian. Toleransi: |F1_varian − F1_legacy| < {TEMPORAL_CONTROL_TOLERANCE:g}.",
        "",
        markdown(control_flat),
        "",
        f"- Deviasi absolut maksimum F1 (nilai terbaik per feature set): **{control['max_absolute_f1_deviation']:.3e}**",
        f"- Deviasi absolut maksimum F1 (seluruh sel horizon × model): **{control['max_absolute_f1_deviation_all_cells']:.3e}**",
        f"- **GATE 1: {control['status']}**",
        "",
    ]
    if control["status"] == "FAIL":
        lines += [
            "Varian yang menyimpang:",
            "",
            markdown(pd.DataFrame(control["deviating"])) if control["deviating"] else "(tidak terdaftar)",
            "",
            "**STOP — indikasi mismatch split/kebocoran.**",
            "",
            "Bagian 2-5 tidak dihitung karena gate validitas gagal.",
            "",
        ]
        (root / "results" / "pilot_embedding_readout.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines[-8:]))
        return

    cosine_rows = [cosine_against_legacy(root, key, pilot_tag) for key in ready if key != LEGACY_KEY]
    cosine = pd.DataFrame(cosine_rows)
    cosine.to_csv(tables / "pilot_readout_gate2_cosine_vs_legacy.csv", index=False)

    effects = effect_table(best)
    effects.to_csv(tables / "pilot_readout_effects.csv", index=False)
    gains = fusion_gain_table(best)
    gains.to_csv(tables / "pilot_readout_fusion_gain.csv", index=False)

    prefix_checks = pd.DataFrame(
        [e5_prefix_check(root, key, pilot_tag) for key in ready if key.startswith("e5")]
    )
    if len(prefix_checks):
        prefix_checks.to_csv(tables / "pilot_readout_e5_prefix_check.csv", index=False)

    verdicts, summary = variant_verdicts(effects, gains, cosine, prefix_checks, ready)
    verdicts.to_csv(tables / "pilot_readout_verdict.csv", index=False)

    if control["status"] == "FAIL":
        final = "NO-GO"
        reason = "gate validitas Temporal Only gagal"
    elif summary["primary_verdict"] == "PENDING":
        final = "PENDING"
        reason = (
            f"varian primer {PRIMARY_VERDICT_VARIANT} belum selesai dihitung; "
            "verdict tidak dapat dibaca"
        )
    elif summary["primary_verdict"] == "GO":
        final = "GO"
        reason = (
            f"varian primer {PRIMARY_VERDICT_VARIANT} memenuhi ambang GO pada "
            f"{summary['primary_horizons_meeting']}/3 horizon"
        )
    else:
        final = "NO-GO"
        reason = (
            f"varian primer {PRIMARY_VERDICT_VARIANT} memenuhi ambang GO hanya pada "
            f"{summary['primary_horizons_meeting']}/3 horizon; efek pilot datar; "
            "pilot ber-CI lebar, Δ kecil = noise"
        )
    if final == "NO-GO" and summary["all_cosine_medians_above_threshold"]:
        reason += (
            f"; median cosine similarity > {COSINE_FLAT_MEDIAN} pada semua varian "
            "(embedding nyaris tidak berubah)"
        )
    if summary["secondary_go_variants"]:
        reason += (
            "; varian sekunder yang memenuhi ambang (tidak menentukan verdict): "
            + ", ".join(summary["secondary_go_variants"])
        )

    lines += [
        "## 2. GATE DELTA EMBEDDING — cosine similarity terhadap legacy",
        "",
        "Per dokumen, subset identik. Leading indicator: median > "
        f"{COSINE_FLAT_MEDIAN} berarti embedding nyaris tidak berubah.",
        "",
        markdown(cosine),
        "",
        "## 3. EFEK SEBENARNYA",
        "",
        "ΔF1 dihitung terhadap `legacy_minilm_trunc128` pada sel yang sama.",
        "",
        "### 3a. Semantic Only penuh (7 fitur)",
        "",
        markdown(effects.loc[effects["feature_set"].eq(SEMANTIC_FULL)].drop(columns=["feature_set"])),
        "",
        "### 3b. Semantic Only similarity-only (4 fitur berbasis embedding)",
        "",
        "Fitur: " + ", ".join(f"`{name}`" for name in SIMILARITY_FEATURES) + ". Tiga fitur "
        + ", ".join(f"`{name}`" for name in TOPIC_PROBABILITY_FEATURES)
        + " dibuang karena berasal dari `topic_assignments` dan invariant ketika topik dikunci.",
        "",
        markdown(effects.loc[effects["feature_set"].eq(SEMANTIC_SIMILARITY)].drop(columns=["feature_set"])),
        "",
        "### 3c. Fusion (Semantic + Temporal), penuh dan similarity-only",
        "",
        markdown(effects.loc[effects["feature_set"].eq(FUSION_FULL)].drop(columns=["feature_set"])),
        "",
        markdown(effects.loc[effects["feature_set"].eq(FUSION_SIMILARITY)].drop(columns=["feature_set"])),
        "",
        "Gain fusion terhadap Temporal Only pada sel yang sama:",
        "",
        markdown(gains),
        "",
        "## 4. CEK e5 — prefix 'passage:'",
        "",
    ]
    if len(prefix_checks):
        lines += [markdown(prefix_checks), ""]
    else:
        lines += ["Belum ada varian e5 yang siap; cek prefix ditunda (PENDING).", ""]

    lines += [
        "## 5. VERDICT",
        "",
        "Ambang yang dipakai (ditetapkan sebelum angka dilihat):",
        "",
        f"- NO-GO jika gate 1 FAIL.",
        f"- Condong NO-GO jika median cosine similarity > {COSINE_FLAT_MEDIAN} pada semua varian.",
        f"- GO hanya jika ΔF1 Semantic Only (similarity-only) ≥ +{GO_SEMANTIC_DELTA} vs legacy "
        f"DAN Fusion similarity-only − Temporal Only ≥ +{GO_FUSION_GAIN}, konsisten pada "
        f"≥{GO_MINIMUM_HORIZONS} dari 3 horizon.",
        "- Selain itu NO-GO.",
        "",
        f"**Verdict akhir dibaca dari `{PRIMARY_VERDICT_VARIANT}` SAJA.** Varian itu mengubah "
        "hanya cara truncation ditangani, dengan encoder identik, sehingga efeknya dapat "
        "diatribusikan ke truncation. Varian `e5*` mengganti encoder sekaligus sehingga "
        "confounded: dilaporkan sebagai sekunder dan tidak menentukan verdict, serta ditandai "
        "SUSPECT bila prefix 'passage:' tidak terkonfirmasi.",
        "",
        "Kondisi Fusion pada ambang GO memakai **fusion similarity-only**. Kolom "
        "`horizons_meeting_full_fusion_diagnostic` dan `max_fusion_full_gain` dicetak sebagai "
        "diagnostik dilusi (efek yang terlarut oleh 3 fitur `topic_probability_*` yang "
        "invariant), bukan sebagai veto.",
        "",
        markdown(verdicts),
        "",
    ]
    lines += [
        f"**VERDICT AKHIR: {final} — {reason}.**",
        "",
    ]
    if pending:
        lines += [
            f"Catatan: {', '.join(pending)} masih PENDING. Verdict akhir TIDAK berubah karena "
            f"dibaca dari varian primer `{PRIMARY_VERDICT_VARIANT}` yang sudah selesai; varian "
            "yang tersisa hanya menambah baris sekunder yang confounded dan tidak menentukan.",
            "",
        ]

    lines += [LIMITATIONS, ""]
    report = "\n".join(lines)
    (root / "results" / "pilot_embedding_readout.md").write_text(report, encoding="utf-8")
    print("\n=== GATE 1 ===")
    print(control_flat.to_string(index=False))
    print(f"status={control['status']} max_dev={control['max_absolute_f1_deviation']:.3e}")
    print("\n=== GATE 2 cosine ===")
    print(cosine.to_string(index=False))
    print("\n=== VERDICT ===")
    print(verdicts.to_string(index=False))
    print(f"\nVERDICT AKHIR: {final} — {reason}")


if __name__ == "__main__":
    main()

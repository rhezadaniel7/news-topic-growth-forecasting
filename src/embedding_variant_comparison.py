"""Kumpulkan hasil perbandingan strategi embedding.

Membaca metadata embedding, statistik fitur semantik, dan hasil ablation yang
sudah ditulis per varian; tidak melatih model dan tidak meng-encode apa pun.

Kontrol internal yang dipakai: feature set `Temporal Only` tidak memakai embedding
sama sekali, sehingga metriknya harus identik di semua varian. Perbedaan pada
`Semantic Only` dan `Semantic + Temporal` karena itu dapat diatribusikan ke
kualitas embedding, bukan ke perubahan label atau split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pilot_subset import PILOT_TAG
from .recompute_semantic_features import semantic_output_path
from .step7_embeddings import LEGACY_VARIANT_KEY, VARIANTS, variant_paths

COMPARISON_ORDER = [
    LEGACY_VARIANT_KEY,
    "minilm_trunc128",
    "minilm_chunk128",
    "e5small_trunc512",
    "e5small_chunk512",
]
FEATURE_SETS = ["Temporal Only", "Semantic Only", "Semantic + Temporal"]


def run_dir(root: Path, variant_key: str, pilot_tag: str) -> Path:
    return root / "results" / "embedding_runs" / f"{variant_key}__{pilot_tag}"


def encoding_facts(root: Path, variant_key: str, pilot_tag: str) -> dict:
    if variant_key == LEGACY_VARIANT_KEY:
        metadata = json.loads(
            (root / "data" / "intermediate" / "embedding_metadata.json").read_text(encoding="utf-8")
        )
        scope = "run penuh 67.160 dokumen di mesin lain"
    else:
        paths = variant_paths(root, VARIANTS[variant_key], pilot_tag)
        if not paths["metadata"].exists():
            return {}
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        scope = f"subset pilot {pilot_tag}"
    return {
        "embedding_model": metadata.get("embedding_model"),
        "max_seq_length": metadata.get("model_max_sequence_length"),
        "chunking_enabled": metadata.get("chunking_enabled", False),
        "mean_chunks_per_document": metadata.get("mean_chunks_per_document"),
        "documents_over_token_limit_percent": metadata.get("documents_over_token_limit_percent"),
        "still_truncated_percent": metadata.get(
            "documents_still_truncated_after_chunking_percent",
            metadata.get("documents_over_token_limit_percent"),
        ),
        "effective_token_coverage": metadata.get("effective_token_coverage"),
        "embedding_dimension": metadata.get("embedding_dimension"),
        "encoding_seconds": metadata.get("processing_seconds"),
        "encoding_scope": scope,
        "embedding_fingerprint": metadata.get("embedding_fingerprint"),
    }


def semantic_facts(root: Path, variant_key: str, pilot_tag: str) -> dict:
    path = semantic_output_path(root, variant_key, pilot_tag)
    if not path.exists():
        return {}
    semantic = pd.read_parquet(path)
    return {
        "semantic_rows": int(len(semantic)),
        "semantic_missing_rate": round(float(semantic["semantic_similarity_mean"].isna().mean()), 6),
        "similarity_mean": round(float(semantic["semantic_similarity_mean"].mean()), 6),
        "similarity_std_mean": round(float(semantic["semantic_similarity_std"].mean()), 6),
        "similarity_between_topic_spread": round(
            float(semantic.groupby("topic_id")["semantic_similarity_mean"].mean().std(ddof=0)), 6
        ),
        "drift_mean": round(float(semantic["semantic_drift"].mean()), 6),
    }


def ablation_facts(root: Path, variant_key: str, pilot_tag: str) -> pd.DataFrame:
    base = run_dir(root, variant_key, pilot_tag)
    validation_path = base / "tables" / "model_validation_results.csv"
    test_path = base / "tables" / "final_test_results.csv"
    if not validation_path.exists():
        return pd.DataFrame()
    validation = pd.read_csv(validation_path)
    learners = validation.loc[validation["model"].ne("Majority Class")]
    rows = []
    test_results = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    for horizon in sorted(validation["horizon"].unique()):
        record = {"variant": variant_key, "horizon": int(horizon)}
        for feature_set in FEATURE_SETS:
            subset = learners.loc[
                learners["horizon"].eq(horizon) & learners["feature_set"].eq(feature_set)
            ]
            key = feature_set.replace(" + ", "_").replace(" ", "_").lower()
            record[f"val_f1_{key}"] = float(subset["f1"].max()) if len(subset) else np.nan
            record[f"val_pr_auc_{key}"] = float(subset["pr_auc"].max()) if len(subset) else np.nan
        record["val_f1_fusion_gain"] = (
            record["val_f1_semantic_temporal"] - record["val_f1_temporal_only"]
        )
        record["val_pr_auc_fusion_gain"] = (
            record["val_pr_auc_semantic_temporal"] - record["val_pr_auc_temporal_only"]
        )
        if len(test_results):
            selected = test_results.loc[
                test_results["horizon"].eq(horizon) & test_results["selected_final_model"]
            ]
            if len(selected):
                record["test_selected_feature_set"] = selected.iloc[0]["feature_set"]
                record["test_selected_model"] = selected.iloc[0]["model"]
                record["test_f1"] = float(selected.iloc[0]["f1"])
                record["test_pr_auc"] = float(selected.iloc[0]["pr_auc"])
        rows.append(record)
    return pd.DataFrame(rows)


def collect(root: Path, pilot_tag: str = PILOT_TAG) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    ablation_frames = []
    facts_rows = []
    missing = []
    for variant_key in COMPARISON_ORDER:
        ablation = ablation_facts(root, variant_key, pilot_tag)
        if ablation.empty:
            missing.append(variant_key)
            continue
        ablation_frames.append(ablation)
        facts_rows.append(
            {
                "variant": variant_key,
                "description": (
                    "embedding run sebelumnya, right truncation 128 token"
                    if variant_key == LEGACY_VARIANT_KEY
                    else VARIANTS[variant_key].description
                ),
                **encoding_facts(root, variant_key, pilot_tag),
                **semantic_facts(root, variant_key, pilot_tag),
            }
        )
    ablation = pd.concat(ablation_frames, ignore_index=True) if ablation_frames else pd.DataFrame()
    facts = pd.DataFrame(facts_rows)
    order = {key: index for index, key in enumerate(COMPARISON_ORDER)}
    if len(ablation):
        ablation = ablation.sort_values(
            ["variant", "horizon"], key=lambda s: s.map(order) if s.name == "variant" else s
        ).reset_index(drop=True)
    if len(facts):
        facts = facts.sort_values("variant", key=lambda s: s.map(order)).reset_index(drop=True)
    return ablation, facts, missing


def check_temporal_control(ablation: pd.DataFrame) -> pd.DataFrame:
    """Temporal Only tidak memakai embedding, jadi metriknya wajib identik."""
    control = ablation.pivot(index="horizon", columns="variant", values="val_f1_temporal_only")
    control["max_absolute_difference"] = control.max(axis=1) - control.min(axis=1)
    control["control_holds"] = control["max_absolute_difference"] < 1e-9
    return control


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tables = root / "results" / "tables"
    ablation, facts, missing = collect(root)
    if ablation.empty:
        raise RuntimeError("Belum ada hasil ablation varian embedding.")
    ablation.to_csv(tables / "embedding_variant_ablation.csv", index=False)
    facts.to_csv(tables / "embedding_variant_encoding_facts.csv", index=False)
    control = check_temporal_control(ablation)
    control.to_csv(tables / "embedding_variant_temporal_control.csv")

    if missing:
        print(f"CATATAN: varian tanpa hasil ablation (dilewati): {missing}\n")
    print("=== Fakta encoding dan fitur semantik ===")
    print(
        facts[
            [
                "variant",
                "embedding_model",
                "max_seq_length",
                "chunking_enabled",
                "mean_chunks_per_document",
                "still_truncated_percent",
                "effective_token_coverage",
                "similarity_mean",
                "drift_mean",
                "encoding_seconds",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n=== Kontrol Temporal Only (wajib identik antar varian) ===")
    print(control.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\n=== Ablation official validation (F1 terbaik per feature set) ===")
    print(
        ablation[
            [
                "variant",
                "horizon",
                "val_f1_temporal_only",
                "val_f1_semantic_only",
                "val_f1_semantic_temporal",
                "val_f1_fusion_gain",
                "val_pr_auc_semantic_only",
                "val_pr_auc_fusion_gain",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n=== Rata-rata tiga horizon ===")
    summary = ablation.groupby("variant", sort=False)[
        [
            "val_f1_temporal_only",
            "val_f1_semantic_only",
            "val_f1_semantic_temporal",
            "val_f1_fusion_gain",
            "val_pr_auc_semantic_only",
            "val_pr_auc_fusion_gain",
        ]
    ].mean()
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    summary.to_csv(tables / "embedding_variant_ablation_summary.csv")


if __name__ == "__main__":
    main()

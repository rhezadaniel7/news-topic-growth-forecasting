"""Kumpulkan hasil run_step14_21 dari setiap skema label menjadi satu tabel.

Skrip ini hanya membaca output yang sudah ditulis oleh step10_13 dan step14_21;
tidak melatih model dan tidak menyentuh kontrol leakage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .label_schemes import (
    COMPARISON_ORDER,
    DEFAULT_SCHEME_KEY,
    SCHEMES,
    fit_threshold,
    population_mask,
)


def scheme_result_dir(root: Path, key: str) -> Path:
    if key == DEFAULT_SCHEME_KEY:
        return root / "results"
    return root / "results" / "scheme_runs" / key


def positive_composition(root: Path) -> pd.DataFrame:
    """Isi substantif label positif tiap skema.

    Menjawab pertanyaan "apakah 'growth' bermakna": berapa proporsi observasi
    positif yang sebenarnya hanya berarti "ada satu artikel pada target date",
    dan berapa besar kenaikan aktual terhadap median historis.
    """
    labels = pd.read_parquet(root / "data" / "processed" / "growth_labels_audit.parquet")
    eligible = labels.loc[labels["eligible_observation"]]
    rows = []
    for key in COMPARISON_ORDER:
        scheme = SCHEMES[key]
        for horizon in [1, 2, 3]:
            frame = eligible.loc[eligible["forecast_horizon"].eq(horizon)]
            included = frame.loc[population_mask(frame, scheme)]
            threshold = fit_threshold(
                included.loc[included["split"].eq("train")], scheme, require_train_split=True
            )
            for split in ["train", "validation", "test"]:
                subset = included.loc[included["split"].eq(split)]
                positives = subset.loc[subset[scheme.score_column].gt(threshold)]
                total = len(positives)
                rows.append(
                    {
                        "scheme": key,
                        "horizon": horizon,
                        "split": split,
                        "threshold_value": threshold,
                        "positives": total,
                        "share_positive_median_zero": float(positives["historical_median"].eq(0).mean()) if total else float("nan"),
                        "share_positive_from_inactive_day": float(positives["baseline_volume"].eq(0).mean()) if total else float("nan"),
                        "share_positive_future_is_one_article": float(positives["future_volume"].eq(1).mean()) if total else float("nan"),
                        "share_positive_future_at_most_two": float(positives["future_volume"].le(2).mean()) if total else float("nan"),
                        "median_positive_future_volume": float(positives["future_volume"].median()) if total else float("nan"),
                        "median_positive_rise_over_median": float((positives["future_volume"] - positives["historical_median"]).median()) if total else float("nan"),
                    }
                )
    order = {key: index for index, key in enumerate(COMPARISON_ORDER)}
    return pd.DataFrame(rows).sort_values(
        ["scheme", "horizon", "split"], key=lambda s: s.map(order) if s.name == "scheme" else s
    ).reset_index(drop=True)


def collect(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    balance = pd.read_csv(root / "results" / "tables" / "label_scheme_class_balance.csv")
    per_horizon_rows = []
    missing = []
    for key in COMPARISON_ORDER:
        scheme = SCHEMES[key]
        base = scheme_result_dir(root, key)
        test_path = base / "tables" / "final_test_results.csv"
        validation_path = base / "tables" / "model_validation_results.csv"
        if not test_path.exists():
            missing.append(key)
            continue
        test_results = pd.read_csv(test_path)
        validation = pd.read_csv(validation_path)
        selected = test_results.loc[test_results["selected_final_model"]]
        majority = (
            validation.loc[validation["model"].eq("Majority Class")]
            .groupby("horizon")["f1"]
            .max()
        )
        for row in selected.itertuples(index=False):
            horizon = int(row.horizon)
            counts = balance.loc[
                balance["scheme"].eq(key) & balance["horizon"].eq(horizon)
            ].set_index("split")
            per_horizon_rows.append(
                {
                    "scheme": key,
                    "scheme_name": scheme.name,
                    "definition": scheme.definition,
                    "threshold_kind": scheme.threshold_kind,
                    "population": scheme.population,
                    "horizon": horizon,
                    "label_threshold": float(row.growth_label_threshold),
                    "train_observations": int(counts.loc["train", "observations"]),
                    "train_growth_rate": float(counts.loc["train", "growth_rate"]),
                    "validation_growth_rate": float(counts.loc["validation", "growth_rate"]),
                    "test_growth_rate": float(counts.loc["test", "growth_rate"]),
                    "selected_model": row.model,
                    "selected_feature_set": row.feature_set,
                    "validation_f1": float(row.validation_f1),
                    "validation_pr_auc": float(row.validation_pr_auc),
                    "test_precision": float(row.precision),
                    "test_recall": float(row.recall),
                    "test_f1": float(row.f1),
                    "test_pr_auc": float(row.pr_auc),
                    "test_pr_auc_over_prevalence": (
                        float(row.pr_auc) / float(counts.loc["test", "growth_rate"])
                        if float(counts.loc["test", "growth_rate"])
                        else float("nan")
                    ),
                    "majority_validation_f1": float(majority.get(horizon, float("nan"))),
                }
            )
    if missing:
        raise RuntimeError(
            f"Hasil modeling belum tersedia untuk skema: {missing}. "
            "Jalankan LABEL_SCHEME=<key> python run_step14_21.py lebih dulu."
        )
    per_horizon = pd.DataFrame(per_horizon_rows)
    aggregate = (
        per_horizon.groupby(["scheme", "scheme_name", "threshold_kind", "population"], as_index=False)
        .agg(
            label_threshold=("label_threshold", "first"),
            train_observations=("train_observations", "first"),
            train_growth_rate=("train_growth_rate", "mean"),
            validation_growth_rate=("validation_growth_rate", "mean"),
            test_growth_rate=("test_growth_rate", "mean"),
            mean_test_f1=("test_f1", "mean"),
            mean_test_pr_auc=("test_pr_auc", "mean"),
            mean_test_pr_auc_over_prevalence=("test_pr_auc_over_prevalence", "mean"),
            mean_validation_f1=("validation_f1", "mean"),
            test_f1_h1=("test_f1", "first"),
        )
    )
    order = {key: index for index, key in enumerate(COMPARISON_ORDER)}
    aggregate = aggregate.sort_values("scheme", key=lambda s: s.map(order)).reset_index(drop=True)
    per_horizon = per_horizon.sort_values(
        ["scheme", "horizon"], key=lambda s: s.map(order) if s.name == "scheme" else s
    ).reset_index(drop=True)
    return per_horizon, aggregate


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tables = root / "results" / "tables"
    per_horizon, aggregate = collect(root)
    per_horizon.to_csv(tables / "label_scheme_model_comparison.csv", index=False)
    aggregate.to_csv(tables / "label_scheme_model_comparison_summary.csv", index=False)
    composition = positive_composition(root)
    composition.to_csv(tables / "label_scheme_positive_composition.csv", index=False)

    print("=== Isi label positif (t+1) ===")
    print(
        composition.loc[composition["horizon"].eq(1) & composition["split"].ne("validation")][
            [
                "scheme",
                "split",
                "threshold_value",
                "positives",
                "share_positive_from_inactive_day",
                "share_positive_future_is_one_article",
                "share_positive_future_at_most_two",
                "median_positive_future_volume",
                "median_positive_rise_over_median",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print()

    print("=== Perbandingan skema label (rata-rata 3 horizon) ===")
    print(
        aggregate[
            [
                "scheme",
                "threshold_kind",
                "label_threshold",
                "train_observations",
                "train_growth_rate",
                "validation_growth_rate",
                "test_growth_rate",
                "mean_validation_f1",
                "mean_test_f1",
                "mean_test_pr_auc",
                "mean_test_pr_auc_over_prevalence",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n=== Per horizon ===")
    print(
        per_horizon[
            [
                "scheme",
                "horizon",
                "label_threshold",
                "train_growth_rate",
                "test_growth_rate",
                "selected_model",
                "selected_feature_set",
                "validation_f1",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_pr_auc",
                "majority_validation_f1",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    payload = {
        "schemes": {
            key: {
                "definition": SCHEMES[key].definition,
                "rationale": SCHEMES[key].rationale,
                "results_directory": str(scheme_result_dir(root, key).relative_to(root)),
            }
            for key in COMPARISON_ORDER
        },
        "recommended_scheme": DEFAULT_SCHEME_KEY,
    }
    (root / "results" / "label_scheme_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

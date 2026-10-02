from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .label_schemes import (
    DEFAULT_SCHEME_KEY,
    MAD_SCALE,
    PRIMARY_GROWTH_QUANTILE,
    LabelScheme,
    apply_labels,
    fit_threshold,
    get_scheme,
    population_mask,
)
from .pilot_subset import PILOT_TAG
from .recompute_semantic_features import semantic_output_path
from .step8_9_topic_modeling import TEST_END, TEST_START, TRAIN_START


RANDOM_SEED = 42
SEMANTIC_FEATURES = [
    "semantic_similarity_mean",
    "semantic_similarity_std",
    "semantic_drift",
    "topic_probability_mean",
    "topic_probability_std",
    "topic_probability_change",
    "similarity_change",
]
TEMPORAL_FEATURES = [
    "current_volume",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_std_7",
    "growth_rate",
    "percentage_change",
    "momentum",
    "burst_intensity",
]
FEATURE_SETS = {
    "Temporal Only": TEMPORAL_FEATURES,
    "Semantic Only": SEMANTIC_FEATURES,
    "Semantic + Temporal": SEMANTIC_FEATURES + TEMPORAL_FEATURES,
}
MODEL_CANDIDATES = {
    "Majority Class": [{"strategy": "majority"}],
    "Logistic Regression": [
        {"C": 0.1, "max_iter": 2000, "class_weight": "balanced", "solver": "lbfgs"},
        {"C": 1.0, "max_iter": 2000, "class_weight": "balanced", "solver": "lbfgs"},
    ],
    "Random Forest": [
        {
            "n_estimators": 150,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
        {
            "n_estimators": 250,
            "max_depth": 14,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
    ],
    "XGBoost": [
        {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
        {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
    ],
}

FOLDS = [
    {
        "fold": 1,
        "train_start": pd.Timestamp("2024-07-31"),
        "train_end": pd.Timestamp("2025-01-26"),
        "validation_start": pd.Timestamp("2025-01-27"),
        "validation_end": pd.Timestamp("2025-03-27"),
    },
    {
        "fold": 2,
        "train_start": pd.Timestamp("2024-07-31"),
        "train_end": pd.Timestamp("2025-03-27"),
        "validation_start": pd.Timestamp("2025-03-28"),
        "validation_end": pd.Timestamp("2025-05-26"),
    },
    {
        "fold": 3,
        "train_start": pd.Timestamp("2024-07-31"),
        "train_end": pd.Timestamp("2025-05-26"),
        "validation_start": pd.Timestamp("2025-05-27"),
        "validation_end": pd.Timestamp("2025-07-25"),
    },
]

ACTIVE_SCHEME: LabelScheme = get_scheme()

# Varian embedding untuk run perbandingan. None berarti memakai fitur semantik
# canonical dari run penuh; nilai lain memakai fitur semantik hasil recompute pada
# subset pilot, dengan topic assignment dan label tetap dikunci.
ACTIVE_EMBEDDING_VARIANT: str | None = os.environ.get("EMBEDDING_VARIANT") or None
ACTIVE_PILOT_TAG: str = os.environ.get("PILOT_SUBSET") or PILOT_TAG

# Subset topik "forecastable". Kriterianya dibekukan dari statistik training saja
# dan disimpan pada JSON; modul ini hanya membaca daftar topik, tidak pernah
# menurunkan atau mengubah kriteria.
ACTIVE_TOPIC_SUBSET: str | None = os.environ.get("TOPIC_SUBSET") or None


def load_topic_subset(root: Path, name: str) -> dict:
    path = root / "data" / "processed" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Daftar subset topik beku tidak ada: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("frozen_before_seeing_validation_or_test"):
        raise RuntimeError(
            f"{path} tidak menandai dirinya dibekukan sebelum melihat validation/test; "
            "run dibatalkan agar kriteria subset tidak bisa dipilih berdasarkan hasil."
        )
    if not payload.get("selection_uses_only_training_window"):
        raise RuntimeError(
            f"{path} tidak menandai kriteria berasal dari training window saja; run dibatalkan."
        )
    return payload


def set_active_scheme(scheme: LabelScheme) -> None:
    global ACTIVE_SCHEME
    ACTIVE_SCHEME = scheme


OFFICIAL_TRAIN_END = pd.Timestamp("2025-04-26")
OFFICIAL_VALIDATION_START = pd.Timestamp("2025-04-27")
OFFICIAL_VALIDATION_END = pd.Timestamp("2025-07-25")
PROBABILITY_THRESHOLDS = np.round(np.linspace(0.10, 0.90, 81), 2)


def json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def prepare_final_feature_dataset(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ACTIVE_EMBEDDING_VARIANT:
        semantic_path = semantic_output_path(root, ACTIVE_EMBEDDING_VARIANT, ACTIVE_PILOT_TAG)
        if not semantic_path.exists():
            raise FileNotFoundError(
                f"Fitur semantik varian belum ada: {semantic_path}. Jalankan "
                f"python run_semantic_variant.py {ACTIVE_EMBEDDING_VARIANT} lebih dulu."
            )
        semantic = pd.read_parquet(semantic_path)
        if ACTIVE_SCHEME.is_fitted:
            print(
                f"PERINGATAN: skema label {ACTIVE_SCHEME.key} memakai threshold yang di-fit, "
                "sehingga label pada subset pilot tidak identik dengan run penuh. Untuk "
                "perbandingan varian embedding pakai skema dengan threshold tetap.",
                flush=True,
            )
    else:
        semantic = pd.read_parquet(root / "data" / "processed" / "semantic_features.parquet")
    temporal = pd.read_parquet(root / "data" / "processed" / "temporal_features.parquet")
    labels = pd.read_parquet(root / "data" / "processed" / "growth_labels_audit.parquet")

    semantic_subset = semantic[["topic_id", "date"] + SEMANTIC_FEATURES].copy()
    if ACTIVE_TOPIC_SUBSET:
        subset = load_topic_subset(root, ACTIVE_TOPIC_SUBSET)
        allowed_topics = set(int(topic) for topic in subset["topics"])
        print(
            f"subset topik aktif: {ACTIVE_TOPIC_SUBSET} — {len(allowed_topics)} topik; "
            f"kriteria {subset['criteria']}",
            flush=True,
        )
        if ACTIVE_SCHEME.is_fitted:
            raise RuntimeError(
                f"Skema label {ACTIVE_SCHEME.key} memakai threshold yang di-fit, sehingga "
                "membatasi populasi topik akan mengubah label dan perbandingan dengan run "
                "penuh tidak sah. Subset topik hanya boleh dipakai dengan skema "
                "berthreshold tetap."
            )
    else:
        allowed_topics = None
    temporal_subset = temporal[["topic_id", "date"] + TEMPORAL_FEATURES].copy()
    features = semantic_subset.merge(
        temporal_subset,
        on=["topic_id", "date"],
        how="inner",
        validate="one_to_one",
    )
    scheme = ACTIVE_SCHEME
    label_subset = labels.loc[
        labels["eligible_observation"] & labels[scheme.score_column].notna(),
        [
            "topic_id",
            "origin_date",
            "forecast_horizon",
            "split",
            "growth_score",
            "burst_score",
            "historical_median",
            "historical_mad",
        ],
    ].copy()
    label_subset = label_subset.rename(columns={"origin_date": "date"})
    if allowed_topics is not None:
        label_subset = label_subset.loc[label_subset["topic_id"].isin(allowed_topics)].copy()
        features = features.loc[features["topic_id"].isin(allowed_topics)].copy()
    final = label_subset.merge(features, on=["topic_id", "date"], how="inner", validate="many_to_one")
    # Seleksi populasi skema memakai lag_1, yaitu volume pada t-1 yang sudah
    # diketahui pada origin date; tidak ada informasi target yang dipakai.
    final = final.loc[population_mask(final, scheme)].copy()
    final = final.sort_values(["forecast_horizon", "date", "topic_id"]).reset_index(drop=True)
    if final.duplicated(["topic_id", "date", "forecast_horizon"]).any():
        raise RuntimeError("Duplicate key pada final feature dataset.")

    # Label resmi per horizon: threshold hanya di-fit pada split training.
    final["label"] = np.int8(0)
    horizon_thresholds = {}
    for horizon in [1, 2, 3]:
        horizon_mask = final["forecast_horizon"].eq(horizon)
        horizon_frame = final.loc[horizon_mask]
        threshold = fit_threshold(
            horizon_frame.loc[horizon_frame["split"].eq("train")], scheme, require_train_split=True
        )
        horizon_thresholds[horizon] = threshold
        final.loc[horizon_mask, "label"] = apply_labels(horizon_frame, scheme, threshold)
    final["label"] = final["label"].astype("int8")

    forbidden_predictors = {"growth_score", "burst_score", "historical_median", "historical_mad", "label", "split", "topic_id", "date", "forecast_horizon"}
    if set(SEMANTIC_FEATURES + TEMPORAL_FEATURES) & forbidden_predictors:
        raise RuntimeError("Target/future field masuk feature list.")

    summary_rows = []
    for horizon in [1, 2, 3]:
        for split in ["train", "validation", "test"]:
            subset = final.loc[
                final["forecast_horizon"].eq(horizon) & final["split"].eq(split)
            ]
            summary_rows.append(
                {
                    "record_type": "partition",
                    "label_scheme": scheme.key,
                    "label_threshold": horizon_thresholds[horizon],
                    "horizon": horizon,
                    "split": split,
                    "feature": "",
                    "rows": int(len(subset)),
                    "growth": int(subset["label"].sum()),
                    "normal": int((subset["label"] == 0).sum()),
                    "missing": int(subset[SEMANTIC_FEATURES + TEMPORAL_FEATURES].isna().sum().sum()),
                    "missing_rate": float(subset[SEMANTIC_FEATURES + TEMPORAL_FEATURES].isna().mean().mean()),
                }
            )
    for feature in SEMANTIC_FEATURES + TEMPORAL_FEATURES:
        summary_rows.append(
            {
                "record_type": "feature",
                "label_scheme": scheme.key,
                "label_threshold": np.nan,
                "horizon": 0,
                "split": "all",
                "feature": feature,
                "rows": int(len(final)),
                "growth": np.nan,
                "normal": np.nan,
                "missing": int(final[feature].isna().sum()),
                "missing_rate": float(final[feature].isna().mean()),
            }
        )
    return final, pd.DataFrame(summary_rows)


def fold_threshold(training_frame: pd.DataFrame) -> float:
    """Threshold label untuk satu fold, di-fit hanya dari training window fold."""
    if training_frame.empty:
        raise RuntimeError("Training frame kosong pada fold.")
    return fit_threshold(training_frame, ACTIVE_SCHEME)


def make_pipeline(model_name: str, params: dict, y_train: np.ndarray) -> Pipeline | None:
    if model_name == "Majority Class":
        return None
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "Logistic Regression":
        steps.append(("scaler", StandardScaler()))
        estimator = LogisticRegression(random_state=RANDOM_SEED, **params)
    elif model_name == "Random Forest":
        estimator = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **params)
    elif model_name == "XGBoost":
        negative = int((y_train == 0).sum())
        positive = int((y_train == 1).sum())
        estimator = XGBClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=(negative / positive) if positive else 1.0,
            **params,
        )
    else:
        raise ValueError(model_name)
    steps.append(("model", estimator))
    return Pipeline(steps)


def predict_probability(model_name: str, pipeline: Pipeline | None, x: pd.DataFrame, y_train: np.ndarray) -> np.ndarray:
    if model_name == "Majority Class":
        return np.full(len(x), float(np.mean(y_train)), dtype=float)
    return pipeline.predict_proba(x)[:, 1]


def choose_probability_threshold(y_true: np.ndarray, probability: np.ndarray, model_name: str) -> float:
    if model_name == "Majority Class":
        return 0.5
    candidates = []
    for threshold in PROBABILITY_THRESHOLDS:
        score = f1_score(y_true, probability >= threshold, zero_division=0)
        candidates.append((score, -abs(threshold - 0.5), -threshold, float(threshold)))
    return max(candidates)[3]


def compute_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y_true)) == 2
    return {
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)) if two_classes else np.nan,
        "pr_auc": float(average_precision_score(y_true, probability)) if two_classes else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "probability_threshold": float(threshold),
    }


def apply_fold_labels(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    return apply_labels(frame, ACTIVE_SCHEME, threshold)


def build_fold_data(frame: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame, float, str]:
    train = frame.loc[frame["date"].between(fold["train_start"], fold["train_end"], inclusive="both")].copy()
    validation = frame.loc[frame["date"].between(fold["validation_start"], fold["validation_end"], inclusive="both")].copy()
    threshold = fold_threshold(train)
    y_train = apply_fold_labels(train, threshold)
    y_validation = apply_fold_labels(validation, threshold)
    status = "VALID" if len(np.unique(y_train)) == 2 and len(np.unique(y_validation)) == 2 else "INVALID"
    train["fold_label"] = y_train
    validation["fold_label"] = y_validation
    return train, validation, threshold, status


def tune_and_validate(final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    candidate_rows = []
    cv_best_rows = []
    best_params = {}
    validation_rows = []
    official_models = {}

    for horizon in [1, 2, 3]:
        horizon_frame = final.loc[final["forecast_horizon"].eq(horizon)].copy()
        for feature_set, columns in FEATURE_SETS.items():
            for model_name, candidates in MODEL_CANDIDATES.items():
                candidate_summaries = []
                candidate_fold_storage = {}
                for candidate_id, params in enumerate(candidates, start=1):
                    fold_rows = []
                    for fold in FOLDS:
                        train, validation, label_threshold, status = build_fold_data(horizon_frame, fold)
                        y_train = train["fold_label"].to_numpy()
                        y_validation = validation["fold_label"].to_numpy()
                        base = {
                            "horizon": horizon,
                            "feature_set": feature_set,
                            "model": model_name,
                            "candidate_id": candidate_id,
                            "hyperparameters": json_dumps(params),
                            **{key: value.date().isoformat() if isinstance(value, pd.Timestamp) else value for key, value in fold.items()},
                            "n_train": int(len(train)),
                            "n_validation": int(len(validation)),
                            "growth_train": int(y_train.sum()),
                            "normal_train": int((y_train == 0).sum()),
                            "growth_validation": int(y_validation.sum()),
                            "normal_validation": int((y_validation == 0).sum()),
                            "growth_label_threshold": label_threshold,
                            "status": status,
                        }
                        if status == "INVALID":
                            fold_rows.append(base)
                            continue
                        pipeline = make_pipeline(model_name, params, y_train)
                        if pipeline is not None:
                            pipeline.fit(train[columns], y_train)
                        probability = predict_probability(model_name, pipeline, validation[columns], y_train)
                        probability_threshold = choose_probability_threshold(y_validation, probability, model_name)
                        metrics = compute_metrics(y_validation, probability, probability_threshold)
                        fold_rows.append(base | metrics)
                    candidate_fold_storage[candidate_id] = fold_rows
                    valid_rows = [row for row in fold_rows if row["status"] == "VALID"]
                    summary = {
                        "horizon": horizon,
                        "feature_set": feature_set,
                        "model": model_name,
                        "candidate_id": candidate_id,
                        "hyperparameters": json_dumps(params),
                        "valid_folds": len(valid_rows),
                        "mean_validation_f1": float(np.mean([row["f1"] for row in valid_rows])) if valid_rows else np.nan,
                        "mean_validation_pr_auc": float(np.mean([row["pr_auc"] for row in valid_rows])) if valid_rows else np.nan,
                        "mean_validation_roc_auc": float(np.mean([row["roc_auc"] for row in valid_rows])) if valid_rows else np.nan,
                        "selection_scope": "temporal_cv_train_validation_only",
                    }
                    candidate_rows.append(summary)
                    candidate_summaries.append(summary)
                valid_candidates = [row for row in candidate_summaries if row["valid_folds"] == len(FOLDS)]
                if not valid_candidates:
                    continue
                chosen = max(
                    valid_candidates,
                    key=lambda row: (row["mean_validation_f1"], row["mean_validation_pr_auc"], -row["candidate_id"]),
                )
                chosen_params = candidates[int(chosen["candidate_id"]) - 1]
                best_params[(horizon, feature_set, model_name)] = chosen_params
                for row in candidate_fold_storage[int(chosen["candidate_id"])]:
                    cv_best_rows.append(row | {"selected_hyperparameters": True})

                official_train = horizon_frame.loc[
                    horizon_frame["date"].between(TRAIN_START, OFFICIAL_TRAIN_END, inclusive="both")
                ].copy()
                official_validation = horizon_frame.loc[
                    horizon_frame["date"].between(OFFICIAL_VALIDATION_START, OFFICIAL_VALIDATION_END, inclusive="both")
                ].copy()
                label_threshold = fold_threshold(official_train)
                y_train = apply_fold_labels(official_train, label_threshold)
                y_validation = apply_fold_labels(official_validation, label_threshold)
                if len(np.unique(y_train)) < 2 or len(np.unique(y_validation)) < 2:
                    validation_rows.append(
                        {
                            "horizon": horizon,
                            "feature_set": feature_set,
                            "model": model_name,
                            "status": "INVALID",
                            "reason": "official train or validation has one class",
                        }
                    )
                    continue
                pipeline = make_pipeline(model_name, chosen_params, y_train)
                if pipeline is not None:
                    pipeline.fit(official_train[columns], y_train)
                probability = predict_probability(model_name, pipeline, official_validation[columns], y_train)
                probability_threshold = choose_probability_threshold(y_validation, probability, model_name)
                metrics = compute_metrics(y_validation, probability, probability_threshold)
                validation_rows.append(
                    {
                        "horizon": horizon,
                        "feature_set": feature_set,
                        "model": model_name,
                        "status": "VALID",
                        "n_train": int(len(official_train)),
                        "n_validation": int(len(official_validation)),
                        "growth_train": int(y_train.sum()),
                        "normal_train": int((y_train == 0).sum()),
                        "growth_validation": int(y_validation.sum()),
                        "normal_validation": int((y_validation == 0).sum()),
                        "growth_label_threshold": label_threshold,
                        "hyperparameters": json_dumps(chosen_params),
                        **metrics,
                    }
                )
                official_models[(horizon, feature_set, model_name)] = {
                    "pipeline": pipeline,
                    "probability_threshold": probability_threshold,
                    "growth_label_threshold": label_threshold,
                }
                print(f"validation complete h={horizon} features={feature_set} model={model_name}", flush=True)

    validation = pd.DataFrame(validation_rows)
    selections = {}
    for horizon in [1, 2, 3]:
        choices = validation.loc[
            validation["horizon"].eq(horizon) & validation["status"].eq("VALID")
        ]
        if choices.empty:
            raise RuntimeError(f"Semua model validation invalid untuk horizon {horizon}")
        selected = choices.sort_values(
            ["f1", "pr_auc", "macro_f1", "model", "feature_set"],
            ascending=[False, False, False, True, True],
        ).iloc[0]
        selections[horizon] = {
            "model": selected["model"],
            "feature_set": selected["feature_set"],
            "hyperparameters": best_params[(horizon, selected["feature_set"], selected["model"])],
            "probability_threshold": float(selected["probability_threshold"]),
            "growth_label_threshold": float(selected["growth_label_threshold"]),
            "selection_metric": "official validation F1 growth; tie-break PR-AUC then Macro-F1",
            "validation_f1": float(selected["f1"]),
            "validation_pr_auc": float(selected["pr_auc"]),
        }
    return pd.DataFrame(candidate_rows), pd.DataFrame(cv_best_rows), validation, selections


def final_test_evaluation(
    final: pd.DataFrame,
    validation: pd.DataFrame,
    best_params: pd.DataFrame,
    selections: dict,
    models_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    results = []
    predictions = []
    fitted_selected = {}
    selected_curve_data = {}
    for row in validation.loc[validation["status"].eq("VALID")].itertuples(index=False):
        horizon = int(row.horizon)
        feature_set = row.feature_set
        model_name = row.model
        selected = (
            selections[horizon]["model"] == model_name
            and selections[horizon]["feature_set"] == feature_set
        )
        if not selected:
            continue
        columns = FEATURE_SETS[feature_set]
        params = json.loads(row.hyperparameters)
        horizon_frame = final.loc[final["forecast_horizon"].eq(horizon)].copy()
        train_validation = horizon_frame.loc[horizon_frame["split"].isin(["train", "validation"])].copy()
        test = horizon_frame.loc[horizon_frame["split"].eq("test")].copy()
        y_train_validation = apply_fold_labels(train_validation, float(row.growth_label_threshold))
        y_test = apply_fold_labels(test, float(row.growth_label_threshold))
        pipeline = make_pipeline(model_name, params, y_train_validation)
        if pipeline is not None:
            pipeline.fit(train_validation[columns], y_train_validation)
        probability = predict_probability(model_name, pipeline, test[columns], y_train_validation)
        metrics = compute_metrics(y_test, probability, float(row.probability_threshold))
        results.append(
            {
                "horizon": horizon,
                "feature_set": feature_set,
                "model": model_name,
                "selected_final_model": selected,
                "n_train_validation": int(len(train_validation)),
                "n_test": int(len(test)),
                "growth_test": int(y_test.sum()),
                "normal_test": int((y_test == 0).sum()),
                "growth_label_threshold": float(row.growth_label_threshold),
                "hyperparameters": row.hyperparameters,
                "validation_precision": float(row.precision),
                "validation_recall": float(row.recall),
                "validation_f1": float(row.f1),
                "validation_macro_f1": float(row.macro_f1),
                "validation_roc_auc": float(row.roc_auc),
                "validation_pr_auc": float(row.pr_auc),
                **metrics,
            }
        )
        selected_predictions = pd.DataFrame(
            {
                "horizon": horizon,
                "topic_id": test["topic_id"].to_numpy(),
                "origin_date": test["date"].to_numpy(),
                "actual_label": y_test,
                "prediction_probability": probability,
                "predicted_label": (probability >= float(row.probability_threshold)).astype("int8"),
                "model": model_name,
                "feature_set": feature_set,
                "probability_threshold": float(row.probability_threshold),
            }
        )
        predictions.append(selected_predictions)
        fitted_selected[horizon] = pipeline
        selected_curve_data[horizon] = {
            "y_true": y_test,
            "probability": probability,
            "threshold": float(row.probability_threshold),
            "features": test[columns],
            "feature_names": columns,
        }
        joblib.dump(
            pipeline,
            models_dir / f"final_model_h{horizon}.joblib",
            compress=3,
        )
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True), fitted_selected, selected_curve_data


def feature_importance_for_selected(
    fitted_selected: dict,
    curve_data: dict,
    selections: dict,
) -> pd.DataFrame:
    rows = []
    for horizon, pipeline in fitted_selected.items():
        if pipeline is None:
            continue
        data = curve_data[horizon]
        y_true = data["y_true"]
        feature_names = data["feature_names"]
        permutation = permutation_importance(
            pipeline,
            data["features"],
            y_true,
            scoring="f1",
            n_repeats=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        estimator = pipeline.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            native = np.asarray(estimator.feature_importances_, dtype=float)
            native_method = "tree_importance"
        elif hasattr(estimator, "coef_"):
            native = np.abs(np.asarray(estimator.coef_[0], dtype=float))
            native_method = "absolute_logistic_coefficient"
        else:
            native = np.full(len(feature_names), np.nan)
            native_method = "not_available"
        for index, feature in enumerate(feature_names):
            rows.append(
                {
                    "horizon": horizon,
                    "model": selections[horizon]["model"],
                    "feature_set": selections[horizon]["feature_set"],
                    "feature": feature,
                    "feature_group": "Semantic" if feature in SEMANTIC_FEATURES else "Temporal",
                    "native_importance": float(native[index]),
                    "native_method": native_method,
                    "permutation_importance_mean_f1": float(permutation.importances_mean[index]),
                    "permutation_importance_std_f1": float(permutation.importances_std[index]),
                }
            )
    return pd.DataFrame(rows)


def create_visualizations(
    validation: pd.DataFrame,
    test_results: pd.DataFrame,
    selections: dict,
    curve_data: dict,
    importance: pd.DataFrame,
    figures: Path,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    subset = validation.loc[
        validation["horizon"].eq(1) & validation["feature_set"].eq("Semantic + Temporal")
    ].sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(subset["model"], subset["f1"], color="#0052CC")
    ax.set_title("Validation F1 Model Comparison (t+1, Fusion)")
    ax.set_ylabel("F1 growth")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(figures / "model_comparison.png", dpi=180)
    plt.close(fig)

    ablation = validation.loc[validation["horizon"].eq(1)].copy()
    pivot = ablation.pivot(index="model", columns="feature_set", values="f1")
    ax = pivot.plot(kind="bar", figsize=(11, 6), color=["#FF8B00", "#36B37E", "#0052CC"])
    ax.set_title("Validation Ablation F1 (t+1)")
    ax.set_ylabel("F1 growth")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Feature set", frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "ablation_comparison.png", dpi=180)
    plt.close(ax.figure)

    selected_results = test_results.loc[test_results["selected_final_model"]].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(selected_results["horizon"], selected_results["f1"], marker="o", label="F1 growth")
    ax.plot(selected_results["horizon"], selected_results["pr_auc"], marker="o", label="PR-AUC")
    ax.set_title("Final Test Performance by Forecast Horizon")
    ax.set_xlabel("Horizon (hari)")
    ax.set_ylabel("Score")
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.savefig(figures / "horizon_comparison.png", dpi=180)
    plt.close(fig)

    data = curve_data[1]
    prediction = (data["probability"] >= data["threshold"]).astype(int)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ConfusionMatrixDisplay(confusion_matrix(data["y_true"], prediction, labels=[0, 1]), display_labels=["Normal", "Growth"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Final Test Confusion Matrix (t+1)")
    fig.savefig(figures / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(data["y_true"], data["probability"])
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.plot(recall, precision, color="#0052CC")
    ax.axhline(np.mean(data["y_true"]), color="#6B778C", linestyle="--", label="Prevalence")
    ax.set_title("Final Test Precision–Recall Curve (t+1)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False)
    fig.savefig(figures / "pr_curve.png", dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(data["y_true"], data["probability"])
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.plot(fpr, tpr, color="#0052CC")
    ax.plot([0, 1], [0, 1], color="#6B778C", linestyle="--")
    ax.set_title("Final Test ROC Curve (t+1)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    fig.savefig(figures / "roc_curve.png", dpi=180)
    plt.close(fig)

    imp = importance.loc[importance["horizon"].eq(1)].nlargest(15, "permutation_importance_mean_f1").sort_values("permutation_importance_mean_f1")
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    colors = imp["feature_group"].map({"Semantic": "#0052CC", "Temporal": "#FF8B00"})
    ax.barh(imp["feature"], imp["permutation_importance_mean_f1"], xerr=imp["permutation_importance_std_f1"], color=colors)
    ax.set_title("Permutation Importance of Selected Final Model (t+1)")
    ax.set_xlabel("Mean decrease in F1")
    fig.savefig(figures / "feature_importance.png", dpi=180)
    plt.close(fig)

    selected_validation = []
    for horizon, selection in selections.items():
        row = validation.loc[
            validation["horizon"].eq(horizon)
            & validation["model"].eq(selection["model"])
            & validation["feature_set"].eq(selection["feature_set"])
        ].iloc[0]
        test_row = test_results.loc[
            test_results["horizon"].eq(horizon) & test_results["selected_final_model"]
        ].iloc[0]
        selected_validation.append({"horizon": horizon, "Validation": row["f1"], "Test": test_row["f1"]})
    shift_plot = pd.DataFrame(selected_validation).set_index("horizon")
    ax = shift_plot.plot(kind="bar", figsize=(9, 5), color=["#FF8B00", "#36B37E"])
    ax.set_title("Selected Model F1: Validation vs Test")
    ax.set_xlabel("Horizon (hari)")
    ax.set_ylabel("F1 growth")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "validation_vs_test_performance.png", dpi=180)
    plt.close(ax.figure)


def build_modeling_report(
    summary: pd.DataFrame,
    cv: pd.DataFrame,
    validation: pd.DataFrame,
    test_results: pd.DataFrame,
    importance: pd.DataFrame,
    selections: dict,
    scheme: LabelScheme,
) -> str:
    selected_test = test_results.loc[test_results["selected_final_model"]].sort_values("horizon")
    test_table = selected_test[[
        "horizon", "feature_set", "model", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc"
    ]]
    t1 = validation.loc[validation["horizon"].eq(1)]
    best_by_feature = (
        t1.sort_values(["feature_set", "f1", "pr_auc"], ascending=[True, False, False])
        .groupby("feature_set", as_index=False)
        .first()[["feature_set", "model", "f1", "pr_auc"]]
    )
    valid_folds = int(cv.loc[cv["status"].eq("VALID"), "fold"].nunique())
    shift_rows = []
    for row in selected_test.itertuples(index=False):
        shift_rows.append(
            {
                "horizon": row.horizon,
                "validation_f1": row.validation_f1,
                "test_f1": row.f1,
                "f1_change": row.f1 - row.validation_f1,
                "validation_pr_auc": row.validation_pr_auc,
                "test_pr_auc": row.pr_auc,
                "pr_auc_change": row.pr_auc - row.validation_pr_auc,
            }
        )
    shift_performance = pd.DataFrame(shift_rows)

    learner_only = validation.loc[validation["model"].ne("Majority Class")]
    best_f1 = (
        learner_only.groupby(["horizon", "feature_set"])["f1"].max().unstack("feature_set")
    )
    majority_f1 = (
        validation.loc[validation["model"].eq("Majority Class")]
        .groupby("horizon")["f1"]
        .max()
    )

    def formatted(series: pd.Series) -> str:
        return " / ".join(f"{float(series.get(horizon, np.nan)):.4f}" for horizon in [1, 2, 3])

    semantic_f1 = best_f1.get("Semantic Only", pd.Series(dtype=float))
    temporal_f1 = best_f1.get("Temporal Only", pd.Series(dtype=float))
    fusion_f1 = best_f1.get("Semantic + Temporal", pd.Series(dtype=float))
    semantic_above_majority = int(
        sum(
            1
            for horizon in [1, 2, 3]
            if float(semantic_f1.get(horizon, np.nan)) > float(majority_f1.get(horizon, np.nan))
        )
    )
    temporal_above_semantic = int(
        sum(
            1
            for horizon in [1, 2, 3]
            if float(temporal_f1.get(horizon, np.nan)) > float(semantic_f1.get(horizon, np.nan))
        )
    )
    fusion_above_temporal = int(
        sum(
            1
            for horizon in [1, 2, 3]
            if float(fusion_f1.get(horizon, np.nan)) > float(temporal_f1.get(horizon, np.nan))
        )
    )
    selected_feature_sets = {
        horizon: selections[horizon]["feature_set"] for horizon in sorted(selections)
    }
    test_by_horizon = selected_test.set_index("horizon")
    f1_series = test_by_horizon["f1"]
    pr_series = test_by_horizon["pr_auc"]
    f1_direction = (
        "menurun monoton"
        if f1_series.is_monotonic_decreasing
        else "meningkat monoton"
        if f1_series.is_monotonic_increasing
        else "tidak monoton"
    )
    pr_direction = (
        "menurun monoton"
        if pr_series.is_monotonic_decreasing
        else "meningkat monoton"
        if pr_series.is_monotonic_increasing
        else "tidak monoton"
    )
    prevalence = (
        summary.loc[summary["record_type"].eq("partition")]
        .assign(rate=lambda frame: frame["growth"] / frame["rows"])
        .pivot(index="horizon", columns="split", values="rate")
    )

    return f"""# Modeling report

## 0. Definisi label

Label growth memakai skema **{scheme.key}** ({scheme.name}).

- Definisi: `{scheme.definition}`
- Jenis threshold: `{scheme.threshold_kind}`
- Populasi: `{scheme.population}`
- Alasan pemilihan: {scheme.rationale}

Positive rate label per split (dari partisi resmi):

{prevalence.to_markdown(floatfmt='.4f')}

## 1. Feature dataset

Final dataset menggabungkan semantic, temporal, dan label audit dengan key `(topic_id, date, forecast_horizon)`. Tidak ada duplicate key. Target, future volume, target date, growth score, burst score, dan source ID tidak digunakan sebagai predictor.

{summary.loc[summary['record_type'].eq('partition')].to_markdown(index=False)}

## 2. Temporal folds

Digunakan tiga expanding-window fold dengan validation block yang tidak tumpang tindih. Jumlah fold valid: **{valid_folds}/3**. Test tidak digunakan untuk tuning.

{cv[['fold','horizon','train_start','train_end','validation_start','validation_end','n_train','n_validation','growth_train','normal_train','growth_validation','normal_validation','status']].drop_duplicates().to_markdown(index=False)}

## 3. Model configuration dan 4. hyperparameters

Model: Majority Class, Logistic Regression, Random Forest, dan XGBoost. Setiap learner diuji dengan Temporal Only, Semantic Only, dan Semantic + Temporal. Logistic memakai median imputation dan StandardScaler; tree model memakai median imputation. Class weighting diterapkan hanya saat fitting training. Tidak ada SMOTE.

Konfigurasi final per horizon dibekukan dari official validation:

```json
{json.dumps(selections, ensure_ascii=False, indent=2)}
```

## 5. Validation performance

{validation[['horizon','feature_set','model','precision','recall','f1','macro_f1','roc_auc','pr_auc','probability_threshold']].to_markdown(index=False)}

## 6. Ablation

Perbandingan t+1 terbaik per feature set pada official validation (test tidak dipakai untuk memilih ablation):

{best_by_feature.to_markdown(index=False)}

## 7. Multi-horizon dan 8. final test

{test_table.to_markdown(index=False)}

## 9. Distribution shift

Mean article/day berubah dari 125,55 pada training menjadi 244,64 pada test. Perubahan validation ke test untuk model terpilih:

{shift_performance.to_markdown(index=False)}

Perubahan ini dilaporkan sebagai temporal distribution shift; penyebabnya tidak disimpulkan.

## 10. Feature importance

Feature importance menyatakan predictive importance, bukan hubungan sebab-akibat. Tabel lengkap memuat native importance dan permutation importance.

{importance.loc[importance['horizon'].eq(1)].sort_values('permutation_importance_mean_f1', ascending=False).head(15).to_markdown(index=False)}

## 11. Statistical comparison

Statistical significance was not reliably assessed because the number of independent temporal evaluation origins was limited.

## 12. Limitations

- Embedding menggunakan batas 128 token; 99,41% dokumen terkena right truncation.
- BERTopic dan forecasting readiness sebelumnya berstatus LIMITED.
- Outlier BERTopic sekitar 39% pada training dan 48% pada validation/test.
- Test mengalami perubahan volume dan komposisi sumber yang besar.
- Semantic feature missing pada hari ketika suatu topik tidak memiliki dokumen dan diimputasi berdasarkan training.
- Hanya tiga temporal folds tersedia; hasil antar-feature-set belum mendukung klaim signifikansi statistik.

## 13. RQ1 — Semantic features

F1 validation terbaik Semantic Only untuk t+1/t+2/t+3: **{formatted(semantic_f1)}**; majority baseline pada horizon yang sama: **{formatted(majority_f1)}**. Semantic Only berada di atas majority baseline pada **{semantic_above_majority}/3** horizon. Angka ini menyatakan predictive signal, bukan klaim kausal.

## 14. RQ2 — Temporal features

F1 validation terbaik Temporal Only untuk t+1/t+2/t+3: **{formatted(temporal_f1)}**. Dibanding Semantic Only pada split dan kandidat learner yang sama, Temporal Only lebih tinggi pada **{temporal_above_semantic}/3** horizon.

## 15. RQ3 — Fusion

F1 validation terbaik Semantic + Temporal untuk t+1/t+2/t+3: **{formatted(fusion_f1)}**. Fusion melampaui Temporal Only pada **{fusion_above_temporal}/3** horizon. Feature set yang terpilih per horizon: {selected_feature_sets}. Karena hanya tiga fold, perbedaan antar feature set tidak dinyatakan signifikan.

## 16. RQ4 — Forecast horizon

Pada final test, F1 model terpilih adalah **{selected_test.set_index('horizon').loc[1, 'f1']:.4f}** (t+1), **{selected_test.set_index('horizon').loc[2, 'f1']:.4f}** (t+2), dan **{selected_test.set_index('horizon').loc[3, 'f1']:.4f}** (t+3) — {f1_direction} terhadap horizon. PR-AUC adalah **{selected_test.set_index('horizon').loc[1, 'pr_auc']:.4f}**, **{selected_test.set_index('horizon').loc[2, 'pr_auc']:.4f}**, dan **{selected_test.set_index('horizon').loc[3, 'pr_auc']:.4f}** — {pr_direction}. Tidak ada horizon yang dicampur saat fitting.

## Status penelitian

**LIMITED** — eksperimen dapat dievaluasi secara temporal dan bebas leakage yang terdeteksi, tetapi distribution shift, truncation, topic outlier, serta jumlah fold membatasi generalisasi hasil.
"""


def scheme_output_paths(root: Path, scheme: LabelScheme) -> dict[str, Path]:
    """Output canonical untuk skema default; subfolder terpisah untuk skema lain."""
    if ACTIVE_TOPIC_SUBSET:
        base = root / "results" / "subset_runs" / ACTIVE_TOPIC_SUBSET
        return {
            "tables": base / "tables",
            "figures": base / "figures",
            "predictions": base / "predictions",
            "models": root / "models" / "subset_runs" / ACTIVE_TOPIC_SUBSET,
            "report": base / "modeling_report.md",
            "metadata": base / "modeling_metadata.json",
            "frozen": base / "frozen_model_selection.json",
            "dataset": base / "final_feature_dataset.parquet",
        }
    if ACTIVE_EMBEDDING_VARIANT:
        base = (
            root
            / "results"
            / "embedding_runs"
            / f"{ACTIVE_EMBEDDING_VARIANT}__{ACTIVE_PILOT_TAG}"
        )
        return {
            "tables": base / "tables",
            "figures": base / "figures",
            "predictions": base / "predictions",
            "models": root / "models" / "embedding_runs" / f"{ACTIVE_EMBEDDING_VARIANT}__{ACTIVE_PILOT_TAG}",
            "report": base / "modeling_report.md",
            "metadata": base / "modeling_metadata.json",
            "frozen": base / "frozen_model_selection.json",
            "dataset": base / "final_feature_dataset.parquet",
        }
    if scheme.key == DEFAULT_SCHEME_KEY:
        base = root / "results"
        return {
            "tables": base / "tables",
            "figures": base / "figures",
            "predictions": base / "predictions",
            "models": root / "models",
            "report": base / "modeling_report.md",
            "metadata": base / "modeling_metadata.json",
            "frozen": base / "frozen_model_selection.json",
            "dataset": root / "data" / "processed" / "final_feature_dataset.parquet",
        }
    base = root / "results" / "scheme_runs" / scheme.key
    return {
        "tables": base / "tables",
        "figures": base / "figures",
        "predictions": base / "predictions",
        "models": root / "models" / "scheme_runs" / scheme.key,
        "report": base / "modeling_report.md",
        "metadata": base / "modeling_metadata.json",
        "frozen": base / "frozen_model_selection.json",
        "dataset": base / "final_feature_dataset.parquet",
    }


def main(scheme_key: str | None = None) -> None:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    scheme = get_scheme(scheme_key)
    set_active_scheme(scheme)
    paths = scheme_output_paths(root, scheme)
    tables = paths["tables"]
    figures = paths["figures"]
    predictions_dir = paths["predictions"]
    models_dir = paths["models"]
    for directory in [tables, figures, predictions_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"label scheme aktif: {scheme.key} ({scheme.definition})", flush=True)
    if ACTIVE_EMBEDDING_VARIANT:
        print(
            f"varian embedding aktif: {ACTIVE_EMBEDDING_VARIANT} (subset {ACTIVE_PILOT_TAG}); "
            "topic assignment dan label dikunci dari run penuh",
            flush=True,
        )

    final, summary = prepare_final_feature_dataset(root)
    final.to_parquet(paths["dataset"], index=False, compression="zstd")
    summary.to_csv(tables / "final_feature_dataset_summary.csv", index=False)

    hyperparameters, cv, validation, selections = tune_and_validate(final)
    hyperparameters.to_csv(tables / "model_hyperparameters.csv", index=False)
    cv.to_csv(tables / "temporal_cv_results.csv", index=False)
    validation.to_csv(tables / "model_validation_results.csv", index=False)
    frozen_path = paths["frozen"]
    normalized_selections = json.loads(json.dumps(selections, ensure_ascii=False))
    frozen_payload = {"label_scheme": scheme.key, "selections": normalized_selections}
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        frozen_scheme = frozen.get("label_scheme") if isinstance(frozen, dict) else None
        if frozen_scheme != scheme.key:
            raise RuntimeError(
                f"Frozen model selection pada {frozen_path} dibuat untuk skema label "
                f"{frozen_scheme!r}, bukan {scheme.key!r}. Pindahkan berkas lama secara "
                "eksplisit sebelum menjalankan skema berbeda; test evaluation dihentikan."
            )
        if frozen.get("selections") != normalized_selections:
            raise RuntimeError("Frozen model selection berubah; test evaluation dihentikan")
    else:
        frozen_path.write_text(
            json.dumps(frozen_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    test_results, predictions, fitted_selected, curve_data = final_test_evaluation(
        final, validation, hyperparameters, selections, models_dir
    )
    test_results.to_csv(tables / "final_test_results.csv", index=False)
    test_results.to_csv(tables / "horizon_results.csv", index=False)
    predictions.to_csv(predictions_dir / "final_test_predictions.csv", index=False)

    ablation = validation[[
        "horizon", "model", "feature_set", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc"
    ]].copy()
    ablation["selected_final_model"] = [
        selections[int(row.horizon)]["model"] == row.model
        and selections[int(row.horizon)]["feature_set"] == row.feature_set
        for row in ablation.itertuples(index=False)
    ]
    ablation.to_csv(tables / "ablation_results.csv", index=False)
    importance = feature_importance_for_selected(fitted_selected, curve_data, selections)
    importance.to_csv(tables / "feature_importance.csv", index=False)
    create_visualizations(validation, test_results, selections, curve_data, importance, figures)

    report = build_modeling_report(summary, cv, validation, test_results, importance, selections, scheme)
    paths["report"].write_text(report, encoding="utf-8")
    metadata = {
        "status": "LIMITED",
        "processing_seconds": time.perf_counter() - started,
        "random_seed": RANDOM_SEED,
        "label_scheme": scheme.key,
        "label_scheme_definition": scheme.definition,
        "label_threshold_kind": scheme.threshold_kind,
        "embedding_variant": ACTIVE_EMBEDDING_VARIANT or "canonical_full_run",
        "pilot_tag": ACTIVE_PILOT_TAG if ACTIVE_EMBEDDING_VARIANT else None,
        "topic_subset": ACTIVE_TOPIC_SUBSET or "all_topics",
        "feature_sets": FEATURE_SETS,
        "folds": [
            {key: value.isoformat() if isinstance(value, pd.Timestamp) else value for key, value in fold.items()}
            for fold in FOLDS
        ],
        "model_candidates": MODEL_CANDIDATES,
        "selection": selections,
        "test_access_rule": "test evaluated only after frozen_model_selection.json was written",
        "statistical_significance": "not reliably assessed; three independent temporal validation blocks",
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console_encoding = sys.stdout.encoding or "utf-8"
    print(report.encode(console_encoding, errors="replace").decode(console_encoding))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

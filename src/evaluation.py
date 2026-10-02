from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from .config import MODEL_DIR, RANDOM_SEED, RESULTS_DIR
from .models import available_models, make_pipeline


def metrics(y_true, y_pred, y_prob):
    classes = np.unique(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_positive": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_positive": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_positive": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(classes) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(classes) == 2 else np.nan,
        "confusion_matrix": json.dumps(confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()),
    }


def tune_threshold(y_true, probabilities):
    if len(np.unique(y_true)) < 2:
        return 0.5
    candidates = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y_true, probabilities >= x, zero_division=0) for x in candidates]
    return float(candidates[int(np.argmax(scores))])


def train_models(train, validation, test, feature_columns, feature_set, horizon):
    rows, fitted = [], {}
    y_train, y_val, y_test = train["actual_label"], validation["actual_label"], test["actual_label"]
    if y_train.nunique() < 2:
        return pd.DataFrame([{"model": "all", "feature_set": feature_set, "forecast_horizon": horizon,
                              "status": "not evaluated: training split contains one class"}]), {}, None
    if y_val.nunique() < 2 or y_test.nunique() < 2:
        reason = (f"not evaluated: validation/test split must contain two classes "
                  f"(validation={y_val.value_counts().to_dict()}, test={y_test.value_counts().to_dict()})")
        return pd.DataFrame([{"model": "all", "feature_set": feature_set, "forecast_horizon": horizon,
                              "status": reason}]), {}, None
    for name, estimator in available_models().items():
        start = time.perf_counter()
        pipe = make_pipeline(estimator, scale=name == "logistic_regression")
        pipe.fit(train[feature_columns], y_train)
        val_prob = pipe.predict_proba(validation[feature_columns])[:, 1] if hasattr(pipe, "predict_proba") and len(pipe.classes_) == 2 else pipe.predict(validation[feature_columns]).astype(float)
        threshold = 0.5 if name == "majority_class" else tune_threshold(y_val, val_prob)
        val_prediction = pipe.predict(validation[feature_columns]) if name == "majority_class" else (val_prob >= threshold).astype(int)
        validation_metrics = metrics(y_val, val_prediction, val_prob)
        test_prob = pipe.predict_proba(test[feature_columns])[:, 1] if hasattr(pipe, "predict_proba") and len(pipe.classes_) == 2 else pipe.predict(test[feature_columns]).astype(float)
        prediction = pipe.predict(test[feature_columns]) if name == "majority_class" else (test_prob >= threshold).astype(int)
        rows.append({"model": name, "feature_set": feature_set, "forecast_horizon": horizon,
                     "decision_threshold": threshold, "training_seconds": time.perf_counter() - start,
                     "hyperparameters": json.dumps(estimator.get_params(), default=str, sort_keys=True),
                     "validation_f1_positive": validation_metrics["f1_positive"],
                     "validation_pr_auc": validation_metrics["pr_auc"],
                     "status": "evaluated", **metrics(y_test, prediction, test_prob)})
        fitted[name] = (pipe, threshold, test_prob, prediction)
    return pd.DataFrame(rows), fitted, y_test


def choose_and_explain(results, fitted, test, feature_columns):
    eligible = results[(results["model"] != "majority_class") & results["validation_pr_auc"].notna()]
    if eligible.empty:
        eligible = results[results["model"] != "majority_class"]
    best_name = eligible.sort_values(["validation_pr_auc", "validation_f1_positive"], ascending=False, na_position="last").iloc[0]["model"]
    pipe, threshold, probabilities, predictions = fitted[best_name]
    joblib.dump(pipe, MODEL_DIR / "best_classifier.joblib")
    importance = permutation_importance(pipe, test[feature_columns], test["actual_label"],
                                        scoring="f1", n_repeats=10, random_state=RANDOM_SEED)
    importance_frame = pd.DataFrame({"feature": feature_columns,
                                     "importance_mean": importance.importances_mean,
                                     "importance_std": importance.importances_std})
    predictions_frame = test[["date", "topic_id", "actual_label", "forecast_horizon"]].copy()
    predictions_frame["predicted_label"] = predictions
    predictions_frame["prediction_probability"] = probabilities
    return best_name, threshold, importance_frame.sort_values("importance_mean", ascending=False), predictions_frame


def expanding_window_scores(frame, feature_columns, estimator_factory, min_train_periods=8, folds=3):
    periods = np.array(sorted(frame["date"].unique()))
    if len(periods) < min_train_periods + folds:
        return []
    origins = np.linspace(min_train_periods, len(periods) - 1, folds, dtype=int)
    scores = []
    for origin in sorted(set(origins)):
        tr = frame[frame["date"] < periods[origin]]
        va = frame[frame["date"] == periods[origin]]
        if tr["actual_label"].nunique() < 2 or va.empty:
            continue
        pipe = estimator_factory()
        pipe.fit(tr[feature_columns], tr["actual_label"])
        pred = pipe.predict(va[feature_columns])
        scores.append(float(f1_score(va["actual_label"], pred, zero_division=0)))
    return scores

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from .config import RESULTS_DIR


FIG = RESULTS_DIR / "figures"


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close()


def message(name, title, detail):
    plt.figure(figsize=(8, 4)); plt.axis("off"); plt.title(title); plt.text(.5, .5, detail, ha="center", va="center", wrap=True); save(name)


def make_figures(docs, topic_series, model_results, ablation_results, importance, predictions):
    plt.figure(figsize=(10, 4)); docs.groupby("date").size().plot(); plt.title("Distribusi artikel menurut waktu"); plt.ylabel("Artikel"); save("01_articles_over_time.png")
    plt.figure(figsize=(8, 4)); docs["source"].value_counts().plot(kind="bar"); plt.title("Distribusi artikel menurut source"); save("02_articles_by_source.png")
    plt.figure(figsize=(8, 4)); docs.loc[docs.topic_id >= 0, "topic_id"].value_counts().sort_index().plot(kind="bar"); plt.title("Topic frequency"); save("03_topic_frequency.png")
    pivot = topic_series.pivot(index="date", columns="topic_id", values="topic_volume")
    plt.figure(figsize=(11, 5)); pivot.plot(ax=plt.gca(), legend=False); plt.title("Topic evolution"); save("04_topic_evolution.png")
    plt.figure(figsize=(12, 6)); plt.imshow(pivot.T, aspect="auto", cmap="YlOrRd"); plt.colorbar(label="Volume"); plt.ylabel("Topic"); plt.xlabel("Date index"); plt.title("Heatmap topic × date"); save("05_topic_date_heatmap.png")
    growth = pivot.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    plt.figure(figsize=(11, 5)); growth.plot(ax=plt.gca(), legend=False); plt.title("Growth rate tiap topic"); save("06_topic_growth_rate.png")
    if predictions is not None and not predictions.empty:
        cm = confusion_matrix(predictions.actual_label, predictions.predicted_label, labels=[0, 1])
        plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap="Blues"); plt.colorbar();
        for (i, j), value in np.ndenumerate(cm): plt.text(j, i, value, ha="center", va="center")
        plt.xticks([0,1]); plt.yticks([0,1]); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix"); save("07_confusion_matrix.png")
    else: message("07_confusion_matrix.png", "Confusion matrix", "not evaluated: insufficient temporal history")
    evaluated = model_results[model_results.get("status", "") == "evaluated"] if not model_results.empty else model_results
    if not evaluated.empty and "f1_positive" in evaluated:
        plt.figure(figsize=(9, 4)); evaluated.set_index("model")["f1_positive"].plot(kind="bar"); plt.title("Model comparison (F1 positive)"); save("08_model_comparison.png")
    else: message("08_model_comparison.png", "Model comparison", "not evaluated")
    if not ablation_results.empty and "f1_positive" in ablation_results:
        plt.figure(figsize=(8, 4)); ablation_results.set_index("feature_set")["f1_positive"].plot(kind="bar"); plt.title("Ablation comparison"); save("09_ablation_comparison.png")
    else: message("09_ablation_comparison.png", "Ablation", "not evaluated")
    if importance is not None and not importance.empty:
        top = importance.head(15).sort_values("importance_mean"); plt.figure(figsize=(9, 6)); plt.barh(top.feature, top.importance_mean); plt.title("Permutation importance (predictive contribution)"); save("10_feature_importance.png")
    else: message("10_feature_importance.png", "Feature importance", "not evaluated")
    if predictions is not None and predictions.actual_label.nunique() == 2:
        PrecisionRecallDisplay.from_predictions(predictions.actual_label, predictions.prediction_probability); plt.title("Precision–Recall curve"); save("11_pr_curve.png")
        RocCurveDisplay.from_predictions(predictions.actual_label, predictions.prediction_probability); plt.title("ROC curve"); save("12_roc_curve.png")
    else:
        message("11_pr_curve.png", "Precision–Recall curve", "not evaluated: test split has one class")
        message("12_roc_curve.png", "ROC curve", "not evaluated: test split has one class")


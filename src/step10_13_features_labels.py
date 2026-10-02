from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .label_schemes import (
    BURST_SCALE_FLOOR,
    COMPARISON_ORDER,
    MAD_SCALE,
    PRIMARY_GROWTH_QUANTILE,
    SCHEMES,
    burst_score_values,
    class_balance,
    fit_threshold,
    get_scheme,
    population_mask,
)
from .step7_embeddings import file_sha256
from .step8_9_topic_modeling import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    split_masks,
    write_json,
)


MIN_ACTIVE_TRAINING_DAYS = 14
MIN_CALENDAR_HISTORY = 14
SPARSITY_LOW_MAX = 0.50
SPARSITY_MEDIUM_MAX = 0.80
GROWTH_QUANTILES = [0.75, 0.80, 0.85, 0.90]
EPSILON = 1e-12
SPARSITY_ORDER = ["LOW SPARSITY", "MEDIUM SPARSITY", "HIGH SPARSITY"]
HISTOGRAM_BINS = [
    ("<= -5", -np.inf, -5.0),
    ("(-5, -2]", -5.0, -2.0),
    ("(-2, -1]", -2.0, -1.0),
    ("(-1, 0)", -1.0, 0.0),
    ("== 0", 0.0, 0.0),
    ("(0, 1]", 0.0, 1.0),
    ("(1, 2]", 1.0, 2.0),
    ("(2, 3]", 2.0, 3.0),
    ("(3, 5]", 3.0, 5.0),
    ("(5, 10]", 5.0, 10.0),
    ("> 10", 10.0, np.inf),
]

SEMANTIC_FEATURE_COLUMNS = [
    "semantic_similarity_mean",
    "semantic_similarity_std",
    "semantic_drift",
    "topic_probability_mean",
    "topic_probability_std",
    "topic_probability_change",
    "similarity_change",
]
TEMPORAL_FEATURE_COLUMNS = [
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
    "zero_baseline_flag",
]
FORBIDDEN_FEATURE_COLUMNS = {
    "future_volume",
    "target_volume",
    "target_date",
    "growth_score",
    "burst_score",
    "label",
    "actual_label",
    "forecast_horizon",
}


def split_name_for_dates(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates).dt.normalize()
    result = pd.Series(pd.NA, index=dates.index, dtype="object")
    result.loc[dates.between(TRAIN_START, TRAIN_END, inclusive="both")] = "train"
    result.loc[dates.between(VALIDATION_START, VALIDATION_END, inclusive="both")] = "validation"
    result.loc[dates.between(TEST_START, TEST_END, inclusive="both")] = "test"
    if result.isna().any():
        raise RuntimeError(f"Tanggal di luar split: {dates.loc[result.isna()].unique()[:5]}")
    return result


def sparsity_category(zero_ratio: float) -> str:
    if zero_ratio < SPARSITY_LOW_MAX:
        return "LOW SPARSITY"
    if zero_ratio < SPARSITY_MEDIUM_MAX:
        return "MEDIUM SPARSITY"
    return "HIGH SPARSITY"


def build_topic_daily_volume(assignments: pd.DataFrame, topics: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range(TRAIN_START, TEST_END, freq="D")
    grid = pd.MultiIndex.from_product([topics, dates], names=["topic_id", "date"])
    non_outlier = assignments.loc[assignments["topic_id"].ne(-1)].copy()
    observed = non_outlier.groupby(["topic_id", "date"], observed=True).size().rename("topic_volume")
    daily = observed.reindex(grid, fill_value=0).reset_index()
    daily["topic_volume"] = daily["topic_volume"].astype("int32")
    daily["split"] = split_name_for_dates(daily["date"])

    outlier = assignments.loc[assignments["topic_id"].eq(-1)].groupby("date").size().rename("outlier_volume")
    outlier_daily = pd.DataFrame({"date": dates}).merge(outlier.reset_index(), on="date", how="left")
    outlier_daily["outlier_volume"] = outlier_daily["outlier_volume"].fillna(0).astype("int32")
    outlier_daily["split"] = split_name_for_dates(outlier_daily["date"])
    return daily, outlier_daily


def initial_topic_history(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train = daily.loc[daily["split"].eq("train")]
    for topic, group in daily.groupby("topic_id", sort=True):
        active = group.loc[group["topic_volume"].gt(0)]
        train_topic = train.loc[train["topic_id"].eq(topic)]
        train_active = train_topic.loc[train_topic["topic_volume"].gt(0)]
        zero_days = int(group["topic_volume"].eq(0).sum())
        zero_ratio = zero_days / len(group)
        rows.append(
            {
                "topic_id": int(topic),
                "first_date": active["date"].min(),
                "last_date": active["date"].max(),
                "active_days": int(len(active)),
                "total_documents": int(group["topic_volume"].sum()),
                "mean_daily_volume": float(group["topic_volume"].mean()),
                "median_daily_volume": float(group["topic_volume"].median()),
                "max_daily_volume": int(group["topic_volume"].max()),
                "zero_days": zero_days,
                "zero_volume_ratio": zero_ratio,
                "sparsity_category": sparsity_category(zero_ratio),
                "training_active_days": int(len(train_active)),
                "training_documents": int(train_topic["topic_volume"].sum()),
                "forecast_eligible": bool(len(train_active) >= MIN_ACTIVE_TRAINING_DAYS),
                "minimum_training_active_days": MIN_ACTIVE_TRAINING_DAYS,
            }
        )
    return pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)


def build_semantic_features(
    assignments: pd.DataFrame,
    embeddings: np.ndarray,
    daily: pd.DataFrame,
    topics: list[int],
) -> pd.DataFrame:
    active_rows = []
    assignment_topics = assignments["topic_id"].to_numpy()
    assignment_dates = pd.to_datetime(assignments["date"]).dt.normalize().to_numpy()
    probabilities = assignments["topic_probability"].to_numpy(dtype=np.float32)

    for position, topic in enumerate(topics, start=1):
        topic_indices = np.flatnonzero(assignment_topics == topic)
        topic_frame = pd.DataFrame(
            {"row_index": topic_indices, "date": pd.to_datetime(assignment_dates[topic_indices])}
        ).sort_values(["date", "row_index"])
        cumulative_sum = np.zeros(embeddings.shape[1], dtype=np.float64)
        cumulative_count = 0
        previous_centroid = None
        for current_date, date_group in topic_frame.groupby("date", sort=True):
            indices = date_group["row_index"].to_numpy(dtype=np.int64)
            vectors = np.asarray(embeddings[indices], dtype=np.float32)
            cumulative_sum += vectors.sum(axis=0, dtype=np.float64)
            cumulative_count += len(indices)
            centroid = cumulative_sum / cumulative_count
            centroid /= max(float(np.linalg.norm(centroid)), EPSILON)
            similarities = vectors @ centroid.astype(np.float32)
            drift = (
                np.nan
                if previous_centroid is None
                else 1.0 - float(np.clip(centroid @ previous_centroid, -1.0, 1.0))
            )
            current_probabilities = probabilities[indices]
            active_rows.append(
                {
                    "topic_id": int(topic),
                    "date": pd.Timestamp(current_date),
                    "semantic_similarity_mean": float(similarities.mean()),
                    "semantic_similarity_std": float(similarities.std(ddof=0)),
                    "semantic_drift": drift,
                    "topic_probability_mean": float(current_probabilities.mean()),
                    "topic_probability_std": float(current_probabilities.std(ddof=0)),
                    "semantic_document_count": int(len(indices)),
                    "semantic_history_document_count": int(cumulative_count),
                    "semantic_history_cutoff": pd.Timestamp(current_date),
                }
            )
            previous_centroid = centroid.copy()
        if position % 25 == 0 or position == len(topics):
            print(f"semantic feature progress topics={position}/{len(topics)}", flush=True)

    active_features = pd.DataFrame(active_rows)
    output = daily[["topic_id", "date", "split"]].merge(
        active_features, on=["topic_id", "date"], how="left", validate="one_to_one"
    )
    for column, change_column in [
        ("topic_probability_mean", "topic_probability_change"),
        ("semantic_similarity_mean", "similarity_change"),
    ]:
        previous_available = output.groupby("topic_id", sort=False)[column].transform(
            lambda values: values.ffill().shift(1)
        )
        output[change_column] = output[column] - previous_available
    output["feature_timestamp"] = output["date"]
    return output


def build_temporal_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = daily[["topic_id", "date", "split", "topic_volume"]].copy()
    output = output.sort_values(["topic_id", "date"]).reset_index(drop=True)
    grouped = output.groupby("topic_id", sort=False)["topic_volume"]
    output["current_volume"] = output["topic_volume"].astype(float)
    for lag in range(1, 5):
        output[f"lag_{lag}"] = grouped.shift(lag)
    output["rolling_mean_3"] = grouped.transform(lambda values: values.shift(1).rolling(3, min_periods=3).mean())
    output["rolling_mean_7"] = grouped.transform(lambda values: values.shift(1).rolling(7, min_periods=7).mean())
    output["rolling_std_7"] = grouped.transform(lambda values: values.shift(1).rolling(7, min_periods=7).std(ddof=0))
    output["zero_baseline_flag"] = output["lag_1"].eq(0) & output["lag_1"].notna()
    raw_delta = output["current_volume"] - output["lag_1"]
    output["growth_rate"] = raw_delta / (output["lag_1"].abs() + 1.0)
    output["percentage_change"] = raw_delta / output["lag_1"].abs().clip(lower=1.0)
    output["momentum"] = raw_delta - (output["lag_1"] - output["lag_2"])
    output["burst_intensity"] = (
        output["current_volume"] - output["rolling_mean_7"]
    ) / (1.0 + output["rolling_std_7"])
    output["feature_timestamp"] = output["date"]
    output["retrospective_history_cutoff"] = output["date"] - pd.Timedelta(days=1)

    raw_percentage = raw_delta / output["lag_1"]
    audit = pd.DataFrame(
        [
            {"metric": "total_topic_days", "value": int(len(output))},
            {"metric": "zero_baseline_count", "value": int(output["zero_baseline_flag"].sum())},
            {"metric": "raw_percentage_infinite_candidates", "value": int(np.isinf(raw_percentage).sum())},
            {"metric": "growth_rate_nan", "value": int(output["growth_rate"].isna().sum())},
            {"metric": "growth_rate_infinite", "value": int(np.isinf(output["growth_rate"]).sum())},
            {"metric": "growth_rate_finite", "value": int(np.isfinite(output["growth_rate"]).sum())},
            {"metric": "percentage_change_nan", "value": int(output["percentage_change"].isna().sum())},
            {"metric": "percentage_change_infinite", "value": int(np.isinf(output["percentage_change"]).sum())},
            {"metric": "percentage_change_finite", "value": int(np.isfinite(output["percentage_change"]).sum())},
        ]
    )
    return output.drop(columns=["topic_volume"]), audit


def build_growth_labels(
    daily: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scheme = get_scheme()
    eligible_topics = set(history.loc[history["forecast_eligible"], "topic_id"].astype(int))
    all_rows = []
    for topic, group in daily.groupby("topic_id", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        volumes = group["topic_volume"].to_numpy(dtype=float)
        dates = pd.to_datetime(group["date"])
        positive_positions = np.flatnonzero(volumes > 0)
        first_active_position = int(positive_positions[0]) if len(positive_positions) else len(group)
        for horizon in [1, 2, 3]:
            for origin_position in range(len(group) - horizon):
                target_position = origin_position + horizon
                origin_date = dates.iloc[origin_position]
                target_date = dates.iloc[target_position]
                origin_split = group.loc[origin_position, "split"]
                target_split = group.loc[target_position, "split"]
                history_length = origin_position - first_active_position + 1
                if history_length >= MIN_CALENDAR_HISTORY:
                    historical = volumes[first_active_position : origin_position + 1]
                    historical_median = float(np.median(historical))
                    historical_mad = float(np.median(np.abs(historical - historical_median)))
                    future_volume = float(volumes[target_position])
                    growth_score = (future_volume - historical_median) / (
                        1.0 + MAD_SCALE * historical_mad
                    )
                else:
                    historical_median = np.nan
                    historical_mad = np.nan
                    future_volume = float(volumes[target_position])
                    growth_score = np.nan
                eligible = bool(
                    topic in eligible_topics
                    and origin_split == target_split
                    and np.isfinite(growth_score)
                )
                all_rows.append(
                    {
                        "topic_id": int(topic),
                        "origin_date": origin_date,
                        "target_date": target_date,
                        "split": origin_split,
                        "target_split": target_split,
                        "forecast_horizon": horizon,
                        "current_volume": float(volumes[origin_position]),
                        "baseline_volume": (
                            float(volumes[origin_position - 1]) if origin_position >= 1 else np.nan
                        ),
                        "future_volume": future_volume,
                        "historical_median": historical_median,
                        "historical_mad": historical_mad,
                        "history_calendar_days": max(0, history_length),
                        "growth_score": growth_score,
                        "forecast_eligible_topic": bool(topic in eligible_topics),
                        "eligible_observation": eligible,
                    }
                )
    labels = pd.DataFrame(all_rows)
    labels["burst_score"] = burst_score_values(
        labels["future_volume"], labels["historical_median"], labels["historical_mad"]
    )
    labels.loc[labels["historical_median"].isna(), "burst_score"] = np.nan

    distribution_rows = []
    sensitivity_rows = []
    event_frames = []
    for horizon in [1, 2, 3]:
        horizon_rows = labels.loc[
            labels["forecast_horizon"].eq(horizon) & labels["eligible_observation"]
        ]
        training = horizon_rows.loc[horizon_rows["split"].eq("train"), "growth_score"]
        if training.empty:
            raise RuntimeError(f"Tidak ada training growth score eligible untuk horizon {horizon}")
        quantiles = {q: float(training.quantile(q)) for q in GROWTH_QUANTILES}
        distribution_rows.append(
            {
                "horizon": horizon,
                "count": int(training.count()),
                "min": float(training.min()),
                "max": float(training.max()),
                "median": float(training.median()),
                "mean": float(training.mean()),
                "p50": float(training.quantile(0.50)),
                "p75": quantiles[0.75],
                "p80": quantiles[0.80],
                "p85": quantiles[0.85],
                "p90": quantiles[0.90],
                "p95": float(training.quantile(0.95)),
                "count_greater_than_zero": int(training.gt(0).sum()),
            }
        )
        for quantile, threshold in quantiles.items():
            for split in ["train", "validation", "test"]:
                split_rows = horizon_rows.loc[horizon_rows["split"].eq(split)]
                growth = split_rows["growth_score"].gt(threshold)
                sensitivity_rows.append(
                    {
                        "horizon": horizon,
                        "threshold_quantile": f"p{int(quantile * 100)}",
                        "threshold_value": threshold,
                        "threshold_fit_split": "train",
                        "split": split,
                        "total_observations": int(len(split_rows)),
                        "growth_count": int(growth.sum()),
                        "normal_count": int((~growth).sum()),
                        "growth_rate": float(growth.mean()) if len(growth) else np.nan,
                        "active_topics": int(split_rows["topic_id"].nunique()),
                        "temporal_origins": int(split_rows["origin_date"].nunique()),
                    }
                )
        horizon_mask = labels["forecast_horizon"].eq(horizon) & labels["eligible_observation"]
        eligible_horizon = labels.loc[horizon_mask]
        in_population = population_mask(eligible_horizon, scheme)
        primary_threshold = fit_threshold(
            eligible_horizon.loc[in_population & eligible_horizon["split"].eq("train")],
            scheme,
            require_train_split=True,
        )
        labeled_index = eligible_horizon.index[in_population]
        labels.loc[labeled_index, "label"] = (
            labels.loc[labeled_index, scheme.score_column].gt(primary_threshold).astype("int8")
        )
        labels.loc[labeled_index, "label_scheme"] = scheme.key
        labels.loc[labeled_index, "label_threshold"] = primary_threshold
        training_events = labels.loc[
            horizon_mask & labels["split"].eq("train") & labels["label"].eq(1),
            ["origin_date", "topic_id", "forecast_horizon", "growth_score", "burst_score", "label"],
        ].copy()
        training_events["label_scheme"] = scheme.key
        training_events["threshold_value"] = primary_threshold
        event_frames.append(training_events)

    distribution = pd.DataFrame(distribution_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    events = pd.concat(event_frames, ignore_index=True)
    return labels, distribution, sensitivity, events


def add_growth_candidates(history: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    candidates = (
        labels.loc[
            labels["forecast_horizon"].eq(1)
            & labels["eligible_observation"]
            & labels["growth_score"].gt(0)
        ]
        .groupby("topic_id")
        .size()
    )
    output = history.copy()
    output["growth_event_candidate_count"] = output["topic_id"].map(candidates).fillna(0).astype(int)
    return output


def attach_sparsity(labels: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    categories = history.set_index("topic_id")["sparsity_category"]
    output = labels.copy()
    output["sparsity_category"] = output["topic_id"].map(categories)
    return output


def bin_growth_scores(values: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="object")
    for name, lower, upper in HISTOGRAM_BINS:
        if lower == upper:
            mask = values.eq(lower)
        elif np.isneginf(lower):
            mask = values.le(upper)
        elif np.isposinf(upper):
            mask = values.gt(lower)
        else:
            mask = values.gt(lower) & values.le(upper)
        result.loc[mask & values.notna()] = name
    return result


def build_growth_score_histogram(
    labels: pd.DataFrame, history: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Histogram dan ringkasan growth_score dipisah kelas sparsity topik."""
    eligible = attach_sparsity(labels.loc[labels["eligible_observation"]], history)
    eligible = eligible.assign(bin=bin_growth_scores(eligible["growth_score"]))
    bin_names = [name for name, _, _ in HISTOGRAM_BINS]

    histogram_rows = []
    summary_rows = []
    for horizon in [1, 2, 3]:
        for split in ["train", "validation", "test"]:
            for category in SPARSITY_ORDER:
                subset = eligible.loc[
                    eligible["forecast_horizon"].eq(horizon)
                    & eligible["split"].eq(split)
                    & eligible["sparsity_category"].eq(category)
                ]
                total = len(subset)
                counts = subset["bin"].value_counts()
                for name in bin_names:
                    count = int(counts.get(name, 0))
                    histogram_rows.append(
                        {
                            "horizon": horizon,
                            "split": split,
                            "sparsity_category": category,
                            "bin": name,
                            "count": count,
                            "share": (count / total) if total else np.nan,
                        }
                    )
                scores = subset["growth_score"]
                summary_rows.append(
                    {
                        "horizon": horizon,
                        "split": split,
                        "sparsity_category": category,
                        "topics": int(subset["topic_id"].nunique()),
                        "observations": total,
                        "share_negative": float(scores.lt(0).mean()) if total else np.nan,
                        "share_exactly_zero": float(scores.eq(0).mean()) if total else np.nan,
                        "share_positive": float(scores.gt(0).mean()) if total else np.nan,
                        "share_median_zero": float(subset["historical_median"].eq(0).mean()) if total else np.nan,
                        "share_mad_zero": float(subset["historical_mad"].eq(0).mean()) if total else np.nan,
                        "p50": float(scores.quantile(0.50)) if total else np.nan,
                        "p75": float(scores.quantile(0.75)) if total else np.nan,
                        "p80": float(scores.quantile(0.80)) if total else np.nan,
                        "p90": float(scores.quantile(0.90)) if total else np.nan,
                        "p95": float(scores.quantile(0.95)) if total else np.nan,
                        "max": float(scores.max()) if total else np.nan,
                    }
                )
    return pd.DataFrame(histogram_rows), pd.DataFrame(summary_rows)


def plot_growth_score_histogram(histogram: pd.DataFrame, path: Path) -> None:
    subset = histogram.loc[histogram["horizon"].eq(1) & histogram["split"].eq("train")]
    bin_names = [name for name, _, _ in HISTOGRAM_BINS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)
    for axis, category in zip(axes, SPARSITY_ORDER):
        frame = subset.loc[subset["sparsity_category"].eq(category)].set_index("bin").reindex(bin_names)
        colors = ["#DE350B" if name == "== 0" else "#0052CC" for name in bin_names]
        axis.bar(bin_names, frame["share"].fillna(0.0), color=colors)
        observations = int(frame["count"].fillna(0).sum())
        axis.set_title(f"{category}\nn={observations}")
        axis.set_xlabel("growth_score")
        axis.tick_params(axis="x", rotation=60)
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("Proporsi observasi eligible")
    fig.suptitle("Distribusi growth_score training (t+1) menurut sparsity topik; bin nol disorot")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_scheme_comparison(labels: pd.DataFrame) -> pd.DataFrame:
    """Class balance train/validation/test untuk setiap skema label kandidat."""
    rows = []
    eligible = labels.loc[labels["eligible_observation"]]
    for horizon in [1, 2, 3]:
        horizon_frame = eligible.loc[eligible["forecast_horizon"].eq(horizon)]
        for key in COMPARISON_ORDER:
            scheme = SCHEMES[key]
            training = horizon_frame.loc[horizon_frame["split"].eq("train")]
            threshold = fit_threshold(training, scheme, require_train_split=True)
            for row in class_balance(horizon_frame, scheme, threshold):
                rows.append({"horizon": horizon, **row, "definition": scheme.definition})
    return pd.DataFrame(rows)


def feasibility_status(growth_count: int, normal_count: int, split: str) -> str:
    if growth_count == 0 or normal_count == 0:
        return "NOT READY"
    minimum = 100 if split == "train" else 30
    return "READY" if min(growth_count, normal_count) >= minimum else "LIMITED"


def build_label_feasibility(sensitivity: pd.DataFrame) -> pd.DataFrame:
    primary = sensitivity.loc[sensitivity["threshold_quantile"].eq("p80")].copy()
    primary["status"] = primary.apply(
        lambda row: feasibility_status(int(row["growth_count"]), int(row["normal_count"]), str(row["split"])),
        axis=1,
    )
    return pd.concat(
        [
            sensitivity.assign(status="SENSITIVITY"),
            primary.assign(threshold_quantile="p80_primary"),
        ],
        ignore_index=True,
    )


def build_distribution_shift(
    assignments: pd.DataFrame,
    daily: pd.DataFrame,
    outlier_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = assignments.copy()
    assignments["split"] = split_name_for_dates(assignments["date"])
    article_daily = assignments.groupby(["split", "date"], observed=True).size().rename("article_count").reset_index()
    source_counts = assignments.groupby(["split", "source"], observed=True).size().rename("articles").reset_index()
    source_counts["percentage"] = 100.0 * source_counts["articles"] / source_counts.groupby("split")["articles"].transform("sum")

    rows = []
    for split in ["train", "validation", "test"]:
        article_values = article_daily.loc[article_daily["split"].eq(split), "article_count"]
        topic_values = daily.loc[daily["split"].eq(split), "topic_volume"]
        outlier_values = outlier_daily.loc[outlier_daily["split"].eq(split), "outlier_volume"]
        rows.extend(
            [
                {"split": split, "dimension": "total_daily_article_volume", "metric": "total", "value": float(article_values.sum())},
                {"split": split, "dimension": "total_daily_article_volume", "metric": "median", "value": float(article_values.median())},
                {"split": split, "dimension": "total_daily_article_volume", "metric": "mean", "value": float(article_values.mean())},
                {"split": split, "dimension": "total_daily_article_volume", "metric": "std", "value": float(article_values.std(ddof=1))},
                {"split": split, "dimension": "topic_volume_distribution", "metric": "median", "value": float(topic_values.median())},
                {"split": split, "dimension": "topic_volume_distribution", "metric": "mean", "value": float(topic_values.mean())},
                {"split": split, "dimension": "topic_volume_distribution", "metric": "std", "value": float(topic_values.std(ddof=1))},
                {"split": split, "dimension": "topic_volume_distribution", "metric": "zero_ratio", "value": float(topic_values.eq(0).mean())},
                {"split": split, "dimension": "outlier_daily_volume", "metric": "total", "value": float(outlier_values.sum())},
                {"split": split, "dimension": "outlier_daily_volume", "metric": "mean", "value": float(outlier_values.mean())},
            ]
        )
        for source_row in source_counts.loc[source_counts["split"].eq(split)].itertuples(index=False):
            rows.append(
                {
                    "split": split,
                    "dimension": "source_distribution",
                    "metric": f"{source_row.source}_percentage",
                    "value": float(source_row.percentage),
                }
            )
    return pd.DataFrame(rows), article_daily, source_counts


def plot_distribution_shift(article_daily: pd.DataFrame, source_counts: pd.DataFrame, output: Path) -> None:
    colors = {"train": "#0052CC", "validation": "#FF8B00", "test": "#36B37E"}
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), constrained_layout=True)
    for split in ["train", "validation", "test"]:
        subset = article_daily.loc[article_daily["split"].eq(split)]
        axes[0].plot(subset["date"], subset["article_count"], label=split.title(), color=colors[split], linewidth=1.2)
    axes[0].set_title("Volume Artikel Harian per Split")
    axes[0].set_ylabel("Artikel")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.25)

    values = [article_daily.loc[article_daily["split"].eq(split), "article_count"] for split in ["train", "validation", "test"]]
    box = axes[1].boxplot(values, labels=["Train", "Validation", "Test"], patch_artist=True)
    for patch, split in zip(box["boxes"], ["train", "validation", "test"]):
        patch.set_facecolor(colors[split])
        patch.set_alpha(0.65)
    axes[1].set_title("Distribusi Volume Harian")
    axes[1].set_ylabel("Artikel per hari")
    axes[1].grid(axis="y", alpha=0.25)

    pivot = source_counts.pivot(index="split", columns="source", values="percentage").reindex(["train", "validation", "test"]).fillna(0)
    source_colors = {"kompas": "#0052CC", "tempo": "#FF8B00", "detik": "#36B37E"}
    bottom = np.zeros(len(pivot))
    for source in pivot.columns:
        axes[2].bar(
            ["Train", "Validation", "Test"],
            pivot[source].to_numpy(),
            bottom=bottom,
            label=source.title(),
            color=source_colors.get(source, "#6B778C"),
        )
        bottom += pivot[source].to_numpy()
    axes[2].set_title("Komposisi Sumber per Split")
    axes[2].set_ylabel("Persentase artikel")
    axes[2].set_ylim(0, 100)
    axes[2].legend(frameon=False, ncol=3)
    axes[2].grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def audit_leakage(
    semantic: pd.DataFrame,
    temporal: pd.DataFrame,
    labels: pd.DataFrame,
    embedding_metadata: dict,
    topic_metadata: dict,
    processed_path: Path,
) -> pd.DataFrame:
    feature_columns = set(SEMANTIC_FEATURE_COLUMNS + TEMPORAL_FEATURE_COLUMNS)
    checks = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "passed": bool(passed), "detail": detail})

    add("target_not_in_features", not bool(feature_columns & {"label", "actual_label"}), str(sorted(feature_columns & {"label", "actual_label"})))
    add("future_volume_not_in_features", "future_volume" not in feature_columns, "future_volume hanya ada pada label audit")
    add("future_timestamp_not_in_features", "target_date" not in feature_columns, "target_date hanya ada pada label audit")
    add(
        "feature_timestamp_not_after_origin",
        bool((pd.to_datetime(semantic["feature_timestamp"]) <= pd.to_datetime(semantic["date"])).all()
             and (pd.to_datetime(temporal["feature_timestamp"]) <= pd.to_datetime(temporal["date"])).all()),
        "semantic dan temporal feature_timestamp <= origin date",
    )
    add(
        "semantic_centroid_not_future",
        bool((semantic.loc[semantic["semantic_history_cutoff"].notna(), "semantic_history_cutoff"]
              <= semantic.loc[semantic["semantic_history_cutoff"].notna(), "date"]).all()),
        "centroid kumulatif berhenti pada origin date",
    )
    expected_roll3 = temporal.assign(_volume=temporal["current_volume"]).groupby("topic_id", sort=False)["_volume"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=3).mean()
    )
    expected_roll7 = temporal.assign(_volume=temporal["current_volume"]).groupby("topic_id", sort=False)["_volume"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=7).mean()
    )
    rolling_ok = bool(
        np.allclose(temporal["rolling_mean_3"], expected_roll3, equal_nan=True)
        and np.allclose(temporal["rolling_mean_7"], expected_roll7, equal_nan=True)
        and (temporal["retrospective_history_cutoff"] < temporal["date"]).all()
    )
    add("rolling_features_past_only", rolling_ok, "rolling windows memakai shift(1)")
    add("growth_score_not_predictor", "growth_score" not in feature_columns, "growth_score hanya label audit")

    eligible_training = labels.loc[labels["eligible_observation"] & labels["split"].eq("train")]
    threshold_ok = bool(
        not eligible_training.empty
        and eligible_training["origin_date"].max() <= TRAIN_END
        and eligible_training["target_date"].max() <= TRAIN_END
    )
    add("threshold_training_only", threshold_ok, f"max origin={eligible_training['origin_date'].max()}; max target={eligible_training['target_date'].max()}")
    add(
        "embedding_fixed_pretrained_encoder",
        bool(
            embedding_metadata.get("embedding_model") == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            and embedding_metadata.get("dataset_hash") == file_sha256(processed_path)
        ),
        "pretrained SentenceTransformer; tidak ada fit pada corpus future",
    )
    add(
        "topic_model_train_only",
        topic_metadata.get("fit_scope") == "training_only",
        f"fit_scope={topic_metadata.get('fit_scope')}; train_period={topic_metadata.get('train_period')}",
    )
    add("forbidden_columns_absent", not bool(feature_columns & FORBIDDEN_FEATURE_COLUMNS), str(sorted(feature_columns & FORBIDDEN_FEATURE_COLUMNS)))
    audit = pd.DataFrame(checks)
    if not audit["passed"].all():
        raise RuntimeError("Pre-feature leakage audit gagal: " + repr(audit.loc[~audit["passed"]].to_dict("records")))
    return audit


def feature_distribution(frame: pd.DataFrame, columns: list[str], family: str) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "feature_family": family,
                "feature": column,
                "count": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "finite": int(np.isfinite(values).sum()),
                "min": float(finite.min()) if len(finite) else np.nan,
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "median": float(finite.median()) if len(finite) else np.nan,
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_report(
    daily: pd.DataFrame,
    history: pd.DataFrame,
    feature_stats: pd.DataFrame,
    growth_distribution: pd.DataFrame,
    sensitivity: pd.DataFrame,
    label_feasibility: pd.DataFrame,
    shifts: pd.DataFrame,
    temporal_audit: pd.DataFrame,
    leakage: pd.DataFrame,
    histogram_summary: pd.DataFrame,
    scheme_comparison: pd.DataFrame,
) -> str:
    activity = {
        threshold: int(history["active_days"].ge(threshold).sum()) for threshold in [14, 30, 60, 90]
    }
    training_activity = {
        threshold: int(history["training_active_days"].ge(threshold).sum()) for threshold in [14, 30, 60, 90]
    }
    sparsity = history["sparsity_category"].value_counts().to_dict()
    scheme = get_scheme()
    sparsity_training = histogram_summary.loc[
        histogram_summary["horizon"].eq(1) & histogram_summary["split"].eq("train")
    ][
        [
            "sparsity_category",
            "topics",
            "observations",
            "share_negative",
            "share_exactly_zero",
            "share_positive",
            "share_median_zero",
            "share_mad_zero",
            "p80",
            "p90",
            "max",
        ]
    ]
    scheme_table = scheme_comparison.loc[scheme_comparison["horizon"].eq(1)][
        [
            "scheme",
            "population",
            "threshold_kind",
            "threshold_value",
            "split",
            "observations",
            "growth_count",
            "normal_count",
            "growth_rate",
        ]
    ]
    primary = label_feasibility.loc[label_feasibility["threshold_quantile"].eq("p80_primary")]
    zero_primary_horizons = growth_distribution.loc[growth_distribution["p80"].eq(0), "horizon"].astype(int).tolist()
    overall_ready = "READY"
    if (primary["status"] == "NOT READY").any():
        label_status = "FAIL"
        forecasting = "NOT READY"
    elif (primary["status"] == "LIMITED").any():
        label_status = "LIMITED"
        forecasting = "LIMITED"
    else:
        label_status = "READY"
        forecasting = "LIMITED"  # topic model/truncation and distribution shift remain material
    feature_status = "READY" if leakage["passed"].all() else "FAIL"
    return f"""# STEP 10–13 report

## 1. Topic time series

- Topik non-outlier: {history['topic_id'].nunique():,}
- Tanggal: {daily['date'].nunique():,}
- Topic-day: {len(daily):,}
- Total volume non-outlier: {int(daily['topic_volume'].sum()):,}
- Semua kombinasi topic-date tersedia; tanggal tanpa dokumen diberi volume nol.
- Dokumen outlier tidak dihitung sebagai volume topik dan diaudit secara terpisah.

## 2. Topic sparsity

Threshold ditetapkan sebelum evaluasi label: LOW <50% hari nol; MEDIUM 50–<80%; HIGH ≥80%.

- Kategori: {sparsity}
- Active days seluruh periode ≥14/30/60/90: {activity}
- Active days training ≥14/30/60/90: {training_activity}
- Topic layak berdasarkan ≥14 active days training: {int(history['forecast_eligible'].sum())}/{len(history)}

## 3. Semantic features

Centroid topik bersifat kumulatif dan hanya memakai dokumen hingga origin t. Similarity mengukur dokumen hari t terhadap centroid kumulatif t. Drift adalah `1 − cosine(centroid_t, centroid_t−1)`. Hari tanpa dokumen menghasilkan semantic feature missing, bukan nilai buatan.

{feature_stats.loc[feature_stats['feature_family'].eq('semantic')].to_markdown(index=False)}

## 4. Temporal features

`current_volume` menggunakan volume origin t. Lag dan rolling memakai masa lalu; rolling menerapkan `shift(1)`. Growth rate memakai denominator `abs(lag_1)+1`; percentage change memakai `max(abs(lag_1),1)`. `zero_baseline_flag` mempertahankan kasus baseline nol tanpa menghasilkan infinity.

{temporal_audit.to_markdown(index=False)}

{feature_stats.loc[feature_stats['feature_family'].eq('temporal')].to_markdown(index=False)}

## 5. Growth score distribution training

Growth score = `(future_volume − historical_median) / (1 + 1.4826 × MAD)`. Histori dimulai pada kemunculan pertama topik, minimal {MIN_CALENDAR_HISTORY} hari kalender, dan origin-target wajib berada dalam split yang sama.

{growth_distribution.to_markdown(index=False)}

## 6. Threshold sensitivity

Threshold p75/p80/p85/p90 seluruhnya dihitung dari training.

WARNING: p80 bernilai nol pada horizon {zero_primary_horizons}. Konsekuensinya diperiksa pada bagian 6b: `growth_score > 0` identik dengan `future_volume > median historis`, dan karena median historis nol pada mayoritas observasi eligible, label kolaps menjadi "ada artikel pada target date". Kuantil p80 karena itu tidak lagi dipakai sebagai definisi label utama.

{sensitivity.loc[sensitivity['split'].eq('train')].to_markdown(index=False)}

## 6b. Zero-inflation growth_score menurut sparsity

Distribusi growth_score training (t+1) dipisah kelas sparsity topik. Kolom `share_exactly_zero` adalah proporsi observasi dengan growth_score tepat nol; `share_median_zero` dan `share_mad_zero` adalah proporsi observasi dengan median dan MAD historis nol.

{sparsity_training.to_markdown(index=False)}

Figure: `results/figures/growth_score_histogram_by_sparsity.png`. Tabel bin penuh: `results/tables/growth_score_histogram_by_sparsity.csv`.

## 6c. Perbandingan skema label

Empat skema dibandingkan pada populasi eligible yang sama. Semua threshold yang di-fit hanya memakai baris split training; skema burst tidak melakukan fitting sama sekali.

{scheme_table.to_markdown(index=False)}

Skema label yang dipakai sebagai label utama: **{scheme.key}** — {scheme.definition}.

Alasan: {scheme.rationale}

Tabel lengkap tiga horizon: `results/tables/label_scheme_class_balance.csv`.

## 7. Positive-event distribution dan 8. horizon feasibility

{primary.to_markdown(index=False)}

## 9. Temporal distribution shift

{shifts.to_markdown(index=False)}

Perubahan distribusi hanya dilaporkan; audit ini tidak menyimpulkan penyebabnya. Source digunakan sebagai audit dan tidak dimasukkan sebagai predictor.

## 10. Leakage audit

{leakage.to_markdown(index=False)}

## 11. Recommendation

- FEATURE ENGINEERING STATUS: **{feature_status}**
- LABEL STATUS: **{label_status}**
- FORECASTING READINESS: **{forecasting}**

`FORECASTING READINESS` tetap LIMITED walaupun label memiliki dua kelas, karena topic model berstatus LIMITED, truncation embedding sangat tinggi, outlier future sekitar 48%, dan terdapat distribution shift volume/source. Classifier belum dijalankan.
"""


def main() -> None:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    tables = root / "results" / "tables"
    figures = root / "results" / "figures"
    processed_dir = root / "data" / "processed"
    intermediate_dir = root / "data" / "intermediate"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    processed_path = processed_dir / "news_cleaned.parquet"
    assignment_path = processed_dir / "topic_assignments_train_only.parquet"
    embedding_path = intermediate_dir / "embeddings.npy"
    embedding_metadata = json.loads((intermediate_dir / "embedding_metadata.json").read_text(encoding="utf-8"))
    topic_metadata = json.loads((root / "results" / "bertopic_run_metadata.json").read_text(encoding="utf-8"))
    if topic_metadata.get("selected_configuration") != "B":
        raise RuntimeError("Konfigurasi BERTopic terpilih bukan B.")

    cleaned = pd.read_parquet(processed_path, columns=["date", "source", "Link"])
    assignments = pd.read_parquet(assignment_path)
    embeddings = np.load(embedding_path, mmap_mode="r")
    if len(cleaned) != len(assignments) or len(cleaned) != embeddings.shape[0]:
        raise RuntimeError("Jumlah baris cleaned/assignment/embedding tidak konsisten.")
    if not (
        pd.to_datetime(cleaned["date"]).reset_index(drop=True).equals(pd.to_datetime(assignments["date"]).reset_index(drop=True))
        and cleaned["Link"].reset_index(drop=True).equals(assignments["url"].reset_index(drop=True))
    ):
        raise RuntimeError("Urutan topic assignment tidak sama dengan news_cleaned.parquet.")
    masks = split_masks(cleaned)
    actual_splits = np.empty(len(cleaned), dtype=object)
    for split, mask in masks.items():
        actual_splits[mask] = split
    if not np.array_equal(actual_splits, assignments["split"].to_numpy()):
        raise RuntimeError("Split topic assignment berubah.")

    topics = sorted(pd.read_csv(tables / "topic_summary.csv")["topic_id"].astype(int).tolist())
    daily, outlier_daily = build_topic_daily_volume(assignments, topics)
    daily.to_csv(tables / "topic_daily_volume.csv", index=False)
    outlier_daily.to_csv(tables / "outlier_daily_volume.csv", index=False)

    history = initial_topic_history(daily)
    semantic = build_semantic_features(assignments, embeddings, daily, topics)
    semantic.to_parquet(processed_dir / "semantic_features.parquet", index=False, compression="zstd")
    semantic.head(1000).to_csv(tables / "semantic_features_preview.csv", index=False)

    temporal, temporal_audit = build_temporal_features(daily)
    temporal.to_parquet(processed_dir / "temporal_features.parquet", index=False, compression="zstd")
    temporal.head(1000).to_csv(tables / "temporal_features_preview.csv", index=False)
    temporal_audit.to_csv(tables / "temporal_feature_numerical_audit.csv", index=False)

    scheme = get_scheme()
    print(f"label scheme aktif: {scheme.key} ({scheme.definition})", flush=True)
    labels, growth_distribution, sensitivity, events = build_growth_labels(daily, history)

    baseline_check = labels.merge(
        temporal[["topic_id", "date", "lag_1"]].rename(columns={"date": "origin_date"}),
        on=["topic_id", "origin_date"],
        how="left",
        validate="many_to_one",
    )
    mismatch = int(
        (
            baseline_check["baseline_volume"].fillna(-1.0)
            != baseline_check["lag_1"].fillna(-1.0)
        ).sum()
    )
    if mismatch:
        raise RuntimeError(f"baseline_volume tidak konsisten dengan temporal lag_1: {mismatch} baris")

    labels.to_parquet(processed_dir / "growth_labels_audit.parquet", index=False, compression="zstd")
    growth_distribution.to_csv(tables / "growth_score_distribution_training.csv", index=False)
    sensitivity.to_csv(tables / "growth_threshold_sensitivity.csv", index=False)
    events.to_csv(tables / "growth_events_training.csv", index=False)

    histogram, histogram_summary = build_growth_score_histogram(labels, history)
    histogram.to_csv(tables / "growth_score_histogram_by_sparsity.csv", index=False)
    histogram_summary.to_csv(tables / "growth_score_summary_by_sparsity.csv", index=False)
    plot_growth_score_histogram(histogram, figures / "growth_score_histogram_by_sparsity.png")

    scheme_comparison = build_scheme_comparison(labels)
    scheme_comparison.to_csv(tables / "label_scheme_class_balance.csv", index=False)

    print("\n=== growth_score training (t+1) menurut sparsity ===", flush=True)
    print(
        histogram_summary.loc[
            histogram_summary["horizon"].eq(1) & histogram_summary["split"].eq("train")
        ].to_string(index=False),
        flush=True,
    )
    print("\n=== histogram growth_score training (t+1) ===", flush=True)
    print(
        histogram.loc[histogram["horizon"].eq(1) & histogram["split"].eq("train")]
        .pivot(index="bin", columns="sparsity_category", values="share")
        .reindex([name for name, _, _ in HISTOGRAM_BINS])[SPARSITY_ORDER]
        .to_string(float_format=lambda value: f"{value:.4f}"),
        flush=True,
    )
    print("\n=== class balance per skema label (t+1) ===", flush=True)
    print(
        scheme_comparison.loc[scheme_comparison["horizon"].eq(1)][
            ["scheme", "threshold_value", "split", "observations", "growth_count", "normal_count", "growth_rate"]
        ].to_string(index=False),
        flush=True,
    )

    history = add_growth_candidates(history, labels)
    history.to_csv(tables / "topic_history_final.csv", index=False)

    feasibility = build_label_feasibility(sensitivity)
    feasibility.to_csv(tables / "label_feasibility.csv", index=False)

    shifts, article_daily, source_counts = build_distribution_shift(assignments, daily, outlier_daily)
    shifts.to_csv(tables / "temporal_distribution_shift.csv", index=False)
    source_counts.to_csv(tables / "source_distribution_by_split.csv", index=False)
    plot_distribution_shift(article_daily, source_counts, figures / "temporal_distribution_shift.png")

    leakage = audit_leakage(
        semantic,
        temporal,
        labels,
        embedding_metadata,
        topic_metadata,
        processed_path,
    )
    leakage.to_csv(tables / "pre_feature_leakage_audit.csv", index=False)

    feature_stats = pd.concat(
        [
            feature_distribution(semantic, SEMANTIC_FEATURE_COLUMNS, "semantic"),
            feature_distribution(temporal, TEMPORAL_FEATURE_COLUMNS, "temporal"),
        ],
        ignore_index=True,
    )
    feature_stats.to_csv(tables / "feature_distribution.csv", index=False)

    report = build_report(
        daily,
        history,
        feature_stats,
        growth_distribution,
        sensitivity,
        feasibility,
        shifts,
        temporal_audit,
        leakage,
        histogram_summary,
        scheme_comparison,
    )
    (root / "results" / "step10_13_report.md").write_text(report, encoding="utf-8")
    metadata = {
        "status": "complete",
        "processing_seconds": time.perf_counter() - started,
        "topic_count": len(topics),
        "date_count": int(daily["date"].nunique()),
        "topic_day_count": len(daily),
        "minimum_training_active_days": MIN_ACTIVE_TRAINING_DAYS,
        "minimum_calendar_history": MIN_CALENDAR_HISTORY,
        "sparsity_thresholds": {
            "low": f"zero_ratio < {SPARSITY_LOW_MAX}",
            "medium": f"{SPARSITY_LOW_MAX} <= zero_ratio < {SPARSITY_MEDIUM_MAX}",
            "high": f"zero_ratio >= {SPARSITY_MEDIUM_MAX}",
        },
        "growth_score": "(future_volume - historical_median) / (1 + 1.4826 * MAD)",
        "burst_score": f"(future_volume - historical_median) / max(1.4826 * MAD, {BURST_SCALE_FLOOR})",
        "threshold_quantiles": GROWTH_QUANTILES,
        "baseline_threshold_quantile": PRIMARY_GROWTH_QUANTILE,
        "baseline_threshold_zero_warning": True,
        "label_scheme": scheme.key,
        "label_scheme_definition": scheme.definition,
        "label_scheme_thresholds": {
            str(horizon): float(value)
            for horizon, value in labels.loc[labels["label"].notna()]
            .groupby("forecast_horizon")["label_threshold"]
            .first()
            .items()
        },
        "label_scheme_candidates": {key: SCHEMES[key].definition for key in COMPARISON_ORDER},
        "threshold_fit_split": "train",
        "semantic_centroid": "cumulative topic centroid through origin t; drift=1-cosine(centroid_t, centroid_t-1)",
        "dataset_hash": file_sha256(processed_path),
        "embedding_file_hash": file_sha256(embedding_path),
        "topic_assignment_hash": file_sha256(assignment_path),
    }
    write_json(root / "results" / "step10_13_metadata.json", metadata)
    console_encoding = sys.stdout.encoding or "utf-8"
    print(report.encode(console_encoding, errors="replace").decode(console_encoding))


if __name__ == "__main__":
    main()

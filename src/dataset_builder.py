from __future__ import annotations

import pandas as pd


SEMANTIC_COLUMNS = ["semantic_similarity_mean", "semantic_similarity_std", "semantic_drift",
                    "topic_probability_mean", "topic_probability_std", "topic_probability_change",
                    "similarity_change"]


def assemble_dataset(temporal: pd.DataFrame, semantic: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_cols = ["topic_id", "date", "target_date", "target_volume", "growth_score", "actual_label", "forecast_horizon"]
    return temporal.merge(semantic, on=["topic_id", "date"], how="left").merge(labels[label_cols], on=["topic_id", "date"], how="inner")


def temporal_split(frame: pd.DataFrame, train_cutoff, test_cutoff):
    valid = frame.dropna(subset=["actual_label", "target_date"]).copy()
    valid["actual_label"] = valid["actual_label"].astype(int)
    # Split ditentukan oleh waktu target, bukan hanya waktu origin. Dengan ini,
    # label training tidak pernah berasal dari volume pada periode validation/test.
    target = pd.to_datetime(valid["target_date"])
    train = valid[target < pd.Timestamp(train_cutoff)].copy()
    validation = valid[(target >= pd.Timestamp(train_cutoff)) & (target < pd.Timestamp(test_cutoff))].copy()
    test = valid[target >= pd.Timestamp(test_cutoff)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"Split temporal kosong: train={len(train)}, validation={len(validation)}, test={len(test)}")
    return train, validation, test

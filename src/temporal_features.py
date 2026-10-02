from __future__ import annotations

import numpy as np
import pandas as pd


TEMPORAL_COLUMNS = ["current_topic_volume", "lag_1", "lag_2", "lag_3", "lag_4",
                    "rolling_mean_3", "rolling_mean_4", "rolling_std_4", "growth_rate",
                    "percentage_change", "momentum", "burst_intensity"]


def build_temporal_features(series: pd.DataFrame) -> pd.DataFrame:
    out = series[["topic_id", "date", "topic_volume"]].copy()
    grouped = out.groupby("topic_id", group_keys=False)["topic_volume"]
    out["current_topic_volume"] = out["topic_volume"]
    for lag in range(1, 5):
        out[f"lag_{lag}"] = grouped.shift(lag)
    out["rolling_mean_3"] = grouped.transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    out["rolling_mean_4"] = grouped.transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
    out["rolling_std_4"] = grouped.transform(lambda x: x.shift(1).rolling(4, min_periods=2).std())
    denom = out["lag_1"].abs() + 1.0
    out["growth_rate"] = (out["current_topic_volume"] - out["lag_1"]) / denom
    out["percentage_change"] = out["growth_rate"]
    out["momentum"] = (out["current_topic_volume"] - out["lag_1"]) - (out["lag_1"] - out["lag_2"])
    out["burst_intensity"] = (out["current_topic_volume"] - out["rolling_mean_4"]) / (1.0 + out["rolling_std_4"].fillna(0))
    out["feature_timestamp"] = out["date"]
    out.attrs["rolling_shifted"] = True
    return out.drop(columns=["topic_volume"])


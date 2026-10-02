from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GROWTH_QUANTILE, MIN_HISTORY


def build_labels(series: pd.DataFrame, horizon: int, train_cutoff) -> tuple[pd.DataFrame, float | None]:
    rows = []
    for topic, group in series.groupby("topic_id", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        volumes = group["topic_volume"].to_numpy(dtype=float)
        dates = pd.to_datetime(group["date"]).to_numpy()
        for idx in range(len(group)):
            target_idx = idx + horizon
            target_volume = volumes[target_idx] if target_idx < len(group) else np.nan
            history = volumes[:idx + 1]
            if len(history) < MIN_HISTORY or np.isnan(target_volume):
                score = np.nan
            else:
                median = float(np.median(history))
                mad = float(np.median(np.abs(history - median)))
                score = (target_volume - median) / (1.0 + 1.4826 * mad)
            rows.append({"topic_id": topic, "date": pd.Timestamp(dates[idx]),
                         "target_date": pd.Timestamp(dates[target_idx]) if target_idx < len(dates) else pd.NaT,
                         "target_volume": target_volume, "growth_score": score, "forecast_horizon": horizon})
    out = pd.DataFrame(rows)
    # Baik origin maupun target label harus berada sebelum cutoff training.
    # Ini mencegah label training di batas horizon membaca volume validation.
    eligible = (out["date"].lt(pd.Timestamp(train_cutoff))
                & out["target_date"].lt(pd.Timestamp(train_cutoff))
                & out["growth_score"].notna())
    threshold = float(out.loc[eligible, "growth_score"].quantile(GROWTH_QUANTILE)) if eligible.any() else None
    # Pertumbuhan harus secara ketat melampaui threshold. Pemakaian >= akan
    # mengubah semua observasi berskor nol menjadi positif ketika kuantil train=0.
    out["actual_label"] = np.where(out["growth_score"].notna() & (out["growth_score"] > threshold), 1, 0) if threshold is not None else np.nan
    out.loc[out["growth_score"].isna(), "actual_label"] = np.nan
    out.attrs["threshold_train_only"] = True
    out.attrs["threshold_fit_max_date"] = out.loc[eligible, "date"].max() if eligible.any() else pd.NaT
    out.attrs["threshold_fit_max_target_date"] = out.loc[eligible, "target_date"].max() if eligible.any() else pd.NaT
    return out, threshold

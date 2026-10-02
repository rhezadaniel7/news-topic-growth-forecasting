from __future__ import annotations

import pandas as pd


FORBIDDEN = {"target_volume", "actual_label", "predicted_label", "prediction_probability", "growth_score", "target_period"}


def validate_raw(frame: pd.DataFrame, mapping: dict):
    if frame.empty:
        raise ValueError("Dataset kosong.")
    parsed = pd.to_datetime(frame[mapping["date"]], errors="coerce", utc=True)
    if parsed.notna().sum() == 0:
        raise ValueError(f"Kolom tanggal {mapping['date']} tidak memiliki timestamp valid.")
    if frame[mapping["title"]].fillna("").astype(str).str.strip().eq("").all():
        raise ValueError(f"Kolom judul {mapping['title']} seluruhnya kosong.")


def audit_leakage(features: pd.DataFrame, labels: pd.DataFrame, documents: pd.DataFrame,
                  feature_columns: list[str], train_cutoff, topic_metadata: dict) -> pd.DataFrame:
    checks = []
    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    overlap = set(feature_columns) & FORBIDDEN
    add("target_not_in_features", not overlap, sorted(overlap))
    add("feature_timestamp_not_future", (pd.to_datetime(features["feature_timestamp"]) <= pd.to_datetime(features["date"])).all(), "feature_timestamp <= origin date")
    valid = labels.dropna(subset=["target_date"])
    add("target_after_origin", (pd.to_datetime(valid["target_date"]) > pd.to_datetime(valid["date"])).all(), "target_date > date")
    add("rolling_windows_shifted", features.attrs.get("rolling_shifted") is True, "rolling statistics use shift(1)")
    threshold_date = labels.attrs.get("threshold_fit_max_date")
    threshold_target_date = labels.attrs.get("threshold_fit_max_target_date")
    threshold_ok = (labels.attrs.get("threshold_train_only") is True and pd.notna(threshold_date)
                    and pd.notna(threshold_target_date)
                    and pd.Timestamp(threshold_date) < pd.Timestamp(train_cutoff)
                    and pd.Timestamp(threshold_target_date) < pd.Timestamp(train_cutoff))
    add("threshold_train_only", threshold_ok, f"latest origin={threshold_date}; latest target={threshold_target_date}; cutoff={train_cutoff}")
    add("topic_model_train_only", topic_metadata.get("fit_scope") == "train_only", topic_metadata)
    add("no_duplicate_url", not ("url" in documents and documents["url"].duplicated().any()), int(documents["url"].duplicated().sum()) if "url" in documents else "URL unavailable")
    add("no_duplicate_article_across_split", not documents.get("article_key", pd.Series(dtype=str)).duplicated().any(), "cleaned article keys unique")
    result = pd.DataFrame(checks)
    failed = result.loc[~result["passed"]]
    if not failed.empty:
        raise RuntimeError("Leakage audit gagal: " + failed.to_dict("records").__repr__())
    return result

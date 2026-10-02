from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import RANDOM_SEED
from .dataset_builder import SEMANTIC_COLUMNS
from .evaluation import train_models
from .temporal_features import TEMPORAL_COLUMNS


FEATURE_SETS = {
    "temporal_only": TEMPORAL_COLUMNS,
    "semantic_only": SEMANTIC_COLUMNS,
    "semantic_temporal": SEMANTIC_COLUMNS + TEMPORAL_COLUMNS,
}


def run_ablation(train, validation, test, horizon):
    rows = []
    for name, columns in FEATURE_SETS.items():
        results, _, _ = train_models(train, validation, test, columns, name, horizon)
        selected = results[results["model"] == "logistic_regression"]
        if not selected.empty:
            rows.append(selected.iloc[0].to_dict())
    return pd.DataFrame(rows)


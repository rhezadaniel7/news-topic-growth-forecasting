from __future__ import annotations

import importlib.util

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_PARAMETERS, RANDOM_SEED


def available_models():
    lr = MODEL_PARAMETERS["logistic_regression"]
    rf = MODEL_PARAMETERS["random_forest"]
    models = {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(random_state=RANDOM_SEED, **lr),
        "random_forest": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **rf),
    }
    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(random_state=RANDOM_SEED, n_jobs=-1, eval_metric="logloss", **MODEL_PARAMETERS["xgboost"])
    else:
        models["hist_gradient_boosting_fallback"] = HistGradientBoostingClassifier(random_state=RANDOM_SEED)
    return models


def make_pipeline(estimator, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


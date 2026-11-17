"""Save and load the trained model so it works on any operating system.

Pickling an XGBoost model stores a memory snapshot that includes
platform-specific state. A model pickled on Linux (Colab, Render) then fails to
load on Windows with "XGBoostError: input stream corrupted".

To avoid that, every XGBoost booster is saved in XGBoost's portable UBJSON model
format, and the rest of the scikit-learn pipeline (preprocessing, calibration)
is pickled as usual. The same file loads on Linux, Windows and macOS.
"""

import joblib
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

FORMAT_VERSION = 1


def _xgb_estimators(model):
    """Every XGBoost estimator inside the model, always in the same order."""
    found = []

    def walk(obj):
        if obj is None:
            return
        if isinstance(obj, xgb.XGBModel):
            found.append(obj)
        elif isinstance(obj, Pipeline):
            for _, step in obj.steps:
                walk(step)
        elif isinstance(obj, CalibratedClassifierCV):
            walk(obj.estimator)
            for calibrated in getattr(obj, "calibrated_classifiers_", []):
                walk(calibrated.estimator)

    walk(model)
    return found


def save_portable(model, path):
    estimators = _xgb_estimators(model)
    originals = [est._Booster for est in estimators]
    boosters = [bytes(booster.save_raw("ubj")) for booster in originals]
    try:
        # Take the boosters out while pickling, so their snapshot isn't stored.
        for est in estimators:
            est._Booster = None
        joblib.dump(
            {"format_version": FORMAT_VERSION, "model": model, "boosters": boosters},
            path,
        )
    finally:
        for est, booster in zip(estimators, originals):
            est._Booster = booster
    print(f"Saved {path} with {len(boosters)} portable XGBoost boosters")


def load_portable(path):
    data = joblib.load(path)
    if not (isinstance(data, dict) and "boosters" in data):
        # Plain pickle from an older version of the project.
        return data

    model = data["model"]
    estimators = _xgb_estimators(model)
    if len(estimators) != len(data["boosters"]):
        raise ValueError(
            f"{path}: found {len(estimators)} XGBoost estimators but "
            f"{len(data['boosters'])} saved boosters"
        )
    for est, raw in zip(estimators, data["boosters"]):
        booster = xgb.Booster()
        booster.load_model(bytearray(raw))
        est._Booster = booster
    return model

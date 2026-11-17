import os

import numpy as np
import pandas as pd

from model_io import _xgb_estimators, load_portable, save_portable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "credit_risk_model.pkl")


def sample_rows():
    df = pd.read_csv(os.path.join(ROOT, "credit_risk_dataset.csv"))
    return df.drop(columns="loan_status").sample(200, random_state=0)


def test_model_file_loads_with_portable_boosters():
    model = load_portable(MODEL_PATH)
    estimators = _xgb_estimators(model)
    assert len(estimators) > 0
    assert all(est._Booster is not None for est in estimators)


def test_model_handles_missing_values():
    # The raw data has missing employment length and interest rate; the
    # pipeline's imputers must handle them instead of failing.
    rows = sample_rows()
    rows.iloc[:20, rows.columns.get_loc("loan_int_rate")] = np.nan
    rows.iloc[:20, rows.columns.get_loc("person_emp_length")] = np.nan
    prob = load_portable(MODEL_PATH).predict_proba(rows)[:, 1]
    assert np.all((prob >= 0) & (prob <= 1))


def test_save_and_load_give_identical_predictions(tmp_path):
    model = load_portable(MODEL_PATH)
    rows = sample_rows()
    before = model.predict_proba(rows)[:, 1]

    path = tmp_path / "copy.pkl"
    save_portable(model, path)
    after = load_portable(path).predict_proba(rows)[:, 1]

    np.testing.assert_array_equal(before, after)
    # Saving must leave the in-memory model usable.
    np.testing.assert_array_equal(before, model.predict_proba(rows)[:, 1])

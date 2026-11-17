import os

import numpy as np
import pandas as pd
import pytest

import db
import drift_check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "credit_risk_dataset.csv")


@pytest.fixture(scope="module")
def training():
    return drift_check.load_training_features(CSV)


def test_training_split_matches_notebook(training):
    X_train, y_train = training
    # Notebook: 32,409 rows after cleaning, 80% for training.
    assert len(X_train) == 25927
    assert y_train.mean() == pytest.approx(0.219, abs=1e-3)


def test_psi_numeric_same_vs_shifted():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(50, 10, 20000))
    same = pd.Series(rng.normal(50, 10, 2000))
    shifted = pd.Series(rng.normal(65, 10, 2000))
    assert drift_check.psi_numeric(reference, same) < drift_check.WATCH_AT
    assert drift_check.psi_numeric(reference, shifted) >= drift_check.DRIFT_AT


def test_psi_numeric_ignores_missing_values():
    reference = pd.Series([1.0, 2.0, np.nan, 3.0, 4.0] * 100)
    current = pd.Series([1.0, 2.0, 3.0, 4.0] * 50)
    assert drift_check.psi_numeric(reference, current) == pytest.approx(0.0, abs=1e-9)


def test_psi_categorical_new_category():
    reference = pd.Series(["A"] * 500 + ["B"] * 500)
    current = pd.Series(["A"] * 50 + ["B"] * 50 + ["C"] * 100)
    assert drift_check.psi_categorical(reference, reference) == pytest.approx(0.0)
    assert drift_check.psi_categorical(reference, current) >= drift_check.DRIFT_AT


@pytest.mark.parametrize("psi, expected", [(0.0, "OK"), (0.099, "OK"), (0.1, "WATCH"), (0.249, "WATCH"), (0.25, "DRIFT")])
def test_status_bands(psi, expected):
    assert drift_check.status(psi) == expected


def _write_log(path, rows):
    db.init_db(path)
    for features in rows.to_dict("records"):
        db.log_prediction(path, features, 0.2, 0, 0.604)


def test_report_flags_only_the_shifted_features(training, tmp_path):
    X_train, _ = training
    incoming = X_train.dropna().sample(400, random_state=1).copy()
    incoming["loan_int_rate"] = incoming["loan_int_rate"] + 5

    path = str(tmp_path / "log.db")
    _write_log(path, incoming)
    report = drift_check.drift_report(X_train, db.load_predictions(path)).set_index("feature")

    assert report.loc["loan_int_rate", "status"] == "DRIFT"
    assert (report.drop(index="loan_int_rate")["status"] == "OK").all()


def test_main_exit_codes(training, tmp_path, capsys):
    X_train, _ = training
    path = str(tmp_path / "log.db")

    # No log yet, then too few rows: nothing to judge, exit 0.
    assert drift_check.main(["--db", path, "--data", CSV]) == 0
    _write_log(path, X_train.dropna().sample(50, random_state=2))
    assert drift_check.main(["--db", path, "--data", CSV]) == 0
    assert "Too few" in capsys.readouterr().out

    # Normal traffic: exit 0. Shifted traffic: exit 1.
    _write_log(path, X_train.dropna().sample(450, random_state=3))
    assert drift_check.main(["--db", path, "--data", CSV]) == 0

    shifted = X_train.dropna().sample(2000, random_state=4).copy()
    shifted["person_income"] = shifted["person_income"] * 0.4
    _write_log(path, shifted)
    assert drift_check.main(["--db", path, "--data", CSV]) == 1

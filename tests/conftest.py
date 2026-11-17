import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main  # noqa: E402


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """Every test gets its own empty prediction log."""
    path = str(tmp_path / "predictions.db")
    monkeypatch.setattr(main, "PREDICTION_DB", path)
    return path


@pytest.fixture
def client(db_path):
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def risky_applicant():
    return {
        "person_age": 22,
        "person_income": 15000,
        "person_home_ownership": "RENT",
        "person_emp_length": 1,
        "loan_intent": "MEDICAL",
        "loan_grade": "D",
        "loan_amnt": 9000,
        "loan_int_rate": 16.5,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 2,
    }


@pytest.fixture
def safe_applicant():
    return {
        "person_age": 40,
        "person_income": 120000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 12,
        "loan_intent": "HOMEIMPROVEMENT",
        "loan_grade": "A",
        "loan_amnt": 5000,
        "loan_int_rate": 7.5,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 15,
    }

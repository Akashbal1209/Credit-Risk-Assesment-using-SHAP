import pytest
from fastapi.testclient import TestClient

import db
import main


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "prediction_log": True}


def test_web_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_response_shape(client, risky_applicant):
    body = client.post("/predict", json=risky_applicant).json()
    assert set(body) == {"default_probability", "default_prediction", "threshold", "Result"}
    assert 0.0 <= body["default_probability"] <= 1.0
    # Decision rule must be exactly the one in the README: probability >= cut-off -> high risk.
    assert body["default_prediction"] == int(body["default_probability"] >= body["threshold"])
    assert body["Result"] == ("High Risk" if body["default_prediction"] == 1 else "Low Risk")


def test_threshold_is_the_trained_one(client, safe_applicant):
    assert client.post("/predict", json=safe_applicant).json()["threshold"] == pytest.approx(0.604, abs=5e-4)


def test_known_applicants(client, risky_applicant, safe_applicant):
    # Fixed applicants with known scores. If the model file changes by accident,
    # these numbers change and the test fails.
    risky = client.post("/predict", json=risky_applicant).json()
    safe = client.post("/predict", json=safe_applicant).json()
    assert risky["Result"] == "High Risk"
    assert safe["Result"] == "Low Risk"
    assert risky["default_probability"] == pytest.approx(0.958861, abs=1e-4)
    assert safe["default_probability"] == pytest.approx(0.016154, abs=1e-4)


def test_borderline_applicant_uses_trained_cutoff(client):
    # Scores 0.576: high risk with a plain 0.5 cut-off, low risk with the trained
    # 0.604 cut-off. Catches the API falling back to 0.5 by mistake.
    applicant = {
        "person_age": 22,
        "person_income": 12000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 7,
        "loan_intent": "EDUCATION",
        "loan_grade": "D",
        "loan_amnt": 1500,
        "loan_int_rate": 14.84,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 3,
    }
    body = client.post("/predict", json=applicant).json()
    assert body["default_probability"] == pytest.approx(0.576241, abs=1e-4)
    assert body["default_prediction"] == 0
    assert body["Result"] == "Low Risk"


def test_loan_percent_income_is_recalculated(client, db_path, risky_applicant):
    # A wrong value sent by the client must be ignored and recalculated.
    client.post("/predict", json={**risky_applicant, "loan_percent_income": 0.01})
    logged = db.load_predictions(db_path)
    assert logged.loc[0, "loan_percent_income"] == pytest.approx(round(9000 / 15000, 2))


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"person_age": 150}, "age above 100"),
        ({"person_age": 17}, "age below 18"),
        ({"person_income": 0}, "zero income"),
        ({"loan_amnt": -5}, "negative loan amount"),
        ({"loan_int_rate": 55}, "interest rate above 40"),
        ({"loan_grade": "Z"}, "unknown grade"),
        ({"person_home_ownership": "CASTLE"}, "unknown home ownership"),
        ({"loan_intent": "VACATION"}, "unknown loan intent"),
        ({"person_age": 25, "person_emp_length": 30}, "employment longer than age"),
        ({"cb_person_default_on_file": "Y", "loan_grade": "A"}, "prior default with grade A"),
        ({"cb_person_default_on_file": "Y", "loan_grade": "B"}, "prior default with grade B"),
    ],
)
def test_invalid_input_returns_422(client, risky_applicant, change, reason):
    assert client.post("/predict", json={**risky_applicant, **change}).status_code == 422, reason


def test_missing_field_returns_422(client, risky_applicant):
    del risky_applicant["loan_grade"]
    assert client.post("/predict", json=risky_applicant).status_code == 422


def test_predictions_are_logged_and_counted(client, risky_applicant, safe_applicant):
    for _ in range(3):
        client.post("/predict", json=risky_applicant)
    client.post("/predict", json=safe_applicant)
    client.post("/predict", json={**risky_applicant, "person_age": 150})  # rejected, not logged

    stats = client.get("/stats").json()
    assert stats["total_predictions"] == 4
    assert stats["high_risk"] == 3
    assert stats["high_risk_rate"] == 0.75
    grades = {g["loan_grade"]: g for g in stats["by_loan_grade"]}
    assert grades["D"]["applications"] == 3 and grades["D"]["high_risk"] == 3
    assert grades["A"]["applications"] == 1 and grades["A"]["high_risk"] == 0


def test_stats_on_empty_log(client):
    stats = client.get("/stats").json()
    assert stats["total_predictions"] == 0
    assert stats["high_risk_rate"] is None
    assert stats["by_loan_grade"] == []


def test_prediction_still_works_if_logging_fails(client, risky_applicant, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(db, "log_prediction", broken)
    response = client.post("/predict", json=risky_applicant)
    assert response.status_code == 200
    assert response.json()["Result"] == "High Risk"


def test_api_runs_without_a_log(tmp_path, monkeypatch, risky_applicant):
    # The log folder can't be created (its parent is a file): predictions keep
    # working, /health says so, and /stats answers 503 instead of crashing.
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")
    monkeypatch.setattr(main, "PREDICTION_DB", str(blocker / "predictions.db"))
    with TestClient(main.app) as c:
        assert c.get("/health").json()["prediction_log"] is False
        assert c.post("/predict", json=risky_applicant).status_code == 200
        assert c.get("/stats").status_code == 503

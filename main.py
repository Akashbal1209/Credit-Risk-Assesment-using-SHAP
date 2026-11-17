import logging
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

import db
from model_io import load_portable

logger = logging.getLogger("uvicorn.error")

# Resolve every path from this file's folder, so the app works no matter
# which directory uvicorn is started from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "credit_risk_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "best_threshold.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")
# SQLite file that logs every prediction. Override with PREDICTION_DB.
PREDICTION_DB = os.environ.get("PREDICTION_DB", os.path.join(BASE_DIR, "data", "predictions.db"))

# No applicant with a prior default has grade A or B in the training data.
GRADES_BLOCKED_ON_DEFAULT = {"A", "B"}

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load_portable, not joblib.load: the XGBoost part is stored in a format
    # that loads on any OS (see model_io.py).
    ml_model["model"] = load_portable(MODEL_PATH)
    ml_model["threshold"] = float(joblib.load(THRESHOLD_PATH))
    # The log is an extra: if the database can't be created, predictions still work.
    try:
        db.init_db(PREDICTION_DB)
        ml_model["db_ready"] = True
    except Exception:
        logger.exception("Prediction log disabled: could not open %s", PREDICTION_DB)
        ml_model["db_ready"] = False
    yield
    ml_model.clear()


app = FastAPI(title="Credit Risk Assessment API", lifespan=lifespan)


# The inputs the model was trained on. Allowed categories and ranges match the
# training data, so a typo or an impossible value gets a 422 instead of a
# silent, meaningless prediction.
class LoanApplication(BaseModel):
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: float = Field(ge=0, le=60)
    loan_intent: Literal[
        "EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(gt=0)
    loan_int_rate: float = Field(gt=0, le=40)
    # Optional and ignored if sent: it is always recalculated from loan_amnt and
    # person_income, so it can never disagree with them.
    loan_percent_income: Optional[float] = None
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(ge=0, le=80)

    @model_validator(mode="after")
    def check_combinations(self):
        if self.person_emp_length > self.person_age:
            raise ValueError("person_emp_length can't be more than person_age")
        if self.cb_person_default_on_file == "Y" and self.loan_grade in GRADES_BLOCKED_ON_DEFAULT:
            raise ValueError("applicants with a prior default must have loan_grade C or lower")
        self.loan_percent_income = round(self.loan_amnt / self.person_income, 2)
        return self


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in ml_model,
        "prediction_log": bool(ml_model.get("db_ready")),
    }


@app.post("/predict")
def predict(data: LoanApplication):
    features = data.model_dump()
    input_df = pd.DataFrame([features])

    probability = float(ml_model["model"].predict_proba(input_df)[:, 1][0])
    threshold = ml_model["threshold"]
    prediction = int(probability >= threshold)

    if ml_model.get("db_ready"):
        try:
            db.log_prediction(PREDICTION_DB, features, probability, prediction, threshold)
        except Exception:
            # Never fail a prediction because the log couldn't be written.
            logger.exception("Could not log prediction")

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk",
    }


@app.get("/stats")
def stats():
    """Summary of the logged predictions: totals, high-risk rate, and per loan grade."""
    if not ml_model.get("db_ready"):
        raise HTTPException(status_code=503, detail="Prediction log is not available")
    return db.get_stats(PREDICTION_DB)


# Mounted last so /predict, /health and /stats are matched before the static files.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

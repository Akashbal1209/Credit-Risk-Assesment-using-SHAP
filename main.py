import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from model_io import load_portable

# Resolve every path from this file's folder, so the app works no matter
# which directory uvicorn is started from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "credit_risk_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "best_threshold.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# No applicant with a prior default has grade A or B in the training data.
GRADES_BLOCKED_ON_DEFAULT = {"A", "B"}

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load_portable, not joblib.load: the XGBoost part is stored in a format
    # that loads on any OS (see model_io.py).
    ml_model["model"] = load_portable(MODEL_PATH)
    ml_model["threshold"] = float(joblib.load(THRESHOLD_PATH))
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
    return {"status": "ok", "model_loaded": "model" in ml_model}


@app.post("/predict")
def predict(data: LoanApplication):
    input_df = pd.DataFrame([data.model_dump()])

    probability = float(ml_model["model"].predict_proba(input_df)[:, 1][0])
    threshold = ml_model["threshold"]
    prediction = int(probability >= threshold)

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk",
    }


# Mounted last so /predict and /health are matched before the static files.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

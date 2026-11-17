# Credit Risk Assessment System

A machine learning web app that estimates the **probability that a loan applicant will default**.
You fill in the applicant's details in the browser, a FastAPI backend runs a calibrated XGBoost model,
and the result card shows the default probability, the risk verdict and the cut-off used.

> **Live demo:** <https://credit-risk-assesment-using-shap-08yl.onrender.com>
> (Free Render instances sleep when idle, so the first request can take up to a minute.)

---

## Features

- **Calibrated XGBoost model** — the probability it shows is close to the real default rate
  (Brier score improved from 0.0608 to 0.0511 after calibration).
- **Tuned decision threshold** — chosen on training data only, so the test scores are honest.
- **Explainability with SHAP** — global feature importance and per-applicant waterfall plots in the notebook.
- **Strict API validation** — wrong categories, impossible values or impossible combinations
  (employment longer than age, grade A/B with a prior default) get a `422` instead of a silent bad prediction.
- **Server-side feature calculation** — `loan_percent_income` is always calculated from loan amount and income,
  so the client can't send a value that disagrees with them.
- **Keyboard and screen-reader friendly** — chip groups work with arrow keys, and the hidden side of the card
  can't be reached with Tab.
- **Frontend rules taken from the data** — interest rate must sit inside the range seen for the chosen loan grade,
  and an applicant with a prior default cannot get grade A or B (no such rows exist in the training data).
- **Cross-platform model file** — the XGBoost part is stored in XGBoost's portable format, so a model trained
  on Linux also loads on Windows and macOS.
- **Single deployable service** — FastAPI serves both the `/predict` API and the static frontend.

## Tech Stack

| Layer      | Tools |
|------------|-------|
| ML         | scikit-learn (Pipeline, ColumnTransformer, CalibratedClassifierCV), XGBoost, SHAP |
| Data       | pandas, NumPy, Matplotlib, Seaborn |
| Backend    | FastAPI, Pydantic, Uvicorn, joblib |
| Frontend   | HTML, CSS, vanilla JavaScript |
| Deployment | Render |

## Project Structure

```
.
├── main.py                  # FastAPI app: /predict, /health, serves the frontend
├── model_io.py              # Saves/loads the model in a format that works on any OS
├── credit_risk_model.pkl    # Calibrated XGBoost pipeline (preprocessing + model)
├── best_threshold.pkl       # Decision threshold for "High Risk"
├── Credit_Risk.ipynb        # Full workflow: EDA → cleaning → training → tuning → calibration → SHAP
├── credit_risk_dataset.csv  # Training data
├── requirements.txt
├── .python-version
└── static/
    ├── index.html
    ├── style.css
    └── script.js
```

## Dataset

[Credit Risk Dataset (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) — 32,581 loan records, 11 input features and one target.

| Column | Meaning |
|--------|---------|
| `person_age` | Applicant age |
| `person_income` | Annual income |
| `person_home_ownership` | RENT / OWN / MORTGAGE / OTHER |
| `person_emp_length` | Employment length in years |
| `loan_intent` | EDUCATION / MEDICAL / VENTURE / PERSONAL / DEBTCONSOLIDATION / HOMEIMPROVEMENT |
| `loan_grade` | A (best) to G (worst) |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate (%) |
| `loan_percent_income` | Loan amount ÷ income |
| `cb_person_default_on_file` | Prior default on record (Y / N) |
| `cb_person_cred_hist_length` | Credit history length in years |
| **`loan_status`** | **Target — 1 = defaulted, 0 = repaid** |

The target is imbalanced: about 22% of applicants defaulted.

## ML Workflow

1. **EDA** — data types, missing values (`person_emp_length`, `loan_int_rate`), class balance, outliers, correlations.
2. **Cleaning** — removed 165 duplicate rows, ages outside 18–100, employment length above 60 or above age,
   and non-positive income or loan amount. Rows with missing values are kept and imputed inside the pipeline.
   32,409 rows remain.
3. **Split** — 80/20 stratified train-test split.
4. **Class imbalance** — `class_weight="balanced"` for Logistic Regression, `scale_pos_weight` (≈3.57) for XGBoost.
5. **Preprocessing pipelines** — median imputation (+ scaling for Logistic Regression) for numeric columns,
   constant imputation + one-hot encoding for categorical columns. Everything lives inside one sklearn
   `Pipeline`, so the API sends raw values and gets the same preprocessing as training.
6. **5-fold stratified cross-validation** — Logistic Regression baseline vs XGBoost.
7. **Hyperparameter tuning** — `RandomizedSearchCV`, 150 candidates × 5 folds, optimising average precision.
8. **Probability calibration** — sigmoid (Platt) scaling with `CalibratedClassifierCV`.
9. **Threshold selection** — out-of-fold predictions on the training set, threshold with the best F1 score.
   The test set is not used for this step.
10. **Explainability** — SHAP summary plot (global) and waterfall plot (single applicant).
11. **Error analysis** — false positives vs false negatives on the test set.

## Results (test set: 6,482 applicants, 1,418 defaults)

| Model | Threshold | ROC-AUC | Precision | Recall | F1 | Accuracy |
|-------|-----------|---------|-----------|--------|----|----------|
| Logistic Regression (baseline) | 0.5 | 0.872 | 0.539 | 0.782 | 0.638 | 0.806 |
| XGBoost (tuned) | 0.5 | 0.954 | 0.840 | 0.807 | 0.823 | 0.924 |
| **XGBoost (tuned + calibrated) — deployed** | **0.604** | **0.953** | **0.957** | **0.758** | **0.846** | **0.940** |

At the deployed threshold the model makes 48 false positives (safe applicants flagged) and 343 false negatives
(defaulters missed). A lender who cares more about catching defaulters can lower the threshold to trade
precision for recall.

## API

### `POST /predict`

Request:

```json
{
  "person_age": 28,
  "person_income": 54000,
  "person_home_ownership": "RENT",
  "person_emp_length": 4,
  "loan_intent": "EDUCATION",
  "loan_grade": "B",
  "loan_amnt": 12000,
  "loan_int_rate": 11.4,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```

Response:

```json
{
  "default_probability": 0.384,
  "default_prediction": 0,
  "threshold": 0.604,
  "Result": "Low Risk"
}
```

`default_prediction` is `1` (High Risk) when `default_probability >= threshold`.

`loan_percent_income` is calculated on the server as `round(loan_amnt / person_income, 2)`. It may be sent,
but the sent value is ignored.

Invalid input returns `422` with the reason, for example:

- a category outside the training data (such as `loan_grade: "Z"`)
- a value out of range (such as `person_age: 15`)
- `person_emp_length` greater than `person_age`
- `loan_grade` A or B together with `cb_person_default_on_file: "Y"`

### `GET /health`

Returns `{"status": "ok", "model_loaded": true}`.

Interactive API docs are available at `/docs` when the server is running.

## Run Locally

Requires Python 3.13.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
```

Activate the virtual environment:

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- macOS / Linux: `source .venv/bin/activate`

Install and start:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000 for the app, or http://127.0.0.1:8000/docs for the API docs.

## Retrain the Model

Open `Credit_Risk.ipynb` (Jupyter or Google Colab), keep `credit_risk_dataset.csv` and `model_io.py` next to it
(in Colab, upload both) and run all cells. The last two cells write `credit_risk_model.pkl` and `best_threshold.pkl`.
Copy both into the project root.

The model is saved with `model_io.save_portable` instead of a plain pickle. A plain pickle of an XGBoost model
trained on Linux (such as Colab) fails to load on Windows with `XGBoostError: input stream corrupted`;
the portable file loads on Linux, Windows and macOS.
Train with the same library versions as `requirements.txt`, otherwise the pickle may not load on the server.
The notebook also needs a few extra packages that the server does not:

```bash
pip install -r requirements.txt
pip install jupyter matplotlib seaborn shap
```

## Deploy on Render

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Health check path:** `/health`

## Limitations

- Trained on one public dataset; it is not validated on real lending data from any institution.
- The model output is meant to support a decision, not to make one on its own.
- `loan_grade` and `loan_int_rate` are usually set by a lender after its own risk review, so they already carry
  part of the risk signal the model learns from.

## Author

**Aman Raj**

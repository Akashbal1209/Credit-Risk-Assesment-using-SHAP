"""Check whether incoming applicants still look like the training data.

A model is only reliable for the kind of people it was trained on. If incomes,
loan sizes or interest rates in real traffic shift away from the training data,
the scores slowly stop meaning what they used to, even though the API keeps
answering normally. This script catches that.

It compares every feature in the prediction log (data/predictions.db) with the
training split of the notebook, using the Population Stability Index (PSI):

    PSI < 0.10        OK     - no real change
    0.10 <= PSI < 0.25 WATCH - some change, keep an eye on it
    PSI >= 0.25       DRIFT  - big change, the model should be checked/retrained

Usage:
    python drift_check.py
    python drift_check.py --db data/predictions.db --data credit_risk_dataset.csv

Exit code 1 when any feature shows DRIFT, so it can run in a scheduled job.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_COLS = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

WATCH_AT = 0.10
DRIFT_AT = 0.25
EPS = 1e-4  # stands in for an empty bin, so log(0) never happens


def load_training_features(csv_path):
    """The exact training split from the notebook: same cleaning, same split."""
    df = pd.read_csv(csv_path).drop_duplicates()
    df = df[df["person_age"].between(18, 100)]
    emp = df["person_emp_length"]
    df = df[emp.isna() | ((emp <= df["person_age"]) & (emp <= 60))]
    df = df[(df["person_income"] > 0) & (df["loan_amnt"] > 0)]

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, y_train


def _psi(ref_share, cur_share):
    ref_share = np.clip(ref_share, EPS, None)
    cur_share = np.clip(cur_share, EPS, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_numeric(reference, current, bins=10):
    # Bin edges from the training data's deciles, open-ended at both sides so
    # values outside the training range still land in a bin.
    reference = reference.dropna()
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.concatenate([[-np.inf], edges, [np.inf]])
    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current.dropna(), bins=edges)[0]
    return _psi(ref_counts / ref_counts.sum(), cur_counts / cur_counts.sum())


def psi_categorical(reference, current):
    categories = sorted(set(reference.dropna()) | set(current.dropna()))
    ref_share = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    cur_share = current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return _psi(ref_share, cur_share)


def status(psi):
    if psi >= DRIFT_AT:
        return "DRIFT"
    if psi >= WATCH_AT:
        return "WATCH"
    return "OK"


def drift_report(reference, current):
    rows = []
    for col in NUMERIC_COLS:
        psi = psi_numeric(reference[col], current[col])
        rows.append({
            "feature": col,
            "psi": psi,
            "status": status(psi),
            "training": f"median {reference[col].median():,.2f}",
            "incoming": f"median {current[col].median():,.2f}",
        })
    for col in CATEGORICAL_COLS:
        psi = psi_categorical(reference[col], current[col])
        top_ref = reference[col].value_counts(normalize=True)
        top_cur = current[col].value_counts(normalize=True)
        rows.append({
            "feature": col,
            "psi": psi,
            "status": status(psi),
            "training": f"{top_ref.index[0]} {top_ref.iloc[0]:.0%}",
            "incoming": f"{top_cur.index[0]} {top_cur.iloc[0]:.0%}",
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=os.path.join(BASE_DIR, "data", "predictions.db"),
                        help="prediction log written by the API")
    parser.add_argument("--data", default=os.path.join(BASE_DIR, "credit_risk_dataset.csv"),
                        help="raw dataset the model was trained on")
    parser.add_argument("--min-rows", type=int, default=100,
                        help="fewer logged predictions than this are too few to judge drift")
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"No prediction log at {args.db}. Run the API and make some predictions first.")
        return 0

    current = db.load_predictions(args.db)
    if len(current) < args.min_rows:
        print(f"Only {len(current)} logged predictions (need at least {args.min_rows}). "
              "Too few to tell real drift from noise.")
        return 0

    reference, y_train = load_training_features(args.data)
    report = drift_report(reference, current)

    print(f"Training rows : {len(reference):,}")
    print(f"Incoming rows : {len(current):,}  ({current['created_at'].min()} to {current['created_at'].max()})")
    print()
    print(report.to_string(index=False, formatters={"psi": "{:.3f}".format}))
    print()
    # The model is calibrated, so on applicants like the training data the
    # average predicted probability should sit near the training default rate.
    print(f"Training default rate              : {y_train.mean():.1%}")
    print(f"Average predicted default (incoming): {current['default_probability'].mean():.1%}")
    print(f"High-risk rate (incoming)          : {current['default_prediction'].mean():.1%}")
    print()

    drifted = report.loc[report["status"] == "DRIFT", "feature"].tolist()
    watch = report.loc[report["status"] == "WATCH", "feature"].tolist()
    if drifted:
        print(f"DRIFT in: {', '.join(drifted)}. Check the model on recent data before trusting its scores.")
        return 1
    if watch:
        print(f"Some change in: {', '.join(watch)}. No action needed yet; keep watching.")
    else:
        print("No drift: incoming applicants look like the training data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

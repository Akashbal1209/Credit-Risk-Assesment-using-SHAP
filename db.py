"""Prediction log stored in SQLite.

Every /predict call is saved with its inputs and result. That makes two things
possible that a stateless API can't do:
- /stats: how many applications came in and how many were flagged high risk.
- drift_check.py: compare incoming applicants with the training data, to notice
  when the model starts seeing people it wasn't trained on.

SQLite is a single file and needs no server. On Render's free plan the disk is
temporary, so the log is lost when the service restarts; for a permanent log the
same table can live in Postgres.
"""

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

FEATURE_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at                 TEXT    NOT NULL,  -- UTC, ISO 8601
    person_age                 INTEGER NOT NULL,
    person_income              REAL    NOT NULL,
    person_home_ownership      TEXT    NOT NULL,
    person_emp_length          REAL    NOT NULL,
    loan_intent                TEXT    NOT NULL,
    loan_grade                 TEXT    NOT NULL,
    loan_amnt                  REAL    NOT NULL,
    loan_int_rate              REAL    NOT NULL,
    loan_percent_income        REAL    NOT NULL,
    cb_person_default_on_file  TEXT    NOT NULL,
    cb_person_cred_hist_length INTEGER NOT NULL,
    default_probability        REAL    NOT NULL,
    default_prediction         INTEGER NOT NULL CHECK (default_prediction IN (0, 1)),
    threshold                  REAL    NOT NULL
)
"""


def _connect(path):
    # A new connection per call: FastAPI runs sync endpoints in a thread pool,
    # and a SQLite connection must not be shared between threads.
    # timeout: wait for a lock instead of failing when two requests write at once.
    return sqlite3.connect(path, timeout=10)


def init_db(path):
    """Create the folder and the table if they don't exist yet."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with closing(_connect(path)) as conn, conn:
        # WAL lets /stats read while /predict writes.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_CREATE_TABLE)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at)")


def log_prediction(path, features, probability, prediction, threshold):
    columns = ["created_at", *FEATURE_COLUMNS, "default_probability", "default_prediction", "threshold"]
    values = [
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        *(features[col] for col in FEATURE_COLUMNS),
        float(probability),
        int(prediction),
        float(threshold),
    ]
    placeholders = ", ".join("?" for _ in columns)
    with closing(_connect(path)) as conn, conn:
        conn.execute(f"INSERT INTO predictions ({', '.join(columns)}) VALUES ({placeholders})", values)


def get_stats(path):
    """Summary of everything logged so far, computed in SQL."""
    with closing(_connect(path)) as conn:
        total, high_risk, avg_prob, first_at, last_at = conn.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(default_prediction), 0),
                   AVG(default_probability),
                   MIN(created_at),
                   MAX(created_at)
            FROM predictions
            """
        ).fetchone()

        by_grade = conn.execute(
            """
            SELECT loan_grade,
                   COUNT(*)                 AS applications,
                   SUM(default_prediction)  AS high_risk,
                   AVG(default_probability) AS avg_probability
            FROM predictions
            GROUP BY loan_grade
            ORDER BY loan_grade
            """
        ).fetchall()

    return {
        "total_predictions": total,
        "high_risk": high_risk,
        "high_risk_rate": round(high_risk / total, 4) if total else None,
        "average_default_probability": round(avg_prob, 4) if avg_prob is not None else None,
        "first_prediction_at": first_at,
        "last_prediction_at": last_at,
        "by_loan_grade": [
            {
                "loan_grade": grade,
                "applications": n,
                "high_risk": hr,
                "high_risk_rate": round(hr / n, 4),
                "average_default_probability": round(avg, 4),
            }
            for grade, n, hr, avg in by_grade
        ],
    }


def load_predictions(path):
    """All logged rows as a DataFrame (used by drift_check.py)."""
    import pandas as pd

    with closing(_connect(path)) as conn:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY id", conn)

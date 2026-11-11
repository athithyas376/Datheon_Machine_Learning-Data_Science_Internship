"""
predict.py - Inference helpers: single + bulk prediction, risk bands and
retention recommendations. Used by app.py and usable from the command line:

    python predict.py            # scores a sample customer
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from train_model import (CATEGORICAL_FEATURES, FEATURES, ID_COL,
                         NUMERIC_FEATURES, REQUIRED_COLUMNS, clean_features)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

LOW_MAX = 0.30      # probability < 0.30  -> Low
HIGH_MIN = 0.60     # probability >= 0.60 -> High, otherwise Medium

ALLOWED_VALUES = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

_cache = {}


class ValidationError(ValueError):
    pass


def load_model():
    if "model" not in _cache:
        path = os.path.join(MODEL_DIR, "best_model.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError("Model not trained yet. Run: python train_model.py")
        _cache["model"] = joblib.load(path)
    return _cache["model"]


def load_metrics():
    if "metrics" not in _cache:
        path = os.path.join(MODEL_DIR, "metrics.json")
        if not os.path.exists(path):
            raise FileNotFoundError("metrics.json missing. Run: python train_model.py")
        with open(path) as fh:
            _cache["metrics"] = json.load(fh)
    return _cache["metrics"]


def threshold() -> float:
    m = load_metrics()
    return float(m["models"][m["best_model"]]["threshold"])


def risk_level(p: float) -> str:
    return "Low" if p < LOW_MAX else ("High" if p >= HIGH_MIN else "Medium")


def validate_frame(df: pd.DataFrame):
    """Validate an uploaded frame; return (clean_df, list_of_warnings)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError("Missing required column(s): " + ", ".join(missing))
    if df.empty:
        raise ValidationError("The file contains no rows.")
    df = clean_features(df)
    warnings = []
    bad_num = int(df[["tenure", "MonthlyCharges", "SeniorCitizen"]].isna().any(axis=1).sum())
    if bad_num:
        warnings.append(f"{bad_num} row(s) had non-numeric tenure/MonthlyCharges/SeniorCitizen; "
                        "missing values were imputed.")
    if int(df["TotalCharges"].isna().sum()):
        warnings.append(f"{int(df['TotalCharges'].isna().sum())} row(s) had blank TotalCharges; imputed.")
    for col in CATEGORICAL_FEATURES:
        unknown = set(df[col].dropna().unique()) - set(ALLOWED_VALUES[col])
        unknown -= {"nan"}
        if unknown:
            warnings.append(f"Column '{col}' has unseen value(s) {sorted(unknown)[:3]}; "
                            "treated as unknown category.")
    return df, warnings


def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with churn_probability, predicted_churn and risk_level."""
    model = load_model()
    proba = model.predict_proba(df[FEATURES])[:, 1]
    out = pd.DataFrame(index=df.index)
    out["churn_probability"] = np.round(proba, 4)
    out["predicted_churn"] = np.where(proba >= threshold(), "Yes", "No")
    out["risk_level"] = [risk_level(p) for p in proba]
    return out


def recommendations(c: dict, p: float) -> list:
    """Rule-based retention actions driven by the customer's profile."""
    recs = []
    if c.get("Contract") == "Month-to-month":
        recs.append("Offer a discounted 1- or 2-year contract; month-to-month customers churn the most.")
    if float(c.get("tenure") or 0) <= 12:
        recs.append("Start a first-year onboarding programme with check-in calls and a loyalty bonus.")
    if c.get("InternetService") == "Fiber optic" and float(c.get("MonthlyCharges") or 0) > 70:
        recs.append("Review the fibre plan price: offer a right-sized bundle or a limited-time discount.")
    if c.get("InternetService") != "No" and c.get("TechSupport") == "No":
        recs.append("Include free Tech Support for 3-6 months to lift satisfaction.")
    if c.get("InternetService") != "No" and c.get("OnlineSecurity") == "No":
        recs.append("Bundle Online Security at a promotional rate to deepen engagement.")
    if c.get("PaymentMethod") == "Electronic check":
        recs.append("Encourage automatic payment (card / bank transfer) with a small bill credit.")
    if float(c.get("MonthlyCharges") or 0) > 90:
        recs.append("Proactively review the bill with a premium-customer account manager.")
    if p >= HIGH_MIN:
        recs.insert(0, "Escalate to the retention team for a personal outreach within 48 hours.")
    if not recs:
        recs.append("Customer looks stable: keep service quality high and send periodic loyalty rewards.")
    return recs[:6]


def predict_one(payload: dict) -> dict:
    """Validate and score one customer submitted as a dict."""
    row = {}
    for col in REQUIRED_COLUMNS:
        if col not in payload or payload[col] in (None, ""):
            if col == "TotalCharges":
                row[col] = None
                continue
            raise ValidationError(f"Field '{col}' is required.")
        row[col] = payload[col]
    for col in ["tenure", "MonthlyCharges"]:
        try:
            if float(row[col]) < 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValidationError(f"'{col}' must be a non-negative number.")
    if row["TotalCharges"] is None:
        row["TotalCharges"] = float(row["tenure"]) * float(row["MonthlyCharges"])
    else:
        try:
            if float(row["TotalCharges"]) < 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValidationError("'TotalCharges' must be a non-negative number.")
    if float(row["tenure"]) > 100:
        raise ValidationError("'tenure' must be between 0 and 100 months.")
    for col in CATEGORICAL_FEATURES:
        if str(row[col]) not in ALLOWED_VALUES[col]:
            raise ValidationError(f"'{col}' must be one of {ALLOWED_VALUES[col]}.")
    df, _ = validate_frame(pd.DataFrame([row]))
    res = score_frame(df).iloc[0]
    p = float(res["churn_probability"])
    return {
        "churn_probability": round(p * 100, 1),
        "predicted_churn": res["predicted_churn"],
        "risk_level": res["risk_level"],
        "threshold": round(threshold() * 100, 1),
        "model": load_metrics()["best_model"],
        "recommendations": recommendations(row, p),
    }


if __name__ == "__main__":
    sample = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 280.0,
    }
    print(json.dumps(predict_one(sample), indent=2))

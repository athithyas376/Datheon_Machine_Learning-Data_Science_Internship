"""
app.py - Flask application serving the churn dashboard and its REST API.

    pip install -r requirements.txt
    python app.py          # trains models automatically on first run
"""
import io
import json
import os
import re
import threading
import uuid

import numpy as np
import pandas as pd
from flask import (Flask, Response, jsonify, render_template, request,
                   send_file)

import predict as P
import train_model as T

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MAX_ROWS = 50000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB uploads
os.makedirs(UPLOAD_DIR, exist_ok=True)

_lock = threading.Lock()
_state = {}


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def models_ready() -> bool:
    return all(os.path.exists(os.path.join(P.MODEL_DIR, f)) for f in
               ["best_model.joblib", "metrics.json", "customer_scores.csv"])


def get_data() -> pd.DataFrame:
    """Dataset joined with out-of-fold churn probabilities and risk levels."""
    if "df" not in _state:
        df = T.load_dataset()
        scores = pd.read_csv(os.path.join(P.MODEL_DIR, "customer_scores.csv"))
        df = df.merge(scores, on=T.ID_COL, how="left")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)
        df["risk_level"] = df["churn_probability"].map(P.risk_level)
        df["Senior"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})
        df["ChurnLabel"] = df["Churn"].map({1: "Yes", 0: "No"})
        _state["df"] = df
    return _state["df"]


def reset_cache():
    _state.clear()
    P._cache.clear()


def clean(obj):
    """Make nested structures JSON safe (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    return obj


def ok(payload, status=200):
    return jsonify(clean(payload)), status


def err(message, status=400, **extra):
    return jsonify({"error": message, **extra}), status


def group_churn(df, col, order=None):
    g = df.groupby(col)["Churn"].agg(["count", "sum"]).rename(
        columns={"count": "total", "sum": "churned"})
    if order:
        g = g.reindex([o for o in order if o in g.index])
    g["retained"] = g["total"] - g["churned"]
    g["churn_rate"] = (g["churned"] / g["total"] * 100).round(1)
    return {"labels": [str(i) for i in g.index], "total": g["total"].tolist(),
            "churned": g["churned"].tolist(), "retained": g["retained"].tolist(),
            "churn_rate": g["churn_rate"].tolist()}


TENURE_BINS = [-1, 12, 24, 48, 60, 100]
TENURE_LABELS = ["0-12 mo", "13-24 mo", "25-48 mo", "49-60 mo", "61+ mo"]
CHARGE_BINS = [0, 30, 50, 70, 90, 200]
CHARGE_LABELS = ["<$30", "$30-50", "$50-70", "$70-90", "$90+"]


def filter_customers(args) -> pd.DataFrame:
    df = get_data()
    q = (args.get("q") or "").strip().lower()
    if q:
        mask = (df[T.ID_COL].str.lower().str.contains(q, regex=False)
                | df["Contract"].str.lower().str.contains(q, regex=False)
                | df["InternetService"].str.lower().str.contains(q, regex=False)
                | df["PaymentMethod"].str.lower().str.contains(q, regex=False)
                | df["gender"].str.lower().eq(q))
        df = df[mask]
    churn = args.get("churn", "")
    if churn in ("Yes", "No"):
        df = df[df["ChurnLabel"] == churn]
    contract = args.get("contract", "")
    if contract in P.ALLOWED_VALUES["Contract"]:
        df = df[df["Contract"] == contract]
    risk = args.get("risk", "")
    if risk in ("Low", "Medium", "High"):
        df = df[df["risk_level"] == risk]
    return df


CUSTOMER_COLS = [T.ID_COL, "gender", "Senior", "tenure", "Contract", "InternetService",
                 "PaymentMethod", "MonthlyCharges", "TotalCharges", "ChurnLabel",
                 "churn_probability", "risk_level"]


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
PAGES = {
    "/": ("index.html", "overview"),
    "/analytics": ("analytics.html", "analytics"),
    "/prediction": ("prediction.html", "prediction"),
    "/bulk-prediction": ("bulk_prediction.html", "bulk"),
    "/model-performance": ("model_performance.html", "model"),
    "/retention": ("retention.html", "retention"),
    "/customers": ("customers.html", "customers"),
    "/about": ("about.html", "about"),
}


def make_page(template, active):
    def view():
        return render_template(template, active=active, allowed=P.ALLOWED_VALUES)
    view.__name__ = "page_" + active
    return view


for route, (tpl, act) in PAGES.items():
    app.add_url_rule(route, view_func=make_page(tpl, act))


# --------------------------------------------------------------------------
# API: dashboard data
# --------------------------------------------------------------------------
@app.get("/api/stats")
def api_stats():
    df = get_data()
    churned = int(df["Churn"].sum())
    total = len(df)
    retained_df = df[df["Churn"] == 0]
    high_retained = retained_df[retained_df["risk_level"] == "High"]
    hist, edges = np.histogram(df["MonthlyCharges"], bins=20)
    return ok({
        "total_customers": total,
        "churned": churned,
        "retained": total - churned,
        "churn_rate": round(churned / total * 100, 2),
        "avg_monthly_charges": round(float(df["MonthlyCharges"].mean()), 2),
        "revenue_at_risk": round(float(high_retained["MonthlyCharges"].sum()), 2),
        "revenue_lost": round(float(df[df["Churn"] == 1]["MonthlyCharges"].sum()), 2),
        "high_risk_retained": int(len(high_retained)),
        "charges_hist": {
            "centers": [round(float((edges[i] + edges[i + 1]) / 2), 1) for i in range(len(hist))],
            "all": hist.tolist(),
            "churned": np.histogram(df[df.Churn == 1]["MonthlyCharges"], bins=edges)[0].tolist(),
            "retained": np.histogram(df[df.Churn == 0]["MonthlyCharges"], bins=edges)[0].tolist(),
        },
        "contract": group_churn(df, "Contract", P.ALLOWED_VALUES["Contract"]),
        "risk": {
            "levels": ["Low", "Medium", "High"],
            "all": [int((df.risk_level == r).sum()) for r in ["Low", "Medium", "High"]],
            "retained": [int((retained_df.risk_level == r).sum()) for r in ["Low", "Medium", "High"]],
            "actual_churn_rate": [round(float(df[df.risk_level == r]["Churn"].mean() * 100), 1)
                                  for r in ["Low", "Medium", "High"]],
            "bands": {"low_max": P.LOW_MAX, "high_min": P.HIGH_MIN},
        },
        "model": P.load_metrics()["best_model"],
    })


@app.get("/api/analytics")
def api_analytics():
    df = get_data().copy()
    df["TenureBand"] = pd.cut(df["tenure"], TENURE_BINS, labels=TENURE_LABELS)
    df["ChargeBand"] = pd.cut(df["MonthlyCharges"], CHARGE_BINS, labels=CHARGE_LABELS)
    return ok({
        "gender": group_churn(df, "gender"),
        "senior": group_churn(df, "Senior"),
        "partner": group_churn(df, "Partner"),
        "dependents": group_churn(df, "Dependents"),
        "internet": group_churn(df, "InternetService"),
        "contract": group_churn(df, "Contract", P.ALLOWED_VALUES["Contract"]),
        "payment": group_churn(df, "PaymentMethod"),
        "tenure": group_churn(df, "TenureBand", TENURE_LABELS),
        "charges": group_churn(df, "ChargeBand", CHARGE_LABELS),
        "tech_support": group_churn(df, "TechSupport"),
        "tenure_values": {"churned": df[df.Churn == 1]["tenure"].tolist(),
                          "retained": df[df.Churn == 0]["tenure"].tolist()},
    })


# --------------------------------------------------------------------------
# API: prediction
# --------------------------------------------------------------------------
@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return err("Request body must be a JSON object.")
    try:
        return ok(P.predict_one(payload))
    except P.ValidationError as e:
        return err(str(e), 422)
    except FileNotFoundError as e:
        return err(str(e), 503)


@app.post("/api/bulk-predict")
def api_bulk_predict():
    f = request.files.get("file")
    if f is None or not f.filename:
        return err("No file uploaded. Choose a CSV file first.")
    if not f.filename.lower().endswith(".csv"):
        return err("Only .csv files are supported.")
    try:
        raw = pd.read_csv(f, dtype=str, keep_default_na=False)
    except Exception:
        return err("Could not parse the file as CSV.", 422)
    if len(raw) > MAX_ROWS:
        return err(f"File has {len(raw)} rows; the limit is {MAX_ROWS}.", 422)
    try:
        cleaned, warnings = P.validate_frame(raw)
        scored = P.score_frame(cleaned)
    except P.ValidationError as e:
        return err(str(e), 422, required=T.REQUIRED_COLUMNS)
    except Exception as e:  # unexpected model failure
        return err(f"Prediction failed: {e}", 500)

    raw.columns = [c.strip() for c in raw.columns]
    result = pd.concat([raw.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    result["churn_probability_pct"] = (result["churn_probability"] * 100).round(1)
    file_id = uuid.uuid4().hex
    result.to_csv(os.path.join(UPLOAD_DIR, f"predictions_{file_id}.csv"), index=False)

    id_col = T.ID_COL if T.ID_COL in result.columns else None
    rows = []
    for i, r in result.sort_values("churn_probability", ascending=False).head(1000).iterrows():
        rows.append({"id": r[id_col] if id_col else f"Row {i + 1}",
                     "tenure": r["tenure"], "contract": r["Contract"],
                     "monthly": r["MonthlyCharges"],
                     "probability": float(r["churn_probability_pct"]),
                     "predicted": r["predicted_churn"], "risk": r["risk_level"]})
    counts = scored["risk_level"].value_counts()
    return ok({
        "file_id": file_id,
        "summary": {
            "total": int(len(scored)),
            "high_risk": int(counts.get("High", 0)),
            "medium_risk": int(counts.get("Medium", 0)),
            "low_risk": int(counts.get("Low", 0)),
            "predicted_churners": int((scored["predicted_churn"] == "Yes").sum()),
            "avg_probability": round(float(scored["churn_probability"].mean() * 100), 1),
        },
        "warnings": warnings,
        "rows": rows,
        "truncated": bool(len(result) > 1000),
    })


@app.get("/api/download/<file_id>")
def api_download(file_id):
    if not re.fullmatch(r"[0-9a-f]{32}", file_id):
        return err("Invalid file id.", 404)
    path = os.path.join(UPLOAD_DIR, f"predictions_{file_id}.csv")
    if not os.path.exists(path):
        return err("Prediction file not found or expired.", 404)
    return send_file(path, mimetype="text/csv", as_attachment=True,
                     download_name="churn_predictions.csv")


@app.get("/api/sample-csv")
def api_sample_csv():
    df = T.load_dataset().sample(25, random_state=7)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df = df.drop(columns=[T.TARGET])
    return Response(df.to_csv(index=False), mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=sample_customers.csv"})


# --------------------------------------------------------------------------
# API: model performance
# --------------------------------------------------------------------------
@app.get("/api/model-metrics")
def api_model_metrics():
    return ok(P.load_metrics())


@app.get("/api/feature-importance")
def api_feature_importance():
    m = P.load_metrics()
    name = request.args.get("model", m["best_model"])
    if name not in m["models"]:
        return err("Unknown model.", 404)
    r = m["models"][name]
    return ok({"model": name, "top": r["feature_importance"][:10],
               "all": r["feature_importance"], "grouped": r["feature_importance_grouped"]})


@app.post("/api/retrain")
def api_retrain():
    if not _lock.acquire(blocking=False):
        return err("Training is already in progress.", 409)
    try:
        T.main()
        reset_cache()
        return ok({"status": "ok", "best_model": P.load_metrics()["best_model"]})
    except Exception as e:
        return err(f"Training failed: {e}", 500)
    finally:
        _lock.release()


# --------------------------------------------------------------------------
# API: customers table
# --------------------------------------------------------------------------
@app.get("/api/customers")
def api_customers():
    df = filter_customers(request.args)
    sort = request.args.get("sort", "churn_probability")
    if sort not in CUSTOMER_COLS:
        sort = "churn_probability"
    asc = request.args.get("order", "desc") == "asc"
    df = df.sort_values(sort, ascending=asc, kind="mergesort")
    try:
        page = max(1, int(request.args.get("page", 1)))
        per = min(100, max(5, int(request.args.get("per_page", 15))))
    except ValueError:
        return err("page and per_page must be integers.")
    total = len(df)
    pages = max(1, -(-total // per))
    page = min(page, pages)
    chunk = df.iloc[(page - 1) * per: page * per][CUSTOMER_COLS]
    return ok({"total": total, "page": page, "pages": pages, "per_page": per,
               "rows": chunk.to_dict("records")})


@app.get("/api/customers/export")
def api_customers_export():
    df = filter_customers(request.args)
    return Response(df[CUSTOMER_COLS].to_csv(index=False), mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=customers_filtered.csv"})


# --------------------------------------------------------------------------
# API: retention insights
# --------------------------------------------------------------------------
@app.get("/api/retention")
def api_retention():
    df = get_data().copy()
    m = P.load_metrics()
    df["TenureBand"] = pd.cut(df["tenure"], TENURE_BINS, labels=TENURE_LABELS)
    df["ChargeBand"] = pd.cut(df["MonthlyCharges"], CHARGE_BINS, labels=CHARGE_LABELS)

    def band_rates(col, order):
        g = group_churn(df, col, order)
        return [{"label": l, "customers": t, "churn_rate": r}
                for l, t, r in zip(g["labels"], g["total"], g["churn_rate"])]

    contract = band_rates("Contract", P.ALLOWED_VALUES["Contract"])
    charges = band_rates("ChargeBand", CHARGE_LABELS)
    tenure = band_rates("TenureBand", TENURE_LABELS)
    overall = df["Churn"].mean() * 100

    def worst(items):
        return max(items, key=lambda x: x["churn_rate"])

    wc, wch, wt = worst(contract), worst(charges), worst(tenure)
    suggestions = {
        "contract": [
            f"'{wc['label']}' customers churn at {wc['churn_rate']}% vs {overall:.1f}% overall: "
            "offer discounted 12/24-month upgrades.",
            "Reward long-term contracts with price locks, free upgrades or loyalty credits.",
            "Add exit-intent offers for month-to-month customers who contact billing or cancel.",
        ],
        "charges": [
            f"The '{wch['label']}' band has the highest churn ({wch['churn_rate']}%): "
            "review plan fit and offer right-sizing or bundled discounts.",
            "Introduce tiered loyalty discounts for customers paying above $70 per month.",
            "Send proactive bill-shock alerts and annual plan reviews.",
        ],
        "tenure": [
            f"Customers in '{wt['label']}' churn at {wt['churn_rate']}%: "
            "focus on onboarding and early-life engagement.",
            "Run 30/60/90-day check-in campaigns and first-year milestone rewards.",
            "Move new customers to auto-pay and multi-service bundles to build stickiness.",
        ],
    }

    risky = (df[(df.Churn == 0) & (df.risk_level == "High")]
             .sort_values("churn_probability", ascending=False).head(25))
    high = []
    for r in risky.to_dict("records"):
        recs = P.recommendations(r, r["churn_probability"])
        high.append({"id": r[T.ID_COL], "tenure": r["tenure"], "contract": r["Contract"],
                     "monthly": r["MonthlyCharges"],
                     "probability": round(r["churn_probability"] * 100, 1),
                     "action": recs[1] if len(recs) > 1 else recs[0]})

    best = m["models"][m["best_model"]]
    return ok({
        "overall_churn_rate": round(float(overall), 2),
        "high_risk_total": int((df.risk_level == "High").sum()),
        "high_risk_retained": int(((df.Churn == 0) & (df.risk_level == "High")).sum()),
        "factors": best["feature_importance_grouped"][:8],
        "contract": contract, "charges": charges, "tenure": tenure,
        "suggestions": suggestions, "high_risk_customers": high,
    })


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(_):
    return err("File too large (max 8 MB).", 413)


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return err("Endpoint not found.", 404)
    return render_template("404.html", active=""), 404


@app.errorhandler(500)
def server_error(e):
    if request.path.startswith("/api/"):
        return err("Internal server error.", 500)
    return render_template("404.html", active="", message="Something went wrong."), 500


def ensure_models():
    if not models_ready():
        print("No trained models found - training now (about 20 seconds)...")
        T.main()


if __name__ == "__main__":
    ensure_models()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

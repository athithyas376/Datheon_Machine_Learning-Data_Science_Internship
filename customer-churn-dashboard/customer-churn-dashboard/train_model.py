"""
train_model.py - Train and evaluate Random Forest and XGBoost churn classifiers.

Workflow
--------
1. Load the IBM Telco dataset, convert TotalCharges to numeric, clean target.
2. Stratified 80/20 train/test split (fixed seed). The test set is never used
   for fitting, imputation, scaling, encoding, or threshold selection.
3. Preprocessing pipeline (fit on training data only):
     numeric     -> median imputer -> StandardScaler
     categorical -> most-frequent imputer -> OneHotEncoder(handle_unknown="ignore")
4. Class imbalance handled with class weights (RF: balanced_subsample,
   XGBoost: scale_pos_weight = n_negative / n_positive).
5. 5-fold StratifiedKFold CV on the training set -> validation ROC-AUC.
   Best model = highest mean CV ROC-AUC.
6. Decision threshold chosen on out-of-fold training predictions (max F1).
7. Final evaluation on the held-out test set: ROC-AUC, precision, recall, F1,
   confusion matrix, ROC curve, feature importance.
8. Out-of-fold churn probabilities for every customer are saved so the
   customer-management page can show honest (non-memorised) risk levels.

Run:  python train_model.py
"""
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TARGET = "Churn"
ID_COL = "customerID"
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = FEATURES  # columns an uploaded CSV must contain


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Type-fix feature columns (no fitting, so it is leakage-free)."""
    out = df.copy()
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str).str.strip()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"].astype(str).str.strip(),
                                        errors="coerce")
    out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")
    sc = out["SeniorCitizen"].astype(str).str.strip().str.lower()
    out["SeniorCitizen"] = sc.map({"1": 1, "0": 0, "yes": 1, "no": 0, "true": 1,
                                   "false": 0, "1.0": 1, "0.0": 0})
    return out


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = clean_features(df.assign(**{c: df[c] for c in df.columns}))
    df[TARGET] = df[TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0})
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])


def build_models(scale_pos_weight: float) -> dict:
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced_subsample",
        n_jobs=-1, random_state=SEED)
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_lambda=1.0, scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", tree_method="hist",
        n_jobs=-1, random_state=SEED)
    return {"Random Forest": rf, "XGBoost": xgb}


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def best_f1_threshold(y_true, proba) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.argmax(f1))])


def feature_names(pipe: Pipeline) -> list:
    return list(pipe.named_steps["prep"].get_feature_names_out())


def original_feature(name: str) -> str:
    """Map an encoded feature name back to its source column."""
    name = name.split("__", 1)[1]
    for col in CATEGORICAL_FEATURES:
        if name.startswith(col + "_"):
            return col
    return name


def pretty(name: str) -> str:
    name = name.split("__", 1)[1] if "__" in name else name
    return name.replace("_", ": ", 1) if any(name.startswith(c + "_") for c in CATEGORICAL_FEATURES) else name


def sample_curve(x, y, n=120):
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return [round(float(v), 4) for v in x[idx]], [round(float(v), 4) for v in y[idx]]


def evaluate(name, pipe, X_train, y_train, X_test, y_test, cv, threshold):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    fpr, tpr, _ = roc_curve(y_test, proba)
    fpr_s, tpr_s = sample_curve(fpr, tpr)

    importances = pipe.named_steps["model"].feature_importances_
    names = feature_names(pipe)
    imp_df = pd.DataFrame({"feature": names, "importance": importances})
    imp_df["label"] = imp_df["feature"].map(pretty)
    imp_df["group"] = imp_df["feature"].map(original_feature)
    top = imp_df.sort_values("importance", ascending=False)
    grouped = (imp_df.groupby("group")["importance"].sum()
               .sort_values(ascending=False))

    return {
        "threshold": round(threshold, 4),
        "test": {
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred)), 4),
            "recall": round(float(recall_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": fpr_s, "tpr": tpr_s},
        "feature_importance": [
            {"feature": r.label, "importance": round(float(r.importance), 5)}
            for r in top.head(20).itertuples()],
        "feature_importance_grouped": [
            {"feature": k, "importance": round(float(v), 5)}
            for k, v in grouped.head(15).items()],
    }


def main():
    t0 = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_dataset()
    X, y = df[FEATURES], df[TARGET]
    print(f"Loaded {len(df)} customers | churn rate {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, model in build_models(spw).items():
        print(f"\n== {name} ==")
        pipe = make_pipeline(model)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                    scoring="roc_auc", n_jobs=1)
        oof = cross_val_predict(pipe, X_train, y_train, cv=cv,
                                method="predict_proba", n_jobs=1)[:, 1]
        threshold = best_f1_threshold(y_train, oof)
        pipe.fit(X_train, y_train)
        res = evaluate(name, pipe, X_train, y_train, X_test, y_test, cv, threshold)
        res["cv"] = {"roc_auc_mean": round(float(cv_scores.mean()), 4),
                     "roc_auc_std": round(float(cv_scores.std()), 4),
                     "folds": [round(float(s), 4) for s in cv_scores]}
        results[name], fitted[name] = res, pipe
        print(f"CV ROC-AUC {cv_scores.mean():.4f} +/- {cv_scores.std():.4f} | "
              f"threshold {threshold:.3f} | test {res['test']}")

    best_name = max(results, key=lambda n: results[n]["cv"]["roc_auc_mean"])
    print(f"\nBest model (validation ROC-AUC): {best_name}")

    # Refit the best model on ALL data for deployment; save every model.
    joblib.dump(fitted["Random Forest"], os.path.join(MODEL_DIR, "random_forest.joblib"))
    joblib.dump(fitted["XGBoost"], os.path.join(MODEL_DIR, "xgboost.joblib"))
    deploy = make_pipeline(build_models(float((y == 0).sum() / (y == 1).sum()))[best_name])
    deploy.fit(X, y)
    joblib.dump(deploy, os.path.join(MODEL_DIR, "best_model.joblib"))

    # Honest out-of-fold scores for every customer (used by data-management page).
    best_pipe = make_pipeline(build_models(spw)[best_name])
    oof_all = cross_val_predict(best_pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    scores = df[[ID_COL]].copy() if ID_COL in df.columns else pd.DataFrame(index=df.index)
    scores["churn_probability"] = np.round(oof_all, 4)
    scores.to_csv(os.path.join(MODEL_DIR, "customer_scores.csv"), index=False)

    metrics = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "dataset": {"rows": int(len(df)), "features": len(FEATURES),
                    "churn_rate": round(float(y.mean()), 4),
                    "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "imbalance_strategy": {"Random Forest": "class_weight='balanced_subsample'",
                               "XGBoost": f"scale_pos_weight={spw:.3f}"},
        "best_model": best_name,
        "models": results,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nSaved models + metrics to {MODEL_DIR} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

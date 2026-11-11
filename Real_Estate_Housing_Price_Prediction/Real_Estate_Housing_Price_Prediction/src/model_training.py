"""Model training: OLS, Ridge (L2) and Lasso (L1) with GridSearchCV.

Every model is a full ``Pipeline([preprocessor, estimator])`` and GridSearchCV
is run on that whole pipeline, so imputation / scaling are re-fitted inside each
CV fold (no leakage from the validation fold).
"""
from __future__ import annotations

import platform
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline

from . import config as C
from . import evaluation as ev
from . import prediction as pr
from .data_loader import prepare_dataset
from .feature_engineering import describe_spec, restrict_spec, suggest_engineering_spec
from .preprocessing import (
    build_preprocessor,
    compute_vif,
    correlation_matrix,
    describe_fitted_preprocessor,
    duplicate_report,
    missing_value_report,
    split_data,
    transform_to_frame,
    vif_prune,
)


ProgressFn = Optional[Callable[[float, str], None]]


def _notify(cb: ProgressFn, frac: float, msg: str) -> None:
    if cb is not None:
        cb(min(max(frac, 0.0), 1.0), msg)


def build_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])


# --------------------------------------------------------------------------- #
# Train the three models for one preprocessing configuration
# --------------------------------------------------------------------------- #
def train_three_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    cv: int = C.CV_FOLDS,
    random_state: int = C.RANDOM_STATE,
    n_jobs: int = -1,
    ridge_alphas: Sequence[float] = C.RIDGE_ALPHAS,
    lasso_alphas: Sequence[float] = C.LASSO_ALPHAS,
    progress: ProgressFn = None,
    progress_range: tuple = (0.0, 1.0),
    label: str = "",
) -> Dict[str, Any]:
    """Fit OLS, and GridSearchCV-tuned Ridge and Lasso."""
    lo, hi = progress_range
    step = (hi - lo) / 3
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = "neg_root_mean_squared_error"
    timings: Dict[str, float] = {}
    prefix = f"{label}: " if label else ""

    # --- OLS --------------------------------------------------------------
    _notify(progress, lo, f"{prefix}training OLS…")
    t = time.perf_counter()
    ols = build_pipeline(preprocessor, LinearRegression())
    ols_cv = -cross_val_score(ols, X_train, y_train, cv=splitter, scoring=scoring, n_jobs=n_jobs)
    ols.fit(X_train, y_train)
    timings["OLS"] = time.perf_counter() - t

    # --- Ridge ------------------------------------------------------------
    _notify(progress, lo + step, f"{prefix}tuning Ridge with GridSearchCV…")
    t = time.perf_counter()
    ridge_gs = GridSearchCV(
        build_pipeline(preprocessor, Ridge(random_state=random_state)),
        param_grid={"model__alpha": list(ridge_alphas)},
        scoring=scoring,
        cv=splitter,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    ).fit(X_train, y_train)
    timings["Ridge"] = time.perf_counter() - t

    # --- Lasso ------------------------------------------------------------
    _notify(progress, lo + 2 * step, f"{prefix}tuning Lasso with GridSearchCV…")
    t = time.perf_counter()
    lasso_gs = GridSearchCV(
        build_pipeline(preprocessor, Lasso(max_iter=5_000, tol=1e-3, random_state=random_state)),
        param_grid={"model__alpha": list(lasso_alphas)},
        scoring=scoring,
        cv=splitter,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    ).fit(X_train, y_train)
    timings["Lasso"] = time.perf_counter() - t
    _notify(progress, hi, f"{prefix}done")

    def grid_frame(gs: GridSearchCV) -> pd.DataFrame:
        res = pd.DataFrame(gs.cv_results_)
        out = pd.DataFrame(
            {
                "alpha": res["param_model__alpha"].astype(float),
                "cv_rmse": -res["mean_test_score"],
                "cv_rmse_std": res["std_test_score"],
                "train_rmse": -res["mean_train_score"],
                "rank": res["rank_test_score"],
            }
        )
        return out.sort_values("alpha").reset_index(drop=True)

    return {
        "models": {"OLS": ols, "Ridge": ridge_gs.best_estimator_, "Lasso": lasso_gs.best_estimator_},
        "cv_rmse": {
            "OLS": float(ols_cv.mean()),
            "Ridge": float(-ridge_gs.best_score_),
            "Lasso": float(-lasso_gs.best_score_),
        },
        "cv_rmse_std": {
            "OLS": float(ols_cv.std()),
            "Ridge": float(ridge_gs.cv_results_["std_test_score"][ridge_gs.best_index_]),
            "Lasso": float(lasso_gs.cv_results_["std_test_score"][lasso_gs.best_index_]),
        },
        "best_params": {
            "OLS": {},
            "Ridge": {"alpha": float(ridge_gs.best_params_["model__alpha"])},
            "Lasso": {"alpha": float(lasso_gs.best_params_["model__alpha"])},
        },
        "grids": {"Ridge": grid_frame(ridge_gs), "Lasso": grid_frame(lasso_gs)},
        "timings": timings,
    }


# --------------------------------------------------------------------------- #
# Feature-set experiments (before / after feature engineering)
# --------------------------------------------------------------------------- #
def _experiment_configs(
    X_train: pd.DataFrame, y_train: pd.Series, numeric: List[str], categorical: List[str]
) -> List[Dict[str, Any]]:
    spec = suggest_engineering_spec(X_train, y_train, numeric)
    configs: List[Dict[str, Any]] = [
        {
            "name": "Baseline (original features)",
            "key": "baseline",
            "numeric": numeric,
            "mode": "none",
            "spec": None,
        }
    ]
    # multicollinearity handling: drop numeric features with VIF >= 5 (iteratively)
    if len(numeric) > 2:
        kept, log = vif_prune(X_train[numeric], threshold=C.VIF_MODERATE)
        if len(log) and kept:
            configs.append(
                {
                    "name": f"VIF-pruned (VIF<{C.VIF_MODERATE:g})",
                    "key": "vif_pruned",
                    "numeric": kept,
                    "mode": "none",
                    "spec": None,
                    "dropped": log.dropped_feature.tolist(),
                }
            )
    if spec["pairs"] or spec["ratios"]:
        configs.append(
            {
                "name": "Interaction features",
                "key": "interaction",
                "numeric": numeric,
                "mode": "interaction",
                "spec": spec,
            }
        )
    if 2 <= len(numeric) <= 12:
        configs.append(
            {
                "name": "Polynomial (degree 2)",
                "key": "poly",
                "numeric": numeric,
                "mode": "poly",
                "spec": None,
            }
        )
    return configs


def run_feature_experiments(
    X_train, y_train, X_test, y_test, numeric, categorical,
    cv=C.CV_FOLDS, random_state=C.RANDOM_STATE, n_jobs=-1, progress: ProgressFn = None,
    progress_range=(0.0, 1.0),
) -> Dict[str, Any]:
    configs = _experiment_configs(X_train, y_train, numeric, categorical)
    lo, hi = progress_range
    span = (hi - lo) / len(configs)
    rows, runs = [], {}
    for i, cfg in enumerate(configs):
        pre = build_preprocessor(cfg["numeric"], categorical, cfg["mode"], cfg["spec"])
        run = train_three_models(
            X_train, y_train, pre, cv=cv, random_state=random_state, n_jobs=n_jobs,
            progress=progress, progress_range=(lo + i * span, lo + (i + 1) * span), label=cfg["name"],
        )
        n_out = len(run["models"]["OLS"].named_steps["preprocessor"].get_feature_names_out())
        row = {"feature_set": cfg["name"], "key": cfg["key"], "n_features_after_encoding": n_out}
        best_cv_model = min(C.MODEL_NAMES, key=lambda m: run["cv_rmse"][m])
        for m in C.MODEL_NAMES:
            pred = run["models"][m].predict(X_test)
            met = ev.regression_metrics(y_test, pred)
            row[f"{m}_cv_rmse"] = run["cv_rmse"][m]
            row[f"{m}_test_rmse"] = met["RMSE"]
            row[f"{m}_test_r2"] = met["R2"]
        row["best_model_by_cv"] = best_cv_model
        row["best_cv_rmse"] = run["cv_rmse"][best_cv_model]
        row["best_test_rmse"] = row[f"{best_cv_model}_test_rmse"]
        row["best_test_r2"] = row[f"{best_cv_model}_test_r2"]
        rows.append(row)
        runs[cfg["key"]] = {"config": cfg, "run": run}
    return {"table": pd.DataFrame(rows), "runs": runs}


# --------------------------------------------------------------------------- #
# Full project run
# --------------------------------------------------------------------------- #
def target_unit_for(target: str) -> Dict[str, Any]:
    if target == C.CALIFORNIA_TARGET:
        return {"scale": C.CALIFORNIA_TARGET_SCALE, "prefix": "$", "label": "Median house value (USD)"}
    return {"scale": 1.0, "prefix": "", "label": target}


def train_project(
    df: pd.DataFrame,
    target: str,
    features: Optional[Sequence[str]] = None,
    dataset_name: str = "California Housing",
    cv: int = C.CV_FOLDS,
    test_size: float = C.TEST_SIZE,
    random_state: int = C.RANDOM_STATE,
    n_jobs: int = -1,
    progress: ProgressFn = None,
) -> Dict[str, Any]:
    """Run the complete pipeline and return a self-contained, picklable result."""
    t0 = time.perf_counter()
    _notify(progress, 0.0, "Validating and cleaning data…")
    raw_missing = missing_value_report(df[[c for c in df.columns]])
    prep = prepare_dataset(df, target, features)
    X, y = prep["X"], prep["y"]
    numeric, categorical = prep["numeric"], prep["categorical"]

    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    # --- diagnostics computed on the TRAINING split only -------------------
    vif_base = compute_vif(X_train[numeric]) if len(numeric) > 1 else pd.DataFrame(columns=["feature", "VIF", "severity"])
    corr = correlation_matrix(prep["data"][numeric + [target]])

    # --- feature experiments (3 models x each feature set) ------------------
    _notify(progress, 0.05, "Comparing feature sets…")
    exp = run_feature_experiments(
        X_train, y_train, X_test, y_test, numeric, categorical,
        cv=cv, random_state=random_state, n_jobs=n_jobs, progress=progress, progress_range=(0.05, 0.92),
    )
    table = exp["table"]
    # Feature set is chosen with cross-validation on TRAIN data only (never the test set)
    chosen_key = table.loc[table.best_cv_rmse.idxmin(), "key"]
    chosen = exp["runs"][chosen_key]
    run, cfg = chosen["run"], chosen["config"]
    models = run["models"]

    # VIF of the engineered design matrix (centred interactions) for the dashboard
    pre_fitted = models["OLS"].named_steps["preprocessor"]
    train_design = transform_to_frame(pre_fitted, X_train)
    design_numeric = [c for c in train_design.columns if train_design[c].nunique() > 2]
    vif_final = compute_vif(train_design[design_numeric]) if len(design_numeric) > 1 else vif_base

    # --- evaluation ----------------------------------------------------------
    _notify(progress, 0.94, "Evaluating models…")
    metrics = ev.evaluate_models(models, X_train, y_train, X_test, y_test, run["cv_rmse"], run["best_params"])
    best_name, best_text = ev.select_best_model(metrics)
    preds = pd.DataFrame({"actual": y_test.values})
    for name, mdl in models.items():
        preds[name] = mdl.predict(X_test)
    preds.index = y_test.index
    resid_stats = {m: ev.residual_analysis(preds["actual"], preds[m]) for m in C.MODEL_NAMES}

    cap_value = float(y.max())
    cap_share = float((y == cap_value).mean())
    target_cap = {"value": cap_value, "share": cap_share} if cap_share >= 0.005 else None

    coefs = {m: ev.coefficient_table(models[m]) for m in C.MODEL_NAMES}
    interpretation = {
        m: ev.interpret_residuals(resid_stats[m], preds["actual"], preds[m], target_cap) for m in C.MODEL_NAMES
    }

    result: Dict[str, Any] = {
        "meta": {
            "dataset_name": dataset_name,
            "target": target,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "model_numeric_features": cfg["numeric"],
            "feature_set": cfg["name"],
            "feature_mode": cfg["mode"],
            "engineering_spec": cfg["spec"],
            "n_rows_raw": int(len(df)),
            "n_rows_used": int(len(X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_size": test_size,
            "random_state": random_state,
            "cv_folds": cv,
            "n_duplicates_removed": prep["n_duplicates_removed"],
            "n_missing_target_removed": prep["n_missing_target_removed"],
            "unit": target_unit_for(target),
            "target_cap": target_cap,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "train_seconds": round(time.perf_counter() - t0, 1),
        },
        "models": models,
        "best_params": run["best_params"],
        "cv_rmse": run["cv_rmse"],
        "cv_rmse_std": run["cv_rmse_std"],
        "grids": run["grids"],
        "timings": run["timings"],
        "metrics": metrics,
        "best_model": best_name,
        "best_model_explanation": best_text,
        "predictions": preds,
        "residual_stats": resid_stats,
        "residual_interpretation": interpretation,
        "coefficients": coefs,
        "experiments": table,
        "experiment_dropped_features": {k: v["config"].get("dropped", []) for k, v in exp["runs"].items()},
        "interaction_spec": (exp["runs"].get("interaction") or {}).get("config", {}).get("spec"),
        "engineering_table": describe_spec((exp["runs"].get("interaction") or {}).get("config", {}).get("spec") or {}),
        "diagnostics": {
            "missing_report": raw_missing,
            "duplicates": {"removed": prep["n_duplicates_removed"], "missing_target_removed": prep["n_missing_target_removed"]},
            "vif_original": vif_base,
            "vif_final_design": vif_final,
            "correlation": corr,
            "preprocessing": describe_fitted_preprocessor(pre_fitted),
        },
        "splits": {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
        "form_spec": pr.build_form_spec(X_train, numeric, categorical),
        "clean_data": prep["data"],
    }
    _notify(progress, 1.0, "Training complete")
    return result


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_artifacts(result: Dict[str, Any], models_dir=C.MODELS_DIR, metrics_dir=C.METRICS_DIR) -> Dict[str, str]:
    """Persist the three models (.pkl), metadata, the dashboard bundle and metric files."""
    from pathlib import Path
    import json

    models_dir, metrics_dir = Path(models_dir), Path(metrics_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for name, fname in C.MODEL_FILES.items():
        joblib.dump(result["models"][name], models_dir / fname)
        paths[name] = str(models_dir / fname)
    pr.save_metadata(result, models_dir / C.METADATA_FILE)
    joblib.dump(result, models_dir / C.BUNDLE_FILE, compress=3)

    result["metrics"].to_csv(metrics_dir / "model_comparison.csv", index=False)
    result["experiments"].to_csv(metrics_dir / "feature_set_comparison.csv", index=False)
    for m, g in result["grids"].items():
        g.to_csv(metrics_dir / f"gridsearch_{m.lower()}.csv", index=False)
    result["predictions"].to_csv(metrics_dir / "test_predictions.csv", index_label="row_id")
    result["diagnostics"]["vif_original"].to_csv(metrics_dir / "vif_original.csv", index=False)
    result["diagnostics"]["vif_final_design"].to_csv(metrics_dir / "vif_final_design.csv", index=False)
    summary = {
        "best_model": result["best_model"],
        "best_params": result["best_params"],
        "feature_set": result["meta"]["feature_set"],
        "test_metrics": {
            r["Model"]: {"RMSE": r["Test RMSE"], "MAE": r["Test MAE"], "R2": r["Test R²"]}
            for _, r in result["metrics"].iterrows()
        },
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return paths

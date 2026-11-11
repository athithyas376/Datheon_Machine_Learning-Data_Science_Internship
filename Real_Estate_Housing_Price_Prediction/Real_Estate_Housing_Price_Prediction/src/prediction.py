"""Prediction utilities: load trained models and score new inputs.

Models are complete scikit-learn Pipelines, therefore a prediction uses exactly
the same preprocessing (imputation, feature engineering, scaling, encoding) that
was fitted on the training data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config as C


# --------------------------------------------------------------------------- #
# Loading / saving
# --------------------------------------------------------------------------- #
def load_models(models_dir=C.MODELS_DIR) -> Dict[str, Any]:
    """Load OLS, Ridge and Lasso pipelines. Raises FileNotFoundError if absent."""
    models_dir = Path(models_dir)
    out = {}
    for name, fname in C.MODEL_FILES.items():
        path = models_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}. Run `python main.py` first.")
        out[name] = joblib.load(path)
    return out


def load_bundle(models_dir=C.MODELS_DIR) -> Dict[str, Any]:
    """Load the full training result saved by ``main.py``."""
    path = Path(models_dir) / C.BUNDLE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Training bundle not found: {path}")
    bundle = joblib.load(path)
    saved = bundle["meta"].get("sklearn_version")
    if saved != sklearn.__version__:
        raise RuntimeError(
            f"Models were trained with scikit-learn {saved} but {sklearn.__version__} is installed."
        )
    return bundle


def save_metadata(result: Dict[str, Any], path) -> None:
    meta = dict(result["meta"])
    meta.pop("engineering_spec", None)
    meta["best_model"] = result["best_model"]
    meta["best_params"] = result["best_params"]
    meta["form_spec"] = result["form_spec"]
    Path(path).write_text(json.dumps(meta, indent=2, default=str))


def load_metadata(models_dir=C.MODELS_DIR) -> Dict[str, Any]:
    return json.loads((Path(models_dir) / C.METADATA_FILE).read_text())


# --------------------------------------------------------------------------- #
# Input form specification
# --------------------------------------------------------------------------- #
def build_form_spec(X_train: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Describe each feature so a prediction form can be generated dynamically."""
    spec: Dict[str, Dict[str, Any]] = {}
    for col in X_train.columns:
        s = X_train[col]
        if col in numeric:
            s = pd.to_numeric(s, errors="coerce").dropna()
            lo, hi = float(s.min()), float(s.max())
            med = float(s.median())
            typical = float(s.quantile(0.99) - s.quantile(0.01))  # robust to outliers
            is_int = bool(np.all(np.isclose(s, np.round(s))))
            if is_int and typical >= 20:
                step = 1.0
            else:
                step = float(10 ** np.floor(np.log10(typical / 100))) if typical > 0 else 0.01
                step = max(step, 1e-6)
            spec[col] = {
                "kind": "numeric", "min": lo, "max": hi, "median": med,
                "p01": float(s.quantile(0.01)), "p99": float(s.quantile(0.99)),
                "step": float(step), "integer": is_int,
                "label": C.CALIFORNIA_LABELS.get(col, col),
            }
        else:
            vc = s.dropna().astype(str).value_counts()
            spec[col] = {
                "kind": "categorical", "options": vc.index.tolist(), "default": str(vc.index[0]),
                "label": C.CALIFORNIA_LABELS.get(col, col),
            }
    return spec


def check_input_ranges(values: Dict[str, Any], form_spec: Dict[str, Dict[str, Any]]) -> List[str]:
    """Warnings for inputs outside the range seen in training (extrapolation)."""
    warns = []
    for col, v in values.items():
        sp = form_spec.get(col)
        if not sp or sp["kind"] != "numeric":
            continue
        if v < sp["min"] or v > sp["max"]:
            warns.append(f"{col} = {v:g} is outside the training range [{sp['min']:g}, {sp['max']:g}]; the prediction is an extrapolation.")
    return warns


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def build_input_frame(values: Dict[str, Any], feature_columns: Sequence[str], numeric: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in values]
    if missing:
        raise ValueError(f"Missing input value(s) for: {missing}")
    row = {}
    for c in feature_columns:
        row[c] = float(values[c]) if c in numeric else values[c]
    return pd.DataFrame([row], columns=list(feature_columns))


def predict_all(models: Dict[str, Any], X: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Predictions from every model for the rows of X (same preprocessing as training)."""
    return {name: np.asarray(mdl.predict(X), dtype=float) for name, mdl in models.items()}


def format_value(value: float, unit: Optional[Dict[str, Any]] = None) -> str:
    unit = unit or {"scale": 1.0, "prefix": ""}
    v = value * unit.get("scale", 1.0)
    prefix = unit.get("prefix", "")
    return f"{prefix}{v:,.0f}" if abs(v) >= 1000 else f"{prefix}{v:,.3f}"


def model_explanation(name: str, best_params: Optional[Dict[str, Any]] = None, n_zero: Optional[int] = None) -> str:
    a = (best_params or {}).get(name, {}).get("alpha")
    if name == "OLS":
        return ("Ordinary Least Squares fits the coefficients that minimise the sum of squared errors with "
                "no penalty. It is easy to interpret, but its coefficients can become unstable when features are correlated.")
    if name == "Ridge":
        extra = f" The tuned penalty strength is alpha = {a:.4g}." if a is not None else ""
        return ("Ridge regression adds an L2 penalty (sum of squared coefficients) to the OLS loss. It shrinks "
                "coefficients smoothly, which reduces variance and copes well with multicollinearity." + extra)
    if name == "Lasso":
        extra = f" The tuned penalty strength is alpha = {a:.4g}." if a is not None else ""
        if n_zero is not None:
            extra += f" {n_zero} coefficient(s) were shrunk exactly to zero."
        return ("Lasso regression adds an L1 penalty (sum of absolute coefficients). It can drive some coefficients "
                "exactly to zero, performing automatic feature selection." + extra)
    return ""

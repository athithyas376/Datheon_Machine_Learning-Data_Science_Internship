"""Preprocessing: diagnostics (missing values, duplicates, correlation, VIF) and
the leakage-free scikit-learn preprocessing pipeline."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from . import config as C
from .data_loader import column_kind
from .feature_engineering import InteractionFeatures, Winsorizer, restrict_spec

FEATURE_MODES = ("none", "interaction", "poly")


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def get_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = [c for c in X.columns if column_kind(X[c]) == "numeric"]
    categorical = [c for c in X.columns if column_kind(X[c]) == "categorical"]
    return numeric, categorical


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rep = pd.DataFrame(
        {
            "column": df.columns,
            "missing": df.isna().sum().values,
            "missing_pct": (100 * df.isna().sum() / max(n, 1)).round(3).values,
            "dtype": [str(t) for t in df.dtypes],
        }
    )
    return rep.sort_values("missing", ascending=False).reset_index(drop=True)


def duplicate_report(df: pd.DataFrame) -> Dict[str, object]:
    mask = df.duplicated(keep="first")
    return {
        "n_duplicates": int(mask.sum()),
        "pct": round(100 * mask.mean(), 4) if len(df) else 0.0,
        "examples": df[df.duplicated(keep=False)].head(10),
    }


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).corr(method=method)


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if np.isfinite(r) and abs(r) >= threshold:
                rows.append({"feature_1": cols[i], "feature_2": cols[j], "correlation": round(float(r), 4)})
    out = pd.DataFrame(rows, columns=["feature_1", "feature_2", "correlation"])
    return out.reindex(out.correlation.abs().sort_values(ascending=False).index).reset_index(drop=True)


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor for each numeric column.

    VIF_j = 1 / (1 - R²_j) where R²_j comes from regressing column j on all the
    other columns (computed on median-imputed, standardised data).  Perfectly
    collinear columns are reported with the cap 1e6.
    """
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.loc[:, num.nunique(dropna=True) > 1]
    if num.shape[1] < 2:
        return pd.DataFrame({"feature": num.columns, "VIF": [1.0] * num.shape[1], "severity": ["low"] * num.shape[1]})
    num = num.fillna(num.median())
    Z = (num - num.mean()) / num.std(ddof=0)
    Zv = Z.to_numpy(dtype=float)
    vifs = []
    for j in range(Zv.shape[1]):
        y = Zv[:, j]
        Xo = np.delete(Zv, j, axis=1)
        Xo = np.column_stack([np.ones(len(Xo)), Xo])
        beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        sse = float(np.sum((y - Xo @ beta) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - sse / sst if sst > 0 else 0.0
        vifs.append(min(1.0 / max(1.0 - r2, 1e-6), 1e6))
    out = pd.DataFrame({"feature": num.columns, "VIF": np.round(vifs, 3)})
    out["severity"] = np.where(
        out.VIF >= C.VIF_THRESHOLD, "severe", np.where(out.VIF >= C.VIF_MODERATE, "moderate", "low")
    )
    return out.sort_values("VIF", ascending=False).reset_index(drop=True)


def vif_prune(df: pd.DataFrame, threshold: float = C.VIF_THRESHOLD) -> Tuple[List[str], pd.DataFrame]:
    """Iteratively drop the feature with the highest VIF until all VIFs < threshold.

    Returns the kept columns and a log of dropped features.
    """
    cols = list(df.select_dtypes(include=[np.number]).columns)
    log = []
    while len(cols) > 1:
        vif = compute_vif(df[cols])
        worst = vif.iloc[0]
        if worst.VIF < threshold:
            break
        log.append({"step": len(log) + 1, "dropped_feature": worst.feature, "VIF_at_drop": float(worst.VIF)})
        cols.remove(worst.feature)
    return cols, pd.DataFrame(log, columns=["step", "dropped_feature", "VIF_at_drop"])


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    feature_mode: str = "none",
    engineering_spec: Optional[Dict[str, object]] = None,
) -> ColumnTransformer:
    """Build the (unfitted) ColumnTransformer.

    numeric     : median imputation -> [interactions | polynomial] -> StandardScaler
    categorical : most-frequent imputation -> OneHotEncoder (first level dropped)
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")
    numeric_cols, categorical_cols = list(numeric_cols), list(categorical_cols)
    if not numeric_cols and not categorical_cols:
        raise ValueError("At least one feature is required.")

    transformers = []
    if numeric_cols:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if feature_mode == "interaction":
            spec = restrict_spec(engineering_spec or {}, numeric_cols)
            if spec["pairs"] or spec["ratios"]:
                steps.append(
                    (
                        "engineer",
                        InteractionFeatures(
                            columns=numeric_cols, pairs=spec["pairs"], ratios=spec["ratios"], center=True
                        ),
                    )
                )
        elif feature_mode == "poly":
            # clip outliers first: squares of extreme values would otherwise dominate
            steps.append(("winsorize", Winsorizer(quantiles=(0.01, 0.99))))
            steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
        steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = C.TEST_SIZE, random_state: int = C.RANDOM_STATE):
    """Reproducible 80:20 split (done before any fitting -> no leakage)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def transformed_feature_names(fitted_preprocessor: ColumnTransformer) -> List[str]:
    return [str(n) for n in fitted_preprocessor.get_feature_names_out()]


def transform_to_frame(fitted_preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    arr = fitted_preprocessor.transform(X)
    return pd.DataFrame(arr, columns=transformed_feature_names(fitted_preprocessor), index=X.index)


def describe_fitted_preprocessor(fitted: ColumnTransformer) -> Dict[str, pd.DataFrame]:
    """Learned preprocessing parameters (all taken from the training data)."""
    out: Dict[str, pd.DataFrame] = {}
    names = {name: (pipe, cols) for name, pipe, cols in fitted.transformers_ if name != "remainder"}
    if "num" in names:
        pipe, cols = names["num"]
        imp = pipe.named_steps["imputer"]
        out["numeric_imputer"] = pd.DataFrame({"feature": cols, "median_used": np.round(imp.statistics_, 4)})
        scaler = pipe.named_steps["scaler"]
        feat_names = pipe[:-1].get_feature_names_out(cols)
        out["scaler"] = pd.DataFrame(
            {"feature": feat_names, "mean": np.round(scaler.mean_, 4), "std": np.round(scaler.scale_, 4)}
        )
    if "cat" in names:
        pipe, cols = names["cat"]
        imp = pipe.named_steps["imputer"]
        out["categorical_imputer"] = pd.DataFrame({"feature": cols, "most_frequent_used": imp.statistics_})
        enc = pipe.named_steps["onehot"]
        rows = []
        for col, cats, drop in zip(cols, enc.categories_, enc.drop_idx_ if enc.drop_idx_ is not None else [None] * len(cols)):
            for k, cat in enumerate(cats):
                rows.append({"feature": col, "category": cat, "encoded_as": "baseline (dropped)" if drop is not None and k == drop else f"{col}_{cat}"})
        out["onehot"] = pd.DataFrame(rows)
    return out

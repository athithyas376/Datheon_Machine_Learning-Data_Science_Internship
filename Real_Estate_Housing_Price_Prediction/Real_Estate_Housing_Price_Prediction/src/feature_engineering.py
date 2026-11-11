"""Feature engineering: interaction and ratio features as a scikit-learn transformer.

The transformer is placed *inside* the preprocessing pipeline, so everything it
learns (column means used for centering) comes from training data only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C


def interaction_name(a: str, b: str) -> str:
    return f"{a}_x_{b}"


def ratio_name(num: str, den: str) -> str:
    return f"{num}_per_{den}"


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Append pairwise interaction (product) and ratio features.

    Parameters
    ----------
    columns : list of str
        Names of the input columns, in the order they arrive.
    pairs : list of (str, str)
        Interaction features ``a * b``.
    ratios : list of (str, str)
        Ratio features ``numerator / denominator``.
    center : bool
        Mean-center both operands (means learned in ``fit``) before multiplying.
        Centering removes most of the *structural* multicollinearity between a
        product term and its parents, which keeps VIFs low.
    clip_quantiles : (float, float) or None
        Operands are clipped to these training-data quantiles before they are
        multiplied / divided.  Raw housing data contains extreme outliers
        (e.g. 1 243 occupants per household); without clipping a single block
        group would dominate every product term.  The original (un-clipped)
        columns are passed through unchanged.
    """

    def __init__(self, columns=None, pairs=None, ratios=None, center=True, clip_quantiles=(0.01, 0.99)):
        self.columns = columns
        self.pairs = pairs
        self.ratios = ratios
        self.center = center
        self.clip_quantiles = clip_quantiles

    # ------------------------------------------------------------------ #
    def _index(self, name: str) -> int:
        try:
            return list(self.columns).index(name)
        except ValueError as exc:
            raise ValueError(f"Column '{name}' is not among the input columns.") from exc

    def fit(self, X, y=None):
        if self.columns is None:
            raise ValueError("InteractionFeatures requires the 'columns' argument.")
        arr = np.asarray(X, dtype=float)
        if arr.shape[1] != len(self.columns):
            raise ValueError("Number of input columns does not match 'columns'.")
        self.n_features_in_ = arr.shape[1]
        if self.clip_quantiles is not None:
            lo, hi = self.clip_quantiles
            self.lower_ = np.nanquantile(arr, lo, axis=0)
            self.upper_ = np.nanquantile(arr, hi, axis=0)
            arr = np.clip(arr, self.lower_, self.upper_)
        self.means_ = np.nanmean(arr, axis=0)
        for a, b in self.pairs or []:
            self._index(a), self._index(b)
        for n, d in self.ratios or []:
            self._index(n), self._index(d)
        return self

    def transform(self, X):
        raw = np.asarray(X, dtype=float)
        arr = np.clip(raw, self.lower_, self.upper_) if self.clip_quantiles is not None else raw
        new_cols = []
        for a, b in self.pairs or []:
            ia, ib = self._index(a), self._index(b)
            xa, xb = arr[:, ia], arr[:, ib]
            if self.center:
                xa, xb = xa - self.means_[ia], xb - self.means_[ib]
            new_cols.append(xa * xb)
        for n, d in self.ratios or []:
            num, den = arr[:, self._index(n)], arr[:, self._index(d)]
            den = np.where(np.abs(den) < 1e-9, 1e-9, den)
            new_cols.append(num / den)
        if not new_cols:
            return raw
        return np.column_stack([raw] + new_cols)

    def get_feature_names_out(self, input_features=None):
        names = list(input_features) if input_features is not None else list(self.columns)
        names += [interaction_name(a, b) for a, b in self.pairs or []]
        names += [ratio_name(n, d) for n, d in self.ratios or []]
        return np.asarray(names, dtype=object)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip every column to training-data quantiles (limits learned in ``fit``)."""

    def __init__(self, quantiles=(0.01, 0.99)):
        self.quantiles = quantiles

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(arr, self.quantiles[0], axis=0)
        self.upper_ = np.nanquantile(arr, self.quantiles[1], axis=0)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


# --------------------------------------------------------------------------- #
# Which features to engineer
# --------------------------------------------------------------------------- #
def california_spec() -> Dict[str, List[Tuple[str, str]]]:
    """Domain-driven features for the California Housing data."""
    return {
        "pairs": [
            ("MedInc", "HouseAge"),        # wealthier areas that are older
            ("AveRooms", "HouseAge"),      # size x age
            ("Population", "AveOccup"),    # density proxy
            ("Latitude", "Longitude"),     # geographic interaction (coastline orientation)
        ],
        "ratios": [("AveBedrms", "AveRooms")],  # share of rooms that are bedrooms
    }


def suggest_engineering_spec(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: Sequence[str],
    top_k: int = 3,
) -> Dict[str, object]:
    """Choose interaction features.

    * California schema -> the domain-driven spec above.
    * Any other dataset  -> all pairwise products among the ``top_k`` numeric
      features most correlated with the target.  Pass *training* data only so the
      choice itself cannot leak information from the test set.
    """
    numeric_cols = list(numeric_cols)
    if set(C.CALIFORNIA_NUMERIC).issubset(numeric_cols):
        spec = california_spec()
        spec["description"] = "Domain-driven California Housing interactions"
        return spec

    if len(numeric_cols) < 2:
        return {"pairs": [], "ratios": [], "description": "Not enough numeric features"}

    num = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
    num = num.fillna(num.median())
    corr = num.corrwith(pd.Series(np.asarray(y, dtype=float), index=num.index)).abs()
    top = corr.fillna(0).sort_values(ascending=False).head(min(top_k, len(numeric_cols))).index.tolist()
    pairs = [(top[i], top[j]) for i in range(len(top)) for j in range(i + 1, len(top))]
    return {
        "pairs": pairs,
        "ratios": [],
        "description": f"Pairwise products of the {len(top)} features most correlated with the target: {top}",
    }


def restrict_spec(spec: Dict[str, object], available: Sequence[str]) -> Dict[str, object]:
    """Drop spec entries that refer to columns that are not available."""
    avail = set(available)
    return {
        "pairs": [p for p in spec.get("pairs", []) if set(p) <= avail],
        "ratios": [r for r in spec.get("ratios", []) if set(r) <= avail],
        "description": spec.get("description", ""),
    }


def describe_spec(spec: Dict[str, object]) -> pd.DataFrame:
    """Table of engineered features for display."""
    rows = []
    for a, b in spec.get("pairs", []):
        rows.append({"feature": interaction_name(a, b), "type": "interaction", "formula": f"(clip({a}) − mean) × (clip({b}) − mean)"})
    for n, d in spec.get("ratios", []):
        rows.append({"feature": ratio_name(n, d), "type": "ratio", "formula": f"clip({n}) / clip({d})"})
    return pd.DataFrame(rows, columns=["feature", "type", "formula"])

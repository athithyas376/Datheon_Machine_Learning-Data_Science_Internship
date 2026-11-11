"""Dataset loading, inspection and validation.

Design rule: the application never silently substitutes an unrelated dataset for
California Housing.  If the data cannot be found or does not match the expected
schema a :class:`DataLoadError` with an actionable message is raised.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from . import config as C

csv_error = csv.Error


class DataLoadError(Exception):
    """Raised when a dataset cannot be loaded or is not usable."""


# --------------------------------------------------------------------------- #
# California Housing
# --------------------------------------------------------------------------- #
def _derive_from_csv_version(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert the block-group CSV columns to the canonical California names.

    The CSV stores *totals* (rooms, bedrooms, population) per block group; the
    canonical (scikit-learn) dataset stores *per-household averages*:

        AveRooms  = total_rooms    / households
        AveBedrms = total_bedrooms / households
        AveOccup  = population     / households
        MedHouseVal = median_house_value / 100 000
    """
    cols = {c.strip().lower(): c for c in raw.columns}
    missing = sorted(C.CALIFORNIA_CSV_COLUMNS - set(cols))
    if missing:
        raise DataLoadError(
            "The file is not the California Housing CSV: missing column(s) "
            + ", ".join(missing)
        )
    g = lambda name: pd.to_numeric(raw[cols[name]], errors="coerce")  # noqa: E731
    households = g("households").replace(0, np.nan)
    out = pd.DataFrame(
        {
            "MedInc": g("median_income"),
            "HouseAge": g("housing_median_age"),
            "AveRooms": g("total_rooms") / households,
            "AveBedrms": g("total_bedrooms") / households,
            "Population": g("population"),
            "AveOccup": g("population") / households,
            "Latitude": g("latitude"),
            "Longitude": g("longitude"),
        }
    )
    if "ocean_proximity" in cols:
        out[C.CALIFORNIA_CATEGORICAL] = raw[cols["ocean_proximity"]].astype("object")
    out[C.CALIFORNIA_TARGET] = g("median_house_value") / C.CALIFORNIA_TARGET_SCALE
    return out


def is_california_schema(df: pd.DataFrame) -> bool:
    """True when *df* has the canonical California Housing columns."""
    return set(C.CALIFORNIA_NUMERIC + [C.CALIFORNIA_TARGET]).issubset(df.columns)


def load_california_housing(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load the California Housing dataset automatically.

    Order of resolution
    1. ``path`` (if given) – must be a valid California dataset, otherwise an error.
    2. ``data/housing.csv`` shipped with the project.
    3. ``sklearn.datasets.fetch_california_housing`` (needs internet on first use).
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise DataLoadError(f"Dataset file not found: {p}")
        return _parse_california_file(p)

    if C.DEFAULT_DATA_FILE.exists():
        return _parse_california_file(C.DEFAULT_DATA_FILE)

    try:
        from sklearn.datasets import fetch_california_housing

        bunch = fetch_california_housing(as_frame=True)
        df = bunch.frame.copy()
        return df
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise DataLoadError(
            "California Housing data was not found. Place the CSV at "
            f"'{C.DEFAULT_DATA_FILE}' or enable internet access so scikit-learn "
            f"can download it. (scikit-learn error: {exc})"
        ) from exc


def _parse_california_file(path: Path) -> pd.DataFrame:
    df = read_csv_source(path)
    if is_california_schema(df):
        return df
    return _derive_from_csv_version(df)


# --------------------------------------------------------------------------- #
# Generic CSV reading
# --------------------------------------------------------------------------- #
def read_csv_source(source: Any) -> pd.DataFrame:
    """Read a CSV from a path, bytes, or a file-like/Streamlit upload object."""
    if isinstance(source, (str, Path)):
        try:
            raw = Path(source).read_bytes()
        except OSError as exc:
            raise DataLoadError(f"Could not read file: {exc}") from exc
    elif hasattr(source, "getvalue"):
        raw = source.getvalue()
    elif hasattr(source, "read"):
        raw = source.read()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        raise DataLoadError("Unsupported data source type.")

    if not raw or not raw.strip():
        raise DataLoadError("The uploaded file is empty.")

    df: Optional[pd.DataFrame] = None
    last_err: Optional[Exception] = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
        attempts: List[Dict[str, Any]] = [{}]
        try:  # second attempt: sniff the delimiter among common separators
            dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
            attempts.append({"sep": dialect.delimiter})
        except csv.Error:
            pass
        for kwargs in attempts:
            try:
                candidate = pd.read_csv(io.StringIO(text), **kwargs)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, csv_error) as exc:
                last_err = exc
                continue
            if candidate.shape[1] >= 2:
                df = candidate
                break
            last_err = DataLoadError("Only one column detected.")
        if df is not None:
            break
    if df is None:
        raise DataLoadError(
            "Could not parse the file as CSV with at least two columns "
            f"({last_err})."
        )

    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        dup = sorted(set(df.columns[df.columns.duplicated()]))
        raise DataLoadError(f"Duplicate column names are not allowed: {dup}")
    if df.empty:
        raise DataLoadError("The file has a header but no data rows.")
    return df


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def column_kind(series: pd.Series) -> str:
    """Return 'numeric', 'categorical' or 'unsupported' for a column."""
    if is_datetime64_any_dtype(series):
        return "unsupported"
    if is_bool_dtype(series):
        return "categorical"
    if is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Summarise columns, dtypes, missing values and duplicates."""
    n = len(df)
    rows: List[Dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        miss = int(s.isna().sum())
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "kind": column_kind(s),
                "non_null": n - miss,
                "missing": miss,
                "missing_pct": round(100 * miss / n, 3) if n else 0.0,
                "unique": int(s.nunique(dropna=True)),
            }
        )
    info = pd.DataFrame(rows)
    return {
        "n_rows": n,
        "n_cols": df.shape[1],
        "columns": info,
        "numeric_cols": info.loc[info.kind == "numeric", "column"].tolist(),
        "categorical_cols": info.loc[info.kind == "categorical", "column"].tolist(),
        "total_missing": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }


def guess_target(df: pd.DataFrame) -> Optional[str]:
    """Best-effort default target: known names first, else the last numeric column."""
    if C.CALIFORNIA_TARGET in df.columns:
        return C.CALIFORNIA_TARGET
    lowered = {c.lower(): c for c in df.columns}
    for key in ("median_house_value", "medv", "price", "saleprice", "target", "y"):
        if key in lowered and is_numeric_dtype(df[lowered[key]]):
            return lowered[key]
    numeric = [c for c in df.columns if column_kind(df[c]) == "numeric"]
    return numeric[-1] if numeric else None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_dataset(
    df: pd.DataFrame,
    target: Optional[str],
    features: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Validate a dataset for regression.

    Returns a dict with ``ok`` (bool), ``errors`` and ``warnings`` (lists of
    human-readable messages), ``numeric``/``categorical`` (usable feature lists)
    and ``excluded`` ({column: reason}).
    """
    errors: List[str] = []
    warnings: List[str] = []
    excluded: Dict[str, str] = {}
    numeric: List[str] = []
    categorical: List[str] = []

    def result() -> Dict[str, Any]:
        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "numeric": numeric,
            "categorical": categorical,
            "excluded": excluded,
        }

    if df is None or df.empty:
        errors.append("The dataset is empty.")
        return result()
    if not target:
        errors.append("Select a target column.")
        return result()
    if target not in df.columns:
        errors.append(f"Target column '{target}' does not exist in the dataset.")
        return result()
    if column_kind(df[target]) != "numeric":
        errors.append(
            f"Target '{target}' must be numeric for regression (found dtype "
            f"{df[target].dtype})."
        )
        return result()

    y = df[target].replace([np.inf, -np.inf], np.nan)
    n_valid = int(y.notna().sum())
    if n_valid < C.MIN_ROWS:
        errors.append(
            f"Only {n_valid} rows have a usable target value; at least {C.MIN_ROWS} are required."
        )
    if n_valid and y.nunique() < 2:
        errors.append(f"Target '{target}' is constant; there is nothing to predict.")
    if n_valid < len(df):
        warnings.append(
            f"{len(df) - n_valid} row(s) have a missing/infinite target and will be dropped."
        )

    candidates = [c for c in (features if features is not None else df.columns) if c != target]
    unknown = [c for c in candidates if c not in df.columns]
    if unknown:
        errors.append(f"Unknown feature column(s): {unknown}")
        candidates = [c for c in candidates if c in df.columns]

    for col in candidates:
        s = df[col]
        kind = column_kind(s)
        if kind == "unsupported":
            excluded[col] = "datetime column (not supported)"
        elif s.notna().sum() == 0:
            excluded[col] = "all values missing"
        elif s.nunique(dropna=True) <= 1:
            excluded[col] = "constant column"
        elif kind == "categorical" and s.nunique(dropna=True) > C.MAX_CATEGORY_LEVELS:
            excluded[col] = (
                f"too many categories ({s.nunique()} > {C.MAX_CATEGORY_LEVELS}); "
                "looks like an identifier or free text"
            )
        elif kind == "numeric":
            numeric.append(col)
        else:
            categorical.append(col)

    for col, reason in excluded.items():
        warnings.append(f"Feature '{col}' excluded: {reason}.")

    if not numeric and not categorical:
        errors.append("No usable feature columns remain after validation.")
    elif not errors:
        # simple leakage warning: a feature that is (almost) the target itself
        for col in numeric:
            both = pd.concat([df[col], y], axis=1).dropna()
            if len(both) > 2 and both.iloc[:, 0].nunique() > 1:
                r = both.corr().iloc[0, 1]
                if np.isfinite(r) and abs(r) > 0.999:
                    warnings.append(
                        f"Feature '{col}' is almost perfectly correlated with the target "
                        f"(r={r:.4f}); this may be data leakage."
                    )
    if len(df) < 200 and not errors:
        warnings.append("Small dataset (<200 rows): results may be unstable.")
    return result()


def prepare_dataset(
    df: pd.DataFrame, target: str, features: Optional[Iterable[str]] = None
) -> Dict[str, Any]:
    """Select columns, drop duplicate rows and rows without a target.

    Duplicates are removed *before* the train/test split so identical rows can
    never appear on both sides of the split.  No statistics are learned here
    (imputation, scaling, etc. happen later inside the Pipeline).
    """
    check = validate_dataset(df, target, features)
    if not check["ok"]:
        raise DataLoadError(" ".join(check["errors"]))
    feats = check["numeric"] + check["categorical"]
    # keep original column order
    feats = [c for c in df.columns if c in feats]
    work = df[feats + [target]].copy()
    work = work.replace([np.inf, -np.inf], np.nan)

    n_before = len(work)
    work = work.drop_duplicates()
    n_dup = n_before - len(work)
    n_before = len(work)
    work = work[work[target].notna()]
    n_missing_target = n_before - len(work)
    work = work.reset_index(drop=True)

    numeric = [c for c in feats if c in check["numeric"]]
    categorical = [c for c in feats if c in check["categorical"]]
    for c in categorical:  # uniform object dtype; keeps NaN as missing
        work[c] = work[c].astype("object").where(work[c].notna(), np.nan)
    return {
        "data": work,
        "X": work[feats],
        "y": work[target],
        "numeric": numeric,
        "categorical": categorical,
        "target": target,
        "n_duplicates_removed": int(n_dup),
        "n_missing_target_removed": int(n_missing_target),
        "validation": check,
    }

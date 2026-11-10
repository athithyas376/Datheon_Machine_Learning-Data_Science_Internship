"""Loading, splitting and validating data (single inputs and uploaded CSV files)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from . import config


# --------------------------------------------------------------------------
# Loading and splitting
# --------------------------------------------------------------------------
def load_iris_df() -> pd.DataFrame:
    """Load the Iris dataset as a DataFrame with clean column names and a 'species' column."""
    try:
        raw = load_iris(as_frame=True).frame.copy()
    except Exception as error:  # pragma: no cover - only happens if scikit-learn is broken
        raise RuntimeError(f"Could not load the Iris dataset: {error}") from error

    raw.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in raw.columns]
    raw["species"] = raw["target"].map(dict(enumerate(config.CLASS_NAMES)))
    return raw


def split_data(df: pd.DataFrame):
    """80/20 stratified split with a fixed seed. Returns X_train, X_test, y_train, y_test."""
    X = df[config.FEATURE_NAMES]
    y = df["target"]
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """A short hash of the data. Saved with the models so we can tell if the data (and therefore the
    train/test split) has changed since the models were trained."""
    hashed = pd.util.hash_pandas_object(df[config.FEATURE_NAMES + ["target"]], index=True).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()


# --------------------------------------------------------------------------
# Single-flower validation (Flower Prediction page)
# --------------------------------------------------------------------------
def validate_measurements(values: dict, df: pd.DataFrame | None = None) -> tuple[list[str], list[str]]:
    """Check four measurements typed by a user.

    Returns (errors, warnings). Errors block the prediction; warnings only inform the user.
    """
    errors: list[str] = []
    warnings: list[str] = []

    for name in config.FEATURE_NAMES:
        label = config.FEATURE_LABELS[name]
        value = values.get(name)
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label}: '{value}' is not a number.")
            continue
        if not np.isfinite(number):
            errors.append(f"{label}: the value must be a finite number.")
        elif number < config.MIN_MEASUREMENT_CM:
            errors.append(f"{label}: must be at least {config.MIN_MEASUREMENT_CM} cm (you entered {number}).")
        elif number > config.MAX_MEASUREMENT_CM:
            errors.append(f"{label}: must be at most {config.MAX_MEASUREMENT_CM} cm (you entered {number}).")

    if errors:
        return errors, warnings

    # Warnings: values far outside what the model saw during training
    if df is not None:
        for name in config.FEATURE_NAMES:
            low, high = df[name].min(), df[name].max()
            margin = 0.25 * (high - low)
            if values[name] < low - margin or values[name] > high + margin:
                warnings.append(
                    f"{config.FEATURE_LABELS[name]} = {values[name]} is far outside the range in the "
                    f"Iris data ({low:.1f} to {high:.1f} cm). The prediction may be unreliable."
                )
    if values["petal_length"] > values["sepal_length"]:
        warnings.append("Petal length is larger than sepal length, which is unusual for iris flowers. "
                        "Please double-check the numbers.")
    return errors, warnings


# --------------------------------------------------------------------------
# CSV validation (Batch Prediction page)
# --------------------------------------------------------------------------
def normalize_column_name(name: object) -> str:
    """'Sepal Length (cm)' -> 'sepal_length', so slightly different headers are still accepted."""
    text = str(name).strip().lower().replace("(cm)", "").strip()
    return "_".join(text.replace("-", " ").split())


@dataclass
class BatchValidation:
    """Result of checking an uploaded CSV."""
    fatal_errors: list[str] = field(default_factory=list)   # the file cannot be used at all
    features: pd.DataFrame | None = None                    # numeric feature table (same row order as the file)
    row_problems: pd.Series | None = None                   # '' for valid rows, otherwise the reason
    column_map: dict = field(default_factory=dict)          # model feature -> original column name

    @property
    def valid_mask(self) -> pd.Series:
        return self.row_problems == ""


def validate_batch(df_raw: pd.DataFrame) -> BatchValidation:
    """Validate an uploaded table: required columns, numeric values and sensible ranges."""
    result = BatchValidation()

    if df_raw is None or df_raw.shape[0] == 0:
        result.fatal_errors.append("The file has no data rows.")
        return result
    if df_raw.shape[0] > config.MAX_BATCH_ROWS:
        result.fatal_errors.append(
            f"The file has {df_raw.shape[0]:,} rows; the limit is {config.MAX_BATCH_ROWS:,}. Please split the file.")
        return result

    # Match uploaded headers to the four required feature names
    normalized = {col: normalize_column_name(col) for col in df_raw.columns}
    for feature in config.FEATURE_NAMES:
        matches = [orig for orig, norm in normalized.items() if norm == feature]
        if len(matches) > 1:
            result.fatal_errors.append(f"Column '{feature}' appears more than once ({', '.join(map(str, matches))}).")
        elif len(matches) == 1:
            result.column_map[feature] = matches[0]

    missing = [f for f in config.FEATURE_NAMES if f not in result.column_map]
    if missing:
        result.fatal_errors.append(
            "Missing required column(s): " + ", ".join(missing) +
            ". Required columns: " + ", ".join(config.FEATURE_NAMES) + ".")
    if result.fatal_errors:
        return result

    # Convert to numbers; anything that is not a number becomes NaN and is reported per row
    numeric = pd.DataFrame({
        feature: pd.to_numeric(df_raw[result.column_map[feature]], errors="coerce")
        for feature in config.FEATURE_NAMES
    })
    # Collect every problem found in each row (a row can have several)
    problem_lists: list[list[str]] = [[] for _ in range(len(df_raw))]
    for feature in config.FEATURE_NAMES:
        col = numeric[feature]
        not_number = col.isna() | np.isinf(col)
        too_small = (~not_number) & (col < config.MIN_MEASUREMENT_CM)
        too_big = (~not_number) & (col > config.MAX_MEASUREMENT_CM)
        for mask, text in [
            (not_number, f"{feature} is missing or not a number"),
            (too_small, f"{feature} must be at least {config.MIN_MEASUREMENT_CM}"),
            (too_big, f"{feature} must be at most {config.MAX_MEASUREMENT_CM}"),
        ]:
            for position in np.flatnonzero(mask.to_numpy()):
                problem_lists[position].append(text)

    result.features = numeric
    result.row_problems = pd.Series(["; ".join(p) for p in problem_lists], index=df_raw.index, dtype=object)
    return result


def sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Protect against 'CSV injection': text cells that start with = + - @ can run as formulas when the
    downloaded file is opened in Excel. We prefix such text cells with an apostrophe."""
    safe = df.copy()
    for col in safe.columns:
        # Text columns are 'object' in older pandas and 'str' in pandas 3, so check both.
        if pd.api.types.is_object_dtype(safe[col]) or pd.api.types.is_string_dtype(safe[col]):
            safe[col] = safe[col].map(
                lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v)
    return safe

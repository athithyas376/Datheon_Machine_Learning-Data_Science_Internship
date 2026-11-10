"""Batch prediction helpers (kept separate from the Streamlit code so they can be tested on their own)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from . import config


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Read an uploaded CSV. Tries UTF-8 first, then Latin-1. Raises ValueError with a clear message."""
    if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"The file is larger than {config.MAX_UPLOAD_MB} MB.")
    if not raw.strip():
        raise ValueError("The file is empty.")
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty.")
        except pd.errors.ParserError as error:
            raise ValueError(f"The file could not be read as a CSV: {error}")
    raise ValueError("The file's text encoding is not supported. Save it as a UTF-8 CSV.")


def _unique_name(base: str, existing) -> str:
    """Never overwrite a column that the user already has."""
    name = base
    while name in existing:
        name += "_model"
    return name


def build_batch_output(df_raw: pd.DataFrame, validation, pipe) -> tuple[pd.DataFrame, str]:
    """Original data + prediction columns. Invalid rows get no prediction and a reason in 'status'.
    Returns (output table, name of the prediction column)."""
    out = df_raw.copy()
    pred_col = _unique_name("predicted_species", out.columns)
    conf_col = _unique_name("confidence", [*out.columns, pred_col])
    status_col = _unique_name("status", [*out.columns, pred_col, conf_col])

    valid = validation.valid_mask
    out[pred_col] = ""
    out[conf_col] = np.nan
    out[status_col] = np.where(valid, "ok", "skipped: " + validation.row_problems.astype(str))
    if valid.any():
        X = validation.features.loc[valid, config.FEATURE_NAMES]
        proba = pipe.predict_proba(X)
        out.loc[valid, pred_col] = [config.CLASS_NAMES[i] for i in proba.argmax(axis=1)]
        out.loc[valid, conf_col] = proba.max(axis=1).round(4)
    return out, pred_col

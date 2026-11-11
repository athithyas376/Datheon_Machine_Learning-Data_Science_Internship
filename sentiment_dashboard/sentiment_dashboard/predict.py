"""Load the saved models and predict sentiment.

Example::

    from predict import predict_sentiment
    predict_sentiment("This was a wonderful film, I loved it!")
    # {'label': 'positive', 'confidence': 0.83, 'probabilities': {...}, ...}

Or from the terminal::

    python predict.py "The battery died after two days, terrible."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from config import (ARTIFACT_FILES, DEFAULT_MODEL, MODEL_LABELS, MODELS_DIR, NEGATIVE,
                    POSITIVE)
from preprocessing import get_backend_info, preprocess_series, preprocess_text

UNCLASSIFIED = "unclassified"


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model files are missing."""


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _paths(models_dir: Path) -> Dict[str, Path]:
    return {key: Path(models_dir) / name for key, name in ARTIFACT_FILES.items()}


def models_available(models_dir: Path = MODELS_DIR) -> bool:
    return all(p.exists() for p in _paths(models_dir).values())


def artifacts_token(models_dir: Path = MODELS_DIR) -> Optional[int]:
    """A value that changes whenever the models are re-trained (used for caching)."""
    if not models_available(models_dir):
        return None
    return max(p.stat().st_mtime_ns for p in _paths(models_dir).values())


def load_artifacts(models_dir: Path = MODELS_DIR) -> dict:
    """Load vectorizer, both models and metrics from ``models_dir``."""
    paths = _paths(models_dir)
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise ModelNotFoundError(
            "Trained model files not found (" + ", ".join(missing) + "). "
            "Run `python train_model.py` first, or train from the Model Performance page.")
    try:
        with open(paths["metrics"], encoding="utf-8") as fh:
            metrics = json.load(fh)
        return {
            "vectorizer": joblib.load(paths["vectorizer"]),
            "models": {key: joblib.load(paths[key]) for key in MODEL_LABELS},
            "metrics": metrics,
        }
    except Exception as exc:  # corrupted file, sklearn version mismatch, ...
        raise RuntimeError(
            f"The saved models could not be loaded ({exc}). Re-train them with "
            "`python train_model.py`.") from exc


def _get(artifacts: Optional[dict], models_dir: Path = MODELS_DIR) -> dict:
    return artifacts if artifacts is not None else load_artifacts(models_dir)


def _class_index(model, label: str) -> int:
    return list(model.classes_).index(label)


# ----------------------------------------------------------------------------
# Single prediction
# ----------------------------------------------------------------------------
def predict_sentiment(text: str, model_name: str = DEFAULT_MODEL,
                      artifacts: Optional[dict] = None) -> dict:
    """Predict the sentiment of one review.

    Returns a dict with ``label``, ``confidence`` (0-1), ``probabilities``,
    ``cleaned_text`` and ``known_terms`` (how many cleaned words/phrases of the
    review exist in the TF-IDF vocabulary - 0 means the prediction is unreliable).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a review before predicting.")
    if model_name not in MODEL_LABELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_LABELS)}.")

    art = _get(artifacts)
    cleaned = preprocess_text(text)
    if not cleaned:
        raise ValueError("The review has no usable words after cleaning (only symbols, numbers "
                         "or stopwords). Please write a longer sentence.")

    model = art["models"][model_name]
    X = art["vectorizer"].transform([cleaned])
    proba = model.predict_proba(X)[0]
    p_pos = float(proba[_class_index(model, POSITIVE)])
    p_neg = float(proba[_class_index(model, NEGATIVE)])
    label = POSITIVE if p_pos >= p_neg else NEGATIVE
    return {
        "label": label,
        "confidence": max(p_pos, p_neg),
        "probabilities": {POSITIVE: p_pos, NEGATIVE: p_neg},
        "cleaned_text": cleaned,
        "known_terms": int(X.nnz),
        "model_name": model_name,
        "model_label": MODEL_LABELS[model_name],
    }


def _class_weights(model, positive_first: bool = True) -> np.ndarray:
    """Per-feature weight where a positive number pushes towards 'positive'."""
    pos_i = _class_index(model, POSITIVE)
    if hasattr(model, "coef_"):  # Logistic Regression (binary: coef_ is for classes_[1])
        coef = model.coef_[0]
        return coef if pos_i == 1 else -coef
    neg_i = _class_index(model, NEGATIVE)  # Multinomial Naive Bayes
    return model.feature_log_prob_[pos_i] - model.feature_log_prob_[neg_i]


def explain_prediction(text: str, model_name: str = DEFAULT_MODEL,
                       artifacts: Optional[dict] = None, top_n: int = 8) -> List[Tuple[str, float]]:
    """Return the words/phrases that influenced the prediction most.

    Each item is ``(term, contribution)``; positive values pushed the prediction
    towards *positive*, negative values towards *negative*.
    """
    art = _get(artifacts)
    X = art["vectorizer"].transform([preprocess_text(text)]).tocoo()
    names = art["vectorizer"].get_feature_names_out()
    weights = _class_weights(art["models"][model_name])
    items = [(str(names[j]), float(v * weights[j])) for j, v in zip(X.col, X.data)]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return items[:top_n]


# ----------------------------------------------------------------------------
# Batch prediction
# ----------------------------------------------------------------------------
def predict_batch(texts: Sequence[object], model_name: str = DEFAULT_MODEL,
                  artifacts: Optional[dict] = None) -> pd.DataFrame:
    """Predict many reviews. Returns a dataframe aligned with ``texts`` (same order).

    Rows with empty / unusable text get ``predicted_sentiment == 'unclassified'``.
    """
    if model_name not in MODEL_LABELS:
        raise ValueError(f"Unknown model '{model_name}'.")
    art = _get(artifacts)
    model = art["models"][model_name]

    cleaned = preprocess_series(pd.Series(list(texts), dtype=object)).reset_index(drop=True)
    valid = cleaned.str.strip() != ""

    out = pd.DataFrame({
        "predicted_sentiment": UNCLASSIFIED,
        "confidence": np.nan,
        "positive_probability": np.nan,
        "negative_probability": np.nan,
        "known_terms": 0,
    }, index=range(len(cleaned)))

    if valid.any():
        X = art["vectorizer"].transform(cleaned[valid])
        proba = model.predict_proba(X)
        p_pos = proba[:, _class_index(model, POSITIVE)]
        p_neg = proba[:, _class_index(model, NEGATIVE)]
        out.loc[valid, "predicted_sentiment"] = np.where(p_pos >= p_neg, POSITIVE, NEGATIVE)
        out.loc[valid, "confidence"] = np.maximum(p_pos, p_neg)
        out.loc[valid, "positive_probability"] = p_pos
        out.loc[valid, "negative_probability"] = p_neg
        out.loc[valid, "known_terms"] = np.asarray((X != 0).sum(axis=1)).ravel()
    return out


def preprocessing_mismatch(artifacts: dict) -> Optional[str]:
    """Warn if the models were trained with a different text-cleaning backend."""
    trained = (artifacts.get("metrics") or {}).get("preprocessing_backend", {})
    now = get_backend_info()
    if trained and trained.get("using_nltk") != now.get("using_nltk"):
        return ("The models were trained with "
                f"{'NLTK' if trained.get('using_nltk') else 'built-in fallback'} preprocessing but "
                f"this session uses {'NLTK' if now['using_nltk'] else 'the built-in fallback'}. "
                "Predictions may be slightly off - retrain the models to fix this.")
    return None


# ----------------------------------------------------------------------------
# Command line helper
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python predict.py "your review text" [multinomial_nb|logistic_regression]')
        sys.exit(1)
    chosen = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    try:
        result = predict_sentiment(sys.argv[1], chosen)
    except (ValueError, ModelNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Model:      {result['model_label']}")
    print(f"Sentiment:  {result['label'].upper()}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"P(positive)={result['probabilities'][POSITIVE]:.3f}  "
          f"P(negative)={result['probabilities'][NEGATIVE]:.3f}")

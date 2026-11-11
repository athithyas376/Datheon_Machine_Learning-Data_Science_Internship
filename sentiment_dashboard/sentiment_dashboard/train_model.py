"""Train and evaluate the sentiment classifiers.

Usage (from the project folder)::

    python train_model.py                                  # bundled starter data
    python train_model.py --data data/IMDB_Dataset.csv     # your own dataset
    python train_model.py --data a.csv b.csv --test-size 0.25 --max-rows 20000

What happens:
    load CSV -> keep positive/negative rows -> clean text -> stratified
    train/test split -> fit TF-IDF on the TRAIN split only -> train
    Multinomial Naive Bayes and Logistic Regression -> evaluate on the
    held-out test split -> save vectorizer, models and metrics.json to models/.

All numbers written to ``metrics.json`` are computed from the real trained
models. Nothing is hard-coded or estimated.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from config import (ARTIFACT_FILES, BUNDLED_DATASETS, DATA_DIR, LABELS, MIN_LABELLED_ROWS,
                    MIN_PER_CLASS, MODEL_LABELS, MODELS_DIR, NEGATIVE, POSITIVE,
                    RANDOM_STATE, TEST_SIZE, TFIDF_PARAMS)
from preprocessing import get_backend_info, preprocess_series

# ----------------------------------------------------------------------------
# Dataset helpers
# ----------------------------------------------------------------------------
_POSITIVE_VALUES = {"positive", "pos", "1", "true", "good", "p"}
_NEGATIVE_VALUES = {"negative", "neg", "0", "false", "bad", "n", "-1"}

_TEXT_HINTS = ("review", "text", "comment", "content", "sentence", "description", "body")
_LABEL_HINTS = ("sentiment", "label", "polarity", "target", "class", "category")


def read_csv_flexible(source) -> pd.DataFrame:
    """Read a CSV path or an uploaded file object, trying common encodings."""
    last_error: Optional[Exception] = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode the CSV file: {last_error}")


def normalize_label(value: object) -> Optional[str]:
    """Map many label spellings to 'positive' / 'negative'; others -> None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text.endswith(".0"):
        text = text[:-2]
    if text in _POSITIVE_VALUES:
        return POSITIVE
    if text in _NEGATIVE_VALUES:
        return NEGATIVE
    return None  # e.g. "neutral" - not part of this binary task


def normalize_labels(series: pd.Series) -> pd.Series:
    return series.astype(object).map(normalize_label)


def _is_text_column(series: pd.Series) -> bool:
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Guess (text_column, label_column) from column names and content."""
    columns = list(df.columns)
    lowered = {c: str(c).lower() for c in columns}
    is_id = {c: bool(re.search(r"(^|[_\s-])id$|^id($|[_\s-])", lowered[c])) for c in columns}

    label_col = next((c for h in _LABEL_HINTS for c in columns
                      if h in lowered[c] and not is_id[c]), None)

    text_col = next((c for h in _TEXT_HINTS for c in columns
                     if h in lowered[c] and c != label_col and not is_id[c]
                     and _is_text_column(df[c])), None)
    if text_col is None:
        candidates = [c for c in columns if c != label_col and not is_id[c]
                      and _is_text_column(df[c])]
        if candidates:
            text_col = max(candidates, key=lambda c: df[c].astype(str).str.len().mean())
    return text_col, label_col


def load_default_dataset() -> Tuple[pd.DataFrame, str]:
    """Load and merge the datasets bundled in ``data/``."""
    frames, used = [], []
    for name in BUNDLED_DATASETS:
        path = DATA_DIR / name
        if path.exists():
            frame = read_csv_flexible(path)
            text_col, label_col = detect_columns(frame)
            if text_col and label_col:
                frames.append(frame[[text_col, label_col]].rename(
                    columns={text_col: "review_text", label_col: "sentiment"}))
                used.append(name)
    if not frames:
        raise FileNotFoundError(
            f"No bundled dataset found in {DATA_DIR}. Add a CSV with a review "
            "text column and a sentiment column, or pass --data your_file.csv.")
    return pd.concat(frames, ignore_index=True), "Bundled starter data: " + " + ".join(used)


def prepare_dataset(df: pd.DataFrame, text_col: str, label_col: str,
                    max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, dict]:
    """Validate, label-normalise and de-duplicate a raw dataframe."""
    for col in (text_col, label_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' was not found in the dataset.")
    if text_col == label_col:
        raise ValueError("The review column and the sentiment column must be different.")

    work = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label_raw"})
    info = {"n_raw": int(len(work))}

    work = work[work["text"].notna() & (work["text"].astype(str).str.strip() != "")]
    work["label"] = normalize_labels(work["label_raw"])
    info["n_unlabeled_dropped"] = int(work["label"].isna().sum())
    work = work.dropna(subset=["label"])

    before = len(work)
    work = work.drop_duplicates(subset="text")
    info["n_duplicates_dropped"] = int(before - len(work))

    if max_rows and len(work) > max_rows:
        work = work.sample(n=max_rows, random_state=RANDOM_STATE)
    return work[["text", "label"]].reset_index(drop=True), info


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def _to_builtin(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Cannot serialise {type(obj)}")


def _evaluate(model, X_train, y_train, X_test, y_test) -> dict:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "confusion_matrix": confusion_matrix(y_test, test_pred, labels=LABELS).tolist(),
        "report": classification_report(y_test, test_pred, labels=LABELS,
                                        output_dict=True, zero_division=0),
    }


def train_pipeline(df: pd.DataFrame, text_col: str, label_col: str,
                   test_size: float = TEST_SIZE, models_dir: Path = MODELS_DIR,
                   source: str = "uploaded dataset", max_rows: Optional[int] = None,
                   log: Callable[[str], None] = lambda msg: None) -> dict:
    """Train both models, evaluate them on a held-out split and save everything.

    Returns the metrics dictionary (also written to ``models/metrics.json``).
    Raises ``ValueError`` with a beginner-friendly message for bad input.
    """
    data, info = prepare_dataset(df, text_col, label_col, max_rows=max_rows)
    counts = data["label"].value_counts()
    class_counts = {POSITIVE: int(counts.get(POSITIVE, 0)), NEGATIVE: int(counts.get(NEGATIVE, 0))}

    if len(data) < MIN_LABELLED_ROWS or min(class_counts.values()) < MIN_PER_CLASS:
        raise ValueError(
            f"Not enough labelled data: found {class_counts[POSITIVE]} positive and "
            f"{class_counts[NEGATIVE]} negative reviews. At least {MIN_LABELLED_ROWS} reviews "
            f"with {MIN_PER_CLASS}+ per class are required. Check that you picked the right "
            "sentiment column (values such as positive/negative or 1/0).")
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")

    log("Cleaning text (HTML, lowercase, punctuation, stopwords, lemmatisation)...")
    data["clean"] = preprocess_series(data["text"])
    empty_mask = data["clean"].str.strip() == ""
    info["n_empty_after_cleaning"] = int(empty_mask.sum())
    data = data[~empty_mask].reset_index(drop=True)
    counts = data["label"].value_counts()
    class_counts = {POSITIVE: int(counts.get(POSITIVE, 0)), NEGATIVE: int(counts.get(NEGATIVE, 0))}
    if min(class_counts.values()) < MIN_PER_CLASS:
        raise ValueError("After cleaning, one of the classes has too few reviews to train on.")

    n_test = max(2, int(round(len(data) * test_size)))
    log(f"Splitting data: {len(data) - n_test} train / {n_test} test (stratified)...")
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        data["clean"], data["label"], test_size=n_test,
        stratify=data["label"], random_state=RANDOM_STATE)

    log("Fitting TF-IDF on the training split...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)

    models = {
        "multinomial_nb": MultinomialNB(alpha=1.0),
        "logistic_regression": LogisticRegression(max_iter=1000, C=1.0,
                                                  random_state=RANDOM_STATE),
    }

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
        "preprocessing_backend": get_backend_info(),
        "dataset": {
            "source": source,
            **info,
            "n_used": int(len(data)),
            "class_counts": class_counts,
            "avg_words": float(data["text"].astype(str).str.split().str.len().mean()),
            "median_words": float(data["text"].astype(str).str.split().str.len().median()),
            "min_words": int(data["text"].astype(str).str.split().str.len().min()),
            "max_words": int(data["text"].astype(str).str.split().str.len().max()),
        },
        "split": {
            "test_size": float(test_size), "n_train": int(len(y_train)), "n_test": int(len(y_test)),
            "random_state": RANDOM_STATE,
            "train_counts": y_train.value_counts().to_dict(),
            "test_counts": y_test.value_counts().to_dict(),
        },
        "tfidf": {"vocab_size": int(len(vectorizer.vocabulary_)),
                  "max_features": TFIDF_PARAMS["max_features"],
                  "ngram_range": list(TFIDF_PARAMS["ngram_range"])},
        "models": {},
        "warnings": [],
    }

    # Optional cross-validation (more stable than a single small test split).
    do_cv = min(class_counts.values()) >= 5
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE) if do_cv else None

    for key, model in models.items():
        log(f"Training {MODEL_LABELS[key]}...")
        model.fit(X_train, y_train)
        result = _evaluate(model, X_train, y_train, X_test, y_test)
        result["label"] = MODEL_LABELS[key]
        if do_cv:
            pipe = make_pipeline(TfidfVectorizer(**TFIDF_PARAMS), clone(model))
            scores = cross_val_score(pipe, data["clean"], data["label"], cv=cv, scoring="accuracy")
            result["cv_accuracy_mean"] = float(scores.mean())
            result["cv_accuracy_std"] = float(scores.std())
        metrics["models"][key] = result

    # Most influential words according to Logistic Regression.
    lr = models["logistic_regression"]
    weights = lr.coef_[0] if list(lr.classes_)[1] == POSITIVE else -lr.coef_[0]
    names = vectorizer.get_feature_names_out()
    order = np.argsort(weights)
    metrics["top_features"] = {
        "positive": [[names[i], float(weights[i])] for i in order[::-1][:15]],
        "negative": [[names[i], float(weights[i])] for i in order[:15]],
    }

    # Honest warnings about how much to trust the numbers.
    if len(data) < 500:
        metrics["warnings"].append(
            f"Only {len(data)} labelled reviews were used. The metrics are real but based on a "
            "very small dataset, so treat them as a demonstration, not a benchmark. Train on a "
            "larger dataset (e.g. IMDB 50K) for reliable results.")
    if len(y_test) < 100:
        metrics["warnings"].append(
            f"The test set contains only {len(y_test)} reviews, so each mistake changes the "
            "scores a lot.")
    if not metrics["preprocessing_backend"]["using_nltk"]:
        metrics["warnings"].append(
            "NLTK data was not available, so built-in stopword/lemmatisation fallbacks were "
            "used. Run `python -m nltk.downloader stopwords wordnet` and retrain for the "
            "standard NLTK pipeline.")

    log("Saving model artifacts...")
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, models_dir / ARTIFACT_FILES["vectorizer"])
    for key, model in models.items():
        joblib.dump(model, models_dir / ARTIFACT_FILES[key])
    with open(models_dir / ARTIFACT_FILES["metrics"], "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=_to_builtin)
    log("Done.")
    return metrics


# ----------------------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train TF-IDF sentiment classifiers.")
    parser.add_argument("--data", nargs="+", help="One or more CSV files (default: bundled data).")
    parser.add_argument("--text-col", help="Review text column (auto-detected if omitted).")
    parser.add_argument("--label-col", help="Sentiment column (auto-detected if omitted).")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Test fraction (default 0.2).")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows (faster training).")
    args = parser.parse_args(argv)

    try:
        if args.data:
            frames: List[pd.DataFrame] = []
            for file in args.data:
                frame = read_csv_flexible(Path(file))
                t_col, l_col = args.text_col, args.label_col
                auto_t, auto_l = detect_columns(frame)
                t_col, l_col = t_col or auto_t, l_col or auto_l
                if not t_col or not l_col:
                    raise ValueError(f"Could not detect text/label columns in {file}. "
                                     "Use --text-col and --label-col.")
                frames.append(frame[[t_col, l_col]].rename(
                    columns={t_col: "review_text", l_col: "sentiment"}))
            df = pd.concat(frames, ignore_index=True)
            source = "Custom dataset: " + ", ".join(Path(f).name for f in args.data)
        else:
            df, source = load_default_dataset()

        metrics = train_pipeline(df, "review_text", "sentiment", test_size=args.test_size,
                                 source=source, max_rows=args.max_rows,
                                 log=lambda m: print(f"  - {m}"))
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    d = metrics["dataset"]
    print(f"\nDataset: {d['n_used']} reviews "
          f"({d['class_counts'][POSITIVE]} positive / {d['class_counts'][NEGATIVE]} negative)")
    print(f"Split:   {metrics['split']['n_train']} train / {metrics['split']['n_test']} test\n")
    header = f"{'Model':<26}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'Train acc':>11}"
    print(header + "\n" + "-" * len(header))
    for m in metrics["models"].values():
        print(f"{m['label']:<26}{m['accuracy']:>10.3f}{m['precision']:>11.3f}"
              f"{m['recall']:>9.3f}{m['f1']:>8.3f}{m['train_accuracy']:>11.3f}")
    for warning in metrics["warnings"]:
        print(f"\nNOTE: {warning}")
    print(f"\nArtifacts saved to: {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Pipelines, evaluation, cross-validation, and safe saving / loading of trained models."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from . import config
from .data import dataset_fingerprint, load_iris_df, split_data


class ModelValidationError(RuntimeError):
    """Raised when a trained model fails a check. Such a model is never saved."""


class ModelLoadError(RuntimeError):
    """Raised when saved models are missing, damaged, or out of date."""


# --------------------------------------------------------------------------
# Building and evaluating models
# --------------------------------------------------------------------------
def build_pipelines() -> dict[str, Pipeline]:
    """Two pipelines. The scaler is INSIDE each pipeline, so it only ever learns from training data."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=config.DECISION_TREE_MAX_DEPTH,
                                             random_state=config.RANDOM_STATE)),
        ]),
    }


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Accuracy plus macro-averaged precision / recall / F1 (all classes count equally)."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision (macro)": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall (macro)": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1-score (macro)": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def metrics_table(pipelines: dict[str, Pipeline], X_test, y_test) -> pd.DataFrame:
    """One row per model, evaluated on the held-out test set."""
    rows = {name: compute_metrics(y_test, pipe.predict(X_test)) for name, pipe in pipelines.items()}
    return pd.DataFrame(rows).T[config.METRIC_COLUMNS].rename_axis("Model")


def cross_validation_table(pipelines: dict[str, Pipeline], X, y) -> pd.DataFrame:
    """Stratified 5-fold cross-validation. Each fold re-fits the whole pipeline (scaler included),
    so there is no leakage between folds."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rows = {}
    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
        rows[name] = {
            "CV mean accuracy": float(scores["test_accuracy"].mean()),
            "CV std accuracy": float(scores["test_accuracy"].std()),
            "CV mean F1 (macro)": float(scores["test_f1_macro"].mean()),
        }
    return pd.DataFrame(rows).T.rename_axis("Model")


def select_best_model(metrics: pd.DataFrame) -> tuple[str | None, str]:
    """Pick the best model from MEASURED results: highest macro-F1, ties broken by accuracy.
    Returns (model name or None for a tie, short explanation)."""
    ranked = metrics.sort_values(["F1-score (macro)", "Accuracy"], ascending=False)
    top, second = ranked.iloc[0], ranked.iloc[1]
    if np.isclose(top["F1-score (macro)"], second["F1-score (macro)"]) and np.isclose(top["Accuracy"], second["Accuracy"]):
        return None, "Both models scored the same on the test set (a tie)."
    name = ranked.index[0]
    return name, (f"{name} has the highest macro F1-score on the test set "
                  f"({top['F1-score (macro)']:.3f} vs {second['F1-score (macro)']:.3f}).")


def misclassified_samples(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Table of the test flowers this model got wrong."""
    y_pred = pipe.predict(X_test)
    wrong = y_pred != y_test.to_numpy()
    out = X_test[wrong].copy()
    out.insert(0, "row_in_dataset", out.index)
    out["true_species"] = [config.CLASS_NAMES[i] for i in y_test[wrong]]
    out["predicted_species"] = [config.CLASS_NAMES[i] for i in y_pred[wrong]]
    proba = pipe.predict_proba(X_test[wrong])
    out["confidence"] = proba.max(axis=1).round(3)
    return out.reset_index(drop=True)


def report_table(pipe: Pipeline, X_test, y_test) -> pd.DataFrame:
    """Per-class precision / recall / F1 as a table."""
    report = classification_report(y_test, pipe.predict(X_test), target_names=config.CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    return pd.DataFrame(report).T.round(3)


def confusion(pipe: Pipeline, X_test, y_test) -> np.ndarray:
    return confusion_matrix(y_test, pipe.predict(X_test), labels=[0, 1, 2])


# --------------------------------------------------------------------------
# Validation gate before saving
# --------------------------------------------------------------------------
def validate_trained_pipeline(name: str, pipe: Pipeline, X_test, y_test) -> dict[str, float]:
    """Sanity checks. If any check fails, ModelValidationError is raised and the model is NOT saved."""
    try:
        check_is_fitted(pipe)
    except Exception as error:
        raise ModelValidationError(f"{name}: the pipeline is not fitted ({error}).") from error

    predictions = pipe.predict(X_test)
    if len(predictions) != len(X_test) or not set(np.unique(predictions)).issubset({0, 1, 2}):
        raise ModelValidationError(f"{name}: produced invalid predictions.")

    proba = pipe.predict_proba(X_test)
    if not np.all(np.isfinite(proba)) or not np.allclose(proba.sum(axis=1), 1.0):
        raise ModelValidationError(f"{name}: produced invalid probabilities.")

    metrics = compute_metrics(y_test, predictions)
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ModelValidationError(f"{name}: produced non-finite metrics.")
    if metrics["Accuracy"] < config.MIN_ACCEPTABLE_ACCURACY:
        raise ModelValidationError(
            f"{name}: test accuracy {metrics['Accuracy']:.3f} is below the minimum "
            f"{config.MIN_ACCEPTABLE_ACCURACY:.2f}, so the model was not saved.")
    return metrics


# --------------------------------------------------------------------------
# Training + saving (only ever runs on training data; evaluates on held-out test data)
# --------------------------------------------------------------------------
@dataclass
class TrainingResult:
    pipelines: dict
    metrics: pd.DataFrame
    cv: pd.DataFrame
    metadata: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_and_save(models_dir: Path | None = None, save: bool = True) -> TrainingResult:
    """Train both pipelines on the TRAINING split, check them on the held-out TEST split, and save them
    only if every check passes. If anything fails, an exception is raised and previously saved models are
    left untouched."""
    models_dir = Path(models_dir) if models_dir else config.MODELS_DIR

    df = load_iris_df()
    X_train, X_test, y_train, y_test = split_data(df)

    pipelines = build_pipelines()
    for name, pipe in pipelines.items():
        try:
            pipe.fit(X_train, y_train)                      # scaler + model learn from TRAINING data only
        except Exception as error:
            raise ModelValidationError(f"Training failed for {name}: {error}") from error

    for name, pipe in pipelines.items():
        validate_trained_pipeline(name, pipe, X_test, y_test)

    metrics = metrics_table(pipelines, X_test, y_test)
    cv = cross_validation_table(build_pipelines(), df[config.FEATURE_NAMES], df["target"])
    best, reason = select_best_model(metrics)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_names": config.FEATURE_NAMES,
        "class_names": config.CLASS_NAMES,
        "dataset_fingerprint": dataset_fingerprint(df),
        "metrics": metrics.round(6).to_dict(orient="index"),
        "cross_validation": cv.round(6).to_dict(orient="index"),
        "best_model": best,
        "best_model_reason": reason,
        "files": {},
    }

    if save:
        _save_atomically(pipelines, metadata, models_dir)

    return TrainingResult(pipelines=pipelines, metrics=metrics, cv=cv, metadata=metadata)


def _save_atomically(pipelines: dict, metadata: dict, models_dir: Path) -> None:
    """Write everything to a temporary folder first, then move the files into place. A crash while
    writing therefore never leaves a half-written model in models/."""
    models_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging_", dir=models_dir.parent))
    try:
        for name, pipe in pipelines.items():
            file_name = config.MODEL_FILES[name]
            joblib.dump(pipe, staging / file_name)           # the Pipeline includes the fitted scaler
            metadata["files"][file_name] = _sha256(staging / file_name)
        (staging / config.METADATA_FILE).write_text(json.dumps(metadata, indent=2))

        for file_name in [*config.MODEL_FILES.values(), config.METADATA_FILE]:
            os.replace(staging / file_name, models_dir / file_name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def load_saved_models(models_dir: Path | None = None) -> tuple[dict[str, Pipeline], dict]:
    """Load saved pipelines WITHOUT retraining. Raises ModelLoadError (with a clear reason) if the files
    are missing, damaged, or were trained with a different data / scikit-learn version."""
    models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
    meta_path = models_dir / config.METADATA_FILE
    if not meta_path.exists():
        raise ModelLoadError("No saved models found.")

    try:
        metadata = json.loads(meta_path.read_text())
    except (OSError, ValueError) as error:
        raise ModelLoadError(f"Model metadata is unreadable: {error}") from error

    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ModelLoadError(
            f"Models were saved with scikit-learn {metadata.get('sklearn_version')} but "
            f"{sklearn.__version__} is installed.")
    if (metadata.get("random_state"), metadata.get("test_size")) != (config.RANDOM_STATE, config.TEST_SIZE):
        raise ModelLoadError("Saved models used different split settings than the current configuration.")
    if metadata.get("dataset_fingerprint") != dataset_fingerprint(load_iris_df()):
        raise ModelLoadError("The dataset has changed since the models were trained.")

    pipelines = {}
    for name, file_name in config.MODEL_FILES.items():
        path = models_dir / file_name
        if not path.exists():
            raise ModelLoadError(f"Missing model file: {file_name}")
        if _sha256(path) != metadata.get("files", {}).get(file_name):
            raise ModelLoadError(f"{file_name} does not match its recorded checksum (damaged or replaced).")
        try:
            pipe = joblib.load(path)
            check_is_fitted(pipe)
        except Exception as error:
            raise ModelLoadError(f"Could not load {file_name}: {error}") from error
        pipelines[name] = pipe
    return pipelines, metadata


# --------------------------------------------------------------------------
# Plain-language explanations
# --------------------------------------------------------------------------
def tree_rules_in_cm(pipe: Pipeline) -> str:
    """The Decision Tree's rules written in real centimetres (the tree itself works on scaled numbers,
    so each threshold is converted back using the scaler's mean and standard deviation)."""
    scaler, tree = pipe.named_steps["scaler"], pipe.named_steps["model"].tree_
    lines: list[str] = []

    def walk(node: int, depth: int) -> None:
        indent = "    " * depth
        if tree.children_left[node] == -1:                     # leaf
            counts = tree.value[node][0]
            species = config.CLASS_NAMES[int(np.argmax(counts))]
            lines.append(f"{indent}-> predict {species} ({int(tree.n_node_samples[node])} training flowers)")
            return
        f = tree.feature[node]
        threshold_cm = tree.threshold[node] * scaler.scale_[f] + scaler.mean_[f]
        name = config.FEATURE_NAMES[f]
        lines.append(f"{indent}if {name} <= {threshold_cm:.2f} cm:")
        walk(tree.children_left[node], depth + 1)
        lines.append(f"{indent}else:  ({name} > {threshold_cm:.2f} cm)")
        walk(tree.children_right[node], depth + 1)

    walk(0, 0)
    return "\n".join(lines)


def tree_path_in_cm(pipe: Pipeline, sample: pd.DataFrame) -> list[str]:
    """The exact questions the Decision Tree asked for one flower, in centimetres."""
    scaler, model = pipe.named_steps["scaler"], pipe.named_steps["model"]
    scaled = scaler.transform(sample[config.FEATURE_NAMES])
    nodes = model.decision_path(scaled).indices
    tree, steps = model.tree_, []
    for node in nodes:
        if tree.children_left[node] == -1:
            continue
        f = tree.feature[node]
        threshold_cm = tree.threshold[node] * scaler.scale_[f] + scaler.mean_[f]
        value = float(sample.iloc[0][config.FEATURE_NAMES[f]])
        answer = "<=" if value <= threshold_cm else ">"
        steps.append(f"{config.FEATURE_NAMES[f]} = {value:g} cm is {answer} {threshold_cm:.2f} cm")
    return steps


def plain_language_explanation(species: str, proba: np.ndarray, sample: dict, train_df: pd.DataFrame) -> str:
    """A short, honest explanation built from the real numbers (no made-up reasoning)."""
    confidence = float(np.max(proba))
    means = train_df.groupby("species")[config.FEATURE_NAMES].mean()
    petal_len, petal_wid = sample["petal_length"], sample["petal_width"]
    ref = means.loc[species]
    text = (
        f"The model predicts **{species.title()}** with a probability of **{confidence:.0%}**. "
        f"This flower has petals {petal_len:g} cm long and {petal_wid:g} cm wide. "
        f"Typical {species.title()} flowers in the dataset have petals about {ref['petal_length']:.1f} cm long "
        f"and {ref['petal_width']:.1f} cm wide, so these measurements are close to that species."
    )
    order = np.sort(proba)[::-1]
    if confidence < 0.7 or (len(order) > 1 and order[0] - order[1] < 0.3):
        second = config.CLASS_NAMES[int(np.argsort(proba)[::-1][1])]
        text += (f" The model is **not very sure**: {second.title()} is also a possible match, "
                 "which often happens between Versicolor and Virginica because their measurements overlap.")
    return text


# --------------------------------------------------------------------------
# Models for the 2-D decision-boundary picture (visualization only, never saved)
# --------------------------------------------------------------------------
def fit_boundary_model(model_name: str, X_train: pd.DataFrame, y_train, mode: str, features: list[str]):
    """Train a small model on TWO inputs so its decision regions can be drawn on a flat plot.

    mode 'features': use two chosen measurements. mode 'pca': compress all four (scaled) measurements
    into two principal components. Returns (preparation, classifier). Everything is fitted on the
    training split only."""
    classifier = build_pipelines()[model_name].named_steps["model"]
    if mode == "pca":
        prep = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=2, random_state=config.RANDOM_STATE))])
        Z_train = prep.fit_transform(X_train[config.FEATURE_NAMES])
    else:
        prep = Pipeline([("scaler", StandardScaler())])
        Z_train = prep.fit_transform(X_train[features])
    classifier.fit(Z_train, y_train)
    return prep, classifier

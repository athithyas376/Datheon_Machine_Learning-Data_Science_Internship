"""Tests for the core logic (no Streamlit needed).  Run:  python -m pytest tests -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config                                                     # noqa: E402
from src.batch import build_batch_output, read_csv_bytes                   # noqa: E402
from src.data import (dataset_fingerprint, load_iris_df, sanitize_for_csv, split_data,  # noqa: E402
                      validate_batch, validate_measurements)
from src.modeling import (ModelLoadError, ModelValidationError, build_pipelines, load_saved_models,  # noqa: E402
                          select_best_model, train_and_save)


@pytest.fixture(scope="module")
def df():
    return load_iris_df()


# ---------------- data and split ----------------
def test_dataset_shape(df):
    assert df.shape[0] == 150 and df["species"].nunique() == 3
    assert df[config.FEATURE_NAMES].isna().sum().sum() == 0


def test_split_is_80_20_and_stratified(df):
    X_train, X_test, y_train, y_test = split_data(df)
    assert (len(X_train), len(X_test)) == (120, 30)
    assert y_test.value_counts().tolist() == [10, 10, 10]
    assert y_train.value_counts().tolist() == [40, 40, 40]
    assert set(X_train.index).isdisjoint(X_test.index)


def test_scaler_learns_from_training_data_only(df):
    """No leakage: the fitted scaler's mean must equal the TRAINING mean, not the full-data mean."""
    X_train, X_test, y_train, _ = split_data(df)
    pipe = build_pipelines()["Logistic Regression"].fit(X_train, y_train)
    np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, X_train.mean().to_numpy())
    assert not np.allclose(pipe.named_steps["scaler"].mean_, df[config.FEATURE_NAMES].mean().to_numpy())


# ---------------- training, saving, loading ----------------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models") / "models"
    return train_and_save(models_dir=models_dir), models_dir


def test_training_matches_expected_quality(trained):
    result, _ = trained
    assert (result.metrics["Accuracy"] >= config.MIN_ACCEPTABLE_ACCURACY).all()
    assert list(result.metrics.columns) == config.METRIC_COLUMNS


def test_saved_models_reload_and_predict_identically(trained, df):
    result, models_dir = trained
    loaded, meta = load_saved_models(models_dir)
    X_test = split_data(df)[1]
    for name in config.MODEL_NAMES:
        np.testing.assert_array_equal(loaded[name].predict(X_test), result.pipelines[name].predict(X_test))
        assert "scaler" in loaded[name].named_steps          # preprocessing saved together with the model
    assert meta["dataset_fingerprint"] == dataset_fingerprint(df)


def test_metadata_metrics_match_recomputed_metrics(trained, df):
    result, models_dir = trained
    meta = json.loads((models_dir / config.METADATA_FILE).read_text())
    for name in config.MODEL_NAMES:
        for column in config.METRIC_COLUMNS:
            assert meta["metrics"][name][column] == pytest.approx(result.metrics.loc[name, column], abs=1e-5)


def test_failed_validation_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MIN_ACCEPTABLE_ACCURACY", 1.01)   # impossible threshold -> validation fails
    target = tmp_path / "models"
    with pytest.raises(ModelValidationError):
        train_and_save(models_dir=target)
    assert not target.exists() or not any(target.iterdir())
    assert not list(tmp_path.glob(".staging_*"))                   # no temporary leftovers


def test_failed_validation_keeps_previous_models(trained, monkeypatch):
    _, models_dir = trained
    before = {p.name: p.read_bytes() for p in models_dir.iterdir()}
    monkeypatch.setattr(config, "MIN_ACCEPTABLE_ACCURACY", 1.01)
    with pytest.raises(ModelValidationError):
        train_and_save(models_dir=models_dir)
    assert {p.name: p.read_bytes() for p in models_dir.iterdir()} == before


def test_missing_and_tampered_models_are_rejected(trained, tmp_path):
    _, models_dir = trained
    with pytest.raises(ModelLoadError):
        load_saved_models(tmp_path / "nothing_here")
    copy = tmp_path / "copy"
    copy.mkdir()
    for p in models_dir.iterdir():
        (copy / p.name).write_bytes(p.read_bytes())
    (copy / config.MODEL_FILES["Decision Tree"]).write_bytes(b"not a model")
    with pytest.raises(ModelLoadError):
        load_saved_models(copy)


def test_best_model_is_computed_not_hardcoded():
    metrics = pd.DataFrame({"Accuracy": [0.9, 0.95], "Precision (macro)": [0.9, 0.95],
                            "Recall (macro)": [0.9, 0.95], "F1-score (macro)": [0.9, 0.95]},
                           index=pd.Index(["Logistic Regression", "Decision Tree"], name="Model"))
    assert select_best_model(metrics)[0] == "Decision Tree"
    assert select_best_model(metrics.iloc[::-1].assign(**{c: [0.95, 0.9] for c in metrics.columns}))[0] == "Decision Tree"
    tie = metrics.assign(**{c: 0.9 for c in metrics.columns})
    assert select_best_model(tie)[0] is None


# ---------------- single input validation ----------------
GOOD = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


def test_valid_measurements_pass(df):
    errors, warnings = validate_measurements(GOOD, df)
    assert errors == [] and warnings == []


@pytest.mark.parametrize("field,value", [("sepal_length", 0), ("petal_width", -1), ("sepal_width", 99),
                                         ("petal_length", float("nan")), ("petal_length", float("inf")),
                                         ("sepal_length", "abc"), ("petal_width", None)])
def test_bad_measurements_give_errors(df, field, value):
    errors, _ = validate_measurements({**GOOD, field: value}, df)
    assert errors and any(config.FEATURE_LABELS[field] in e for e in errors)


def test_unusual_values_give_warnings_not_errors(df):
    errors, warnings = validate_measurements({**GOOD, "sepal_length": 14.0}, df)
    assert errors == [] and warnings


# ---------------- batch CSV ----------------
def csv(text: str) -> pd.DataFrame:
    return read_csv_bytes(text.encode())


def test_batch_happy_path_and_header_variants(trained):
    result, _ = trained
    raw = csv("Sepal Length (cm),Sepal Width (cm),petal-length,PETAL WIDTH,note\n5.1,3.5,1.4,0.2,a\n6.7,3.0,5.2,2.3,b\n")
    v = validate_batch(raw)
    assert not v.fatal_errors and v.valid_mask.all()
    out, pred_col = build_batch_output(raw, v, result.pipelines["Decision Tree"])
    assert out[pred_col].tolist() == ["setosa", "virginica"]
    assert list(raw.columns) == list(out.columns[:5])              # original columns preserved, in order
    assert (out["status"] == "ok").all()


def test_batch_missing_columns_is_fatal():
    v = validate_batch(csv("sepal_length,sepal_width\n5.1,3.5\n"))
    assert v.fatal_errors and "petal_length" in v.fatal_errors[0]


def test_batch_bad_rows_are_skipped_with_reasons(trained):
    result, _ = trained
    raw = csv("sepal_length,sepal_width,petal_length,petal_width\n5.1,3.5,1.4,0.2\nabc,3.5,1.4,0.2\n5.1,-3,1.4,0.2\n5.1,3.5,,0.2\n5.1,3.5,1.4,99\n")
    v = validate_batch(raw)
    assert v.valid_mask.tolist() == [True, False, False, False, False]
    out, pred_col = build_batch_output(raw, v, result.pipelines["Logistic Regression"])
    assert out[pred_col].tolist()[0] == "setosa" and out[pred_col].tolist()[1:] == [""] * 4
    assert out.loc[1, "status"].startswith("skipped:") and "sepal_length" in out.loc[1, "status"]


def test_batch_all_invalid_and_empty():
    v = validate_batch(csv("sepal_length,sepal_width,petal_length,petal_width\nx,y,z,w\n"))
    assert not v.fatal_errors and not v.valid_mask.any()
    with pytest.raises(ValueError):
        read_csv_bytes(b"")
    v = validate_batch(csv("sepal_length,sepal_width,petal_length,petal_width\n"))
    assert v.fatal_errors


def test_batch_does_not_overwrite_user_columns(trained):
    result, _ = trained
    raw = csv("sepal_length,sepal_width,petal_length,petal_width,predicted_species\n5.1,3.5,1.4,0.2,mine\n")
    out, pred_col = build_batch_output(raw, validate_batch(raw), result.pipelines["Decision Tree"])
    assert pred_col == "predicted_species_model" and out.loc[0, "predicted_species"] == "mine"


def test_duplicate_required_column_is_fatal():
    v = validate_batch(csv("sepal_length,Sepal Length (cm),sepal_width,petal_length,petal_width\n1,1,1,1,1\n"))
    assert v.fatal_errors


def test_row_limit_and_non_utf8():
    big = pd.DataFrame(np.ones((config.MAX_BATCH_ROWS + 1, 4)), columns=config.FEATURE_NAMES)
    assert validate_batch(big).fatal_errors
    latin = "sepal_length,sepal_width,petal_length,petal_width,nom\n5.1,3.5,1.4,0.2,caf\xe9\n".encode("latin-1")
    assert read_csv_bytes(latin).loc[0, "nom"] == "caf\xe9"


def test_csv_injection_is_neutralised():
    df = pd.DataFrame({"note": ["=HYPERLINK(\"x\")", "+1", "-2", "@cmd", "safe"], "n": [1, 2, 3, 4, 5]})
    safe = sanitize_for_csv(df)
    assert safe["note"].tolist() == ["'=HYPERLINK(\"x\")", "'+1", "'-2", "'@cmd", "safe"]
    assert safe["n"].tolist() == [1, 2, 3, 4, 5]

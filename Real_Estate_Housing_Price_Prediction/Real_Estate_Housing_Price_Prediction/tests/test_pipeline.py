"""Automated checks for the ML pipeline (run with `python -m pytest tests -q`)."""
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src import config as C
from src import data_loader as dl
from src import evaluation as ev
from src import model_training as mt
from src import prediction as pr
from src import preprocessing as pp


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["x", "y", "z"], n)})
    df["y"] = 3 * df.a - 2 * df.b + (df.c == "x") * 1.5 + rng.normal(0, 0.3, n)
    df.loc[rng.choice(n, 25), "a"] = np.nan
    df.loc[rng.choice(n, 10), "c"] = np.nan
    return df


def test_california_loads_with_expected_schema():
    df = dl.load_california_housing()
    assert set(C.CALIFORNIA_NUMERIC + [C.CALIFORNIA_TARGET]).issubset(df.columns)
    assert len(df) == 20640


def test_validation_messages():
    df = pd.DataFrame({"a": [1, 2, 3], "t": ["u", "v", "w"]})
    assert not dl.validate_dataset(df, "t")["ok"]
    assert not dl.validate_dataset(df, "missing")["ok"]
    with pytest.raises(dl.DataLoadError):
        dl.read_csv_source(io.BytesIO(b""))


def test_split_is_80_20_and_reproducible(synthetic):
    X, y = synthetic[["a", "b", "c"]], synthetic["y"]
    a = pp.split_data(X, y)
    b = pp.split_data(X, y)
    assert len(a[1]) == 100 and a[0].index.equals(b[0].index)


def test_pipeline_no_nans_and_scaled(synthetic):
    X = synthetic[["a", "b", "c"]]
    pre = pp.build_preprocessor(["a", "b"], ["c"]).fit(X)
    Z = pre.transform(X)
    assert not np.isnan(Z).any()
    assert abs(Z[:, 0].mean()) < 1e-9 and abs(Z[:, 0].std() - 1) < 1e-6


def test_imputer_uses_training_statistics_only(synthetic):
    X = synthetic[["a", "b", "c"]]
    Xtr, Xte = X.iloc[:400], X.iloc[400:]
    pre = pp.build_preprocessor(["a", "b"], ["c"]).fit(Xtr)
    med = pre.named_transformers_["num"].named_steps["imputer"].statistics_[0]
    assert np.isclose(med, Xtr["a"].median())


def test_vif_detects_collinearity():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "x2": x + rng.normal(0, 0.05, 300), "z": rng.normal(size=300)})
    vif = pp.compute_vif(df).set_index("feature").VIF
    assert vif["x"] > 10 and vif["z"] < 2
    kept, log = pp.vif_prune(df, 10)
    assert len(log) >= 1 and "z" in kept


def test_train_project_end_to_end(synthetic, tmp_path):
    res = mt.train_project(synthetic, "y", dataset_name="synthetic", cv=3, n_jobs=1)
    assert set(res["models"]) == {"OLS", "Ridge", "Lasso"}
    m = res["metrics"].set_index("Model")
    # metrics are recomputed from predictions, not stored constants
    X_te, y_te = res["splits"]["X_test"], res["splits"]["y_test"]
    ref = ev.regression_metrics(y_te, res["models"]["Ridge"].predict(X_te))
    assert np.isclose(m.loc["Ridge", "Test RMSE"], ref["RMSE"])
    assert m.loc["OLS", "Test R²"] > 0.9
    assert "alpha" in res["best_params"]["Ridge"] and "alpha" in res["best_params"]["Lasso"]
    # persistence + prediction with the saved pipelines
    mt.save_artifacts(res, tmp_path / "models", tmp_path / "metrics")
    models = pr.load_models(tmp_path / "models")
    row = pr.build_input_frame({"a": 0.5, "b": -1.0, "c": "x"}, ["a", "b", "c"], ["a", "b"])
    out = pr.predict_all(models, row)
    direct = res["models"]["OLS"].predict(row)
    assert np.isclose(out["OLS"][0], direct[0])
    assert abs(out["OLS"][0] - (3 * 0.5 + 2 * 1.0 + 1.5)) < 0.4   # true signal = 5.0
    assert np.isfinite(out["Lasso"][0])


def test_residual_analysis_keys():
    rng = np.random.default_rng(2)
    a = rng.normal(size=500); p = a + rng.normal(0, 0.3, 500)
    st = ev.residual_analysis(a, p)
    assert abs(st["mean"]) < 0.1 and "decile_table" in st
    assert ev.interpret_residuals(st, a, p)

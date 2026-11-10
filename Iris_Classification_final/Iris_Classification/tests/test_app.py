"""Tests that run the real Streamlit app headlessly (Streamlit's AppTest).  Run:  python -m pytest tests -q

Batch CSV upload is tested here with Streamlit's file-uploader support, and in test_core.py at logic level."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

APP = str(ROOT / "app.py")
PAGES = ["Home Dashboard", "Dataset Explorer", "Model Comparison", "Flower Prediction",
         "Batch Prediction", "Model Insights"]


def run_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["page"] = page
    return at.run()


def click(at: AppTest, label: str) -> AppTest:
    matches = [b for b in at.button if b.label == label]
    assert matches, f"button '{label}' not found"
    return matches[0].click().run()


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_home_best_model_is_computed_from_results():
    at = run_page("Home Dashboard")
    kpis = {m.label: m.value for m in at.metric}
    assert kpis["Samples"] == "150" and kpis["Input features"] == "4" and kpis["Flower classes"] == "3"
    assert kpis["Best model"] in ("Logistic Regression", "Decision Tree", "Tie")
    assert any("highest macro F1-score" in s.value or "tie" in s.value.lower() for s in at.success)


def test_sidebar_timestamp_is_clean_and_best_model_card_is_keyed():
    """Regression tests for two visual fixes found in a real-browser check."""
    at = run_page("Home Dashboard")
    sidebar_text = " ".join(c.value for c in at.sidebar.caption)
    assert "Models trained:" in sidebar_text and "+00:00" not in sidebar_text
    # the 'Best model' card has its own key so a CSS rule can shrink its font (long names were cut off with "...")
    assert any("st-key-kpi_best" in m.value for m in at.markdown)


def test_comparison_table_shows_all_four_metrics_and_chart_below():
    at = run_page("Model Comparison")
    assert not at.exception
    table = next(d.value for d in at.dataframe if "F1-score (macro)" in d.value.columns)
    assert list(table.columns) == config.METRIC_COLUMNS


def test_decision_tree_probability_note_is_shown_only_for_the_tree():
    at = run_page("Flower Prediction")
    at.selectbox(key="predict_model").set_value("Decision Tree")
    at = click(at, "Predict")
    assert any("often exactly 100%" in c.value for c in at.caption)
    at.selectbox(key="predict_model").set_value("Logistic Regression")
    at = click(at, "Predict")
    assert not any("often exactly 100%" in c.value for c in at.caption)


def test_home_navigation_buttons():
    at = run_page("Home Dashboard")
    at.button(key="open_Model Comparison").click().run()
    assert at.session_state["page"] == "Model Comparison" and not at.exception
    assert any("Confusion" in s.value for s in at.subheader)


def test_prediction_setosa_and_probabilities():
    at = run_page("Flower Prediction")
    for key, value in {"in_sepal_length": 5.1, "in_sepal_width": 3.5, "in_petal_length": 1.4, "in_petal_width": 0.2}.items():
        at.number_input(key=key).set_value(value)
    at = click(at, "Predict")
    assert not at.exception and not at.error
    assert any("Iris Setosa" in s.value for s in at.success)
    assert any(m.label == "Confidence" for m in at.metric)


def test_prediction_virginica_with_decision_tree():
    at = run_page("Flower Prediction")
    at.selectbox(key="predict_model").set_value("Decision Tree")
    for key, value in {"in_sepal_length": 6.9, "in_sepal_width": 3.1, "in_petal_length": 5.4, "in_petal_width": 2.1}.items():
        at.number_input(key=key).set_value(value)
    at = click(at, "Predict")
    assert any("Iris Virginica" in s.value for s in at.success)
    assert any("Questions the Decision Tree asked" in e.label for e in at.expander)


def test_prediction_with_logistic_regression():
    at = run_page("Flower Prediction")
    at.selectbox(key="predict_model").set_value("Logistic Regression")
    for key, value in {"in_sepal_length": 6.0, "in_sepal_width": 2.9, "in_petal_length": 4.5, "in_petal_width": 1.5}.items():
        at.number_input(key=key).set_value(value)
    at = click(at, "Predict")
    assert not at.exception and not at.error
    assert any("Iris Versicolor" in s.value for s in at.success)
    assert not any("Questions the Decision Tree asked" in e.label for e in at.expander)   # tree-only section


@pytest.mark.parametrize("field,value,text", [("in_sepal_length", 0.0, "at least"),
                                              ("in_petal_width", 50.0, "at most")])
def test_prediction_rejects_invalid_input_with_helpful_message(field, value, text):
    at = run_page("Flower Prediction")
    at.number_input(key=field).set_value(value)
    at = click(at, "Predict")
    assert not at.exception
    assert at.error and any(text in e.value for e in at.error)
    assert not at.success                        # no prediction is shown for invalid input


def test_prediction_unusual_values_warn_but_still_predict():
    at = run_page("Flower Prediction")
    at.number_input(key="in_sepal_length").set_value(14.0)
    at = click(at, "Predict")
    assert at.warning and at.success


def test_example_selector_fills_inputs():
    at = run_page("Flower Prediction")
    at.selectbox(key="example_choice").set_value("Typical Setosa").run()
    assert at.number_input(key="in_petal_length").value < 2.0


def test_dataset_explorer_filters():
    at = run_page("Dataset Explorer")
    at.multiselect(key="explorer_species").set_value(["Setosa"]).run()
    assert not at.exception
    assert any("Showing **50** of 150" in m.value for m in at.markdown)
    # an impossible filter combination must show a friendly warning instead of crashing
    at.multiselect(key="explorer_species").set_value([]).run()
    assert not at.exception and any("No flowers match" in w.value for w in at.warning)


def test_insights_boundary_modes_and_same_feature_error():
    at = run_page("Model Insights")
    at.radio(key="boundary_mode").set_value("PCA (2 components)").run()
    assert not at.exception and any("PC1 keeps" in i.value for i in at.info)
    at.radio(key="boundary_mode").set_value("Two features").run()
    at.selectbox(key="boundary_f2").set_value("petal_length").run()
    assert any("two different features" in e.value for e in at.error)


def upload(csv_text: bytes, name: str = "flowers.csv") -> AppTest:
    at = run_page("Batch Prediction")
    at.file_uploader(key="batch_file").set_value((name, csv_text, "text/csv"))
    return at.run()


def metrics_of(at: AppTest) -> dict:
    return {m.label: m.value for m in at.metric}


def test_batch_page_waits_for_upload():
    at = run_page("Batch Prediction")
    assert any("Upload a CSV" in i.value for i in at.info)


def test_batch_valid_csv_predicts_and_offers_download():
    at = upload(b"sepal_length,sepal_width,petal_length,petal_width\n5.1,3.5,1.4,0.2\n6.0,2.9,4.5,1.5\n6.9,3.1,5.4,2.1\n")
    assert not at.exception and not at.error
    assert metrics_of(at) == {"Rows in file": "3", "Valid rows predicted": "3", "Rows skipped": "0"}
    frames = [d.value for d in at.dataframe]
    result = next(f for f in frames if "predicted_species" in f.columns)
    assert result["predicted_species"].tolist() == ["setosa", "versicolor", "virginica"]
    assert at.get("download_button"), "download button for the results is missing"


def test_batch_mixed_valid_and_invalid_rows():
    at = upload(b"sepal_length,sepal_width,petal_length,petal_width,id\n5.1,3.5,1.4,0.2,a\nbad,3.5,1.4,0.2,b\n5.1,3.5,1.4,-2,c\n")
    assert not at.exception
    assert metrics_of(at) == {"Rows in file": "3", "Valid rows predicted": "1", "Rows skipped": "2"}
    assert any("2 row(s) were skipped" in w.value for w in at.warning)
    result = next(d.value for d in at.dataframe if "status" in d.value.columns)
    assert result["status"].tolist()[0] == "ok" and result["status"].tolist()[1].startswith("skipped")
    assert result["id"].tolist() == ["a", "b", "c"]            # original columns are preserved


def test_batch_missing_columns_shows_error():
    at = upload(b"sepal_length,sepal_width\n5.1,3.5\n")
    assert not at.exception
    assert any("Missing required column" in e.value for e in at.error)


def test_batch_all_invalid_rows_shows_error():
    at = upload(b"sepal_length,sepal_width,petal_length,petal_width\nx,y,z,w\n")
    assert not at.exception and any("None of the rows" in e.value for e in at.error)


@pytest.mark.parametrize("content,text", [(b"", "empty"), (b"\n\n", "empty"),
                                          (b"sepal_length,sepal_width,petal_length,petal_width\n", "no data rows")])
def test_batch_empty_files_show_friendly_error(content, text):
    at = upload(content)
    assert not at.exception and any(text in e.value.lower() for e in at.error)


def test_batch_accepts_header_variants_and_other_model():
    at = run_page("Batch Prediction")
    at.selectbox(key="batch_model").set_value("Logistic Regression")
    at.file_uploader(key="batch_file").set_value(
        ("v.csv", b"Sepal Length (cm),Sepal Width (cm),Petal Length (cm),Petal Width (cm)\n5.0,3.4,1.5,0.2\n", "text/csv"))
    at.run()
    assert not at.exception and not at.error and metrics_of(at)["Valid rows predicted"] == "1"


def test_app_trains_when_models_missing(tmp_path, monkeypatch):
    """If the saved models are missing the app must train and save new ones, not crash."""
    import streamlit as st
    empty = tmp_path / "models"
    monkeypatch.setattr(config, "MODELS_DIR", empty)
    st.cache_resource.clear()                           # forget models cached by earlier tests
    try:
        at = AppTest.from_file(APP, default_timeout=120)
        at.session_state["page"] = "Home Dashboard"
        at.run()
        assert not at.exception and not at.error
        assert any("new models were trained and saved" in i.value for i in at.info)
        assert (empty / config.MODEL_FILES["Decision Tree"]).exists()
        assert (empty / config.METADATA_FILE).exists()
    finally:
        st.cache_resource.clear()                       # do not leak these models into other tests

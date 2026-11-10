"""Iris Flower Classification Dashboard (Streamlit).

Run with:   streamlit run app.py

The app loads the models saved by train_model.py (no retraining on every start). If no valid saved model
exists, it trains and saves one automatically. Every number shown comes from the real models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import config, plots
from src.batch import build_batch_output, read_csv_bytes
from src.data import (load_iris_df, sanitize_for_csv, split_data, validate_batch, validate_measurements)
from src.modeling import (ModelLoadError, confusion, fit_boundary_model, load_saved_models,
                          metrics_table, misclassified_samples, plain_language_explanation, report_table,
                          select_best_model, train_and_save, tree_path_in_cm, tree_rules_in_cm)

st.set_page_config(page_title="Iris Classification Dashboard", page_icon="🌸", layout="wide",
                   initial_sidebar_state="auto")   # open on desktop, collapsed on phones so it never covers the page

PAGES = ["Home Dashboard", "Dataset Explorer", "Model Comparison", "Flower Prediction",
         "Batch Prediction", "Model Insights"]


# ==========================================================================
# Data and model loading (cached, so nothing is reloaded or retrained on every click)
# ==========================================================================
@st.cache_data(show_spinner=False)
def get_data():
    df = load_iris_df()
    return df, split_data(df)


@st.cache_resource(show_spinner="Loading saved models...")
def get_models():
    """Load saved models. Only if they are missing or unusable, train new ones (and save them)."""
    try:
        pipelines, metadata = load_saved_models()
        return {"pipelines": pipelines, "metadata": metadata, "notice": None, "error": None}
    except ModelLoadError as reason:
        try:
            result = train_and_save()
            return {"pipelines": result.pipelines, "metadata": result.metadata, "error": None,
                    "notice": f"Saved models were not usable ({reason}) so new models were trained and saved."}
        except Exception as error:
            return {"pipelines": None, "metadata": None, "notice": None,
                    "error": f"Training failed and nothing was saved: {error}"}


def require_models():
    """Return (pipelines, metadata) or stop the page with a clear error."""
    state = get_models()
    if state["error"]:
        st.error(state["error"])
        st.info("Fix the problem above, then use 'Retrain models' in the sidebar or run `python train_model.py`.")
        st.stop()
    if state["notice"]:
        st.info(state["notice"])
    return state["pipelines"], state["metadata"]


def evaluate(pipelines):
    """Evaluate the saved models on the held-out test set (no training happens here)."""
    _, (X_train, X_test, y_train, y_test) = get_data()
    metrics = metrics_table(pipelines, X_test, y_test)
    best, reason = select_best_model(metrics)
    return metrics, best, reason


def go_to(page: str) -> None:
    st.session_state["page"] = page


def section(title: str, description: str) -> None:
    st.header(title)
    st.caption(description)


# ==========================================================================
# Sidebar
# ==========================================================================
def sidebar() -> str:
    with st.sidebar:
        st.title("Iris Classifier")
        st.caption("Multi-class classification with Logistic Regression and a Decision Tree")
        page = st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.divider()

        state = get_models()
        if state["pipelines"] is not None:
            meta = state["metadata"]
            trained_at = meta["trained_at_utc"].replace("T", " ").replace("+00:00", "")
            st.caption(f"Models trained: {trained_at} UTC  \n"
                       f"scikit-learn {meta['sklearn_version']}")
        else:
            st.caption("Models are not available.")

        with st.expander("Model management"):
            st.write("Retraining uses the same 80/20 split and settings. New models replace the saved ones "
                     "only if all checks pass.")
            if st.button("Retrain models", help="Train both pipelines again and save them if they pass validation."):
                with st.spinner("Training..."):
                    try:
                        train_and_save()
                        get_models.clear()
                        st.success("New models trained, validated and saved.")
                    except Exception as error:
                        st.error(f"Retraining failed. The previously saved models were kept. Reason: {error}")
    return page


# ==========================================================================
# A. Home
# ==========================================================================
def page_home():
    pipelines, meta = require_models()
    df, (X_train, X_test, y_train, y_test) = get_data()
    metrics, best, reason = evaluate(pipelines)

    st.title("Iris Flower Classification Dashboard")
    st.write("This project predicts the species of an iris flower (**Setosa**, **Versicolor** or **Virginica**) "
             "from four measurements: sepal length, sepal width, petal length and petal width. "
             "Two machine-learning models, **Logistic Regression** and a **Decision Tree**, are trained, "
             "compared and made available for live predictions.")

    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1, st.container(border=True):
        st.metric("Samples", f"{len(df)}", help="Number of flowers in the Iris dataset.")
    with c2, st.container(border=True):
        st.metric("Input features", f"{len(config.FEATURE_NAMES)}", help="Sepal length/width and petal length/width, in cm.")
    with c3, st.container(border=True):
        st.metric("Flower classes", f"{df['species'].nunique()}", help="Setosa, Versicolor and Virginica (50 flowers each).")
    with c4, st.container(border=True):
        st.metric("Test flowers", f"{len(X_test)}", help="20% of the data, kept aside and never used for training.")

    st.subheader("Model performance (held-out test set)")
    top_name = best if best else metrics.sort_values("F1-score (macro)", ascending=False).index[0]
    row = metrics.loc[top_name]
    cv = meta["cross_validation"][top_name]
    k1, k2, k3, k4 = st.columns(4)
    # The model name is long, so this card gets a keyed container: a small CSS rule (below) shrinks its font
    # so "Logistic Regression" is never cut off with "...".
    st.markdown("<style>.st-key-kpi_best [data-testid='stMetricValue']{font-size:1.5rem;}</style>",
                unsafe_allow_html=True)
    with k1, st.container(border=True, key="kpi_best"):
        st.metric("Best model", best if best else "Tie", help=reason)
    with k2, st.container(border=True):
        st.metric("Accuracy", f"{row['Accuracy']:.1%}", help="Share of test flowers classified correctly.")
    with k3, st.container(border=True):
        st.metric("F1-score (macro)", f"{row['F1-score (macro)']:.3f}",
                  help="Average of the per-species F1-scores; balances precision and recall.")
    with k4, st.container(border=True):
        st.metric("5-fold CV accuracy", f"{cv['CV mean accuracy']:.1%}",
                  help="Average accuracy over 5 train/test rounds on all data: a more stable estimate than one split.")

    st.success(reason)
    per_flower = 100 / len(X_test)
    st.caption(f"The test set has only {len(X_test)} flowers, so a single flower changes accuracy by about "
               f"{per_flower:.1f} percentage points. See Model Comparison for the cross-validation check.")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plots.class_distribution_fig(df), width="stretch")
    with right:
        st.plotly_chart(plots.metrics_bar_fig(metrics), width="stretch")

    st.subheader("Where to go next")
    cards = [
        ("Dataset Explorer", "Browse the data, filter it and view charts."),
        ("Model Comparison", "Metrics, confusion matrices and cross-validation."),
        ("Flower Prediction", "Enter measurements and get a species."),
        ("Batch Prediction", "Upload a CSV and download predictions."),
        ("Model Insights", "Feature importance, tree and decision boundaries."),
    ]
    for col, (page, text) in zip(st.columns(len(cards)), cards):
        with col, st.container(border=True):
            st.markdown(f"**{page}**")
            st.caption(text)
            st.button("Open", key=f"open_{page}", on_click=go_to, args=(page,), width="stretch")


# ==========================================================================
# B. Dataset Explorer
# ==========================================================================
def page_dataset():
    df, (X_train, X_test, y_train, y_test) = get_data()
    section("Dataset Explorer", "Look at the raw data, check its quality, and explore how the features behave.")

    missing = int(df[config.FEATURE_NAMES].isna().sum().sum())
    dup_mask = df.duplicated(keep=False)
    n_dups = int(df.duplicated().sum())

    a, b, c, d = st.columns(4)
    a.metric("Rows x columns", f"{df.shape[0]} x {len(config.FEATURE_NAMES) + 1}", help="Four features plus the species label.")
    b.metric("Features", len(config.FEATURE_NAMES))
    c.metric("Missing values", missing)
    d.metric("Duplicate records", n_dups, help="Rows identical to an earlier row.")

    with st.expander("Feature names and data quality notes", expanded=False):
        st.write("**Features:** " + ", ".join(config.FEATURE_LABELS[f] for f in config.FEATURE_NAMES))
        st.write("**Classes:** " + ", ".join(c.title() for c in config.CLASS_NAMES))
        if n_dups:
            st.write("The dataset contains duplicate records (this is a known property of the original Iris data). "
                     "They are kept so results stay comparable with other Iris benchmarks.")
            dups = df[dup_mask][config.FEATURE_NAMES + ["species"]].copy()
            dups["used_in"] = ["training" if i in X_train.index else "test" for i in dups.index]
            st.dataframe(dups, width="stretch")

    tab_table, tab_dist, tab_rel = st.tabs(["Table and filters", "Feature distributions", "Relationships"])

    with tab_table:
        f1, f2 = st.columns([1, 2])
        with f1:
            species = st.multiselect("Species", [c.title() for c in config.CLASS_NAMES],
                                     default=[c.title() for c in config.CLASS_NAMES], key="explorer_species",
                                     help="Show only the selected species.")
        with f2:
            with st.expander("Filter by measurement range", expanded=False):
                ranges = {}
                cols = st.columns(2)
                for i, feature in enumerate(config.FEATURE_NAMES):
                    low, high = float(df[feature].min()), float(df[feature].max())
                    ranges[feature] = cols[i % 2].slider(config.FEATURE_LABELS[feature], low, high, (low, high),
                                                         step=0.1, key=f"range_{feature}")
        filtered = df[df["species"].isin([s.lower() for s in species])]
        for feature, (low, high) in ranges.items():
            filtered = filtered[filtered[feature].between(low, high)]

        st.write(f"Showing **{len(filtered)}** of {len(df)} flowers.")
        st.dataframe(filtered[config.FEATURE_NAMES + ["species"]], width="stretch", hide_index=True)
        if len(filtered):
            st.write("**Summary statistics of the filtered data**")
            st.dataframe(filtered[config.FEATURE_NAMES].describe().round(2), width="stretch")
        else:
            st.warning("No flowers match these filters. Widen the species or range filters.")

    with tab_dist:
        left, right = st.columns([1, 3])
        with left:
            feature = st.selectbox("Feature", config.FEATURE_NAMES, format_func=lambda f: config.FEATURE_LABELS[f],
                                   key="explorer_feature")
            kind = st.radio("Chart type", ["Histogram", "Box plot"], key="explorer_kind",
                            help="Histograms show how values are spread; box plots show median, spread and outliers.")
            st.caption("Charts use the filters from the 'Table and filters' tab.")
        with right:
            data = filtered if len(filtered) else df
            if not len(filtered):
                st.warning("The filters match no flowers, so the full dataset is shown.")
            st.plotly_chart(plots.feature_distribution_fig(data, feature, kind), width="stretch")
        st.plotly_chart(plots.class_distribution_fig(df), width="stretch")

    with tab_rel:
        st.plotly_chart(plots.correlation_fig(df), width="stretch")
        st.caption("Petal length and petal width are strongly correlated with each other and with the species.")
        st.plotly_chart(plots.pairplot_fig(df), width="stretch")
        st.caption("Setosa is clearly separate from the other two species; Versicolor and Virginica overlap a little.")


# ==========================================================================
# C. Model Comparison
# ==========================================================================
def page_comparison():
    pipelines, meta = require_models()
    _, (X_train, X_test, y_train, y_test) = get_data()
    section("Model Comparison",
            "Both models are evaluated on the same held-out test flowers that were never used for training.")

    st.info(f"Setup: {len(X_train)} training / {len(X_test)} test flowers (80/20, stratified, random_state="
            f"{config.RANDOM_STATE}). `StandardScaler` sits inside each Scikit-learn Pipeline, so it learns only "
            "from training data. Precision, recall and F1 use **macro** averaging (every species counts equally).")

    metrics, best, reason = evaluate(pipelines)
    (st.success if best else st.warning)(reason)

    # Full-width table (in a narrow column the Recall and F1 columns were cut off), chart underneath.
    st.subheader("Results table")
    st.dataframe(metrics.style.format("{:.4f}").highlight_max(axis=0, color="#cdeee8"), width="stretch")
    st.caption("Green = best value in each column.")
    st.plotly_chart(plots.metrics_bar_fig(metrics), width="stretch")

    st.subheader("Confusion matrices")
    st.caption("Rows are the true species, columns are the predicted species. The diagonal shows correct predictions.")
    cols = st.columns(2)
    for col, (name, pipe) in zip(cols, pipelines.items()):
        with col:
            st.plotly_chart(plots.confusion_fig(confusion(pipe, X_test, y_test), name), width="stretch")

    st.subheader("5-fold cross-validation")
    st.caption("All 150 flowers are split into 5 parts; each part is used once as a test set while the pipeline is "
               "re-trained on the other four. This is more reliable than one 30-flower test set.")
    cv = pd.DataFrame(meta["cross_validation"]).T.rename_axis("Model")
    st.dataframe(cv.style.format("{:.4f}").highlight_max(subset=["CV mean accuracy", "CV mean F1 (macro)"],
                                                           axis=0, color="#cdeee8"), width="stretch")
    gap = abs(cv["CV mean accuracy"].iloc[0] - cv["CV mean accuracy"].iloc[1])
    spread = cv["CV std accuracy"].max()
    if gap < spread:
        st.caption(f"The cross-validated accuracy gap ({gap:.3f}) is smaller than the fold-to-fold variation "
                   f"({spread:.3f}), so the two models perform similarly; the single-split winner should not be over-interpreted.")

    with st.expander("Per-species precision, recall and F1"):
        for name, pipe in pipelines.items():
            st.write(f"**{name}**")
            st.dataframe(report_table(pipe, X_test, y_test), width="stretch")


# ==========================================================================
# D. Flower Prediction
# ==========================================================================
def _example_values(df: pd.DataFrame, choice: str) -> dict | None:
    if choice == "Custom values":
        return None
    means = df.groupby("species")[config.FEATURE_NAMES].mean().round(1)
    return means.loc[choice.replace("Typical ", "").lower()].to_dict()


def _apply_example(df: pd.DataFrame) -> None:
    values = _example_values(df, st.session_state["example_choice"])
    if values:
        for feature, value in values.items():
            st.session_state[f"in_{feature}"] = float(value)


def page_prediction():
    pipelines, meta = require_models()
    df, _ = get_data()
    metrics, best, _ = evaluate(pipelines)
    section("Flower Prediction", "Enter the four measurements of a flower and the model will predict its species.")

    names = list(pipelines)
    model_name = st.selectbox("Model", names, index=names.index(best) if best else 0, key="predict_model",
                              help="The best model on the test set is selected by default.")

    defaults = df.groupby("species")[config.FEATURE_NAMES].mean().round(1).loc["versicolor"].to_dict()
    for feature, value in defaults.items():
        st.session_state.setdefault(f"in_{feature}", float(value))

    st.selectbox("Start from an example", ["Custom values", "Typical Setosa", "Typical Versicolor", "Typical Virginica"],
                 key="example_choice", on_change=_apply_example, args=(df,),
                 help="Fills the boxes with the average measurements of that species. You can still edit them.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        inputs = {}
        for i, feature in enumerate(config.FEATURE_NAMES):
            inputs[feature] = (col1 if i % 2 == 0 else col2).number_input(
                config.FEATURE_LABELS[feature], min_value=0.0, max_value=100.0, step=0.1, format="%.1f",
                key=f"in_{feature}",
                help=f"Allowed range: {config.MIN_MEASUREMENT_CM} to {config.MAX_MEASUREMENT_CM} cm. "
                     f"Iris data range: {df[feature].min():.1f} to {df[feature].max():.1f} cm.")
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        st.caption("Press **Predict** to see the result.")
        return

    errors, warnings = validate_measurements(inputs, df)
    if errors:
        for message in errors:
            st.error(message)
        st.info("Please correct the values above and press Predict again.")
        return
    for message in warnings:
        st.warning(message)

    pipe = pipelines[model_name]
    sample = pd.DataFrame([inputs])[config.FEATURE_NAMES]
    try:
        species = config.CLASS_NAMES[int(pipe.predict(sample)[0])]
        proba = pipe.predict_proba(sample)[0] if hasattr(pipe, "predict_proba") else None
    except Exception as error:
        st.error(f"The prediction failed: {error}")
        return

    st.success(f"Predicted species: **Iris {species.title()}**")
    left, right = st.columns([1, 1])
    with left:
        if proba is not None:
            st.metric("Confidence", f"{float(proba.max()):.1%}",
                      help="The probability the model gives to its chosen species.")
            st.plotly_chart(plots.probability_fig(proba, model_name), width="stretch")
        else:
            st.info("This model does not provide class probabilities.")
    with right:
        st.plotly_chart(plots.position_fig(df, inputs), width="stretch")

    st.subheader("What does this mean?")
    if proba is not None:
        st.write(plain_language_explanation(species, proba, inputs, df))
    if model_name == "Decision Tree":
        st.caption("A Decision Tree's probability is simply the share of training flowers of each species in the final "
                   "box (leaf) the flower lands in, so it is often exactly 100%. That is not the same as being "
                   "certain: Logistic Regression gives smoother probabilities.")
        with st.expander("Questions the Decision Tree asked about this flower"):
            for step in tree_path_in_cm(pipe, sample):
                st.write("- " + step)
    st.caption("This is a statistical prediction from 150 example flowers. It is a learning project, not a botanical identification tool.")


# ==========================================================================
# E. Batch Prediction
# ==========================================================================
def page_batch():
    pipelines, meta = require_models()
    metrics, best, _ = evaluate(pipelines)
    section("Batch Prediction", "Upload a CSV file of flower measurements and download the predicted species.")

    st.write("**Required columns:** " + ", ".join(f"`{f}`" for f in config.FEATURE_NAMES) +
             f". Values must be numbers between {config.MIN_MEASUREMENT_CM} and {config.MAX_MEASUREMENT_CM} cm. "
             "Other columns are kept in the output. Headers like `Sepal Length (cm)` are also accepted.")
    template = pd.DataFrame({"sepal_length": [5.1, 6.0, 6.9], "sepal_width": [3.5, 2.9, 3.1],
                             "petal_length": [1.4, 4.5, 5.4], "petal_width": [0.2, 1.5, 2.1]})
    st.download_button("Download a sample CSV template", template.to_csv(index=False), "iris_batch_template.csv",
                       "text/csv", help="A small valid example you can edit.")

    names = list(pipelines)
    model_name = st.selectbox("Model", names, index=names.index(best) if best else 0, key="batch_model")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], key="batch_file",
                                help=f"Up to {config.MAX_UPLOAD_MB} MB and {config.MAX_BATCH_ROWS:,} rows.")
    if uploaded is None:
        st.info("Upload a CSV file to begin.")
        return

    try:
        df_raw = read_csv_bytes(uploaded.getvalue())
    except ValueError as error:
        st.error(str(error))
        return

    validation = validate_batch(df_raw)
    if validation.fatal_errors:
        for message in validation.fatal_errors:
            st.error(message)
        st.caption("Columns found in your file: " + ", ".join(map(str, df_raw.columns)))
        return

    n_valid, n_total = int(validation.valid_mask.sum()), len(df_raw)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in file", n_total)
    c2.metric("Valid rows predicted", n_valid)
    c3.metric("Rows skipped", n_total - n_valid)
    if n_valid == 0:
        st.error("None of the rows contain valid measurements, so no predictions were made.")
    elif n_valid < n_total:
        st.warning(f"{n_total - n_valid} row(s) were skipped because of invalid values. They are listed in the "
                   "'status' column of the results.")
    else:
        st.success("All rows are valid. Predictions are ready.")

    try:
        result, pred_col = build_batch_output(df_raw, validation, pipelines[model_name])
    except Exception as error:
        st.error(f"The predictions failed: {error}")
        return

    st.subheader("Predictions")
    st.dataframe(result, width="stretch")
    if n_valid:
        counts = (result[pred_col].replace("", np.nan).dropna().str.title().value_counts()
                  .rename_axis("Species").reset_index(name="Flowers"))
        st.write("**Predicted species counts**")
        st.dataframe(counts, hide_index=True)
    st.download_button("Download results as CSV", sanitize_for_csv(result).to_csv(index=False),
                       "iris_predictions.csv", "text/csv", type="primary",
                       help="The original columns plus predicted species, confidence and a status column.")


# ==========================================================================
# F. Model Insights
# ==========================================================================
LIMITATIONS = """
**Limits of a two-dimensional picture**
- The real models use **four** measurements. A flat plot can only show **two** at a time, so it can never show the full
  decision boundary of the four-measurement model.
- The regions drawn here come from a **separate model trained on only those two inputs**. Its accuracy can differ from
  the main models (it is printed above the chart).
- With **PCA**, the two axes are mixtures of all four measurements. They keep as much variation as possible, but they
  are not real measurements, and some information is lost (see the explained-variance numbers).
- Points that look close together in 2-D may be far apart in the hidden dimensions, and the reverse.
- Decision-region colours show where the model *would* predict each species, even in places where no real flower exists.
"""


def page_insights():
    pipelines, meta = require_models()
    df, (X_train, X_test, y_train, y_test) = get_data()
    section("Model Insights", "Understand how the models make decisions and where they make mistakes.")
    tab_imp, tab_tree, tab_bound, tab_eval = st.tabs(
        ["Feature importance", "Decision Tree", "Decision boundary", "Evaluation and errors"])

    tree_pipe = pipelines["Decision Tree"]
    with tab_imp:
        importance = pd.Series(tree_pipe.named_steps["model"].feature_importances_, index=config.FEATURE_NAMES)
        st.plotly_chart(plots.importance_fig(importance), width="stretch")
        top = importance.idxmax()
        unused = [config.FEATURE_LABELS[f].replace(" (cm)", "").lower() for f in importance.index if importance[f] == 0]
        text = (f"The tree relies most on **{config.FEATURE_LABELS[top].replace(' (cm)', '').lower()}** "
                f"({importance[top]:.1%} of the total importance).")
        if unused:
            text += " It did not use: " + ", ".join(unused) + "."
        st.write(text)
        st.caption("Importance shows how much each feature reduced uncertainty inside this tree. It applies only to the "
                   "Decision Tree, and a feature with 0 importance may still be informative (it can be correlated with "
                   "a feature the tree chose instead).")

    with tab_tree:
        st.pyplot(plots.tree_figure(tree_pipe), clear_figure=True)
        st.caption("Each box shows the question asked, the number of training flowers, and the majority species. "
                   "Numbers in the questions are in *scaled* units because the tree was trained after StandardScaler.")
        st.write("**The same rules in centimetres**")
        st.code(tree_rules_in_cm(tree_pipe), language="text")

    with tab_bound:
        c1, c2, c3 = st.columns(3)
        model_name = c1.selectbox("Model", list(pipelines), key="boundary_model")
        mode_label = c2.radio("View", ["Two features", "PCA (2 components)"], key="boundary_mode",
                              help="Two features: pick two measurements. PCA: squeeze all four into two combined axes.")
        mode = "pca" if mode_label.startswith("PCA") else "features"
        features = ["petal_length", "petal_width"]
        if mode == "features":
            with c3:
                f1 = st.selectbox("Horizontal axis", config.FEATURE_NAMES, index=2, key="boundary_f1",
                                  format_func=lambda f: config.FEATURE_LABELS[f])
                f2 = st.selectbox("Vertical axis", config.FEATURE_NAMES, index=3, key="boundary_f2",
                                  format_func=lambda f: config.FEATURE_LABELS[f])
            features = [f1, f2]
        if mode == "features" and features[0] == features[1]:
            st.error("Please choose two different features.")
        else:
            try:
                prep, clf = fit_boundary_model(model_name, X_train, y_train, mode, features)
                Z_test = prep.transform(X_test[config.FEATURE_NAMES] if mode == "pca" else X_test[features])
                acc = float((clf.predict(Z_test) == y_test.to_numpy()).mean())
                st.plotly_chart(plots.boundary_fig(prep, clf, X_train, y_train, X_test, y_test, mode, features,
                                                   model_name, acc), width="stretch")
                full_acc = float((pipelines[model_name].predict(X_test) == y_test.to_numpy()).mean())
                st.caption(f"Circles = training flowers, squares = test flowers. This 2-input model: test accuracy "
                           f"{acc:.3f}. The main four-feature {model_name} model: {full_acc:.3f}.")
                if mode == "pca":
                    ratio = prep.named_steps["pca"].explained_variance_ratio_
                    st.info(f"PC1 keeps {ratio[0]:.1%} and PC2 keeps {ratio[1]:.1%} of the variation in the (scaled) "
                            f"training data: {ratio.sum():.1%} in total.")
                st.markdown(LIMITATIONS)
            except Exception as error:
                st.error(f"The decision boundary could not be drawn: {error}")

    with tab_eval:
        metrics, best, reason = evaluate(pipelines)
        st.write(reason)
        st.dataframe(metrics.style.format("{:.4f}"), width="stretch")
        pick = st.selectbox("Model for the per-species report", list(pipelines), key="insight_report_model")
        st.dataframe(report_table(pipelines[pick], X_test, y_test), width="stretch")
        st.subheader("Misclassified test flowers")
        for name, pipe in pipelines.items():
            wrong = misclassified_samples(pipe, X_test, y_test)
            st.write(f"**{name}**: {len(wrong)} of {len(X_test)} test flowers misclassified")
            if len(wrong):
                st.dataframe(wrong, width="stretch", hide_index=True)
            else:
                st.success("No mistakes on the test set.")
        st.caption("Mistakes almost always happen between Versicolor and Virginica, where the measurements overlap.")


# ==========================================================================
# Main
# ==========================================================================
def main():
    st.session_state.setdefault("page", PAGES[0])
    page = sidebar()
    {
        "Home Dashboard": page_home,
        "Dataset Explorer": page_dataset,
        "Model Comparison": page_comparison,
        "Flower Prediction": page_prediction,
        "Batch Prediction": page_batch,
        "Model Insights": page_insights,
    }[page]()


main()

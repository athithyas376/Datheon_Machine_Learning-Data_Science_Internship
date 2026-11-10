"""Chart builders for the dashboard. Each function returns a figure; app.py only has to display it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.tree import plot_tree

from . import config

COLOR_MAP = config.CLASS_COLORS
CLASS_ORDER = {"species": config.CLASS_NAMES}
MODEL_COLORS = {"Logistic Regression": "#4a6fa5", "Decision Tree": "#e8833a"}


def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(size=13), legend_title_text="")
    return fig


def class_distribution_fig(df: pd.DataFrame) -> go.Figure:
    counts = df["species"].value_counts().reindex(config.CLASS_NAMES).reset_index()
    counts.columns = ["species", "count"]
    fig = px.bar(counts, x="species", y="count", color="species", color_discrete_map=COLOR_MAP,
                 text="count", title="Samples per species")
    fig.update_layout(showlegend=False, xaxis_title="Species", yaxis_title="Number of flowers")
    return _style(fig, 320)


def feature_distribution_fig(df: pd.DataFrame, feature: str, kind: str) -> go.Figure:
    label = config.FEATURE_LABELS[feature]
    if kind == "Box plot":
        fig = px.box(df, x="species", y=feature, color="species", color_discrete_map=COLOR_MAP,
                     category_orders=CLASS_ORDER, points="all", labels={feature: label, "species": "Species"})
    else:
        fig = px.histogram(df, x=feature, color="species", color_discrete_map=COLOR_MAP, barmode="overlay",
                           opacity=0.65, nbins=20, category_orders=CLASS_ORDER,
                           labels={feature: label, "count": "Count"})
        fig.update_layout(yaxis_title="Number of flowers")
    fig.update_layout(title=f"Distribution of {label}", showlegend=(kind != "Box plot"))
    return _style(fig)


def correlation_fig(df: pd.DataFrame) -> go.Figure:
    corr = df[config.FEATURE_NAMES].corr().round(2)
    labels = [config.FEATURE_LABELS[f].replace(" (cm)", "") for f in config.FEATURE_NAMES]
    fig = px.imshow(corr.values, x=labels, y=labels, text_auto=True, zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", title="Correlation between features")
    fig.update_layout(coloraxis_colorbar_title="r")
    return _style(fig, 420)


def pairplot_fig(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter_matrix(df, dimensions=config.FEATURE_NAMES, color="species",
                            color_discrete_map=COLOR_MAP, category_orders=CLASS_ORDER,
                            labels={f: f.replace("_", " ") for f in config.FEATURE_NAMES},
                            title="Pairplot (scatter matrix) of all features")
    fig.update_traces(diagonal_visible=False, marker=dict(size=5, opacity=0.8))
    return _style(fig, 620)


def metrics_bar_fig(metrics: pd.DataFrame) -> go.Figure:
    long = metrics.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", text_auto=".3f",
                 color_discrete_map=MODEL_COLORS, title="Model comparison on the held-out test set")
    fig.update_yaxes(range=[0.0, 1.08])
    fig.update_traces(textposition="outside")
    return _style(fig, 400)


def confusion_fig(cm: np.ndarray, model_name: str) -> go.Figure:
    labels = [c.title() for c in config.CLASS_NAMES]
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted species", y="True species", color="Flowers"),
                    title=f"Confusion matrix: {model_name}")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(side="bottom")
    return _style(fig, 380)


def importance_fig(importances: pd.Series) -> go.Figure:
    data = importances.sort_values().reset_index()
    data.columns = ["feature", "importance"]
    data["feature"] = data["feature"].map(lambda f: config.FEATURE_LABELS[f])
    fig = px.bar(data, x="importance", y="feature", orientation="h", text_auto=".3f",
                 title="Decision Tree feature importance (sums to 1)")
    fig.update_traces(marker_color="#e8833a")
    fig.update_layout(xaxis_title="Importance", yaxis_title="")
    return _style(fig, 320)


def tree_figure(pipe) -> Figure:
    """The trained tree drawn with matplotlib. Thresholds shown on the boxes are in scaled units
    (the tree was trained on standardized features); the rules in centimetres are shown as text."""
    fig = Figure(figsize=(14, 6.5))
    FigureCanvasAgg(fig)          # a rendering canvas is needed so scikit-learn can measure the text boxes
    ax = fig.subplots()
    plot_tree(pipe.named_steps["model"], feature_names=config.FEATURE_NAMES, class_names=config.CLASS_NAMES,
              filled=True, rounded=True, fontsize=9, ax=ax)
    return fig


def probability_fig(proba: np.ndarray, model_name: str) -> go.Figure:
    data = pd.DataFrame({"species": [c.title() for c in config.CLASS_NAMES], "probability": proba})
    fig = px.bar(data, x="probability", y="species", orientation="h", color="species", text=data["probability"].map("{:.1%}".format),
                 color_discrete_map={c.title(): COLOR_MAP[c] for c in config.CLASS_NAMES},
                 title=f"Class probabilities ({model_name})")
    fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1.05], tickformat=".0%", title="Probability"),
                      yaxis_title="")
    return _style(fig, 260)


def position_fig(df: pd.DataFrame, sample: dict) -> go.Figure:
    """Where the new flower sits among the dataset flowers (petal length vs petal width)."""
    fig = px.scatter(df, x="petal_length", y="petal_width", color="species", color_discrete_map=COLOR_MAP,
                     category_orders=CLASS_ORDER, opacity=0.7,
                     labels={"petal_length": "Petal length (cm)", "petal_width": "Petal width (cm)"},
                     title="Your flower (star) compared with the dataset")
    fig.add_trace(go.Scatter(x=[sample["petal_length"]], y=[sample["petal_width"]], mode="markers",
                             marker=dict(symbol="star", size=20, color="black", line=dict(color="white", width=1)),
                             name="Your flower"))
    return _style(fig, 360)


def boundary_fig(prep, classifier, X_train, y_train, X_test, y_test, mode: str, features: list[str],
                 model_name: str, accuracy: float) -> go.Figure:
    """Decision regions of a 2-input model, with training points (circles) and test points (squares)."""
    def to_plane(X: pd.DataFrame) -> np.ndarray:
        if mode == "pca":
            return prep.transform(X[config.FEATURE_NAMES])
        return X[features].to_numpy()

    P_train, P_test = to_plane(X_train), to_plane(X_test)
    allp = np.vstack([P_train, P_test])
    pad = 0.08 * (allp.max(axis=0) - allp.min(axis=0)) + 0.2
    lo, hi = allp.min(axis=0) - pad, allp.max(axis=0) + pad
    xs, ys = np.linspace(lo[0], hi[0], 260), np.linspace(lo[1], hi[1], 260)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]

    if mode == "pca":
        Z = classifier.predict(grid)
    else:
        Z = classifier.predict(prep.transform(pd.DataFrame(grid, columns=features)))
    Z = Z.reshape(xx.shape)

    c = [COLOR_MAP[n] for n in config.CLASS_NAMES]
    colorscale = [[0, c[0]], [1 / 3, c[0]], [1 / 3, c[1]], [2 / 3, c[1]], [2 / 3, c[2]], [1, c[2]]]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=Z, zmin=0, zmax=2, colorscale=colorscale, showscale=False,
                             opacity=0.35, hoverinfo="skip", name="Predicted region"))
    for k, name in enumerate(config.CLASS_NAMES):
        for P, y, symbol, label in [(P_train, y_train, "circle", "train"), (P_test, y_test, "square", "test")]:
            mask = (np.asarray(y) == k)
            fig.add_trace(go.Scatter(
                x=P[mask, 0], y=P[mask, 1], mode="markers", name=f"{name.title()} ({label})",
                marker=dict(color=COLOR_MAP[name], symbol=symbol, size=9 if label == "train" else 11,
                            line=dict(color="black", width=1))))
    if mode == "pca":
        xt, yt = "Principal component 1", "Principal component 2"
    else:
        xt, yt = config.FEATURE_LABELS[features[0]], config.FEATURE_LABELS[features[1]]
    fig.update_layout(title=f"{model_name}: decision regions (test accuracy of this 2-input model = {accuracy:.3f})",
                      xaxis_title=xt, yaxis_title=yt, xaxis_range=[lo[0], hi[0]], yaxis_range=[lo[1], hi[1]])
    return _style(fig, 520)

"""Static (matplotlib / seaborn) charts for the notebook, ``main.py`` and report exports.

The Streamlit dashboard draws interactive Plotly versions of the same data; these
functions produce publication-style PNGs that can be saved and downloaded.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import config as C

MC = C.MODEL_COLORS
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.edgecolor": "#C9D1DA",
        "axes.labelcolor": C.PALETTE["ink"],
        "grid.color": "#E6EBF0",
        "savefig.bbox": "tight",
    }
)


def _finish(fig, title: Optional[str] = None):
    if title:
        fig.suptitle(title, fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ data charts
def plot_correlation_heatmap(corr: pd.DataFrame):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(6, 0.75 * n), max(5, 0.65 * n)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=n <= 14, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Pearson r"}, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation matrix")
    return _finish(fig)


def plot_missing_values(report: pd.DataFrame):
    rep = report[report.missing > 0]
    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.5 * max(len(rep), 1) + 1)))
    if rep.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=13, color=C.PALETTE["teal"])
        ax.set_axis_off()
    else:
        ax.barh(rep.column[::-1], rep.missing_pct[::-1], color=C.PALETTE["gold"])
        for i, (v, n) in enumerate(zip(rep.missing_pct[::-1], rep.missing[::-1])):
            ax.text(v, i, f"  {n:,} ({v:.2f}%)", va="center", fontsize=9)
        ax.set_xlabel("% missing")
        ax.set_xlim(0, max(rep.missing_pct) * 1.4)
    ax.set_title("Missing values by column")
    return _finish(fig)


def plot_target_distribution(y: pd.Series, target: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(y, bins=50, color=C.PALETTE["blue"], edgecolor="white", ax=ax)
    ax.set_xlabel(target)
    ax.set_title(f"Distribution of {target}")
    return _finish(fig)


def plot_feature_distributions(df: pd.DataFrame, columns, ncols: int = 3):
    columns = list(columns)
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows))
    for ax, col in zip(np.atleast_1d(axes).ravel(), columns):
        sns.histplot(df[col].dropna(), bins=40, color=C.PALETTE["teal"], edgecolor="white", ax=ax)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    for ax in np.atleast_1d(axes).ravel()[len(columns):]:
        ax.set_axis_off()
    return _finish(fig, "Feature distributions")


def plot_vif(vif: pd.DataFrame, title: str = "Variance Inflation Factors"):
    d = vif.sort_values("VIF")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.38 * len(d) + 1)))
    colors = np.where(d.VIF >= C.VIF_THRESHOLD, C.PALETTE["red"], np.where(d.VIF >= C.VIF_MODERATE, C.PALETTE["gold"], C.PALETTE["teal"]))
    ax.barh(d.feature, d.VIF, color=colors)
    ax.axvline(C.VIF_MODERATE, color=C.PALETTE["gold"], ls="--", lw=1)
    ax.axvline(C.VIF_THRESHOLD, color=C.PALETTE["red"], ls="--", lw=1)
    ax.set_xlabel("VIF  (dashed: 5 moderate, 10 severe)")
    ax.set_title(title)
    return _finish(fig)


def plot_feature_set_comparison(exp: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    x = np.arange(len(exp))
    w = 0.26
    for ax, kind, ttl in zip(axes, ["cv_rmse", "test_rmse"], ["Cross-validated RMSE (train set)", "Test RMSE"]):
        for k, m in enumerate(C.MODEL_NAMES):
            ax.bar(x + (k - 1) * w, exp[f"{m}_{kind}"], w, label=m, color=MC[m])
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" (", "\n(") for s in exp.feature_set], fontsize=8)
        ax.set_title(ttl)
        ax.set_ylabel("RMSE (y-axis truncated)")
        lo = float(exp[[f"{m}_{kind}" for m in C.MODEL_NAMES]].min().min())
        hi = float(exp[[f"{m}_{kind}" for m in C.MODEL_NAMES]].max().max())
        ax.set_ylim(lo * 0.9, hi * 1.04)
    axes[0].legend(title="Model")
    return _finish(fig, "Before vs after feature engineering")


# ------------------------------------------------------------------ model charts
def plot_gridsearch(grids: Dict[str, pd.DataFrame], best_params: Dict[str, Dict[str, float]]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name in zip(axes, ["Ridge", "Lasso"]):
        g = grids[name]
        ax.plot(g.alpha, g.cv_rmse, "o-", color=MC[name], label="CV RMSE (validation)")
        ax.fill_between(g.alpha, g.cv_rmse - g.cv_rmse_std, g.cv_rmse + g.cv_rmse_std, color=MC[name], alpha=0.15)
        ax.plot(g.alpha, g.train_rmse, "s--", color=C.PALETTE["slate"], alpha=0.7, label="Train RMSE")
        a = best_params[name]["alpha"]
        ax.axvline(a, color=C.PALETTE["red"], ls=":", label=f"best alpha = {a:.3g}")
        ax.set_xscale("log")
        ax.set_xlabel("alpha (log scale)")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{name} – GridSearchCV")
        ax.legend()
    return _finish(fig)


def plot_actual_vs_predicted(preds: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    lim = [min(preds.actual.min(), preds[C.MODEL_NAMES].min().min()), max(preds.actual.max(), preds[C.MODEL_NAMES].max().max())]
    for ax, m in zip(axes, C.MODEL_NAMES):
        ax.scatter(preds.actual, preds[m], s=6, alpha=0.35, color=MC[m], edgecolors="none")
        ax.plot(lim, lim, "--", color=C.PALETTE["red"], lw=1.2, label="perfect prediction")
        ax.set_title(m)
        ax.set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].legend(loc="upper left")
    return _finish(fig, "Actual vs predicted (test set)")


def plot_residual_hist(preds: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, m in zip(axes, C.MODEL_NAMES):
        r = preds.actual - preds[m]
        sns.histplot(r, bins=60, stat="density", color=MC[m], edgecolor="white", ax=ax)
        xs = np.linspace(r.min(), r.max(), 200)
        ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), color=C.PALETTE["ink"], lw=1.2, label="normal fit")
        ax.axvline(0, color=C.PALETTE["red"], ls="--", lw=1)
        ax.set_title(f"{m}  (mean {r.mean():+.3f}, sd {r.std():.3f})")
        ax.set_xlabel("Residual (actual − predicted)")
    axes[0].legend()
    return _finish(fig, "Residual distribution")


def plot_residual_vs_predicted(preds: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    for ax, m in zip(axes, C.MODEL_NAMES):
        r = preds.actual - preds[m]
        ax.scatter(preds[m], r, s=6, alpha=0.3, color=MC[m], edgecolors="none")
        ax.axhline(0, color=C.PALETTE["red"], ls="--", lw=1.2)
        # binned mean residual line reveals systematic structure
        bins = pd.qcut(preds[m], 15, duplicates="drop")
        line = pd.DataFrame({"p": preds[m], "r": r}).groupby(bins, observed=True).mean()
        ax.plot(line.p, line.r, color=C.PALETTE["ink"], lw=2, label="binned mean residual")
        ax.set_title(m)
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].legend()
    return _finish(fig, "Residuals vs predicted")


def plot_prediction_comparison(preds: pd.DataFrame, n: int = 60):
    d = preds.iloc[:n].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(d.index, d.actual, "k-o", ms=4, lw=1.4, label="Actual")
    for m in C.MODEL_NAMES:
        ax.plot(d.index, d[m], "--", lw=1.2, marker="x", ms=4, color=MC[m], label=m)
    ax.set_xlabel(f"Test sample (first {n})")
    ax.set_ylabel("Target value")
    ax.legend(ncol=4)
    ax.set_title("Regression prediction comparison")
    return _finish(fig)


def plot_model_performance(metrics: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, ttl) in zip(axes, [("RMSE", "RMSE (lower is better)"), ("MAE", "MAE (lower is better)"), ("R²", "R² (higher is better)")]):
        tr, te = metrics[f"Train {col}"].values, metrics[f"Test {col}"].values
        x = np.arange(len(metrics))
        ax.bar(x - 0.2, tr, 0.38, label="Train", color="#B8C4D2")
        bars = ax.bar(x + 0.2, te, 0.38, label="Test", color=[MC[m] for m in metrics.Model])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics.Model)
        ax.set_title(ttl + "\n(y-axis truncated)", fontsize=10)
        lo = min(tr.min(), te.min())
        ax.set_ylim(lo * 0.85 if col != "R²" else lo - 0.05, max(tr.max(), te.max()) * 1.06)
    axes[0].legend()
    return _finish(fig, "Model performance comparison")


def plot_error_distribution(preds: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    long = pd.concat([pd.DataFrame({"Model": m, "Absolute error": (preds.actual - preds[m]).abs()}) for m in C.MODEL_NAMES])
    sns.boxplot(data=long, x="Model", y="Absolute error", hue="Model", palette=MC, legend=False, ax=axes[0], fliersize=2)
    axes[0].set_title("Absolute error by model")
    for m in C.MODEL_NAMES:
        sns.kdeplot((preds.actual - preds[m]).abs(), ax=axes[1], color=MC[m], label=m, clip=(0, None), lw=2)
    axes[1].set_xlabel("Absolute error")
    axes[1].set_title("Density of absolute errors")
    axes[1].legend()
    return _finish(fig, "Error distribution analysis")


def plot_qq(preds: pd.DataFrame, model: str):
    r = (preds.actual - preds[model]).to_numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(r, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(MC[model])
    ax.get_lines()[0].set_markersize(3)
    ax.set_title(f"Normal Q-Q plot of residuals – {model}")
    return _finish(fig)


def plot_coefficients(coefs: pd.DataFrame, model: str, top: int = 20):
    d = coefs.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(d) + 1)))
    ax.barh(d.feature, d.coefficient, color=[MC[model] if v >= 0 else C.PALETTE["red"] for v in d.coefficient])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{model}: largest standardised coefficients")
    return _finish(fig)


# ------------------------------------------------------------------ export helpers
def figure_to_png_bytes(fig, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def build_all_figures(result: Dict[str, Any]) -> Dict[str, "plt.Figure"]:
    """Create every static chart for a training result (name -> Figure)."""
    d, p = result["diagnostics"], result["predictions"]
    figs: Dict[str, plt.Figure] = {
        "01_correlation_heatmap": plot_correlation_heatmap(d["correlation"]),
        "02_missing_values": plot_missing_values(d["missing_report"]),
        "03_target_distribution": plot_target_distribution(result["clean_data"][result["meta"]["target"]], result["meta"]["target"]),
        "04_vif_original_features": plot_vif(d["vif_original"], "VIF – original numeric features"),
        "05_feature_set_comparison": plot_feature_set_comparison(result["experiments"]),
        "06_gridsearch_alpha_curves": plot_gridsearch(result["grids"], result["best_params"]),
        "07_actual_vs_predicted": plot_actual_vs_predicted(p),
        "08_residual_distribution": plot_residual_hist(p),
        "09_residual_vs_predicted": plot_residual_vs_predicted(p),
        "10_prediction_comparison": plot_prediction_comparison(p),
        "11_model_performance": plot_model_performance(result["metrics"]),
        "12_error_distribution": plot_error_distribution(p),
        "13_qq_plot_best_model": plot_qq(p, result["best_model"]),
        "14_coefficients_best_model": plot_coefficients(result["coefficients"][result["best_model"]], result["best_model"]),
    }
    if len(d["vif_final_design"]) > 1:
        figs["15_vif_final_design"] = plot_vif(d["vif_final_design"], "VIF – final design matrix")
    return figs


def save_all_charts(result: Dict[str, Any], out_dir=C.CHARTS_DIR, dpi: int = 130) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fig in build_all_figures(result).items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths[name] = str(path)
    return paths

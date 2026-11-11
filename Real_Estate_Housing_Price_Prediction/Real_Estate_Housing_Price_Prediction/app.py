"""Streamlit dashboard – Real Estate Housing Price Prediction (OLS vs Ridge vs Lasso).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import io
import sys
import time
import warnings
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sps
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import model_training as mt  # noqa: E402
from src import prediction as pr  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src import visualization as viz  # noqa: E402

st.set_page_config(
    page_title="Housing Price Studio",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY, GOLD, TEAL, BLUE, RED, SLATE = (C.PALETTE[k] for k in ("navy", "gold", "teal", "blue", "red", "slate"))
MODEL_COLORS = C.MODEL_COLORS
SRC_CAL = "California Housing (bundled)"
SRC_UP = "Upload a CSV"

PAGES = [
    ("🏠", "Home / Overview"),
    ("🔎", "Data Exploration"),
    ("🧹", "Data Preprocessing"),
    ("⚙️", "Model Training"),
    ("📊", "Model Performance"),
    ("📉", "Residual Analysis"),
    ("🏡", "House Price Prediction"),
    ("📥", "Reports"),
]

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; color: {NAVY}; }}
h1 {{ font-family: 'Playfair Display', Georgia, serif; }}
.hero {{ background: linear-gradient(120deg, {NAVY} 0%, #1d4468 55%, {TEAL} 130%); color: #fff; border-radius: 18px;
        padding: 2rem 2.2rem; margin-bottom: 1.2rem; box-shadow: 0 10px 30px rgba(18,38,63,.18); }}
.hero h1 {{ color: #fff; margin: 0 0 .35rem 0; font-size: 2.1rem; }}
.hero p {{ color: #d9e4ef; margin: 0; font-size: 1.02rem; max-width: 62rem; line-height: 1.55; }}
.hero .badge {{ display:inline-block; background: rgba(201,154,59,.18); color:#f3d9a0; border:1px solid rgba(201,154,59,.5);
        padding: 2px 10px; border-radius: 999px; font-size:.75rem; font-weight:600; letter-spacing:.05em; margin-bottom:.7rem; }}
.page-title {{ margin-bottom: .1rem; }}
.page-sub {{ color: {SLATE}; margin-bottom: 1.1rem; }}
[data-testid="stMetric"] {{ background:#fff; border:1px solid #E1E8F0; border-radius:14px; padding: .85rem 1rem;
        box-shadow: 0 1px 2px rgba(18,38,63,.04); }}
[data-testid="stMetricLabel"] p {{ font-size:.8rem; color:{SLATE}; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }}
[data-testid="stMetricValue"] {{ color:{NAVY}; font-weight:700; }}
.card {{ background:#fff; border:1px solid #E1E8F0; border-radius:14px; padding:1rem 1.2rem; height:100%;
        box-shadow: 0 1px 2px rgba(18,38,63,.04); }}
.card h4 {{ margin:.1rem 0 .3rem 0; color:{NAVY}; font-size:.98rem; }}
.card p {{ margin:0; color:#41546b; font-size:.88rem; line-height:1.45; }}
.step-no {{ display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px; border-radius:50%;
        background:{GOLD}; color:#fff; font-weight:700; font-size:.8rem; margin-right:.5rem; }}
.callout {{ border-left:5px solid {TEAL}; background:#EAF6F5; border-radius:10px; padding:.9rem 1.1rem; color:#133b39; margin:.6rem 0 1rem 0; }}
.callout.warn {{ border-color:{GOLD}; background:#FBF3E2; color:#5c4513; }}
.callout.bad {{ border-color:{RED}; background:#FBECEA; color:#6b221b; }}
.result-card {{ background: linear-gradient(135deg, {NAVY}, #214a73); border-radius:18px; padding:1.6rem 1.8rem; color:#fff;
        box-shadow: 0 10px 28px rgba(18,38,63,.22); }}
.result-card .lbl {{ text-transform:uppercase; letter-spacing:.08em; font-size:.75rem; color:#f3d9a0; font-weight:600; }}
.result-card .val {{ font-family:'Playfair Display', Georgia, serif; font-size:3rem; font-weight:700; line-height:1.1; margin:.2rem 0; }}
.result-card .sub {{ color:#cfdbe8; font-size:.9rem; }}
.brand {{ font-family:'Playfair Display', Georgia, serif; font-size:1.35rem; font-weight:700; color:{NAVY}; }}
.brand small {{ display:block; font-family:'Inter',sans-serif; font-size:.72rem; color:{SLATE}; font-weight:500; letter-spacing:.05em; text-transform:uppercase; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 8px 8px 0 0; padding: .45rem 1rem; font-weight:600; }}
div.stButton > button, div.stDownloadButton > button, .stFormSubmitButton > button {{ border-radius: 10px; font-weight:600; }}
@media (max-width: 760px) {{
  .hero {{ padding: 1.3rem 1.2rem; }} .hero h1 {{ font-size: 1.5rem; }} .result-card .val {{ font-size: 2.2rem; }}
}}
</style>
"""


def style_fig(fig: go.Figure, height: int = 400, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=NAVY),
        title=dict(font=dict(size=16)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(gridcolor="#E6EBF0", zerolinecolor="#C9D1DA")
    fig.update_yaxes(gridcolor="#E6EBF0", zerolinecolor="#C9D1DA")
    return fig


def show(fig: go.Figure, height: int = 400, legend: bool = True) -> None:
    st.plotly_chart(style_fig(fig, height, legend))


def callout(text: str, kind: str = "info") -> None:
    cls = {"info": "callout", "warn": "callout warn", "bad": "callout bad"}[kind]
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(f"<h1 class='page-title'>{icon} {title}</h1><div class='page-sub'>{subtitle}</div>", unsafe_allow_html=True)


def safe_slider(label: str, lo: int, hi: int, default: int, key: str, **kw) -> int:
    """st.slider that degrades gracefully when the range collapses (tiny datasets)."""
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        return hi
    return st.slider(label, lo, hi, int(min(max(default, lo), hi)), key=key, **kw)


def csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8")


def fmt_target(value: float, res: dict) -> str:
    return pr.format_value(value, res["meta"]["unit"])


# --------------------------------------------------------------------------- #
# Session state & data activation
# --------------------------------------------------------------------------- #
def init_state() -> None:
    ss = st.session_state
    defaults = {
        "df": None, "dataset_name": None, "target": None, "features": None,
        "is_california": True, "result": None, "active_source": None,
        "prediction_log": [], "last_prediction": None, "load_error": None,
        "bundle_note": None, "charts_cache": {}, "booted": False, "upload_key": None,
        "processed_cache": None,
    }
    for k, v in defaults.items():
        ss.setdefault(k, v)


def reset_dataset_state() -> None:
    ss = st.session_state
    ss.result = None
    ss.prediction_log = []
    ss.last_prediction = None
    ss.charts_cache = {}
    ss.processed_cache = None
    for key in [k for k in ss.keys() if str(k).startswith("inp_")]:
        del ss[key]


def activate_california() -> None:
    ss = st.session_state
    try:
        df = dl.load_california_housing()
    except dl.DataLoadError as exc:
        ss.load_error = str(exc)
        return
    ss.load_error = None
    reset_dataset_state()
    ss.df, ss.dataset_name, ss.target = df, "California Housing", C.CALIFORNIA_TARGET
    ss.features = [c for c in df.columns if c != ss.target]
    ss.is_california = True
    ss.active_source = SRC_CAL
    try:  # reuse models trained by main.py when they are compatible
        bundle = pr.load_bundle()
        if bundle["meta"]["target"] == C.CALIFORNIA_TARGET and bundle["meta"]["dataset_name"] == "California Housing":
            ss.result = bundle
            ss.bundle_note = "Loaded pre-trained models from the `models/` folder (created by `python main.py`)."
        else:
            ss.bundle_note = "Saved models belong to a different dataset and were ignored."
    except FileNotFoundError:
        ss.bundle_note = "No pre-trained models found – they will be trained now."
    except Exception as exc:  # version mismatch or unreadable pickle
        ss.bundle_note = f"Saved models could not be used ({exc}) – retraining."


def run_training(cv: int, save: bool) -> None:
    ss = st.session_state
    bar = st.progress(0.0, text="Starting…")

    def cb(frac: float, msg: str) -> None:
        bar.progress(min(frac, 1.0), text=msg)

    with st.spinner("Training OLS, Ridge and Lasso with GridSearchCV…"):
        try:
            res = mt.train_project(
                ss.df, ss.target, ss.features, dataset_name=ss.dataset_name, cv=cv, progress=cb,
            )
        except Exception as exc:  # noqa: BLE001 - shown to the user
            bar.empty()
            st.error(f"Training failed: {exc}")
            return
    bar.empty()
    ss.result = res
    ss.last_prediction = None
    ss.charts_cache = {}
    ss.processed_cache = None
    for key in [k for k in ss.keys() if str(k).startswith("inp_")]:
        del ss[key]
    if save and ss.is_california:
        try:
            mt.save_artifacts(res)
            st.success("Models and metrics were also saved to the project's `models/` and `outputs/` folders.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Models trained but could not be saved to disk: {exc}")
    st.success(f"Training finished in {res['meta']['train_seconds']} s. Best model: **{res['best_model']}**.")


def bootstrap() -> None:
    """First run: load California Housing and, if needed, train once."""
    ss = st.session_state
    if ss.booted:
        return
    activate_california()
    if ss.df is not None and ss.result is None:
        run_training(C.CV_FOLDS, save=True)
    ss.booted = True


def get_result():
    return st.session_state.result


def need_result(page: str):
    res = get_result()
    if res is None:
        callout(
            f"<b>No trained models yet.</b> Open <b>⚙️ Model Training</b> and press <i>Train models</i> – "
            f"the {page} page will fill in automatically.",
            "warn",
        )
    return res


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar_dataset_panel() -> None:
    ss = st.session_state
    choice = st.sidebar.radio(
        "Data source", [SRC_CAL, SRC_UP], key="source_choice",
        help="Use the bundled California Housing data or upload any compatible regression CSV.",
    )
    if choice == SRC_CAL:
        if ss.active_source == SRC_UP:  # user switched back from a custom dataset
            with st.spinner("Loading California Housing…"):
                activate_california()
                if ss.df is not None and ss.result is None:
                    run_training(C.CV_FOLDS, save=True)
            st.rerun()
        return

    up = st.sidebar.file_uploader("CSV file", type=["csv", "txt"], key="uploader",
                                  help="Any CSV with a numeric target column. Missing values are imputed automatically.")
    if up is None:
        st.sidebar.info("Upload a CSV to configure a custom dataset.")
        return
    try:
        df_up = dl.read_csv_source(up)
    except dl.DataLoadError as exc:
        st.sidebar.error(f"Upload failed: {exc}")
        return

    info = dl.inspect_dataset(df_up)
    st.sidebar.caption(f"{info['n_rows']:,} rows × {info['n_cols']} columns · {info['total_missing']:,} missing · {info['n_duplicates']:,} duplicates")
    numeric_cols = info["numeric_cols"]
    if not numeric_cols:
        st.sidebar.error("No numeric columns found – a numeric target is required for regression.")
        return
    guess = dl.guess_target(df_up)
    target = st.sidebar.selectbox(
        "Target column", numeric_cols, index=numeric_cols.index(guess) if guess in numeric_cols else 0,
        key="up_target", help="The numeric value to predict.",
    )
    candidates = [c for c in df_up.columns if c != target]
    feats = st.sidebar.multiselect("Feature columns", candidates, default=candidates, key=f"up_feats_{target}",
                                   help="Choose which columns the models may use.")
    check = dl.validate_dataset(df_up, target, feats)
    for e in check["errors"]:
        st.sidebar.error(e)
    for w in check["warnings"]:
        st.sidebar.warning(w)
    if check["ok"]:
        st.sidebar.success(f"Valid: {len(check['numeric'])} numeric + {len(check['categorical'])} categorical features.")
    if st.sidebar.button("Use this dataset", disabled=not check["ok"], key="apply_upload"):
        reset_dataset_state()
        ss.df, ss.target = df_up, target
        ss.features = check["numeric"] + check["categorical"]
        ss.dataset_name = up.name
        ss.is_california = False
        ss.active_source = SRC_UP
        ss.load_error = None
        st.sidebar.success("Dataset applied. Go to **Model Training** to train.")
        st.rerun()


def render_sidebar() -> str:
    st.sidebar.markdown("<div class='brand'>🏡 Housing Price Studio<small>OLS · Ridge · Lasso</small></div>", unsafe_allow_html=True)
    st.sidebar.markdown("")
    labels = [f"{i} {n}" for i, n in PAGES]
    page_label = st.sidebar.radio("Navigation", labels, key="nav", label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset**")
    sidebar_dataset_panel()
    ss = st.session_state
    st.sidebar.markdown("---")
    st.sidebar.caption("**Active dataset**")
    if ss.df is not None:
        st.sidebar.write(f"{ss.dataset_name}  \nTarget: `{ss.target}`")
        st.sidebar.write("Models: " + ("✅ trained" if ss.result is not None else "⚠️ not trained"))
    else:
        st.sidebar.write("None loaded")
    return page_label.split(" ", 1)[1]


# --------------------------------------------------------------------------- #
# Page 1 – Home
# --------------------------------------------------------------------------- #
WORKFLOW = [
    ("Load & validate", "Auto-load California Housing or upload a CSV; schema, dtypes and target are checked."),
    ("Clean", "Detect duplicates, profile missing values, impute (median / most-frequent)."),
    ("Engineer features", "Interactions, ratios and polynomial terms – compared with the baseline."),
    ("Split 80 : 20", "Reproducible split (random_state=42) *before* any fitting – no data leakage."),
    ("Train 3 models", "OLS, Ridge (L2) and Lasso (L1); alpha tuned with GridSearchCV."),
    ("Evaluate", "RMSE, MAE and R² from real predictions on the held-out test set."),
    ("Residual analysis", "Distribution, residual-vs-predicted, error structure, limitations."),
    ("Predict & report", "Interactive prediction with the trained pipeline; CSV, chart and report downloads."),
]


def page_home() -> None:
    ss = st.session_state
    st.markdown(
        """
        <div class="hero">
          <div class="badge">MACHINE LEARNING · REAL ESTATE ANALYTICS</div>
          <h1>Real Estate Housing Price Prediction</h1>
          <p>Predict median house values with <b>Ordinary Least Squares</b>, <b>Ridge</b> and <b>Lasso</b> regression.
          Explore the data, inspect multicollinearity, compare feature sets, tune hyper-parameters with GridSearchCV,
          study the residuals and price a home interactively – all with the exact preprocessing pipeline used in training.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if ss.load_error:
        callout(f"<b>Dataset not available.</b> {ss.load_error}", "bad")
    if ss.df is None:
        return
    df, target = ss.df, ss.target
    used = [c for c in (ss.features or []) if c in df.columns] + [target]
    sub = df[used]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{len(df):,}", help="Total rows in the active dataset (before cleaning).")
    c2.metric("Features", f"{len(used) - 1}", help="Number of feature columns selected for modelling.")
    c3.metric("Missing values", f"{int(sub.isna().sum().sum()):,}", help="Empty cells across the selected features and target.")
    c4.metric("Target", target, help="The numeric variable being predicted.")
    c5.metric("Duplicates", f"{int(sub.duplicated().sum()):,}", help="Exact duplicate rows (removed before splitting).")

    res = get_result()
    if res is not None:
        m = res["metrics"].set_index("Model")
        best = res["best_model"]
        st.markdown("#### Latest results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Best model", best, help=res["best_model_explanation"].replace("**", ""))
        r2.metric("Test R²", f"{m.loc[best, 'Test R²']:.4f}", help="Share of target variance explained on unseen data.")
        r3.metric("Test RMSE", f"{m.loc[best, 'Test RMSE']:.4f}", help="Root mean squared error on the test set (target units).")
        r4.metric("Feature set", res["meta"]["feature_set"], help="Chosen by cross-validation on the training data.")
    elif ss.bundle_note:
        st.info(ss.bundle_note)

    st.markdown("#### Project workflow")
    cols = st.columns(4)
    for i, (title, desc) in enumerate(WORKFLOW):
        with cols[i % 4]:
            st.markdown(
                f'<div class="card"><h4><span class="step-no">{i + 1}</span>{title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

    st.markdown("#### Dataset overview")
    info = dl.inspect_dataset(sub)
    left, right = st.columns([3, 2])
    with left:
        cols_df = info["columns"].rename(columns={"missing_pct": "missing %", "non_null": "non-null"})
        st.dataframe(cols_df, hide_index=True)
    with right:
        st.markdown(f"**Dataset:** {ss.dataset_name}")
        st.markdown(f"**Numeric features:** {len([c for c in info['numeric_cols'] if c != target])}")
        st.markdown(f"**Categorical features:** {len(info['categorical_cols'])}")
        st.markdown(f"**Memory:** {info['memory_mb']} MB")
        if ss.is_california:
            with st.expander("How the supplied CSV maps to the standard California Housing features"):
                st.markdown(
                    "The CSV stores *totals per block group*. They are converted to the standard per-household features:\n\n"
                    "- `AveRooms = total_rooms / households`\n- `AveBedrms = total_bedrooms / households`\n"
                    "- `AveOccup = population / households`\n- `MedHouseVal = median_house_value / 100 000`\n\n"
                    "`ocean_proximity` is kept as a categorical feature (one-hot encoded); the 207 missing `total_bedrooms` "
                    "values become missing `AveBedrms` values that are imputed with the training median."
                )


# --------------------------------------------------------------------------- #
# Page 2 – Data exploration
# --------------------------------------------------------------------------- #
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("🎛️ Interactive filters", expanded=False):
        chosen = st.multiselect("Filter on columns", list(df.columns), key="flt_cols",
                                help="Pick columns, then narrow numeric ranges or categories.")
        out = df
        for col in chosen:
            if dl.column_kind(df[col]) == "numeric":
                lo, hi = float(df[col].min()), float(df[col].max())
                if lo == hi:
                    continue
                rng = st.slider(col, lo, hi, (lo, hi), key=f"flt_{col}")
                out = out[out[col].between(rng[0], rng[1]) | out[col].isna()]
            else:
                opts = sorted(df[col].dropna().astype(str).unique().tolist())
                sel = st.multiselect(col, opts, default=opts, key=f"flt_{col}")
                out = out[out[col].astype(str).isin(sel) | out[col].isna()]
    return out


def page_exploration() -> None:
    page_header("🔎", "Data Exploration", "Understand distributions, relationships and data quality before modelling.")
    ss = st.session_state
    if ss.df is None:
        callout("No dataset loaded.", "warn")
        return
    df = ss.df[[c for c in (ss.features or []) if c in ss.df.columns] + [ss.target]]
    numeric = [c for c in df.columns if dl.column_kind(df[c]) == "numeric"]
    categorical = [c for c in df.columns if dl.column_kind(df[c]) == "categorical"]

    tabs = st.tabs(["Preview", "Statistical summary", "Distributions", "Correlation", "Missing values", "Relationships"])

    with tabs[0]:
        filtered = apply_filters(df)
        a, b = st.columns([3, 1])
        with a:
            n_show = safe_slider("Rows to display", 5, min(1000, max(len(filtered), 6)), 20, key="n_preview")
        b.metric("Rows after filters", f"{len(filtered):,}", delta=f"{len(filtered) - len(df):,}" if len(filtered) != len(df) else None)
        st.dataframe(filtered.head(n_show))
        st.download_button("⬇️ Download filtered rows (CSV)", csv_bytes(filtered), "filtered_dataset.csv", "text/csv")

    with tabs[1]:
        st.markdown("**Numeric summary**")
        if numeric:
            st.dataframe(df[numeric].describe().T.round(4))
        if categorical:
            st.markdown("**Categorical summary**")
            st.dataframe(df[categorical].describe().T)
        st.markdown("**Column types**")
        st.dataframe(dl.inspect_dataset(df)["columns"], hide_index=True)

    with tabs[2]:
        if not numeric:
            st.info("No numeric columns.")
        else:
            c1, c2, c3 = st.columns([2, 1, 2])
            feat = c1.selectbox("Feature", numeric, key="dist_feat", help="Choose a numeric column.")
            bins = c2.slider("Bins", 10, 120, 50, key="dist_bins")
            color = c3.selectbox("Colour by", ["(none)"] + categorical, key="dist_color")
            fig = px.histogram(df, x=feat, nbins=bins, marginal="box", color=None if color == "(none)" else color,
                               color_discrete_sequence=[BLUE, TEAL, GOLD, RED, SLATE, "#7A5C9E"], opacity=0.85)
            fig.update_layout(title=f"Distribution of {feat}", barmode="overlay")
            show(fig, 460)
            s = df[feat].dropna()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Mean", f"{s.mean():,.3f}")
            k2.metric("Median", f"{s.median():,.3f}")
            k3.metric("Std dev", f"{s.std():,.3f}")
            k4.metric("Skewness", f"{s.skew():+.2f}", help="|skew| > 1 indicates a strongly skewed distribution.")

    with tabs[3]:
        if len(numeric) < 2:
            st.info("At least two numeric columns are needed.")
        else:
            corr = pp.correlation_matrix(df[numeric])
            fig = px.imshow(corr, text_auto=".2f" if len(numeric) <= 14 else False, zmin=-1, zmax=1,
                            color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(title="Correlation heatmap (Pearson)")
            show(fig, 560, legend=False)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Correlation with the target**")
                tc = corr[ss.target].drop(ss.target).sort_values()
                fig = go.Figure(go.Bar(x=tc.values, y=tc.index, orientation="h",
                                       marker_color=[TEAL if v >= 0 else RED for v in tc.values]))
                fig.update_layout(title=f"Pearson r with {ss.target}")
                show(fig, 380, legend=False)
            with c2:
                thr = st.slider("Highlight feature pairs with |r| ≥", 0.3, 0.95, 0.7, 0.05, key="corr_thr")
                pairs = pp.high_correlation_pairs(corr.drop(index=ss.target, columns=ss.target), thr)
                if pairs.empty:
                    st.success("No feature pair exceeds this threshold.")
                else:
                    st.warning(f"{len(pairs)} highly correlated feature pair(s) – possible multicollinearity.")
                    st.dataframe(pairs, hide_index=True)

    with tabs[4]:
        rep = pp.missing_value_report(df)
        miss = rep[rep.missing > 0]
        if miss.empty:
            st.success("No missing values in the selected columns.")
        else:
            fig = px.bar(miss.sort_values("missing"), x="missing", y="column", orientation="h", text="missing_pct",
                         color_discrete_sequence=[GOLD], hover_data=["dtype"])
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig.update_layout(title="Missing values per column", xaxis_title="Missing cells", yaxis_title="")
            show(fig, max(280, 60 * len(miss) + 120), legend=False)
        st.dataframe(rep, hide_index=True)
        if not miss.empty:
            st.caption("Numeric gaps are imputed with the training-set median, categorical gaps with the most frequent value – inside the pipeline, so no test information leaks.")

    with tabs[5]:
        if len(numeric) < 2:
            st.info("At least two numeric columns are needed.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            x = c1.selectbox("X axis", numeric, index=0, key="rel_x")
            y = c2.selectbox("Y axis", numeric, index=numeric.index(ss.target) if ss.target in numeric else min(1, len(numeric) - 1), key="rel_y")
            color = c3.selectbox("Colour", ["(none)"] + numeric + categorical, key="rel_c")
            with c4:
                n = safe_slider("Sample size", 200, min(8000, len(df)), 3000, key="rel_n",
                                help="Plotting a random sample keeps the chart responsive.")
            samp = df.sample(min(n, len(df)), random_state=C.RANDOM_STATE)
            fig = px.scatter(samp, x=x, y=y, color=None if color == "(none)" else color, opacity=0.55,
                             color_continuous_scale="Viridis", color_discrete_sequence=[BLUE, TEAL, GOLD, RED, SLATE])
            fig.update_layout(title=f"{y} vs {x}")
            show(fig, 480)
            if {"Latitude", "Longitude"}.issubset(df.columns):
                st.markdown("**Geographic view** – each point is a block group")
                samp2 = df.sample(min(6000, len(df)), random_state=C.RANDOM_STATE)
                fig = px.scatter(samp2, x="Longitude", y="Latitude", color=ss.target, opacity=0.6,
                                 color_continuous_scale="Turbo", hover_data=[c for c in ("MedInc", "HouseAge") if c in df.columns])
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                fig.update_layout(title=f"Location coloured by {ss.target}")
                show(fig, 560, legend=False)


# --------------------------------------------------------------------------- #
# Page 3 – Preprocessing
# --------------------------------------------------------------------------- #
def page_preprocessing() -> None:
    page_header("🧹", "Data Preprocessing", "Missing values, duplicates, encoding, scaling, multicollinearity and feature engineering.")
    ss = st.session_state
    res = need_result("preprocessing")
    if res is None or ss.df is None:
        return
    diag, meta = res["diagnostics"], res["meta"]
    tabs = st.tabs(["Missing values", "Duplicates", "Encoding & scaling", "Multicollinearity", "VIF comparison", "Feature engineering"])

    with tabs[0]:
        miss = diag["missing_report"]
        miss = miss[miss.column.isin(list(res["splits"]["X_train"].columns) + [meta["target"]])]
        pre_fit = res["models"]["OLS"].named_steps["preprocessor"]
        after = int(pd.DataFrame(pre_fit.transform(res["splits"]["X_train"])).isna().sum().sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Missing cells (before)", f"{int(miss.missing.sum()):,}")
        k2.metric("Missing cells after imputation (train)", f"{after:,}", help="Measured on the transformed training matrix.")
        k3.metric("Columns with gaps", f"{int((miss.missing > 0).sum())}")
        st.dataframe(miss, hide_index=True)
        c1, c2 = st.columns(2)
        pdesc = diag["preprocessing"]
        with c1:
            st.markdown("**Numeric imputer – medians learned on the training set**")
            if "numeric_imputer" in pdesc:
                st.dataframe(pdesc["numeric_imputer"], hide_index=True)
        with c2:
            st.markdown("**Categorical imputer – most frequent value**")
            if "categorical_imputer" in pdesc:
                st.dataframe(pdesc["categorical_imputer"], hide_index=True)
            else:
                st.caption("No categorical features.")
        callout("Imputation statistics come from the <b>training split only</b>. The same medians are re-used for the test set and for new predictions.")

    with tabs[1]:
        raw = ss.df[[c for c in (ss.features or []) if c in ss.df.columns] + [ss.target]]
        dup = pp.duplicate_report(raw)
        k1, k2, k3 = st.columns(3)
        k1.metric("Duplicate rows found", f"{dup['n_duplicates']:,}")
        k2.metric("Removed before split", f"{diag['duplicates']['removed']:,}")
        k3.metric("Rows without target removed", f"{diag['duplicates']['missing_target_removed']:,}")
        if dup["n_duplicates"] == 0:
            st.success("No duplicate rows detected – nothing had to be removed.")
        else:
            st.warning("Duplicates were dropped before the train/test split so identical rows cannot appear in both sets.")
            st.dataframe(dup["examples"])

    with tabs[2]:
        st.markdown(f"**Numeric features ({len(meta['numeric_features'])}):** " + ", ".join(f"`{c}`" for c in meta["numeric_features"]))
        st.markdown(f"**Categorical features ({len(meta['categorical_features'])}):** " + (", ".join(f"`{c}`" for c in meta["categorical_features"]) or "none"))
        pdesc = diag["preprocessing"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**StandardScaler** – mean & std learned on the training set")
            if "scaler" in pdesc:
                st.dataframe(pdesc["scaler"], hide_index=True)
        with c2:
            st.markdown("**OneHotEncoder** – category levels")
            if "onehot" in pdesc:
                st.dataframe(pdesc["onehot"], hide_index=True)
            else:
                st.caption("No categorical features were encoded.")
        pre_fit = res["models"]["OLS"].named_steps["preprocessor"]
        sample = pp.transform_to_frame(pre_fit, res["splits"]["X_train"].head(8))
        st.markdown("**Transformed training rows (what the models actually see)**")
        st.dataframe(sample.round(3))
        st.caption(f"Feature set in use: **{meta['feature_set']}** → {sample.shape[1]} model inputs.")

    with tabs[3]:
        num = meta["numeric_features"]
        X_train = res["splits"]["X_train"]
        if len(num) < 2:
            st.info("Multicollinearity needs at least two numeric features.")
        else:
            corr = diag["correlation"].drop(index=meta["target"], columns=meta["target"], errors="ignore")
            thr = st.slider("Correlation threshold", 0.3, 0.95, 0.6, 0.05, key="mc_thr")
            pairs = pp.high_correlation_pairs(corr, thr)
            vif = diag["vif_original"]
            c1, c2 = st.columns([3, 2])
            with c1:
                fig = go.Figure(go.Bar(x=vif.VIF[::-1], y=vif.feature[::-1], orientation="h",
                                       marker_color=[RED if v >= C.VIF_THRESHOLD else GOLD if v >= C.VIF_MODERATE else TEAL for v in vif.VIF[::-1]],
                                       text=[f"{v:.2f}" for v in vif.VIF[::-1]], textposition="outside"))
                fig.add_vline(x=C.VIF_MODERATE, line_dash="dash", line_color=GOLD, annotation_text="5")
                fig.add_vline(x=C.VIF_THRESHOLD, line_dash="dash", line_color=RED, annotation_text="10")
                fig.update_layout(title="Variance Inflation Factor – original numeric features (training set)", xaxis_title="VIF")
                show(fig, max(320, 42 * len(vif) + 120), legend=False)
            with c2:
                st.markdown("**Correlated pairs**")
                if pairs.empty:
                    st.success(f"No pair with |r| ≥ {thr:.2f}.")
                else:
                    st.dataframe(pairs, hide_index=True)
                st.markdown("**VIF scale**")
                st.markdown("- **< 5** low  \n- **5 – 10** moderate  \n- **> 10** severe")
            severe = vif[vif.VIF >= C.VIF_THRESHOLD].feature.tolist()
            moderate = vif[(vif.VIF >= C.VIF_MODERATE) & (vif.VIF < C.VIF_THRESHOLD)].feature.tolist()
            if severe:
                callout(f"<b>Severe multicollinearity</b> (VIF ≥ 10): {html.escape(', '.join(severe))}. Ridge/Lasso or feature removal is recommended.", "bad")
            elif moderate:
                callout(f"<b>Moderate multicollinearity</b> (VIF 5–10): {html.escape(', '.join(moderate))}. No feature is severe, but the coefficients of these variables are less stable – "
                        "Ridge/Lasso regularisation is the safeguard used here.", "warn")
            else:
                callout("All VIFs are below 5 – multicollinearity is not a concern.")
            st.markdown("**How multicollinearity is handled in this project**")
            st.markdown(
                "1. **Detect** with the correlation matrix and VIF.\n"
                "2. **Mean-centre** variables before creating product terms (removes structural collinearity).\n"
                "3. **Regularise** – Ridge shrinks and Lasso can zero correlated coefficients.\n"
                "4. **Test removal** – a VIF-pruned feature set is trained and compared (next tab and *Feature engineering*)."
            )

    with tabs[4]:
        num = meta["numeric_features"]
        X_train = res["splits"]["X_train"]
        if len(num) < 2:
            st.info("VIF needs at least two numeric features.")
        else:
            thr = st.slider("Prune features until every VIF is below", 2.0, 10.0, float(C.VIF_MODERATE), 0.5, key="vif_prune_thr",
                            help="Iteratively removes the feature with the highest VIF.")
            kept, log = pp.vif_prune(X_train[num], thr)
            before = diag["vif_original"].rename(columns={"VIF": "VIF before"})[["feature", "VIF before"]]
            after = pp.compute_vif(X_train[kept]).rename(columns={"VIF": "VIF after pruning"})[["feature", "VIF after pruning"]] if len(kept) > 1 else pd.DataFrame(columns=["feature", "VIF after pruning"])
            comp = before.merge(after, on="feature", how="left")
            comp["status"] = np.where(comp.feature.isin(log.dropped_feature), "dropped", "kept")
            c1, c2 = st.columns([3, 2])
            with c1:
                long = comp.melt(id_vars=["feature", "status"], value_vars=["VIF before", "VIF after pruning"], var_name="stage", value_name="VIF").dropna()
                fig = px.bar(long, x="feature", y="VIF", color="stage", barmode="group", text_auto=".2f",
                             color_discrete_map={"VIF before": SLATE, "VIF after pruning": TEAL})
                fig.add_hline(y=C.VIF_THRESHOLD, line_dash="dash", line_color=RED)
                fig.add_hline(y=C.VIF_MODERATE, line_dash="dash", line_color=GOLD)
                fig.update_layout(title="VIF before vs after pruning", xaxis_title="")
                show(fig, 440)
            with c2:
                st.dataframe(comp, hide_index=True)
                if log.empty:
                    st.success("No feature has to be dropped at this threshold.")
                else:
                    st.markdown("**Pruning log**")
                    st.dataframe(log, hide_index=True)
            vf = diag["vif_final_design"]
            st.markdown(f"**VIF of the final model inputs** ({meta['feature_set']}; one-hot dummies excluded)")
            if len(vf) > 1:
                st.dataframe(vf, hide_index=True)
            st.caption("Interaction and polynomial terms are naturally correlated with their parent features; centring and regularisation keep the models stable.")

    with tabs[5]:
        exp = res["experiments"]
        st.markdown("**Engineered features**")
        eng = res["engineering_table"]
        if len(eng):
            st.dataframe(eng, hide_index=True)
            st.caption("Operands are clipped to their 1st–99th training percentiles before products/ratios are formed, so a few extreme block groups cannot dominate a feature.")
        else:
            st.caption("No domain interaction features could be defined for this dataset.")
        long = exp.melt(id_vars=["feature_set"], value_vars=[f"{m}_cv_rmse" for m in C.MODEL_NAMES], var_name="model", value_name="CV RMSE")
        long["model"] = long.model.str.replace("_cv_rmse", "", regex=False)
        fig = px.bar(long, x="feature_set", y="CV RMSE", color="model", barmode="group", text_auto=".3f", color_discrete_map=MODEL_COLORS)
        lo = float(long["CV RMSE"].min())
        fig.update_yaxes(range=[lo * 0.9, float(long["CV RMSE"].max()) * 1.03])
        fig.update_layout(title="Cross-validated RMSE by feature set (training data; y-axis truncated)", xaxis_title="")
        show(fig, 420)
        view = exp[["feature_set", "n_features_after_encoding", "best_model_by_cv", "best_cv_rmse", "best_test_rmse", "best_test_r2"]].round(4)
        st.dataframe(view, hide_index=True)
        base = exp.loc[exp.key == "baseline"].iloc[0]
        chosen = exp.loc[exp.feature_set == meta["feature_set"]].iloc[0]
        gain = 100 * (base.best_test_rmse - chosen.best_test_rmse) / base.best_test_rmse
        callout(
            f"<b>Selected: {meta['feature_set']}</b> (lowest cross-validated RMSE on the training data – the test set was not used for this choice). "
            f"Compared with the baseline, test RMSE changed from <b>{base.best_test_rmse:.4f}</b> to <b>{chosen.best_test_rmse:.4f}</b> "
            f"({gain:+.1f}% {'better' if gain > 0 else 'worse'}) and test R² from <b>{base.best_test_r2:.4f}</b> to <b>{chosen.best_test_r2:.4f}</b>."
            + (" Feature sets that did not improve generalisation were rejected." if meta["feature_set"] != "Baseline (original features)" else ""),
        )


# --------------------------------------------------------------------------- #
# Page 4 – Model training
# --------------------------------------------------------------------------- #
def page_training() -> None:
    page_header("⚙️", "Model Training", "Fit OLS, Ridge and Lasso inside leakage-free pipelines and tune alpha with GridSearchCV.")
    ss = st.session_state
    if ss.df is None:
        callout("No dataset loaded.", "warn")
        return
    res = get_result()

    with st.container():
        c1, c2, c3 = st.columns([1, 1, 2])
        cv = c1.slider("CV folds", 3, 10, int(res["meta"]["cv_folds"]) if res else C.CV_FOLDS, key="cv_folds",
                       help="Number of KFold splits used by GridSearchCV inside the training data.")
        c2.metric("Test split", "20 %", help="Fixed 80:20 train/test ratio with random_state=42.")
        save = c3.checkbox("Save models to the project folder", value=ss.is_california, disabled=not ss.is_california,
                           help="Only the bundled California Housing models are written to models/ (custom datasets stay in this session).")
        if st.button("🚀 Train models" if res is None else "🔁 Re-train models", key="train_btn"):
            run_training(cv, save)
            res = get_result()

    if res is None:
        callout("Press <b>Train models</b> to start. Training takes about 30–60 seconds.", "warn")
        return
    meta = res["meta"]

    st.markdown("#### Train–test split")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Training rows", f"{meta['n_train']:,}")
    k2.metric("Test rows", f"{meta['n_test']:,}")
    k3.metric("Split ratio", f"{round((1 - meta['test_size']) * 100)}:{round(meta['test_size'] * 100)}")
    k4.metric("Random state", meta["random_state"])
    k5.metric("CV folds", meta["cv_folds"])
    st.caption(f"Duplicates removed before splitting: {meta['n_duplicates_removed']} · rows without target removed: {meta['n_missing_target_removed']} · feature set used: **{meta['feature_set']}**")
    st.code(
        "Pipeline([\n"
        "  ('preprocessor', ColumnTransformer([\n"
        "       ('num', median-impute → [feature engineering] → StandardScaler),\n"
        "       ('cat', most-frequent-impute → OneHotEncoder)])),\n"
        "  ('model', LinearRegression | Ridge(alpha) | Lasso(alpha))\n"
        "])   # GridSearchCV refits the whole pipeline in every fold → no leakage",
        language="text",
    )

    st.markdown("#### Model training status")
    rows = []
    for name in C.MODEL_NAMES:
        bp = res["best_params"][name]
        rows.append({
            "Model": name,
            "Status": "✅ trained",
            "Regularisation": {"OLS": "none", "Ridge": "L2", "Lasso": "L1"}[name],
            "Best alpha": f"{bp['alpha']:.6g}" if "alpha" in bp else "—",
            "CV RMSE": round(res["cv_rmse"][name], 4),
            "CV RMSE std": round(res["cv_rmse_std"][name], 4),
            "Fit time (s)": round(res["timings"][name], 1),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.markdown("#### GridSearchCV optimisation")
    tabs = st.tabs(["Ridge (L2)", "Lasso (L1)", "OLS"])
    for tab, name in zip(tabs[:2], ["Ridge", "Lasso"]):
        with tab:
            g = res["grids"][name]
            best_a = res["best_params"][name]["alpha"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=g.alpha, y=g.cv_rmse + g.cv_rmse_std, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=g.alpha, y=g.cv_rmse - g.cv_rmse_std, mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(120,140,160,.18)", name="±1 std (folds)", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=g.alpha, y=g.cv_rmse, mode="lines+markers", name="CV RMSE (validation)", line=dict(color=MODEL_COLORS[name], width=3)))
            fig.add_trace(go.Scatter(x=g.alpha, y=g.train_rmse, mode="lines+markers", name="Train RMSE", line=dict(color=SLATE, dash="dash")))
            ymin, ymax = float((g.cv_rmse - g.cv_rmse_std).min()), float(g.train_rmse.max())
            fig.add_trace(go.Scatter(x=[best_a, best_a], y=[min(ymin, float(g.train_rmse.min())), max(ymax, float(g.cv_rmse.max()))], mode="lines",
                                     name=f"best α = {best_a:.3g}", line=dict(color=RED, dash="dot")))
            fig.update_xaxes(type="log", title="alpha (log scale)")
            fig.update_yaxes(title="RMSE")
            fig.update_layout(title=f"{name}: validation error vs alpha")
            show(fig, 420)
            k1, k2, k3 = st.columns(3)
            k1.metric("Best alpha", f"{best_a:.6g}")
            k2.metric("Best CV RMSE", f"{res['cv_rmse'][name]:.4f}")
            k3.metric("Alphas tested", f"{len(g)}", help=f"Range {g.alpha.min():.0e} – {g.alpha.max():.0e}")
            if best_a in (g.alpha.min(), g.alpha.max()):
                callout(f"The best alpha lies on the <b>edge of the grid</b> ({best_a:.3g}). The optimum may be slightly beyond the tested range; "
                        "widen the grid in <code>src/config.py</code> to check.", "warn")
            st.markdown("**All candidates**")
            st.dataframe(g.sort_values("rank").round(5), hide_index=True)
    with tabs[2]:
        st.markdown("OLS has no hyper-parameters. Its cross-validated RMSE (same folds as Ridge and Lasso) is "
                    f"**{res['cv_rmse']['OLS']:.4f} ± {res['cv_rmse_std']['OLS']:.4f}**.")
        st.markdown(pr.model_explanation("OLS"))


# --------------------------------------------------------------------------- #
# Page 5 – Model performance
# --------------------------------------------------------------------------- #
def page_performance() -> None:
    page_header("📊", "Model Performance", "RMSE, MAE and R² computed from real predictions on the held-out test set.")
    res = need_result("performance")
    if res is None:
        return
    met, preds, meta = res["metrics"], res["predictions"], res["meta"]
    best = res["best_model"]

    st.markdown(
        f'<div class="result-card"><div class="lbl">🏆 Best model (lowest test RMSE)</div><div class="val">{best}</div>'
        f'<div class="sub">Test RMSE {met.set_index("Model").loc[best, "Test RMSE"]:.4f} · MAE {met.set_index("Model").loc[best, "Test MAE"]:.4f} · '
        f'R² {met.set_index("Model").loc[best, "Test R²"]:.4f} &nbsp;|&nbsp; feature set: {meta["feature_set"]}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    callout(res["best_model_explanation"].replace("**", "<b>", 1).replace("**", "</b>", 1))

    st.markdown("#### Comparison table")
    st.dataframe(met.round(4), hide_index=True)

    st.markdown("#### Metric comparison")
    tabs = st.tabs(["RMSE", "MAE", "R²", "Cross-validated RMSE"])
    for tab, col in zip(tabs[:3], ["RMSE", "MAE", "R²"]):
        with tab:
            long = pd.concat([
                pd.DataFrame({"Model": met.Model, "Split": "Train", col: met[f"Train {col}"]}),
                pd.DataFrame({"Model": met.Model, "Split": "Test", col: met[f"Test {col}"]}),
            ])
            fig = px.bar(long, x="Model", y=col, color="Split", barmode="group", text_auto=".4f",
                         color_discrete_map={"Train": "#B8C4D2", "Test": NAVY})
            span = long[col].max() - long[col].min()
            fig.update_yaxes(range=[max(0, long[col].min() - 3 * max(span, 0.01)) if col != "R²" else long[col].min() - 0.1, long[col].max() + 0.25 * max(span, 0.01)])
            fig.update_layout(title=f"{col} – {'lower' if col != 'R²' else 'higher'} is better (y-axis truncated)")
            show(fig, 400)
    with tabs[3]:
        cvdf = pd.DataFrame({"Model": list(res["cv_rmse"]), "CV RMSE": list(res["cv_rmse"].values()), "std": list(res["cv_rmse_std"].values())})
        fig = px.bar(cvdf, x="Model", y="CV RMSE", error_y="std", color="Model", color_discrete_map=MODEL_COLORS, text_auto=".4f")
        fig.update_layout(title=f"{meta['cv_folds']}-fold cross-validated RMSE on the training set")
        show(fig, 400, legend=False)

    st.markdown("#### Actual vs predicted")
    sel = st.selectbox("Model", ["All models (side by side)"] + C.MODEL_NAMES, index=(C.MODEL_NAMES.index(best) + 1), key="avp_model")
    n = safe_slider("Points to plot", 200, len(preds), 3000, key="avp_n", help="Random sample of the test set.")
    samp = preds.sample(min(n, len(preds)), random_state=C.RANDOM_STATE)
    models_to_plot = C.MODEL_NAMES if sel.startswith("All") else [sel]
    long = pd.concat([pd.DataFrame({"Actual": samp.actual, "Predicted": samp[m], "Model": m}) for m in models_to_plot])
    fig = px.scatter(long, x="Actual", y="Predicted", color="Model", facet_col="Model" if len(models_to_plot) > 1 else None,
                     opacity=0.45, color_discrete_map=MODEL_COLORS)
    lim = [float(min(long.Actual.min(), long.Predicted.min())), float(max(long.Actual.max(), long.Predicted.max()))]
    diag_line = dict(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1], line=dict(color=RED, dash="dash"))
    if len(models_to_plot) > 1:
        fig.add_shape(**diag_line, row="all", col="all")
    else:
        fig.add_shape(**diag_line)
    fig.update_layout(title="Actual vs predicted values (dashed line = perfect prediction)")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    show(fig, 470, legend=False)

    st.markdown("#### Prediction comparison (first test samples)")
    k = safe_slider("Samples", 10, min(150, len(preds)), 50, key="cmp_n")
    d = preds.iloc[:k].reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d.actual, name="Actual", mode="lines+markers", line=dict(color="#111", width=2)))
    for m in C.MODEL_NAMES:
        fig.add_trace(go.Scatter(x=d.index, y=d[m], name=m, mode="lines", line=dict(color=MODEL_COLORS[m], dash="dot")))
    fig.update_layout(title="Actual vs the three models", xaxis_title="Test sample", yaxis_title=meta["target"])
    show(fig, 400)

    with st.expander("Model coefficients (standardised features)"):
        m_sel = st.selectbox("Model", C.MODEL_NAMES, index=C.MODEL_NAMES.index(best), key="coef_model")
        coefs = res["coefficients"][m_sel]
        top = safe_slider("Top features", 5, min(40, len(coefs)), 15, key="coef_top")
        d = coefs.head(top).iloc[::-1]
        fig = go.Figure(go.Bar(x=d.coefficient, y=d.feature, orientation="h", marker_color=[MODEL_COLORS[m_sel] if v >= 0 else RED for v in d.coefficient]))
        fig.update_layout(title=f"{m_sel}: largest coefficients")
        show(fig, max(320, 26 * len(d) + 120), legend=False)
        st.caption(f"{int((coefs.coefficient.abs() <= 1e-10).sum())} coefficient(s) are exactly zero.")


# --------------------------------------------------------------------------- #
# Page 6 – Residual analysis
# --------------------------------------------------------------------------- #
def page_residuals() -> None:
    page_header("📉", "Residual Analysis", "Do the errors look random, or do they reveal systematic problems?")
    res = need_result("residual analysis")
    if res is None:
        return
    preds, meta = res["predictions"], res["meta"]
    best = res["best_model"]
    model = st.selectbox("Model", C.MODEL_NAMES, index=C.MODEL_NAMES.index(best), key="res_model")
    st_ = res["residual_stats"][model]
    resid = preds.actual - preds[model]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Mean residual", f"{st_['mean']:+.4f}", help="Close to 0 means no overall bias.")
    k2.metric("Std of residuals", f"{st_['std']:.4f}")
    k3.metric("Skewness", f"{st_['skew']:+.2f}", help="0 = symmetric.")
    k4.metric("Excess kurtosis", f"{st_['excess_kurtosis']:.2f}", help="0 = normal tails; larger = heavier tails.")
    k5.metric("|resid| vs prediction ρ", f"{st_['hetero_spearman']:+.2f}", help="Spearman correlation – non-zero means error size depends on the price level.")

    tabs = st.tabs(["Distribution", "Residual vs predicted", "Q-Q plot", "Error analysis", "Observations & limitations"])
    with tabs[0]:
        bins = st.slider("Bins", 20, 150, 60, key="res_bins")
        fig = px.histogram(x=resid, nbins=bins, histnorm="probability density", color_discrete_sequence=[MODEL_COLORS[model]])
        fig.update_traces(name="Residuals", showlegend=True)
        xs = np.linspace(float(resid.min()), float(resid.max()), 200)
        fig.add_trace(go.Scatter(x=xs, y=sps.norm.pdf(xs, resid.mean(), resid.std()), mode="lines", name="Normal fit", line=dict(color=NAVY)))
        fig.add_vline(x=0, line_dash="dash", line_color=RED)
        fig.update_layout(title=f"Residual distribution – {model}", xaxis_title="Residual (actual − predicted)")
        show(fig, 460)
    with tabs[1]:
        n = safe_slider("Points", 200, len(preds), 4000, key="res_n")
        samp = pd.DataFrame({"Predicted": preds[model], "Residual": resid}).sample(min(n, len(preds)), random_state=C.RANDOM_STATE)
        fig = px.scatter(samp, x="Predicted", y="Residual", opacity=0.4, color_discrete_sequence=[MODEL_COLORS[model]])
        dec = st_["decile_table"]
        fig.add_trace(go.Scatter(x=dec.mean_pred, y=dec.mean_residual, mode="lines+markers", name="Mean residual per decile", line=dict(color=NAVY, width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color=RED)
        fig.update_layout(title=f"Residuals vs predicted – {model}")
        show(fig, 470)
        st.caption("A flat line near zero means no systematic pattern. A curve or a funnel shape signals missing non-linearity or heteroscedasticity.")
        fig = go.Figure(go.Bar(x=dec.decile, y=dec.mean_residual, marker_color=[TEAL if v >= 0 else RED for v in dec.mean_residual],
                               error_y=dict(type="data", array=(dec.std_residual / np.sqrt(dec.n)).tolist())))
        fig.update_layout(title="Mean residual by predicted-value decile (± standard error)", xaxis_title="Prediction decile (1 = cheapest)", yaxis_title="Mean residual")
        show(fig, 340, legend=False)
    with tabs[2]:
        (osm, osr), (slope, intercept, _r) = sps.probplot(resid.to_numpy(), dist="norm")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Residuals", marker=dict(color=MODEL_COLORS[model], size=4, opacity=0.6)))
        fig.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[intercept + slope * osm.min(), intercept + slope * osm.max()], mode="lines", name="Normal reference", line=dict(color=RED, dash="dash")))
        fig.update_layout(title=f"Normal Q-Q plot – {model}", xaxis_title="Theoretical quantiles", yaxis_title="Ordered residuals")
        show(fig, 460)
        st.caption("Points on the dashed line = normally distributed errors. Departures in the tails mean a few very large errors.")
    with tabs[3]:
        long = pd.concat([pd.DataFrame({"Model": m, "Absolute error": (preds.actual - preds[m]).abs()}) for m in C.MODEL_NAMES])
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(long, x="Model", y="Absolute error", color="Model", color_discrete_map=MODEL_COLORS, points=False)
            fig.update_layout(title="Absolute error by model")
            show(fig, 400, legend=False)
        with c2:
            fig = px.histogram(long, x="Absolute error", color="Model", barmode="overlay", histnorm="probability density", nbins=60,
                               opacity=0.55, color_discrete_map=MODEL_COLORS)
            fig.update_layout(title="Distribution of absolute errors")
            show(fig, 400)
        a = (preds.actual - preds[model]).abs()
        q = a.quantile([0.5, 0.9, 0.99])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Median |error|", f"{q[0.5]:.3f}")
        k2.metric("90th percentile", f"{q[0.9]:.3f}")
        k3.metric("99th percentile", f"{q[0.99]:.3f}")
        k4.metric("Largest |error|", f"{a.max():.3f}")
        worst = preds.assign(error=(preds.actual - preds[model]), abs_error=a).sort_values("abs_error", ascending=False).head(10)
        st.markdown("**10 largest errors (test set)**")
        st.dataframe(worst[["actual", model, "error"]].round(3))
    with tabs[4]:
        st.markdown(f"#### Do the residuals indicate systematic error? ({model})")
        for line in res["residual_interpretation"][model]:
            st.markdown(f"- {line}")
        st.markdown("#### Model limitations")
        lims = [
            "Linear models cannot capture arbitrary non-linear or spatial structure beyond the engineered terms; tree-based models or splines would likely reduce the remaining error.",
            "The three models perform almost identically: with thousands of observations and few features, regularisation has little variance to remove.",
            "Predictions are only reliable inside the range of the training data; the prediction page warns about extrapolation.",
            "Metrics come from a single 80:20 split; the cross-validation standard deviation shows how much they could vary.",
        ]
        if meta.get("target_cap"):
            lims.insert(0, f"The target is capped at {meta['target_cap']['value']:.4g} ({100 * meta['target_cap']['share']:.1f}% of rows), so very expensive areas are systematically under-predicted.")
        for l in lims:
            st.markdown(f"- {l}")


# --------------------------------------------------------------------------- #
# Page 7 – Prediction
# --------------------------------------------------------------------------- #
def _set_inputs_from_row(row: pd.Series) -> None:
    for col, v in row.items():
        key = f"inp_{col}"
        sp = st.session_state.result["form_spec"].get(col)
        if sp is None:
            continue
        if sp["kind"] == "numeric":
            st.session_state[key] = float(v) if pd.notna(v) else float(sp["median"])
        else:
            st.session_state[key] = str(v) if pd.notna(v) else sp["default"]


def _random_row() -> None:
    res = st.session_state.result
    row = res["splits"]["X_test"].sample(1, random_state=int(time.time()) % 100000).iloc[0]
    _set_inputs_from_row(row)


def _median_row() -> None:
    res = st.session_state.result
    for col, sp in res["form_spec"].items():
        st.session_state[f"inp_{col}"] = sp["median"] if sp["kind"] == "numeric" else sp["default"]


def render_inputs(spec: dict) -> dict:
    """Render one widget per feature (generated from the training data) and return the values."""
    values = {}
    cols = st.columns(3)
    for i, (col, sp) in enumerate(spec.items()):
        with cols[i % 3]:
            key = f"inp_{col}"
            label = sp.get("label", col)
            if sp["kind"] == "numeric":
                kwargs = {}
                if sp["min"] >= 0:
                    kwargs["min_value"] = 0.0
                if key not in st.session_state:
                    st.session_state[key] = float(sp["median"])
                values[col] = st.number_input(
                    label, key=key, step=float(sp["step"]),
                    format="%.0f" if sp["integer"] and sp["step"] >= 1 else "%.4f",
                    help=f"Training range: {sp['min']:g} – {sp['max']:g} (median {sp['median']:g}). Typical (1–99 %): {sp['p01']:g} – {sp['p99']:g}.",
                    **kwargs,
                )
            else:
                opts = sp["options"]
                if key not in st.session_state or st.session_state[key] not in opts:
                    st.session_state[key] = sp["default"]
                values[col] = st.selectbox(label, opts, key=key, help="Categorical feature – one-hot encoded by the pipeline.")
    return values


def page_prediction() -> None:
    page_header("🏡", "House Price Prediction", "Enter the characteristics of a neighbourhood and price it with the trained models.")
    ss = st.session_state
    res = need_result("prediction")
    if res is None:
        return
    spec, meta = res["form_spec"], res["meta"]
    feature_cols = list(spec.keys())
    numeric = meta["numeric_features"]
    unit = meta["unit"]

    a, b, c = st.columns([2, 2, 3])
    model_name = a.selectbox("Model", C.MODEL_NAMES, index=C.MODEL_NAMES.index(res["best_model"]), key="pred_model",
                             help="The model whose prediction is highlighted. All three are always compared below.")
    live = b.toggle("Real-time prediction", value=False, key="pred_live", help="Update the prediction instantly as you change an input (no button needed).")
    with c:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        ca.button("🎲 Random test row", on_click=_random_row, key="rand_row", help="Fill the form with a real held-out neighbourhood.")
        cb.button("↺ Median values", on_click=_median_row, key="med_row", help="Reset all inputs to training medians.")

    go_predict = False
    if live:
        values = render_inputs(spec)
        go_predict = True
    else:
        with st.form("predict_form"):
            values = render_inputs(spec)
            go_predict = st.form_submit_button("🔮 Predict price")

    if go_predict:
        try:
            X_new = pr.build_input_frame(values, feature_cols, numeric)
            out = pr.predict_all(res["models"], X_new)
            ss.last_prediction = {
                "values": dict(values), "preds": {m: float(v[0]) for m, v in out.items()},
                "model": model_name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "saved": False,
            }
            if not live:
                ss.prediction_log.append({"time": ss.last_prediction["time"], "selected_model": model_name, **values,
                                          **{f"pred_{m}": float(v[0]) for m, v in out.items()}})
                ss.last_prediction["saved"] = True
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")
            ss.last_prediction = None

    lp = ss.last_prediction
    if not lp:
        st.info("Fill in the form and press **Predict price** (or switch on real-time prediction).")
        return
    # keep the highlighted model in sync with the selector
    chosen_pred = lp["preds"][model_name]
    for w in pr.check_input_ranges(lp["values"], spec):
        st.warning(w)

    left, right = st.columns([2, 3])
    with left:
        pct = float((res["splits"]["y_train"] <= chosen_pred).mean() * 100)
        st.markdown(
            f'<div class="result-card"><div class="lbl">Predicted {unit["label"]} · {model_name}</div>'
            f'<div class="val">{fmt_target(chosen_pred, res)}</div>'
            f'<div class="sub">Higher than {pct:.0f}% of training neighbourhoods · prediction made {lp["time"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        n_zero = int((res["coefficients"]["Lasso"].coefficient.abs() <= 1e-10).sum())
        st.markdown(f"**About {model_name}.** " + pr.model_explanation(model_name, res["best_params"], n_zero))
        rmse = res["metrics"].set_index("Model").loc[model_name, "Test RMSE"]
        st.caption(f"Typical error of this model on unseen data (test RMSE): ±{pr.format_value(rmse, unit)}.")
    with right:
        comp = pd.DataFrame({"Model": C.MODEL_NAMES, "Prediction": [lp["preds"][m] for m in C.MODEL_NAMES]})
        comp["Display"] = [fmt_target(v, res) for v in comp.Prediction]
        fig = go.Figure(go.Bar(x=comp.Model, y=comp.Prediction * unit["scale"], marker_color=[MODEL_COLORS[m] for m in comp.Model],
                               text=comp.Display, textposition="outside",
                               marker_line_color=[NAVY if m == model_name else "rgba(0,0,0,0)" for m in comp.Model], marker_line_width=3))
        fig.update_layout(title="All three models compared", yaxis_title=unit["label"])
        vals = comp.Prediction * unit["scale"]
        pad = max(float(vals.max() - vals.min()), abs(float(vals.max())) * 0.02, 1e-9)
        fig.update_yaxes(range=[float(vals.min()) - 3 * pad, float(vals.max()) + 1.5 * pad])
        show(fig, 360, legend=False)
        st.dataframe(comp[["Model", "Display"]].rename(columns={"Display": "Predicted value"}), hide_index=True)
    spread = max(lp["preds"].values()) - min(lp["preds"].values())
    st.caption(f"Spread between the three models: {pr.format_value(spread, unit)} – the models agree closely when the input is typical and diverge when it is unusual.")

    if {"Latitude", "Longitude"}.issubset(lp["values"]) and {"Latitude", "Longitude"}.issubset(res["splits"]["X_train"].columns):
        with st.expander("📍 Where is this on the map?"):
            Xt = res["splits"]["X_train"].assign(**{meta["target"]: res["splits"]["y_train"]}).sample(3000, random_state=C.RANDOM_STATE)
            fig = px.scatter(Xt, x="Longitude", y="Latitude", color=meta["target"], opacity=0.5, color_continuous_scale="Turbo")
            fig.add_trace(go.Scatter(x=[lp["values"]["Longitude"]], y=[lp["values"]["Latitude"]], mode="markers", name="Your input",
                                     marker=dict(symbol="star", size=18, color="#111", line=dict(color="white", width=2))))
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            show(fig, 520)

    if not lp["saved"]:
        if st.button("➕ Save this prediction to the report", key="save_pred"):
            ss.prediction_log.append({"time": lp["time"], "selected_model": lp["model"], **lp["values"], **{f"pred_{m}": v for m, v in lp["preds"].items()}})
            lp["saved"] = True
            st.success("Saved – download it from the Reports page.")
    else:
        st.caption(f"✅ Saved to report ({len(ss.prediction_log)} prediction(s) in this session).")


# --------------------------------------------------------------------------- #
# Page 8 – Reports
# --------------------------------------------------------------------------- #
def _processed_frames(res: dict) -> dict:
    ss = st.session_state
    if ss.processed_cache is not None and ss.processed_cache.get("stamp") == res["meta"]["trained_at"]:
        return ss.processed_cache["frames"]
    pre = res["models"]["OLS"].named_steps["preprocessor"]
    s, target = res["splits"], res["meta"]["target"]
    frames = {}
    for part, X, y in (("train", s["X_train"], s["y_train"]), ("test", s["X_test"], s["y_test"])):
        f = pp.transform_to_frame(pre, X)
        f[target] = y.values
        f.insert(0, "split", part)
        frames[part] = f
    frames["both"] = pd.concat([frames["train"], frames["test"]], ignore_index=True)
    ss.processed_cache = {"stamp": res["meta"]["trained_at"], "frames": frames}
    return frames


def _build_zip(res: dict, charts: dict, report_md: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model_comparison.csv", csv_bytes(res["metrics"]))
        z.writestr("feature_set_comparison.csv", csv_bytes(res["experiments"]))
        for m, g in res["grids"].items():
            z.writestr(f"gridsearch_{m.lower()}.csv", csv_bytes(g))
        z.writestr("test_predictions.csv", csv_bytes(res["predictions"], index=True))
        z.writestr("cleaned_dataset.csv", csv_bytes(res["clean_data"]))
        z.writestr("processed_dataset.csv", csv_bytes(_processed_frames(res)["both"]))
        z.writestr("project_summary.md", report_md.encode("utf-8"))
        if st.session_state.prediction_log:
            z.writestr("prediction_results.csv", csv_bytes(pd.DataFrame(st.session_state.prediction_log)))
        for name, data in charts.items():
            z.writestr(f"charts/{name}.png", data)
    return buf.getvalue()


def page_reports() -> None:
    page_header("📥", "Reports", "Download datasets, results, charts and a generated project summary.")
    ss = st.session_state
    res = need_result("reports")
    if res is None:
        return
    meta = res["meta"]
    tabs = st.tabs(["Datasets", "Model results", "Predictions", "Charts", "Project summary"])

    with tabs[0]:
        frames = _processed_frames(res)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Processed dataset** – exactly what the models receive (imputed, engineered, scaled, encoded)")
            part = st.radio("Split", ["both", "train", "test"], horizontal=True, key="proc_part")
            st.caption(f"{frames[part].shape[0]:,} rows × {frames[part].shape[1]} columns")
            st.download_button("⬇️ Download processed dataset (CSV)", csv_bytes(frames[part]), f"processed_dataset_{part}.csv", "text/csv", key="dl_proc")
        with c2:
            st.markdown("**Cleaned dataset** – selected columns after duplicate / missing-target removal (original units)")
            st.caption(f"{res['clean_data'].shape[0]:,} rows × {res['clean_data'].shape[1]} columns")
            st.download_button("⬇️ Download cleaned dataset (CSV)", csv_bytes(res["clean_data"]), "cleaned_dataset.csv", "text/csv", key="dl_clean")
        st.dataframe(frames[part].head(10).round(3))

    with tabs[1]:
        items = [
            ("Model comparison", res["metrics"], "model_comparison.csv", False),
            ("Feature-set comparison", res["experiments"], "feature_set_comparison.csv", False),
            ("GridSearchCV – Ridge", res["grids"]["Ridge"], "gridsearch_ridge.csv", False),
            ("GridSearchCV – Lasso", res["grids"]["Lasso"], "gridsearch_lasso.csv", False),
            ("Test-set predictions (actual + 3 models)", res["predictions"], "test_predictions.csv", True),
            ("Coefficients – best model", res["coefficients"][res["best_model"]], "coefficients_best_model.csv", False),
        ]
        for i, (label, frame, fname, idx) in enumerate(items):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{label}** · {frame.shape[0]:,} rows")
            c2.download_button("⬇️ CSV", csv_bytes(frame, index=idx), fname, "text/csv", key=f"dl_res_{i}")
        st.markdown("**Model comparison preview**")
        st.dataframe(res["metrics"].round(4), hide_index=True)

    with tabs[2]:
        log = pd.DataFrame(ss.prediction_log)
        if log.empty:
            st.info("No predictions saved yet. Make a prediction on the **House Price Prediction** page (button predictions are saved automatically).")
        else:
            st.dataframe(log, hide_index=True)
            c1, c2 = st.columns(2)
            c1.download_button("⬇️ Download prediction results (CSV)", csv_bytes(log), "prediction_results.csv", "text/csv", key="dl_pred")
            if c2.button("🗑️ Clear prediction log", key="clear_log"):
                ss.prediction_log = []
                st.rerun()

    with tabs[3]:
        st.caption("Publication-style PNG versions of the key charts (the dashboard charts themselves are interactive).")
        if st.button("🖼️ Generate charts", key="gen_charts") or ss.charts_cache:
            if not ss.charts_cache or ss.charts_cache.get("_stamp") != meta["trained_at"]:
                with st.spinner("Rendering charts…"):
                    figs = viz.build_all_figures(res)
                    cache = {"_stamp": meta["trained_at"]}
                    for name, fig in figs.items():
                        cache[name] = viz.figure_to_png_bytes(fig)
                        viz.plt.close(fig)
                    ss.charts_cache = cache
            names = [k for k in ss.charts_cache if k != "_stamp"]
            pick = st.selectbox("Preview chart", names, key="chart_pick")
            st.image(ss.charts_cache[pick])
            c1, c2 = st.columns(2)
            c1.download_button("⬇️ Download this chart (PNG)", ss.charts_cache[pick], f"{pick}.png", "image/png", key="dl_chart")
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
                for n in names:
                    z.writestr(f"{n}.png", ss.charts_cache[n])
            c2.download_button("⬇️ Download all charts (ZIP)", zbuf.getvalue(), "charts.zip", "application/zip", key="dl_charts_zip")

    with tabs[4]:
        report_md = ev.build_summary_report(res)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download summary report (Markdown)", report_md.encode("utf-8"), "project_summary.md", "text/markdown", key="dl_report")
        charts = {k: v for k, v in ss.charts_cache.items() if k != "_stamp"}
        c2.download_button("📦 Download everything (ZIP)", _build_zip(res, charts, report_md), "housing_project_report.zip", "application/zip", key="dl_all",
                           help="CSV results, processed data, summary report and (if generated) charts.")
        st.markdown("---")
        st.markdown(report_md)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
PAGE_FUNCS = {
    "Home / Overview": page_home,
    "Data Exploration": page_exploration,
    "Data Preprocessing": page_preprocessing,
    "Model Training": page_training,
    "Model Performance": page_performance,
    "Residual Analysis": page_residuals,
    "House Price Prediction": page_prediction,
    "Reports": page_reports,
}


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    try:
        if not st.session_state.booted:
            with st.spinner("Preparing the California Housing dataset and models…"):
                bootstrap()
        page = render_sidebar()
        PAGE_FUNCS[page]()
    except dl.DataLoadError as exc:
        st.error(f"Data problem: {exc}")
    except Exception as exc:  # noqa: BLE001 - last line of defence for the UI
        st.error(f"Something went wrong on this page: {exc}")
        with st.expander("Technical details"):
            st.exception(exc)


if __name__ == "__main__":
    main()

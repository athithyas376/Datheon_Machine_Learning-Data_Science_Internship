"""NLP Sentiment Classification on Customer Reviews (TF-IDF) - Streamlit dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html as _html
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DATA_DIR, MODEL_LABELS, NEGATIVE, POSITIVE, TEST_SIZE
from predict import (ModelNotFoundError, UNCLASSIFIED, artifacts_token, explain_prediction,
                     load_artifacts, predict_batch, predict_sentiment, preprocessing_mismatch)
from preprocessing import get_backend_info, preprocess_text, tokenize
from train_model import (detect_columns, load_default_dataset, normalize_labels,
                         read_csv_flexible, train_pipeline)

st.set_page_config(page_title="Sentiment Studio | TF-IDF NLP", page_icon="💬",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
POS_COLOR, NEG_COLOR, ACCENT, CYAN = "#10b981", "#f43f5e", "#3b82f6", "#22d3ee"
COLOR_MAP = {POSITIVE: POS_COLOR, NEGATIVE: NEG_COLOR, UNCLASSIFIED: "#64748b"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --text:#e2e8f0; --muted:#94a3b8; --pos:#10b981; --neg:#f43f5e;
  --glass:rgba(255,255,255,.05); --glass-border:rgba(148,163,184,.18);
}
html, body, .stApp{font-family:'Inter',system-ui,sans-serif;}
.stApp{
  background:
    radial-gradient(1100px 560px at 8% -10%, rgba(59,130,246,.25), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,.13), transparent 55%),
    linear-gradient(160deg,#050b1f 0%,#0a1128 45%,#0f1f4d 100%);
  background-attachment:fixed;
}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1250px;}
h1,h2,h3{letter-spacing:-.02em;}

/* Sidebar */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#07102b 0%,#0b1738 100%);
  border-right:1px solid var(--glass-border);
}
section[data-testid="stSidebar"] div[role="radiogroup"]{gap:.3rem;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label{
  width:100%;padding:.65rem .9rem;border-radius:12px;border:1px solid transparent;
  transition:all .2s ease;cursor:pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{
  background:rgba(59,130,246,.12);border-color:rgba(96,165,250,.3);transform:translateX(3px);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child{display:none;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){
  background:linear-gradient(90deg,rgba(59,130,246,.38),rgba(34,211,238,.12));
  border-color:rgba(96,165,250,.55);box-shadow:0 4px 18px rgba(59,130,246,.25);
}
.brand{display:flex;align-items:center;gap:.75rem;padding:.4rem .2rem 1.1rem;}
.brand-logo{width:44px;height:44px;border-radius:14px;display:grid;place-items:center;font-size:1.4rem;
  background:linear-gradient(135deg,#2563eb,#22d3ee);box-shadow:0 8px 24px rgba(37,99,235,.45);}
.brand-title{font-weight:800;font-size:1.1rem;line-height:1.1;}
.brand-sub{color:var(--muted);font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;}
.side-label{color:var(--muted);font-size:.7rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin:1.2rem 0 .4rem;}

/* Glass cards */
.glass{
  background:var(--glass);border:1px solid var(--glass-border);border-radius:18px;padding:1.25rem 1.4rem;
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  box-shadow:0 10px 30px rgba(0,0,0,.28);transition:transform .25s ease, border-color .25s ease, box-shadow .25s ease;
  margin-bottom:1rem;
}
.glass:hover{transform:translateY(-3px);border-color:rgba(96,165,250,.45);box-shadow:0 16px 38px rgba(37,99,235,.22);}
.glass h4{margin:0 0 .6rem;font-size:1.02rem;font-weight:700;}
.glass p, .glass li{color:#cbd5e1;font-size:.93rem;line-height:1.6;}
.glass ul, .glass ol{padding-left:1.2rem;margin:.2rem 0;}

.hero{
  position:relative;overflow:hidden;border-radius:22px;padding:2rem 2.2rem;margin-bottom:1.2rem;
  background:linear-gradient(135deg,rgba(37,99,235,.55),rgba(14,165,233,.22) 55%,rgba(10,17,40,.4));
  border:1px solid rgba(96,165,250,.35);box-shadow:0 18px 50px rgba(37,99,235,.25);
}
.hero:after{content:"";position:absolute;right:-60px;top:-60px;width:260px;height:260px;border-radius:50%;
  background:radial-gradient(circle,rgba(34,211,238,.35),transparent 70%);}
.hero h1{margin:0 0 .4rem;font-size:2rem;font-weight:800;}
.hero p{margin:0 0 1rem;color:#dbeafe;max-width:760px;line-height:1.65;}

.page-title{display:flex;align-items:center;gap:.9rem;margin-bottom:1.1rem;}
.page-title .ic{width:50px;height:50px;border-radius:15px;display:grid;place-items:center;font-size:1.5rem;
  background:linear-gradient(135deg,rgba(37,99,235,.8),rgba(34,211,238,.45));box-shadow:0 8px 22px rgba(37,99,235,.35);}
.page-title h2{margin:0;font-size:1.6rem;font-weight:800;}
.page-title span{color:var(--muted);font-size:.92rem;}

.badge{display:inline-block;padding:.22rem .7rem;margin:0 .35rem .35rem 0;border-radius:999px;font-size:.74rem;font-weight:600;
  background:rgba(59,130,246,.18);border:1px solid rgba(96,165,250,.4);color:#bfdbfe;}
.badge.pos{background:rgba(16,185,129,.16);border-color:rgba(16,185,129,.5);color:#a7f3d0;}
.badge.neg{background:rgba(244,63,94,.16);border-color:rgba(244,63,94,.5);color:#fecdd3;}
.badge.warn{background:rgba(245,158,11,.16);border-color:rgba(245,158,11,.5);color:#fde68a;}

.kpi{display:flex;gap:1rem;align-items:center;padding:1.1rem 1.2rem;border-radius:18px;margin-bottom:1rem;
  background:var(--glass);border:1px solid var(--glass-border);backdrop-filter:blur(14px);
  box-shadow:0 10px 26px rgba(0,0,0,.26);transition:all .25s ease;}
.kpi:hover{transform:translateY(-4px);border-color:rgba(96,165,250,.5);box-shadow:0 16px 36px rgba(37,99,235,.25);}
.kpi-icon{width:48px;height:48px;border-radius:14px;display:grid;place-items:center;font-size:1.4rem;flex:none;
  background:rgba(59,130,246,.2);}
.kpi-pos .kpi-icon{background:rgba(16,185,129,.2);} .kpi-neg .kpi-icon{background:rgba(244,63,94,.2);}
.kpi-cyan .kpi-icon{background:rgba(34,211,238,.2);}
.kpi-label{color:var(--muted);font-size:.75rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;}
.kpi-value{font-size:1.75rem;font-weight:800;line-height:1.15;}
.kpi-sub{color:var(--muted);font-size:.74rem;}

.result{border-radius:20px;padding:1.5rem;text-align:center;border:1px solid;margin-bottom:1rem;animation:pop .35s ease;}
.result.positive{background:linear-gradient(160deg,rgba(16,185,129,.28),rgba(16,185,129,.07));border-color:rgba(16,185,129,.6);}
.result.negative{background:linear-gradient(160deg,rgba(244,63,94,.28),rgba(244,63,94,.07));border-color:rgba(244,63,94,.6);}
.result .big{font-size:2.3rem;font-weight:800;letter-spacing:.02em;}
.result .sub{color:#cbd5e1;font-size:.9rem;}
@keyframes pop{from{opacity:0;transform:scale(.96) translateY(6px);}to{opacity:1;transform:none;}}
.quote{border-left:3px solid #3b82f6;background:rgba(59,130,246,.08);padding:.7rem 1rem;border-radius:0 12px 12px 0;
  color:#dbeafe;font-style:italic;line-height:1.6;}
.kv{display:flex;justify-content:space-between;padding:.45rem 0;border-bottom:1px solid rgba(148,163,184,.12);font-size:.92rem;}
.kv:last-child{border-bottom:none;} .kv span:first-child{color:var(--muted);} .kv span:last-child{font-weight:600;}

.flow{display:flex;flex-wrap:wrap;gap:.5rem;align-items:stretch;margin:.4rem 0 1rem;}
.step{flex:1 1 130px;text-align:center;padding:.9rem .6rem;border-radius:14px;background:rgba(59,130,246,.12);
  border:1px solid rgba(96,165,250,.35);transition:all .25s ease;font-size:.85rem;}
.step:hover{transform:translateY(-4px);background:rgba(59,130,246,.25);}
.step .n{font-size:1.4rem;display:block;margin-bottom:.2rem;}
.arrow{align-self:center;color:#60a5fa;font-size:1.2rem;}

/* Widgets */
.stButton>button, .stDownloadButton>button{
  border-radius:12px;font-weight:600;border:1px solid rgba(96,165,250,.4);background:rgba(59,130,246,.12);
  color:#e2e8f0;transition:all .2s ease;padding:.5rem 1.1rem;}
.stButton>button:hover, .stDownloadButton>button:hover{
  transform:translateY(-2px);border-color:#60a5fa;background:rgba(59,130,246,.3);box-shadow:0 8px 20px rgba(37,99,235,.35);color:#fff;}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"]{
  background:linear-gradient(135deg,#2563eb,#0ea5e9);border:none;color:#fff;box-shadow:0 8px 22px rgba(37,99,235,.45);}
.stTextArea textarea, .stTextInput input{border-radius:14px;background:rgba(15,27,61,.7);border:1px solid var(--glass-border);}
.stTextArea textarea:focus{border-color:#60a5fa;box-shadow:0 0 0 2px rgba(59,130,246,.35);}
div[data-testid="stFileUploader"] section{border-radius:16px;background:rgba(15,27,61,.55);border:1px dashed rgba(96,165,250,.45);}
button[data-baseweb="tab"]{font-weight:600;}
div[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;border:1px solid var(--glass-border);}
div[data-testid="stExpander"]{border-radius:14px;border:1px solid var(--glass-border);background:var(--glass);}
@media (max-width:760px){.hero h1{font-size:1.5rem}.kpi-value{font-size:1.4rem}.block-container{padding-top:1rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------------
_fig_counter = [0]  # reset on each script run; gives every chart a unique key


def H(markup: str) -> str:
    """Strip indentation so Markdown does not treat HTML as a code block."""
    return "\n".join(line.strip() for line in markup.strip().splitlines())


def esc(text: object) -> str:
    return _html.escape(str(text))


def show_html(markup: str) -> None:
    st.markdown(H(markup), unsafe_allow_html=True)


def show_fig(fig: go.Figure, height: int = 340) -> None:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"), margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)")
    _fig_counter[0] += 1
    key = f"fig_{_fig_counter[0]}"
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:  # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(df: pd.DataFrame, height: int | None = None) -> None:
    kwargs = {"hide_index": True}
    if height:
        kwargs["height"] = height
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, **kwargs)


def page_header(icon: str, title: str, subtitle: str) -> None:
    show_html(f'<div class="page-title"><div class="ic">{icon}</div>'
              f'<div><h2>{esc(title)}</h2><span>{esc(subtitle)}</span></div></div>')
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)


def kpi(col, icon: str, label: str, value: str, sub: str = "", tone: str = "blue") -> None:
    col.markdown(H(f'<div class="kpi kpi-{tone}"><div class="kpi-icon">{icon}</div><div>'
                   f'<div class="kpi-label">{esc(label)}</div><div class="kpi-value">{esc(value)}</div>'
                   f'<div class="kpi-sub">{esc(sub)}</div></div></div>'), unsafe_allow_html=True)


def glass(title: str, body: str) -> None:
    show_html(f'<div class="glass"><h4>{title}</h4>{body}</div>')


def sentiment_badge(label: str) -> str:
    cls = "pos" if label == POSITIVE else "neg"
    icon = "😊" if label == POSITIVE else "😞"
    return f'<span class="badge {cls}">{icon} {esc(label.title())}</span>'


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def empty_state(message: str) -> None:
    show_html(f'<div class="glass" style="text-align:center;color:#94a3b8">{message}</div>')


# ----------------------------------------------------------------------------
# Session state, model loading (cached - models are NEVER trained on refresh)
# ----------------------------------------------------------------------------
st.session_state.setdefault("history", [])
st.session_state.setdefault("review_input", "")
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("show_analysis", False)


@st.cache_resource(show_spinner="Loading trained models...")
def _load_cached(token: int) -> dict:
    return load_artifacts()


def get_artifacts() -> dict | None:
    token = artifacts_token()
    if token is None:
        return None
    try:
        return _load_cached(token)
    except (ModelNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        return None


@st.cache_data(show_spinner=False)
def _bundled_dataset() -> tuple[pd.DataFrame, str]:
    return load_default_dataset()


def best_model_key(metrics: dict) -> str:
    models = metrics["models"]
    return max(models, key=lambda k: (models[k]["f1"], models[k]["accuracy"]))


def need_models_notice() -> None:
    show_html("""<div class="glass"><h4>🧠 No trained models found yet</h4>
    <p>Train the models once and they are saved to the <code>models/</code> folder, so the dashboard
    never has to train again. You can either run <code>python train_model.py</code> in a terminal, or
    use the button below to train on the bundled starter dataset.</p></div>""")
    if st.button("🚀 Train on bundled starter dataset", type="primary", key="train_first"):
        run_training_from_bundled(TEST_SIZE)


def run_training_from_bundled(test_size: float) -> None:
    try:
        df, source = _bundled_dataset()
        with st.spinner("Training models..."):
            train_pipeline(df, "review_text", "sentiment", test_size=test_size, source=source)
    except Exception as exc:
        st.error(f"Training failed: {exc}")
        return
    st.cache_resource.clear()
    st.session_state["flash"] = "✅ Models trained and saved. All pages now use the new results."
    st.rerun()


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
NAV = ["🏠  Home Dashboard", "💬  Sentiment Analysis", "📦  Batch Analysis",
       "🔍  Dataset Explorer", "📊  Model Performance", "📘  About Project"]

artifacts = get_artifacts()
metrics = artifacts["metrics"] if artifacts else None

with st.sidebar:
    show_html("""<div class="brand"><div class="brand-logo">💬</div><div>
    <div class="brand-title">Sentiment Studio</div><div class="brand-sub">TF-IDF · NLP · ML</div></div></div>""")
    page = st.radio("Navigation", NAV, label_visibility="collapsed")

    show_html('<div class="side-label">Prediction model</div>')
    default_idx = list(MODEL_LABELS).index(best_model_key(metrics)) if metrics else 1
    model_key = st.selectbox("Model", list(MODEL_LABELS), index=default_idx,
                             format_func=MODEL_LABELS.get, label_visibility="collapsed",
                             help="Which trained model is used on the prediction pages. "
                                  "Defaults to the best model by F1-score.")
    show_html('<div class="side-label">Status</div>')
    if metrics:
        show_html(f'<span class="badge pos">● Models ready</span><br>'
                  f'<span class="kv-sub" style="color:#94a3b8;font-size:.75rem">Trained {esc(metrics["trained_at"])}</span>')
    else:
        show_html('<span class="badge warn">● Not trained yet</span>')

if artifacts:
    mismatch = preprocessing_mismatch(artifacts)
    if mismatch:
        st.warning(mismatch)


# ----------------------------------------------------------------------------
# PAGE 1 - Home
# ----------------------------------------------------------------------------
def page_home() -> None:
    show_html("""<div class="hero"><h1>NLP Sentiment Classification on Customer Reviews</h1>
    <p>An end-to-end machine learning system that cleans raw review text, converts it into
    <b>TF-IDF</b> feature vectors and classifies each review as <b>Positive</b> or <b>Negative</b> using
    <b>Multinomial Naive Bayes</b> and <b>Logistic Regression</b>.</p>
    <span class="badge">Python</span><span class="badge">NLTK</span><span class="badge">TF-IDF</span>
    <span class="badge">Scikit-learn</span><span class="badge">Streamlit</span><span class="badge">Plotly</span></div>""")
    page_flash = st.session_state.pop("flash", None)
    if page_flash:
        st.success(page_flash)

    if not metrics:
        need_models_notice()
        return

    ds, split = metrics["dataset"], metrics["split"]
    best = best_model_key(metrics)
    best_m = metrics["models"][best]
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "📝", "Reviews processed", f"{ds['n_used']:,}", "labelled, cleaned & de-duplicated")
    kpi(c2, "😊", "Positive reviews", f"{ds['class_counts'][POSITIVE]:,}", pct(ds["class_counts"][POSITIVE] / ds["n_used"]) + " of data", "pos")
    kpi(c3, "😞", "Negative reviews", f"{ds['class_counts'][NEGATIVE]:,}", pct(ds["class_counts"][NEGATIVE] / ds["n_used"]) + " of data", "neg")
    kpi(c4, "🎯", "Model accuracy", pct(best_m["accuracy"]), f"{best_m['label']} · test set n={split['n_test']}", "cyan")

    for warning in metrics.get("warnings", [])[:2]:
        st.info(f"ℹ️ {warning}")

    left, right = st.columns([1.1, 1])
    with left:
        fig = go.Figure(go.Pie(labels=[POSITIVE.title(), NEGATIVE.title()],
                               values=[ds["class_counts"][POSITIVE], ds["class_counts"][NEGATIVE]],
                               hole=.62, marker=dict(colors=[POS_COLOR, NEG_COLOR], line=dict(color="#0a1128", width=3)),
                               textinfo="label+percent", sort=False))
        fig.update_layout(title="Sentiment distribution (training data)", showlegend=False)
        show_fig(fig, 360)
    with right:
        backend = metrics["preprocessing_backend"]
        rows = [("Average review length", f"{ds['avg_words']:.1f} words"),
                ("Median / min / max", f"{ds['median_words']:.0f} / {ds['min_words']} / {ds['max_words']} words"),
                ("Training / test split", f"{split['n_train']} / {split['n_test']}"),
                ("TF-IDF vocabulary", f"{metrics['tfidf']['vocab_size']:,} terms"),
                ("Text cleaning", "NLTK" if backend["using_nltk"] else "Built-in fallback"),
                ("Reviews analysed this session", str(len(st.session_state['history'])))]
        glass("📈 Review statistics", "".join(f'<div class="kv"><span>{k}</span><span>{esc(v)}</span></div>' for k, v in rows))
        glass("🗂️ Data source", f"<p>{esc(ds['source'])}</p>")

    st.markdown("#### 🕒 Recent review predictions")
    history = st.session_state["history"]
    if history:
        recent = pd.DataFrame(history[::-1][:10])
        show_table(recent)
    else:
        empty_state("No predictions yet. Open <b>Sentiment Analysis</b> and analyse a review - it will appear here.")


# ----------------------------------------------------------------------------
# PAGE 2 - Sentiment Analysis (single review)
# ----------------------------------------------------------------------------
EXAMPLES = {
    "😊 Loved it": "Absolutely love this product! The quality is excellent and it works flawlessly.",
    "😞 Awful": "Terrible experience. It broke after two days and customer support was useless.",
    "🎬 Great film": "A wonderful, heartwarming movie with brilliant acting. I enjoyed every minute.",
    "🤔 Negation": "This is not good at all, I'm really disappointed with the quality.",
}


def _set_example(text: str) -> None:
    st.session_state["review_input"] = text
    st.session_state["last_result"] = None
    st.session_state["show_analysis"] = False


def _clear_input() -> None:
    st.session_state["review_input"] = ""
    st.session_state["last_result"] = None
    st.session_state["show_analysis"] = False


def _run_single(text: str, analyze: bool) -> None:
    try:
        result = predict_sentiment(text, model_key, artifacts)
        result["explanation"] = explain_prediction(text, model_key, artifacts)
        result["review"] = text.strip()
    except ValueError as exc:
        st.warning(f"⚠️ {exc}")
        return
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return
    st.session_state["last_result"] = result
    st.session_state["show_analysis"] = analyze
    st.session_state["history"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "review": result["review"][:90] + ("…" if len(result["review"]) > 90 else ""),
        "sentiment": result["label"].title(),
        "confidence": pct(result["confidence"]),
        "model": result["model_label"],
    })
    st.session_state["history"] = st.session_state["history"][-100:]


def page_sentiment() -> None:
    page_header("💬", "Sentiment Analysis", "Type or paste a customer review and the model will classify it")
    if not artifacts:
        need_models_notice()
        return

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("**Try an example**")
        ex_cols = st.columns(len(EXAMPLES))
        for col, (name, text) in zip(ex_cols, EXAMPLES.items()):
            col.button(name, key=f"ex_{name}", on_click=_set_example, args=(text,))
        st.text_area("Customer review", key="review_input", height=170,
                     placeholder="e.g. The delivery was fast and the product works perfectly...")
        b1, b2, b3, _ = st.columns([1.4, 1, 1, 1.2])
        predict_clicked = b1.button("🔮 Predict Sentiment", type="primary", key="btn_predict")
        analyze_clicked = b2.button("🧪 Analyze", key="btn_analyze",
                                    help="Predict and show which words drove the decision")
        b3.button("🧹 Clear", key="btn_clear", on_click=_clear_input)
        if predict_clicked or analyze_clicked:
            _run_single(st.session_state["review_input"], analyze=analyze_clicked)

    result = st.session_state["last_result"]
    with right:
        if not result:
            empty_state("🔮<br><br>Your result will appear here.<br>Enter a review and press <b>Predict Sentiment</b>.")
            return
        label = result["label"]
        icon = "😊" if label == POSITIVE else "😞"
        show_html(f'<div class="result {label}"><div class="big">{icon} {label.upper()}</div>'
                  f'<div class="sub">Confidence {pct(result["confidence"])} · {esc(result["model_label"])}</div></div>')
        if result["known_terms"] == 0:
            st.warning("None of the words in this review appear in the model's vocabulary, so this "
                       "prediction is only a guess based on class frequencies. Try a different review "
                       "or train on a larger dataset.")
        elif result["confidence"] < 0.6:
            st.info("The model is unsure about this review (confidence below 60%).")

    # Visualisations
    g1, g2 = st.columns(2)
    color = POS_COLOR if label == POSITIVE else NEG_COLOR
    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=result["confidence"] * 100, number={"suffix": "%", "font": {"size": 38}},
            title={"text": "Prediction confidence"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}, "bgcolor": "rgba(255,255,255,.04)",
                   "steps": [{"range": [0, 60], "color": "rgba(148,163,184,.12)"},
                             {"range": [60, 100], "color": "rgba(59,130,246,.12)"}]}))
        show_fig(gauge, 280)
    with g2:
        probs = result["probabilities"]
        bar = go.Figure(go.Bar(x=[probs[POSITIVE] * 100, probs[NEGATIVE] * 100], y=["Positive", "Negative"],
                               orientation="h", marker_color=[POS_COLOR, NEG_COLOR],
                               text=[pct(probs[POSITIVE]), pct(probs[NEGATIVE])], textposition="inside"))
        bar.update_xaxes(range=[0, 100], title="Probability (%)")
        bar.update_layout(title="Positive vs Negative probability")
        show_fig(bar, 280)

    glass("📄 Review preview", f'<div class="quote">{esc(result["review"][:600])}{"…" if len(result["review"]) > 600 else ""}</div>')

    if st.session_state["show_analysis"]:
        st.markdown("#### 🧪 Why did the model decide this?")
        a1, a2 = st.columns([1, 1.2])
        with a1:
            glass("After preprocessing", f'<p><code>{esc(result["cleaned_text"])}</code></p>'
                  f'<p>Terms found in the TF-IDF vocabulary: <b>{result["known_terms"]}</b></p>')
        with a2:
            expl = result["explanation"]
            if expl:
                df = pd.DataFrame(expl, columns=["term", "contribution"]).iloc[::-1]
                fig = go.Figure(go.Bar(x=df["contribution"], y=df["term"], orientation="h",
                                       marker_color=[POS_COLOR if v > 0 else NEG_COLOR for v in df["contribution"]]))
                fig.update_layout(title="Words pushing towards Positive (green) or Negative (red)")
                show_fig(fig, 320)
            else:
                st.info("No known words to explain.")


# ----------------------------------------------------------------------------
# PAGE 3 - Batch analysis
# ----------------------------------------------------------------------------
def page_batch() -> None:
    page_header("📦", "Batch Review Analysis", "Upload a CSV with many reviews and classify them all at once")
    if not artifacts:
        need_models_notice()
        return

    up_col, opt_col = st.columns([1.4, 1])
    with up_col:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"], key="batch_upload")
    with opt_col:
        use_sample = st.checkbox("Use the bundled demo file instead", value=False,
                                 help="Lets you try the page without your own data.")
        st.caption("Your CSV needs one column with the review text.")

    df = None
    name = ""
    try:
        if uploaded is not None and not use_sample:
            df, name = read_csv_flexible(uploaded), uploaded.name
        elif use_sample:
            df, name = read_csv_flexible(DATA_DIR / "demo_sample_reviews.csv"), "demo_sample_reviews.csv"
    except Exception as exc:
        st.error(f"Could not read that file: {exc}. Please upload a valid CSV.")
        return

    if df is None:
        empty_state("📂<br><br>Upload a CSV (or tick the demo box) to begin.")
        return
    if df.empty:
        st.error("The file has no rows.")
        return

    guess_text, guess_label = detect_columns(df)
    cols = list(df.columns)
    c1, c2 = st.columns(2)
    text_col = c1.selectbox("Review column", cols, index=cols.index(guess_text) if guess_text in cols else 0)
    label_options = ["— none —"] + [c for c in cols if c != text_col]
    label_default = label_options.index(guess_label) if guess_label in label_options else 0
    label_col = c2.selectbox("Actual sentiment column (optional, to measure accuracy)", label_options, index=label_default)

    st.caption(f"Loaded **{name}** — {len(df):,} rows × {len(cols)} columns")
    run = st.button("🚀 Predict sentiments", type="primary", key="btn_batch")

    if run:
        if len(df) > 100_000:
            st.warning("Large file - only the first 100,000 rows will be analysed.")
            df = df.head(100_000)
        with st.spinner(f"Analysing {len(df):,} reviews..."):
            try:
                preds = predict_batch(df[text_col].tolist(), model_key, artifacts)
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")
                return
        result = pd.concat([df.reset_index(drop=True), preds], axis=1)
        for c in ("confidence", "positive_probability", "negative_probability"):
            result[c] = result[c].round(4)
        accuracy = None
        if label_col != "— none —":
            actual = normalize_labels(result[label_col])
            mask = actual.notna() & (result["predicted_sentiment"] != UNCLASSIFIED)
            if mask.any():
                accuracy = (float((actual[mask] == result.loc[mask, "predicted_sentiment"]).mean()), int(mask.sum()))
        st.session_state["batch"] = {"name": name, "df": result, "model": MODEL_LABELS[model_key], "acc": accuracy}

    batch = st.session_state.get("batch")
    if not batch:
        return
    res = batch["df"]
    counts = res["predicted_sentiment"].value_counts()
    n_pos, n_neg, n_un = int(counts.get(POSITIVE, 0)), int(counts.get(NEGATIVE, 0)), int(counts.get(UNCLASSIFIED, 0))

    st.markdown(f"#### Results for `{batch['name']}` · {batch['model']}")
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "📝", "Reviews analysed", f"{len(res):,}", f"{n_un} unclassified (empty text)")
    kpi(k2, "😊", "Positive", f"{n_pos:,}", pct(n_pos / max(len(res), 1)), "pos")
    kpi(k3, "😞", "Negative", f"{n_neg:,}", pct(n_neg / max(len(res), 1)), "neg")
    if batch["acc"]:
        kpi(k4, "🎯", "Agreement with your labels", pct(batch["acc"][0]), f"on {batch['acc'][1]:,} labelled rows", "cyan")
    else:
        kpi(k4, "📊", "Avg. confidence", pct(float(res["confidence"].mean())) if res["confidence"].notna().any() else "–", "", "cyan")

    if batch["acc"] and batch["name"] == "demo_sample_reviews.csv":
        st.caption("⚠️ The demo file is part of the training data, so this agreement score is optimistic. "
                   "Use reviews the model has never seen for a fair check.")
    unknown = int((res["known_terms"] == 0).sum()) - n_un
    if unknown > 0:
        st.info(f"ℹ️ {unknown} review(s) contained no words known to the model; their predictions are unreliable.")

    v1, v2 = st.columns(2)
    with v1:
        pie = go.Figure(go.Pie(labels=[POSITIVE.title(), NEGATIVE.title()], values=[n_pos, n_neg], hole=.55,
                               marker=dict(colors=[POS_COLOR, NEG_COLOR], line=dict(color="#0a1128", width=3))))
        pie.update_layout(title="Predicted sentiment share")
        show_fig(pie, 320)
    with v2:
        valid = res[res["predicted_sentiment"] != UNCLASSIFIED]
        if len(valid):
            hist = px.histogram(valid, x="confidence", color="predicted_sentiment", nbins=20,
                                color_discrete_map=COLOR_MAP, barmode="overlay", opacity=.8)
            hist.update_layout(title="Confidence distribution", xaxis_title="Confidence", yaxis_title="Reviews",
                               legend_title_text="")
            show_fig(hist, 320)

    st.markdown("#### Prediction table")
    choice = st.radio("Filter", ["All", "Positive", "Negative"], horizontal=True, label_visibility="collapsed")
    view = res if choice == "All" else res[res["predicted_sentiment"] == choice.lower()]
    show_table(view, height=380)

    csv_bytes = res.to_csv(index=False).encode("utf-8")
    d1, d2, _ = st.columns([1.3, 1, 2])
    d1.download_button("⬇️ Download predictions (CSV)", csv_bytes, file_name="sentiment_predictions.csv",
                       mime="text/csv", type="primary")
    if d2.button("🗑️ Clear results", key="clear_batch"):
        st.session_state.pop("batch", None)
        st.rerun()


# ----------------------------------------------------------------------------
# PAGE 4 - Dataset explorer
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Cleaning text for word analysis...")
def _top_words(texts: tuple, top_n: int) -> list:
    counter: Counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    return counter.most_common(top_n)


def page_explorer() -> None:
    page_header("🔍", "Dataset Explorer", "Inspect the data: quality, balance, review length and common words")
    source = st.radio("Data source", ["Bundled starter dataset", "Upload my own CSV"], horizontal=True)
    df, label = None, ""
    if source == "Bundled starter dataset":
        try:
            df, label = _bundled_dataset()
            st.caption(f"{label} (the 64-row file is a hand-written *demo* sample)")
        except Exception as exc:
            st.error(str(exc))
            return
    else:
        file = st.file_uploader("Upload dataset (CSV)", type=["csv"], key="explorer_upload")
        if file is None:
            empty_state("📂<br><br>Upload a CSV file to explore it.")
            return
        try:
            df = read_csv_flexible(file)
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")
            return
    if df.empty:
        st.error("The dataset is empty.")
        return

    cols = list(df.columns)
    g_text, g_label = detect_columns(df)
    c1, c2 = st.columns(2)
    text_col = c1.selectbox("Review column", cols, index=cols.index(g_text) if g_text in cols else 0, key="ex_text")
    label_opts = ["— none —"] + [c for c in cols if c != text_col]
    label_col = c2.selectbox("Sentiment column", label_opts,
                             index=label_opts.index(g_label) if g_label in label_opts else 0, key="ex_label")

    texts = df[text_col].astype(str).where(df[text_col].notna(), "")
    n_missing = int(df.isna().sum().sum())
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "🧾", "Total records", f"{len(df):,}", f"{len(cols)} columns")
    kpi(k2, "❓", "Missing values", f"{n_missing:,}", "across all columns", "neg" if n_missing else "pos")
    kpi(k3, "♻️", "Duplicate reviews", f"{int(texts.duplicated().sum()):,}", "identical text", "cyan")
    words = texts.str.split().str.len()
    kpi(k4, "📏", "Avg. length", f"{words.mean():.1f} words", f"max {int(words.max())}")

    tab_prev, tab_miss, tab_dist, tab_len, tab_words = st.tabs(
        ["👀 Preview", "❓ Missing values", "⚖️ Sentiment distribution", "📏 Review length", "🔤 Word frequency"])

    with tab_prev:
        n = st.slider("Rows to preview", 5, min(200, len(df)), min(15, len(df))) if len(df) > 5 else len(df)
        show_table(df.head(n), height=380)
    with tab_miss:
        miss = df.isna().sum().rename("missing").reset_index().rename(columns={"index": "column"})
        miss.columns = ["column", "missing"]
        if miss["missing"].sum() == 0:
            st.success("✅ No missing values in any column.")
        fig = px.bar(miss, x="column", y="missing", color_discrete_sequence=[ACCENT])
        fig.update_layout(title="Missing values per column")
        show_fig(fig, 300)
    with tab_dist:
        if label_col == "— none —":
            st.info("Choose a sentiment column above to see the class distribution.")
        else:
            raw = df[label_col].astype(str).str.strip().str.lower().value_counts().reset_index()
            raw.columns = ["label", "count"]
            cmap = {**COLOR_MAP, "neutral": "#f59e0b"}
            a, b = st.columns(2)
            with a:
                fig = px.bar(raw, x="label", y="count", color="label", color_discrete_map=cmap, text="count")
                fig.update_layout(title="Records per sentiment label", showlegend=False)
                show_fig(fig, 320)
            with b:
                fig = px.pie(raw, names="label", values="count", hole=.55, color="label", color_discrete_map=cmap)
                fig.update_layout(title="Share of each label")
                show_fig(fig, 320)
            dropped = int(normalize_labels(df[label_col]).isna().sum())
            if dropped:
                st.caption(f"ℹ️ {dropped} record(s) are not positive/negative (e.g. neutral) and are ignored during training.")
    with tab_len:
        plot_df = pd.DataFrame({"words": words, "chars": texts.str.len()})
        if label_col != "— none —":
            plot_df["label"] = df[label_col].astype(str).str.strip().str.lower()
        fig = px.histogram(plot_df, x="words", color="label" if "label" in plot_df else None, nbins=30,
                           color_discrete_map=COLOR_MAP, barmode="overlay", opacity=.8)
        fig.update_layout(title="Review length (words)", xaxis_title="Words per review", yaxis_title="Reviews")
        show_fig(fig, 340)
        stats = plot_df[["words", "chars"]].describe().round(1).T.reset_index().rename(columns={"index": "measure"})
        show_table(stats)
    with tab_words:
        a, b = st.columns([1, 1])
        top_n = a.slider("Number of words", 5, 40, 20)
        choice = b.radio("Show words from", ["All reviews", "Positive", "Negative"], horizontal=True,
                         disabled=label_col == "— none —")
        subset = texts
        if label_col != "— none —" and choice != "All reviews":
            norm = normalize_labels(df[label_col])
            subset = texts[norm == choice.lower()]
        subset = subset.head(20_000)
        if subset.empty:
            st.info("No reviews to analyse for this selection.")
        else:
            top = _top_words(tuple(subset), top_n)
            wdf = pd.DataFrame(top, columns=["word", "count"]).iloc[::-1]
            color = {"All reviews": ACCENT, "Positive": POS_COLOR, "Negative": NEG_COLOR}[choice]
            fig = px.bar(wdf, x="count", y="word", orientation="h", color_discrete_sequence=[color])
            fig.update_layout(title=f"Top {top_n} words after cleaning ({choice.lower()})")
            show_fig(fig, max(340, 22 * top_n + 90))
            if len(texts) > 20_000:
                st.caption("Word analysis uses the first 20,000 reviews for speed.")


# ----------------------------------------------------------------------------
# PAGE 5 - Model performance
# ----------------------------------------------------------------------------
def _training_panel() -> None:
    with st.expander("🛠️ Train / retrain the models", expanded=not metrics):
        st.caption("Training only happens when you press the button - never on page refresh. "
                   "Results replace the saved files in the `models/` folder.")
        src = st.radio("Training data", ["Bundled starter dataset", "Upload my own CSV"], horizontal=True, key="train_src")
        test_pct = st.slider("Test set size (%)", 10, 40, int(TEST_SIZE * 100), 5, key="train_test")
        df, text_col, label_col, source = None, "review_text", "sentiment", ""
        if src == "Bundled starter dataset":
            try:
                df, source = _bundled_dataset()
            except Exception as exc:
                st.error(str(exc))
        else:
            file = st.file_uploader("Labelled CSV (review text + sentiment)", type=["csv"], key="train_upload")
            if file is not None:
                try:
                    df = read_csv_flexible(file)
                    g_t, g_l = detect_columns(df)
                    cols = list(df.columns)
                    a, b = st.columns(2)
                    text_col = a.selectbox("Review column", cols, index=cols.index(g_t) if g_t in cols else 0, key="tr_t")
                    label_col = b.selectbox("Sentiment column", cols, index=cols.index(g_l) if g_l in cols else min(1, len(cols) - 1), key="tr_l")
                    source = f"Uploaded dataset: {file.name}"
                except Exception as exc:
                    st.error(f"Could not read that file: {exc}")
                    df = None
        if st.button("🚀 Train models", type="primary", key="btn_train", disabled=df is None):
            try:
                with st.spinner("Cleaning text, extracting TF-IDF features and training both models..."):
                    train_pipeline(df, text_col, label_col, test_size=test_pct / 100, source=source)
            except ValueError as exc:
                st.error(f"⚠️ {exc}")
            except Exception as exc:
                st.error(f"Training failed unexpectedly: {exc}")
            else:
                st.cache_resource.clear()
                st.session_state["flash"] = "✅ Training finished. The numbers below are from the new models."
                st.rerun()


def page_performance() -> None:
    page_header("📊", "Model Performance", "Real evaluation results measured on a held-out test set")
    if not metrics:
        need_models_notice()
        _training_panel()
        return

    models = metrics["models"]
    split, ds = metrics["split"], metrics["dataset"]
    best = best_model_key(metrics)
    for warning in metrics.get("warnings", []):
        st.warning(f"⚠️ {warning}")

    show_html(f'<span class="badge">Train: {split["n_train"]} reviews</span>'
              f'<span class="badge">Test: {split["n_test"]} reviews</span>'
              f'<span class="badge">TF-IDF terms: {metrics["tfidf"]["vocab_size"]:,}</span>'
              f'<span class="badge pos">Best by F1: {esc(models[best]["label"])}</span>')

    # Metrics table + comparison chart
    rows = []
    for key, m in models.items():
        row = {"Model": m["label"], "Accuracy": m["accuracy"], "Precision (macro)": m["precision"],
               "Recall (macro)": m["recall"], "F1-score (macro)": m["f1"], "Train accuracy": m["train_accuracy"]}
        if "cv_accuracy_mean" in m:
            row["5-fold CV accuracy"] = m["cv_accuracy_mean"]
        rows.append(row)
    table = pd.DataFrame(rows)

    st.markdown("#### 🏁 Model comparison")
    kcols = st.columns(len(models))
    for col, (key, m) in zip(kcols, models.items()):
        kpi(col, "🏆" if key == best else "🤖", m["label"], pct(m["accuracy"]),
            f"F1 {m['f1']:.3f} · P {m['precision']:.3f} · R {m['recall']:.3f}", "cyan" if key == best else "blue")

    long = table.melt(id_vars="Model", value_vars=["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"],
                      var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", text_auto=".3f",
                 color_discrete_sequence=[ACCENT, CYAN])
    fig.update_layout(title="Test-set metrics", yaxis_range=[0, 1.05], legend_title_text="")
    show_fig(fig, 360)

    shown = table.copy()
    for c in shown.columns[1:]:
        shown[c] = shown[c].map(lambda v: f"{v:.3f}")
    show_table(shown)

    tab_cm, tab_rep, tab_tt, tab_feat = st.tabs(["🧩 Confusion matrices", "📋 Classification reports",
                                                  "🔁 Training vs testing", "⭐ Influential words"])
    with tab_cm:
        cols = st.columns(len(models))
        for col, (key, m) in zip(cols, models.items()):
            with col:
                cm = np.array(m["confusion_matrix"])
                labels = [NEGATIVE.title(), POSITIVE.title()]
                fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                                labels=dict(x="Predicted", y="Actual", color="Reviews"))
                fig.update_layout(title=m["label"], coloraxis_showscale=False)
                show_fig(fig, 340)
        st.caption("Rows = actual sentiment, columns = predicted sentiment. The diagonal shows correct predictions.")
    with tab_rep:
        for key, m in models.items():
            st.markdown(f"**{m['label']}**")
            rep = m["report"]
            rr = pd.DataFrame([{"class": k, "precision": v["precision"], "recall": v["recall"],
                                "f1-score": v["f1-score"], "support": int(v["support"])}
                               for k, v in rep.items() if isinstance(v, dict)])
            rr[["precision", "recall", "f1-score"]] = rr[["precision", "recall", "f1-score"]].round(3)
            show_table(rr)
    with tab_tt:
        tt = pd.DataFrame([{"Model": m["label"], "Split": s, "Accuracy": v}
                           for m in models.values() for s, v in (("Training", m["train_accuracy"]), ("Testing", m["accuracy"]))])
        fig = px.bar(tt, x="Model", y="Accuracy", color="Split", barmode="group", text_auto=".3f",
                     color_discrete_sequence=["#6366f1", CYAN])
        fig.update_layout(title="Training vs testing accuracy", yaxis_range=[0, 1.05], legend_title_text="")
        show_fig(fig, 340)
        st.caption("A large gap (high training accuracy, lower testing accuracy) means the model has "
                   "memorised the training reviews - a sign of **overfitting**, common with small datasets.")
        if any("cv_accuracy_mean" in m for m in models.values()):
            cv = [f"{m['label']}: {pct(m['cv_accuracy_mean'])} ± {pct(m['cv_accuracy_std'])}"
                  for m in models.values() if "cv_accuracy_mean" in m]
            st.markdown("**5-fold cross-validation accuracy** (more stable than one small split): " + " · ".join(cv))
    with tab_feat:
        tf = metrics["top_features"]
        a, b = st.columns(2)
        for col, side, color in ((a, "positive", POS_COLOR), (b, "negative", NEG_COLOR)):
            d = pd.DataFrame(tf[side], columns=["term", "weight"]).iloc[::-1]
            fig = go.Figure(go.Bar(x=d["weight"], y=d["term"], orientation="h", marker_color=color))
            fig.update_layout(title=f"Strongest {side} terms (Logistic Regression)")
            with col:
                show_fig(fig, 460)

    _training_panel()


# ----------------------------------------------------------------------------
# PAGE 6 - About
# ----------------------------------------------------------------------------
def page_about() -> None:
    page_header("📘", "About the Project", "Problem, methodology and how the system works")
    glass("🎯 Project overview",
          "<p>This project builds a complete NLP sentiment classifier for customer product and movie reviews. "
          "Raw text is cleaned, converted into numerical TF-IDF features and fed to two classic machine-learning "
          "models that decide whether a review is <b>positive</b> or <b>negative</b>. The dashboard lets you test "
          "single reviews, score whole CSV files, explore datasets and inspect real evaluation metrics.</p>")
    c1, c2 = st.columns(2)
    with c1:
        glass("❓ Problem statement",
              "<p>Companies receive thousands of reviews and cannot read them all. Automatically detecting whether "
              "customers are happy or unhappy helps prioritise complaints, track satisfaction and improve products.</p>")
    with c2:
        glass("✅ Objectives", "<ul><li>Clean and normalise raw review text</li><li>Extract TF-IDF features</li>"
              "<li>Train Multinomial Naive Bayes &amp; Logistic Regression</li><li>Evaluate honestly on unseen data</li>"
              "<li>Provide an interactive prediction helper</li></ul>")

    st.markdown("#### 🔄 System workflow")
    steps = [("📥", "Load reviews"), ("🧹", "Clean text"), ("✂️", "Split train / test"), ("🔢", "TF-IDF vectors"),
             ("🤖", "Train models"), ("📊", "Evaluate"), ("💾", "Save models"), ("🔮", "Predict")]
    flow = '<div class="arrow">➜</div>'.join(f'<div class="step"><span class="n">{i}</span>{t}</div>' for i, t in steps)
    show_html(f'<div class="flow">{flow}</div>')

    c1, c2 = st.columns(2)
    with c1:
        glass("🧹 NLP preprocessing steps",
              "<ol><li><b>Remove HTML tags</b> - e.g. <code>&lt;br /&gt;</code></li><li><b>Lowercase</b> - 'Great' = 'great'</li>"
              "<li><b>Remove URLs &amp; punctuation</b></li><li><b>Expand contractions</b> - don't → do not</li>"
              "<li><b>Remove stopwords</b> - the, is, and… (negations like <i>not</i> are kept)</li>"
              "<li><b>Lemmatization</b> - movies → movie, loved → love</li></ol>")
    with c2:
        info = get_backend_info()
        glass("🧰 Methodology",
              "<ul><li>Stratified 80/20 train-test split (same class balance in both)</li>"
              "<li>TF-IDF is fitted on the <b>training data only</b> to avoid data leakage</li>"
              "<li>Metrics are computed on the unseen test set</li>"
              "<li>Models are trained once, saved with <code>joblib</code> and re-loaded for predictions</li>"
              f"<li>Current text-cleaning backend: <b>{esc('NLTK' if info['using_nltk'] else 'built-in fallback')}</b></li></ul>")

    glass("🔢 TF-IDF explained",
          "<p><b>TF-IDF</b> (Term Frequency - Inverse Document Frequency) turns text into numbers. A word gets a high "
          "score when it appears often in one review (<b>TF</b>) but rarely across all reviews (<b>IDF</b>), so "
          "distinctive words like <i>excellent</i> or <i>broken</i> count more than common ones like <i>product</i>.</p>")
    st.latex(r"\text{TF-IDF}(t,d)=\text{TF}(t,d)\times \log\!\left(\frac{1+N}{1+\text{df}(t)}\right)+\text{TF}(t,d)")
    st.caption("scikit-learn's smoothed variant, where N = number of reviews and df(t) = reviews containing term t. "
               "Vectors are L2-normalised and limited to the 5,000 most frequent terms (unigrams + bigrams).")

    c1, c2 = st.columns(2)
    with c1:
        glass("📐 Multinomial Naive Bayes",
              "<p>A probabilistic model based on Bayes' theorem. It assumes words are independent and learns how "
              "likely each word is in positive vs negative reviews. It is extremely fast and works well with "
              "word-count style features such as TF-IDF.</p>")
    with c2:
        glass("📈 Logistic Regression",
              "<p>A linear model that learns a weight for every term and turns the weighted sum into a probability "
              "with the sigmoid function. Weights are easy to interpret - positive weights mean positive sentiment.</p>")
    glass("⚠️ Limitations",
          "<ul><li>Bag-of-words ignores word order beyond 2-word phrases, so sarcasm and complex negation can be misread.</li>"
          "<li>Only positive and negative classes are supported (neutral reviews are ignored).</li>"
          "<li>Accuracy depends strongly on the amount and quality of training data - the bundled files are tiny; "
          "train on a large dataset such as IMDB 50K for dependable results.</li></ul>")


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
PAGES = {NAV[0]: page_home, NAV[1]: page_sentiment, NAV[2]: page_batch,
         NAV[3]: page_explorer, NAV[4]: page_performance, NAV[5]: page_about}

try:
    PAGES[page]()
except Exception as exc:  # last line of defence so the app never shows a raw crash
    st.error("Something went wrong while rendering this page.")
    with st.expander("Technical details"):
        st.exception(exc)

"""Model evaluation, best-model selection, residual analysis and report text.

All numbers are computed from real model predictions – nothing is hard-coded.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config as C


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def coefficient_table(pipeline) -> pd.DataFrame:
    """Coefficients of a fitted preprocessor+linear-model pipeline."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coef = np.ravel(pipeline.named_steps["model"].coef_)
    out = pd.DataFrame({"feature": [str(n) for n in names], "coefficient": coef})
    out["abs_coefficient"] = out.coefficient.abs()
    return out.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def evaluate_models(
    models: Dict[str, Any],
    X_train, y_train, X_test, y_test,
    cv_rmse: Optional[Dict[str, float]] = None,
    best_params: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Comparison table with train/test RMSE, MAE, R² and cross-validated RMSE."""
    rows = []
    for name in C.MODEL_NAMES:
        mdl = models[name]
        tr = regression_metrics(y_train, mdl.predict(X_train))
        te = regression_metrics(y_test, mdl.predict(X_test))
        coefs = coefficient_table(mdl)
        bp = (best_params or {}).get(name, {})
        rows.append(
            {
                "Model": name,
                "Best Params": f"alpha={bp['alpha']:.6g}" if "alpha" in bp else "—",
                "CV RMSE": (cv_rmse or {}).get(name, np.nan),
                "Train RMSE": tr["RMSE"],
                "Train MAE": tr["MAE"],
                "Train R²": tr["R2"],
                "Test RMSE": te["RMSE"],
                "Test MAE": te["MAE"],
                "Test R²": te["R2"],
                "R² Gap (Train−Test)": tr["R2"] - te["R2"],
                "Non-zero Coefs": int((coefs.coefficient.abs() > 1e-10).sum()),
                "Total Coefs": int(len(coefs)),
            }
        )
    return pd.DataFrame(rows)


def select_best_model(metrics: pd.DataFrame) -> Tuple[str, str]:
    """Pick the model with the lowest *test* RMSE and explain the choice."""
    ranked = metrics.sort_values("Test RMSE").reset_index(drop=True)
    best, others = ranked.iloc[0], ranked.iloc[1:]
    runner = others.iloc[0]
    rel = 100 * (runner["Test RMSE"] - best["Test RMSE"]) / runner["Test RMSE"]
    parts = [
        f"**{best['Model']}** is the best model: it has the lowest test-set RMSE "
        f"({best['Test RMSE']:.4f}), together with MAE {best['Test MAE']:.4f} and R² {best['Test R²']:.4f}.",
        f"The runner-up, {runner['Model']}, has RMSE {runner['Test RMSE']:.4f}"
        + (f" – the winner is {rel:.2f}% lower." if rel >= 0.05 else " – the two are tied to within 0.05%."),
    ]
    if rel < 1.0:
        parts.append(
            "The gap is below 1%, so the models are practically equivalent in accuracy; "
            f"{best['Model']} wins on the stated criterion (lowest test RMSE)."
        )
    gap = best["R² Gap (Train−Test)"]
    parts.append(
        f"Its train–test R² gap is {gap:+.4f} and its cross-validated RMSE is {best['CV RMSE']:.4f}, "
        + ("which indicates good generalisation (no sign of over-fitting)." if abs(gap) < 0.03 else "so keep an eye on over-fitting.")
    )
    if best["Model"] == "Lasso":
        parts.append(
            f"Lasso also performs feature selection: {int(best['Non-zero Coefs'])} of "
            f"{int(best['Total Coefs'])} coefficients are non-zero, giving a more compact model."
        )
    elif best["Model"] == "Ridge":
        parts.append(
            "Ridge shrinks correlated coefficients towards zero, which stabilises estimates when "
            "features are collinear."
        )
    else:
        parts.append(
            "With plenty of observations relative to the number of features, the un-penalised OLS fit "
            "has little variance to remove, so regularisation did not improve on it."
        )
    return str(best["Model"]), " ".join(parts)


# --------------------------------------------------------------------------- #
# Residual analysis
# --------------------------------------------------------------------------- #
def residual_by_decile(actual, predicted, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"actual": np.asarray(actual, float), "pred": np.asarray(predicted, float)})
    df["residual"] = df.actual - df.pred
    df["bin"] = pd.qcut(df.pred, q=n_bins, duplicates="drop")
    g = df.groupby("bin", observed=True)
    out = g.agg(mean_pred=("pred", "mean"), mean_residual=("residual", "mean"), std_residual=("residual", "std"), n=("residual", "size")).reset_index(drop=True)
    out.insert(0, "decile", range(1, len(out) + 1))
    return out


def residual_analysis(actual, predicted) -> Dict[str, Any]:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    r = a - p
    sd = float(r.std(ddof=1))
    dec = residual_by_decile(a, p)
    rho, rho_p = stats.spearmanr(np.abs(r), p)
    try:
        norm_p = float(stats.normaltest(r).pvalue)
    except Exception:  # noqa: BLE001
        norm_p = float("nan")
    return {
        "mean": float(r.mean()),
        "std": sd,
        "skew": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "normality_p": norm_p,
        "corr_resid_pred": float(np.corrcoef(r, p)[0, 1]),
        "hetero_spearman": float(rho),
        "hetero_p": float(rho_p),
        "decile_bias_index": float(dec.mean_residual.abs().max() / sd) if sd > 0 else 0.0,
        "share_beyond_2sd": float((np.abs(r - r.mean()) > 2 * sd).mean()),
        "max_over_prediction": float(r.min()),
        "max_under_prediction": float(r.max()),
        "decile_table": dec,
    }


def interpret_residuals(st: Dict[str, Any], actual, predicted, target_cap: Optional[Dict[str, float]] = None) -> List[str]:
    """Plain-language reading of the residual diagnostics (generated from numbers)."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    dec = st["decile_table"]
    notes: List[str] = []
    sd = st["std"]

    reasons = []
    if st["decile_bias_index"] > 0.25:
        reasons.append("the mean residual differs between low and high predicted prices")
    if abs(st["hetero_spearman"]) > 0.15 and st["hetero_p"] < 0.01:
        reasons.append("error variance grows with the predicted value (heteroscedasticity)")
    if target_cap and (p > target_cap["value"]).any():
        reasons.append("the capped target cannot be reproduced by a linear model")
    if abs(st["skew"]) > 0.5 or st["excess_kurtosis"] > 1:
        reasons.append("the residuals are skewed / heavy-tailed")
    if reasons:
        notes.append("**Verdict: the residuals are not purely random noise.** Systematic effects remain because "
                     + "; ".join(reasons) + ".")
    else:
        notes.append("**Verdict: the residuals look like unstructured noise** – no systematic error pattern was detected.")

    if abs(st["mean"]) < 0.02 * sd:
        notes.append(f"Overall bias is negligible (mean residual {st['mean']:+.4f}).")
    else:
        notes.append(f"Mean residual is {st['mean']:+.4f}: predictions are on average "
                     f"{'too low' if st['mean'] > 0 else 'too high'}.")

    lo, hi = dec.iloc[0], dec.iloc[-1]
    if st["decile_bias_index"] > 0.25:
        notes.append(
            f"Systematic structure is visible: the lowest-priced predictions have mean residual "
            f"{lo.mean_residual:+.3f} while the highest have {hi.mean_residual:+.3f} "
            f"(largest decile bias = {st['decile_bias_index']:.2f}σ). The model is not equally accurate across the price range."
        )
    else:
        notes.append(
            f"Residuals are centred around zero in every prediction decile (largest decile bias = "
            f"{st['decile_bias_index']:.2f}σ), so there is no strong systematic over/under-prediction pattern."
        )
    if abs(st["hetero_spearman"]) > 0.15 and st["hetero_p"] < 0.01:
        notes.append(
            f"Error size changes with the predicted value (Spearman ρ between |residual| and prediction = "
            f"{st['hetero_spearman']:+.2f}) – evidence of heteroscedasticity."
        )
    else:
        notes.append(
            f"Error size is roughly constant across predictions (Spearman ρ = {st['hetero_spearman']:+.2f})."
        )
    if abs(st["skew"]) > 0.5 or st["excess_kurtosis"] > 1:
        notes.append(
            f"The residual distribution is non-normal (skew {st['skew']:+.2f}, excess kurtosis "
            f"{st['excess_kurtosis']:.2f}): a few large errors dominate RMSE. Confidence intervals based on normal errors would be unreliable."
        )
    else:
        notes.append(f"The residual distribution is close to symmetric (skew {st['skew']:+.2f}).")
    if target_cap:
        above = int((p > target_cap["value"]).sum())
        notes.append(
            f"The target is capped at {target_cap['value']:.4g} ({100*target_cap['share']:.1f}% of rows sit exactly at the cap). "
            f"Linear models cannot represent this censoring, which produces the large positive residuals at the top of the "
            f"actual-vs-predicted plot. {above} test predictions exceed the cap."
        )
    below0 = int((p < 0).sum())
    if below0:
        notes.append(f"{below0} prediction{'s are' if below0 != 1 else ' is'} negative, which is impossible for a price – a linear model does not bound its output.")
    return notes


# --------------------------------------------------------------------------- #
# Report text
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_summary_report(result: Dict[str, Any]) -> str:
    """Markdown project summary generated from a training result."""
    m, met = result["meta"], result["metrics"]
    show = met[["Model", "Best Params", "CV RMSE", "Test RMSE", "Test MAE", "Test R²", "Non-zero Coefs"]]
    exp = result["experiments"][
        ["feature_set", "n_features_after_encoding", "best_model_by_cv", "best_cv_rmse", "best_test_rmse", "best_test_r2"]
    ]
    best = result["best_model"]
    st = result["residual_stats"][best]
    lines = [
        f"# Project Summary – {m['dataset_name']}",
        "",
        f"*Generated: {m['trained_at']}  ·  scikit-learn {m['sklearn_version']}  ·  random_state={m['random_state']}*",
        "",
        "## 1. Dataset",
        f"- Target: **{m['target']}**",
        f"- Rows (raw → used): {m['n_rows_raw']:,} → {m['n_rows_used']:,}  "
        f"(duplicates removed: {m['n_duplicates_removed']}, rows without target removed: {m['n_missing_target_removed']})",
        f"- Features: {len(m['numeric_features'])} numeric, {len(m['categorical_features'])} categorical",
        f"- Split: {m['n_train']:,} train / {m['n_test']:,} test ({int(round((1 - m['test_size']) * 100))}:{int(round(m['test_size'] * 100))}), "
        f"{m['cv_folds']}-fold CV inside the training set",
        "",
        "## 2. Preprocessing",
        "- Numeric: median imputation → feature engineering (if selected) → StandardScaler",
        "- Categorical: most-frequent imputation → OneHotEncoder",
        "- All statistics are fitted on the training data only (scikit-learn Pipeline + ColumnTransformer).",
        "",
        "## 3. Feature-set comparison (selected by cross-validation on the training set)",
        _md_table(exp),
        "",
        f"Selected feature set: **{m['feature_set']}**",
        "",
        "## 4. Model comparison (test set)",
        _md_table(met[["Model", "Best Params", "CV RMSE", "Test RMSE", "Test MAE", "Test R²", "Non-zero Coefs"]]),
        "",
        "## 5. Best model",
        result["best_model_explanation"],
        "",
        f"## 6. Residual analysis ({best})",
        *[f"- {line}" for line in result["residual_interpretation"][best]],
        "",
        "## 7. Limitations",
        "- Linear models cannot capture strong non-linearities or spatial effects beyond the engineered features.",
        "- Results describe this dataset; predictions outside the training range are extrapolations.",
        "- Only one random train/test split is reported (cross-validation quantifies the variability).",
    ]
    return "\n".join(lines) + "\n"

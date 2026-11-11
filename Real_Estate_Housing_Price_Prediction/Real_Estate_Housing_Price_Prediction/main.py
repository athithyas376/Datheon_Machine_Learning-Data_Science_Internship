"""Command-line entry point: run the complete training pipeline.

    python main.py                       # California Housing (data/housing.csv)
    python main.py --data my.csv --target price
    python main.py --cv 10

Outputs
    models/*.pkl                 trained OLS, Ridge, Lasso pipelines (+ metadata, dashboard bundle)
    outputs/metrics/*.csv|json   comparison tables, GridSearchCV results, summary
    outputs/charts/*.png         diagnostic and evaluation charts
    outputs/reports/             processed dataset + Markdown summary report
    assets/dashboard_images/     preview images of key charts
"""
from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNG files only

from sklearn.exceptions import ConvergenceWarning  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import model_training as mt  # noqa: E402
from src import prediction as pr  # noqa: E402
from src import visualization as viz  # noqa: E402
from src.preprocessing import transform_to_frame  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train OLS / Ridge / Lasso housing price models.")
    p.add_argument("--data", type=str, default=None, help="CSV path (default: bundled California Housing)")
    p.add_argument("--target", type=str, default=None, help="Target column for a custom CSV")
    p.add_argument("--cv", type=int, default=C.CV_FOLDS, help="Cross-validation folds (default 5)")
    p.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    return p.parse_args(argv)


def log_progress(frac: float, msg: str) -> None:
    print(f"[{frac*100:5.1f}%] {msg}", flush=True)


def export_reports(result) -> None:
    """Processed dataset, comparison tables and the Markdown summary."""
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    s = result["splits"]
    pre = result["models"]["OLS"].named_steps["preprocessor"]
    target = result["meta"]["target"]
    for part, X, y in (("train", s["X_train"], s["y_train"]), ("test", s["X_test"], s["y_test"])):
        frame = transform_to_frame(pre, X)
        frame[target] = y.values
        frame.insert(0, "split", part)
        frame.to_csv(C.REPORTS_DIR / f"processed_{part}.csv", index=False)
    result["clean_data"].to_csv(C.REPORTS_DIR / "cleaned_dataset.csv", index=False)
    (C.REPORTS_DIR / "project_summary.md").write_text(ev.build_summary_report(result), encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.data:
            df = dl.read_csv_source(args.data)
            if dl.is_california_schema(df):
                name = "California Housing"
            else:
                name = Path(args.data).stem
        else:
            df = dl.load_california_housing()
            name = "California Housing"
        target = args.target or dl.guess_target(df)
        print(f"Dataset: {name}  shape={df.shape}  target={target}")
        check = dl.validate_dataset(df, target)
        for w in check["warnings"]:
            print("  warning:", w)
        if not check["ok"]:
            for e in check["errors"]:
                print("  ERROR:", e)
            return 2

        result = mt.train_project(df, target, dataset_name=name, cv=args.cv, progress=log_progress)
    except dl.DataLoadError as exc:
        print(f"Data error: {exc}")
        return 2

    paths = mt.save_artifacts(result)
    export_reports(result)

    if not args.no_charts:
        charts = viz.save_all_charts(result)
        C.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        for key in ("07_actual_vs_predicted", "09_residual_vs_predicted", "11_model_performance", "05_feature_set_comparison"):
            if key in charts:
                shutil.copy(charts[key], C.ASSETS_DIR / Path(charts[key]).name)
        print(f"Saved {len(charts)} charts to {C.CHARTS_DIR}")

    # ---------------- console summary -----------------
    import pandas as pd

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== Feature-set comparison (selected by CV on training data) ===")
    print(result["experiments"][["feature_set", "n_features_after_encoding", "best_model_by_cv", "best_cv_rmse", "best_test_rmse", "best_test_r2"]].to_string(index=False))
    print(f"\nSelected feature set: {result['meta']['feature_set']}")
    print("\n=== Model comparison ===")
    print(result["metrics"][["Model", "Best Params", "CV RMSE", "Test RMSE", "Test MAE", "Test R²"]].round(4).to_string(index=False))
    print("\nBest model:", result["best_model"])
    print(result["best_model_explanation"].replace("**", ""))
    print("\nResidual analysis (best model):")
    for line in result["residual_interpretation"][result["best_model"]]:
        print("  -", line.replace("**", ""))
    print("\nSaved models:", ", ".join(Path(p).name for p in paths.values()))
    print(f"Total time: {result['meta']['train_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

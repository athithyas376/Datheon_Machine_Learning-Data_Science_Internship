"""Train, validate and save the Iris models.

Usage:   python train_model.py

What it does:
  1. Loads the Iris data and makes an 80/20 stratified split (random_state=42).
  2. Trains Logistic Regression and Decision Tree pipelines (StandardScaler inside) on the TRAINING data only.
  3. Checks both models on the held-out TEST data. If any check fails, NOTHING is saved.
  4. Saves the fitted pipelines with joblib to models/ and writes evaluation files to results/.
"""
import json
import sys

from src import config
from src.data import load_iris_df, split_data
from src.modeling import (ModelValidationError, confusion, misclassified_samples, report_table,
                          train_and_save)


def main() -> int:
    try:
        result = train_and_save()
    except ModelValidationError as error:
        print(f"TRAINING FAILED - nothing was saved.\n  Reason: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # any unexpected problem: do not save anything
        print(f"UNEXPECTED ERROR - nothing was saved.\n  {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    # Evaluation files (re-created from the models that were just trained)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_iris_df()
    _, X_test, _, y_test = split_data(df)

    result.metrics.round(4).to_csv(config.RESULTS_DIR / "model_comparison.csv")
    result.cv.round(4).to_csv(config.RESULTS_DIR / "cross_validation_results.csv")
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(
        {"test_metrics": result.metadata["metrics"], "cross_validation": result.metadata["cross_validation"],
         "best_model": result.metadata["best_model"], "reason": result.metadata["best_model_reason"]}, indent=2))

    for name, pipe in result.pipelines.items():
        slug = name.lower().replace(" ", "_")
        report_table(pipe, X_test, y_test).to_csv(config.RESULTS_DIR / f"classification_report_{slug}.csv")
        misclassified_samples(pipe, X_test, y_test).to_csv(config.RESULTS_DIR / f"misclassified_{slug}.csv", index=False)
        print(f"\n{name} confusion matrix (rows = true, columns = predicted: {config.CLASS_NAMES}):")
        print(confusion(pipe, X_test, y_test))

    print("\nTest-set metrics:")
    print(result.metrics.round(4).to_string())
    print("\n5-fold cross-validation:")
    print(result.cv.round(4).to_string())
    print("\n" + result.metadata["best_model_reason"])
    print(f"\nSaved models to: {config.MODELS_DIR}")
    print(f"Saved results to: {config.RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

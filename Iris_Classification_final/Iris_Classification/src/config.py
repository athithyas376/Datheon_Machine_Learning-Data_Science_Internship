"""Project-wide settings. Every other file imports from here, so values are defined once."""
from pathlib import Path

# ---- Folders (relative to the project root, so the project works from any location) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---- Data ----
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEATURE_LABELS = {
    "sepal_length": "Sepal length (cm)",
    "sepal_width": "Sepal width (cm)",
    "petal_length": "Petal length (cm)",
    "petal_width": "Petal width (cm)",
}
CLASS_NAMES = ["setosa", "versicolor", "virginica"]  # index 0, 1, 2 in the target column
CLASS_COLORS = {"setosa": "#2a9d8f", "versicolor": "#e8833a", "virginica": "#4a6fa5"}

# ---- Training settings (same values as the notebook) ----
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DECISION_TREE_MAX_DEPTH = 3

# ---- Model files ----
MODEL_NAMES = ["Logistic Regression", "Decision Tree"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
}
METADATA_FILE = "model_metadata.json"

# A model is only saved if its test accuracy is at least this value.
# (Random guessing on 3 balanced classes gives about 0.33, so 0.80 catches a broken model.)
MIN_ACCEPTABLE_ACCURACY = 0.80

# ---- Input validation ----
MAX_MEASUREMENT_CM = 15.0     # hard upper limit for any measurement (Iris flowers are far smaller)
MIN_MEASUREMENT_CM = 0.1      # measurements must be positive
MAX_BATCH_ROWS = 10_000       # largest CSV accepted in the Batch Prediction page
MAX_UPLOAD_MB = 5

METRIC_COLUMNS = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]

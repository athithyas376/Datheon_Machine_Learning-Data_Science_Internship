"""Shared configuration for the sentiment classification project.

Keeping paths, label names and model settings in one place means that
``train_model.py``, ``predict.py`` and ``app.py`` always agree with each other.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Datasets shipped with the project (merged for the default training run).
#   sentiment_reviews.csv    -> the small file supplied with the task (10 rows)
#   demo_sample_reviews.csv  -> a hand-written DEMO sample (64 rows) so metrics
#                               can be computed out of the box
BUNDLED_DATASETS = ["sentiment_reviews.csv", "demo_sample_reviews.csv"]

# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------
POSITIVE = "positive"
NEGATIVE = "negative"
LABELS = [NEGATIVE, POSITIVE]  # fixed order used for confusion matrices

# ----------------------------------------------------------------------------
# Saved artifacts
# ----------------------------------------------------------------------------
ARTIFACT_FILES = {
    "vectorizer": "tfidf_vectorizer.joblib",
    "multinomial_nb": "multinomial_nb.joblib",
    "logistic_regression": "logistic_regression.joblib",
    "metrics": "metrics.json",
}

MODEL_LABELS = {
    "multinomial_nb": "Multinomial Naive Bayes",
    "logistic_regression": "Logistic Regression",
}
DEFAULT_MODEL = "logistic_regression"

# ----------------------------------------------------------------------------
# Training settings
# ----------------------------------------------------------------------------
TFIDF_PARAMS = {
    "max_features": 5000,
    "ngram_range": (1, 2),  # unigrams + bigrams (captures phrases like "not good")
    "sublinear_tf": True,
    "min_df": 1,
}
TEST_SIZE = 0.20
RANDOM_STATE = 42
MIN_LABELLED_ROWS = 10
MIN_PER_CLASS = 3

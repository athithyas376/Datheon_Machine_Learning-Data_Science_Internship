"""Central configuration: paths, constants, schema mappings and palette."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_DIR = OUTPUTS_DIR / "metrics"
ASSETS_DIR = PROJECT_ROOT / "assets" / "dashboard_images"
DEFAULT_DATA_FILE = DATA_DIR / "housing.csv"

MODEL_FILES = {
    "OLS": "ols_model.pkl",
    "Ridge": "ridge_model.pkl",
    "Lasso": "lasso_model.pkl",
}
BUNDLE_FILE = "training_bundle.pkl"   # full training result used by the dashboard
METADATA_FILE = "metadata.json"

# --------------------------------------------------------------------------- #
# Reproducibility / modelling constants
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20            # 80:20 split
CV_FOLDS = 5
MODEL_NAMES = ["OLS", "Ridge", "Lasso"]
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
LASSO_ALPHAS = np.logspace(-4, 0, 9)
VIF_THRESHOLD = 10.0        # severe multicollinearity
VIF_MODERATE = 5.0
MAX_CATEGORY_LEVELS = 50    # categorical columns with more levels are excluded
MIN_ROWS = 50

# --------------------------------------------------------------------------- #
# California Housing schema
# --------------------------------------------------------------------------- #
CALIFORNIA_TARGET = "MedHouseVal"
CALIFORNIA_NUMERIC = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
CALIFORNIA_CATEGORICAL = "OceanProximity"   # present in the supplied CSV only
CALIFORNIA_LABELS = {
    "MedInc": "Median income (tens of thousands of USD)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms per household",
    "AveBedrms": "Average bedrooms per household",
    "Population": "Block-group population",
    "AveOccup": "Average household occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "OceanProximity": "Ocean proximity",
}
# Column names of the widely distributed CSV version of the dataset
CALIFORNIA_CSV_COLUMNS = {
    "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income", "median_house_value",
}
CALIFORNIA_TARGET_SCALE = 100_000  # MedHouseVal is expressed in units of $100,000

# --------------------------------------------------------------------------- #
# Palette (used by matplotlib charts and the dashboard)
# --------------------------------------------------------------------------- #
PALETTE = {
    "navy": "#12263F",
    "ink": "#1F2D3D",
    "slate": "#5B6B7F",
    "mist": "#EEF2F6",
    "gold": "#C99A3B",
    "teal": "#1B8A83",
    "blue": "#2F5D9E",
    "red": "#C0453A",
}
MODEL_COLORS = {"OLS": "#2F5D9E", "Ridge": "#1B8A83", "Lasso": "#D08A1E"}

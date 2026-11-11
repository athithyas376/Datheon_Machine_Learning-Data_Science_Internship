# 🏡 Real Estate Housing Price Prediction using Ridge and Lasso Regression

A complete, modular machine-learning project that predicts **median house values** for the California Housing dataset and compares **Ordinary Least Squares (OLS)**, **Ridge (L2)** and **Lasso (L1)** regression – with an interactive Streamlit dashboard, an executed Jupyter notebook, trained models and automated tests.

## 1. Project overview
| Item | Description |
|---|---|
| Task | Continuous regression – predict `MedHouseVal` (in units of $100 000) |
| Data | California Housing, 20 640 block groups (`data/housing.csv`) |
| Models | OLS, Ridge, Lasso (alpha tuned with `GridSearchCV`) |
| Metrics | RMSE, MAE, R² – all computed from real predictions |
| Extras | VIF multicollinearity analysis, interaction / polynomial features, residual diagnostics, CSV/PNG/ZIP report export, CSV upload for custom datasets |

## 2. Problem statement
Predict the median house value of a census block group from its median income, house age, room/bedroom averages, population, occupancy, location and ocean proximity – while handling missing values, outliers, correlated predictors and a capped target correctly and without data leakage.

## 3. Objectives
1. Load, inspect and validate data (duplicates, missing values, dtypes).
2. Impute (median / most-frequent), encode (OneHotEncoder) and standardise (StandardScaler) inside Pipelines.
3. Detect and handle multicollinearity (correlation matrix, VIF, regularisation, VIF pruning experiment).
4. Engineer interaction / polynomial features and compare before vs after.
5. Train and tune OLS, Ridge and Lasso; evaluate on a reproducible 80:20 hold-out set.
6. Analyse residuals for systematic error and select the best model.
7. Deliver an interactive dashboard for exploration, training, prediction and reporting.

## 4. Dataset information
`data/housing.csv` is the widely distributed CSV version of the California Housing data. It stores *totals per block group*; `src/data_loader.py` converts them to the standard features:

| Feature | Derived as |
|---|---|
| `MedInc` | `median_income` |
| `HouseAge` | `housing_median_age` |
| `AveRooms` | `total_rooms / households` |
| `AveBedrms` | `total_bedrooms / households` (207 missing values → imputed) |
| `Population` | `population` |
| `AveOccup` | `population / households` |
| `Latitude`, `Longitude` | as given |
| `OceanProximity` | `ocean_proximity` (categorical, one-hot encoded) |
| **`MedHouseVal`** (target) | `median_house_value / 100 000` |

The loader never silently substitutes another dataset: if `data/housing.csv` is missing it tries scikit-learn's `fetch_california_housing` (needs internet) and otherwise raises a clear error.
Custom CSV files can be uploaded in the dashboard (any numeric target; features of numeric and categorical type are configured automatically).

## 5. Algorithms used
* **OLS** – `LinearRegression`, minimises squared error.
* **Ridge** – adds an L2 penalty α·Σβ²; α searched over 13 values (1e-3 … 1e3).
* **Lasso** – adds an L1 penalty α·Σ|β|; α searched over 9 values (1e-4 … 1). Performs feature selection.
* **GridSearchCV** – 5-fold shuffled `KFold` (`random_state=42`), scoring = negative RMSE, run on the *whole* pipeline so preprocessing is re-fitted in every fold.
* **Feature sets compared** (selected by cross-validation on the training data only): baseline · VIF-pruned (VIF < 5) · interaction features (`MedInc×HouseAge`, `AveRooms×HouseAge`, `Population×AveOccup`, `Latitude×Longitude`, `AveBedrms/AveRooms`; clipped to the 1st–99th percentile and mean-centred) · degree-2 polynomial.

## 6. System architecture
```
                 ┌────────────── data/housing.csv  or  uploaded CSV ──────────────┐
                 ▼                                                                │
  src/data_loader.py  → load · inspect · validate · clean (dedupe, drop NaN target)│
                 ▼                                                                │
  src/preprocessing.py → 80:20 split → ColumnTransformer                          │
        numeric: median impute → [feature engineering] → StandardScaler           │
        categorical: most-frequent impute → OneHotEncoder                         │
  src/feature_engineering.py → InteractionFeatures / Winsorizer transformers      │
                 ▼                                                                │
  src/model_training.py → OLS · GridSearchCV(Ridge) · GridSearchCV(Lasso)         │
                 ▼                                                                │
  src/evaluation.py → RMSE/MAE/R² · best model · residual analysis · report text  │
  src/visualization.py → matplotlib/seaborn PNG charts                            │
                 ▼                                                                │
  models/*.pkl + outputs/ ◄── main.py (CLI)        app.py (Streamlit) ────────────┘
                 ▼
  src/prediction.py → load pipelines → predict (same preprocessing as training)
```

### Folder structure
```
Real_Estate_Housing_Price_Prediction/
├── app.py                  Streamlit dashboard (8 pages)
├── main.py                 CLI: full training pipeline + exports
├── requirements.txt
├── README.md
├── data/housing.csv        California Housing data
├── notebooks/housing_price_prediction.ipynb   executed notebook (24 sections)
├── src/                    config, data_loader, preprocessing, feature_engineering,
│                           model_training, evaluation, prediction, visualization
├── models/                 ols_model.pkl, ridge_model.pkl, lasso_model.pkl,
│                           metadata.json, training_bundle.pkl (used by the dashboard)
├── outputs/charts|reports|metrics/    generated PNG charts, CSVs, summary report
├── assets/dashboard_images/           preview charts
├── tests/test_pipeline.py             automated tests
└── .streamlit/config.toml             dashboard theme
```

## 7. Installation
Requires Python 3.10+.
```bash
cd Real_Estate_Housing_Price_Prediction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 8. Train the models (optional – already done)
```bash
python main.py                                   # California Housing
python main.py --data my.csv --target price      # custom CSV
```
This writes the three `.pkl` models, metrics, 15 charts, the processed datasets and a Markdown summary report (≈ 40 s).
> The `.pkl` files are scikit-learn pipelines, which are tied to the scikit-learn version that created them. If you use a different version the dashboard detects this and retrains automatically; you can also simply run `python main.py` again.

## 9. How to run the Jupyter notebook
```bash
jupyter notebook notebooks/housing_price_prediction.ipynb
```
Choose **Kernel ▸ Restart & Run All** (≈ 1–2 minutes; the GridSearchCV cell is the longest). The notebook is delivered already executed, so results are visible without running it.

## 10. How to launch the Streamlit dashboard
```bash
streamlit run app.py
```
On first launch the app loads the data and the pre-trained models (or trains them once if none are found).

| Page | What you can do |
|---|---|
| 🏠 Home / Overview | Dataset KPIs (records, features, missing values, target), workflow, latest results |
| 🔎 Data Exploration | Filterable preview, statistics, distributions, correlation heatmap, missing values, relationship & map charts |
| 🧹 Data Preprocessing | Missing-value treatment, duplicates, encoding/scaling details, VIF (interactive pruning), feature-engineering results |
| ⚙️ Model Training | Split info, train/re-train with chosen CV folds, GridSearchCV alpha curves and best alphas, training status |
| 📊 Model Performance | RMSE / MAE / R² comparison, best-model card, actual-vs-predicted, coefficients |
| 📉 Residual Analysis | Histogram, residual-vs-predicted, Q-Q plot, error analysis, observations & limitations |
| 🏡 House Price Prediction | Choose OLS/Ridge/Lasso, enter features (dynamic form), **Predict** or real-time mode, compare all three models |
| 📥 Reports | Download processed / cleaned data, comparison tables, predictions, PNG charts, Markdown summary, ZIP of everything |

**Custom datasets:** choose *Upload a CSV* in the sidebar, pick the target and features (validation messages explain any problem), click *Use this dataset*, then train on the *Model Training* page. Preprocessing, feature engineering and the prediction form adapt to the columns of your file.

## 11. Run the tests
```bash
python -m pytest tests -q
```

## 12. Expected outputs (California Housing, random_state = 42)
Results from the shipped run (your numbers should match when package versions are similar):

| Model | Best alpha | CV RMSE | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|---|
| OLS | – | 0.5930 | 0.6014 | 0.4216 | 0.7240 |
| **Ridge** | 0.001 | 0.5930 | **0.6014** | 0.4217 | **0.7240** |
| Lasso | 0.0001 | 0.6045 | 0.6133 | 0.4357 | 0.7129 |

* Feature sets (best test RMSE): baseline 0.7331 → interactions 0.7078 → **polynomial degree 2 0.6014** (selected by cross-validation). VIF pruning *worsened* the result (0.7384) – the collinear variables carry signal, so regularisation rather than removal is used.
* All VIFs of the original features are below 10 (maximum: `Latitude` 9.2); four are moderate (> 5).
* Ridge is marginally best on the test set, but OLS and Ridge are practically identical (large sample, few features); Lasso zeroes 3 of 48 coefficients at the cost of ≈ 2 % higher RMSE.
* Residuals: mean ≈ 0 but error variance grows with the predicted price, residuals are right-skewed/heavy-tailed, and the capped target (4.7 % of rows) causes large errors for the most expensive areas – i.e. **systematic structure remains**.

Generated files: `models/*.pkl`, `outputs/metrics/*.csv|json`, `outputs/charts/*.png`, `outputs/reports/{processed_train,processed_test,cleaned_dataset}.csv`, `outputs/reports/project_summary.md`.

## 13. Limitations
* Linear models cannot capture strong non-linearity or spatial effects beyond the engineered terms; they can predict negative or above-cap prices.
* The target is capped at $500 000 (censoring) and the data are from the 1990 census – not suitable for current market valuation.
* Metrics come from one 80:20 split (cross-validation std shows the variability); predictions outside the training range are extrapolations (the dashboard warns).
* Pickled models depend on the scikit-learn version.
* The dashboard's plots are interactive Plotly charts; PNG exports are produced with matplotlib.

## 14. Future enhancements
Log-transformed or censored (Tobit-style) targets · tree-based models and gradient boosting · spatial features (distance to cities/coast, k-NN neighbourhood prices) · prediction intervals · SHAP explanations · model registry / experiment tracking · Docker image and CI.

# ChurnIQ: Predictive Customer Churn Classifier (Random Forest & XGBoost)

A full-stack churn-prediction dashboard built on the **IBM Telco Customer Churn** dataset
(7,043 customers, 21 columns). Flask serves a REST API and a Bootstrap 5 / Plotly.js
dashboard that is wired to real, trained scikit-learn and XGBoost models. No dummy data.

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. On first run the app trains both models automatically
(about 20 seconds) and saves them to `models/`. Pre-trained models are already included.
All JS/CSS libraries (Bootstrap, Bootstrap Icons, Plotly) are vendored in `static/vendor/`,
so the app works offline. Only the optional Inter web font comes from Google Fonts.

Python 3.9+ is recommended. Use another port with `PORT=8000 python app.py`.

## Dataset setup

The CSV is already at `dataset/Telco-Customer-Churn.csv`. To use a fresh copy, download
"Telco Customer Churn" from Kaggle (blastchar/telco-customer-churn) or the IBM sample-data
page, save it with that exact name in `dataset/`, then run `python train_model.py`
(or press **Retrain models** on the Model Performance page).

## Pages

| Route | Purpose |
|---|---|
| `/` | KPIs, churn donut, monthly-charge histogram, contract mix, risk overview |
| `/analytics` | Gender, senior, partner/dependents, internet, contract, payment, tenure, charges |
| `/prediction` | Single-customer form: probability gauge, Low/Medium/High risk, retention plan |
| `/bulk-prediction` | Upload CSV, validate, score, preview, download results CSV |
| `/model-performance` | ROC-AUC comparison, ROC curves, confusion matrix, P/R/F1, top-10 features, comparison table, retrain |
| `/retention` | High-risk active customers, churn factors, contract / charge / tenure playbooks |
| `/customers` | Searchable, sortable, paginated table; filters for churn, contract, risk; export filtered CSV |
| `/about` | Workflow, architecture, technologies, dataset facts |

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/stats` | KPIs, histogram, contract and risk breakdowns |
| GET | `/api/analytics` | Churn breakdowns by every dimension |
| POST | `/api/predict` | JSON customer -> probability, risk, recommendations |
| POST | `/api/bulk-predict` | multipart `file` (CSV) -> summary + rows + `file_id` |
| GET | `/api/download/<file_id>` | Download scored CSV |
| GET | `/api/sample-csv` | Sample upload template (25 customers) |
| GET | `/api/model-metrics` | All metrics, curves, confusion matrices |
| GET | `/api/feature-importance?model=XGBoost` | Top-10 / full / grouped importances |
| GET | `/api/customers` | `q, churn, contract, risk, page, per_page, sort, order` |
| GET | `/api/customers/export` | Filtered CSV export (same filters) |
| GET | `/api/retention` | Factors, playbooks, top high-risk customers |
| POST | `/api/retrain` | Retrain and reload models |

Errors return JSON `{"error": "..."}` with 4xx/5xx status codes.

## Machine-learning design

- **Cleaning:** `TotalCharges` coerced to numeric (11 blanks for tenure-0 customers, imputed with the training median); `customerID` excluded from features; target mapped to 0/1.
- **No leakage:** stratified 80/20 split first. Imputers, scaler and one-hot encoder live inside a `Pipeline` / `ColumnTransformer` fitted on training folds only. The threshold is chosen on out-of-fold training predictions, never on the test set.
- **Imbalance (about 27% churn):** Random Forest uses `class_weight="balanced_subsample"`, XGBoost uses `scale_pos_weight` (neg/pos).
- **Validation:** 5-fold `StratifiedKFold` ROC-AUC. The best model is the one with the highest mean CV ROC-AUC.
- **Threshold:** maximises F1 on out-of-fold predictions (RF about 0.43, XGBoost about 0.62).
- **Risk bands:** Low < 30%, Medium 30-60%, High >= 60% churn probability.
- **Customer-table risk scores** are out-of-fold predictions, so they are not inflated by memorisation.
- **Reproducible:** fixed seed 42 everywhere.

Typical results (hold-out test set): XGBoost ROC-AUC about 0.846, Random Forest about 0.842, both above the 0.80 target.

## Bulk CSV format

Required columns: `gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges,
TotalCharges`. `customerID` is optional; `Churn` is ignored. Use the **Sample CSV** button for a template.
Limits: 8 MB, 50,000 rows. Unseen category values and blank `TotalCharges` are handled and reported as warnings.

## Project structure

```
customer-churn-dashboard/
├── app.py              Flask app + REST API
├── train_model.py      Preprocessing, RF + XGBoost training, CV, metrics, persistence
├── predict.py          Inference, validation, risk bands, recommendations
├── requirements.txt
├── models/             *.joblib, metrics.json, customer_scores.csv (generated)
├── dataset/            Telco-Customer-Churn.csv
├── templates/          base + 8 pages (+ 404)
├── static/css/style.css, static/js/dashboard.js, static/vendor/, static/images/
└── uploads/            Scored CSVs from bulk predictions
```

## Troubleshooting

- *"Model not trained yet"*: run `python train_model.py`.
- *Port in use*: `PORT=5001 python app.py`.
- Models were saved with specific scikit-learn/XGBoost versions; if you see unpickling errors on a different version, retrain with `python train_model.py`.

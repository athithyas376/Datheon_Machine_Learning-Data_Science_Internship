# Project Summary – California Housing

*Generated: 2026-11-11T09:58:26  ·  scikit-learn 1.9.1  ·  random_state=42*

## 1. Dataset
- Target: **MedHouseVal**
- Rows (raw → used): 20,640 → 20,640  (duplicates removed: 0, rows without target removed: 0)
- Features: 8 numeric, 1 categorical
- Split: 16,512 train / 4,128 test (80:20), 5-fold CV inside the training set

## 2. Preprocessing
- Numeric: median imputation → feature engineering (if selected) → StandardScaler
- Categorical: most-frequent imputation → OneHotEncoder
- All statistics are fitted on the training data only (scikit-learn Pipeline + ColumnTransformer).

## 3. Feature-set comparison (selected by cross-validation on the training set)
| feature_set | n_features_after_encoding | best_model_by_cv | best_cv_rmse | best_test_rmse | best_test_r2 |
|---|---|---|---|---|---|
| Baseline (original features) | 12 | Ridge | 0.7121 | 0.7331 | 0.5899 |
| VIF-pruned (VIF<5) | 10 | Ridge | 0.7281 | 0.7384 | 0.5839 |
| Interaction features | 17 | Ridge | 0.6996 | 0.7078 | 0.6177 |
| Polynomial (degree 2) | 48 | Ridge | 0.5930 | 0.6014 | 0.7240 |

Selected feature set: **Polynomial (degree 2)**

## 4. Model comparison (test set)
| Model | Best Params | CV RMSE | Test RMSE | Test MAE | Test R² | Non-zero Coefs |
|---|---|---|---|---|---|---|
| OLS | — | 0.5930 | 0.6014 | 0.4216 | 0.7240 | 48 |
| Ridge | alpha=0.001 | 0.5930 | 0.6014 | 0.4217 | 0.7240 | 48 |
| Lasso | alpha=0.0001 | 0.6045 | 0.6133 | 0.4357 | 0.7129 | 45 |

## 5. Best model
**Ridge** is the best model: it has the lowest test-set RMSE (0.6014), together with MAE 0.4217 and R² 0.7240. The runner-up, OLS, has RMSE 0.6014 – the two are tied to within 0.05%. The gap is below 1%, so the models are practically equivalent in accuracy; Ridge wins on the stated criterion (lowest test RMSE). Its train–test R² gap is +0.0153 and its cross-validated RMSE is 0.5930, which indicates good generalisation (no sign of over-fitting). Ridge shrinks correlated coefficients towards zero, which stabilises estimates when features are collinear.

## 6. Residual analysis (Ridge)
- **Verdict: the residuals are not purely random noise.** Systematic effects remain because error variance grows with the predicted value (heteroscedasticity); the capped target cannot be reproduced by a linear model; the residuals are skewed / heavy-tailed.
- Overall bias is negligible (mean residual +0.0089).
- Residuals are centred around zero in every prediction decile (largest decile bias = 0.20σ), so there is no strong systematic over/under-prediction pattern.
- Error size changes with the predicted value (Spearman ρ between |residual| and prediction = +0.35) – evidence of heteroscedasticity.
- The residual distribution is non-normal (skew +1.04, excess kurtosis 4.91): a few large errors dominate RMSE. Confidence intervals based on normal errors would be unreliable.
- The target is capped at 5 (4.7% of rows sit exactly at the cap). Linear models cannot represent this censoring, which produces the large positive residuals at the top of the actual-vs-predicted plot. 47 test predictions exceed the cap.
- 1 prediction is negative, which is impossible for a price – a linear model does not bound its output.

## 7. Limitations
- Linear models cannot capture strong non-linearities or spatial effects beyond the engineered features.
- Results describe this dataset; predictions outside the training range are extrapolations.
- Only one random train/test split is reported (cross-validation quantifies the variability).

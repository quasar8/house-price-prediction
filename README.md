
# Ames Housing Price Prediction: Classification & Regression for Real Estate Appraisal
 
## 🎯 Project Overview
 
This project tackles that in two phases on the Ames Housing dataset: classifying homes as **Expensive** or **Not Expensive**, then predicting their **exact sale price**. The classification models ranked **5th (97.88% accuracy)** on the course leaderboard, while the regression phase's **XGBoost** model reached a cross-validated **RMSE of 0.133** on log-transformed price.

## 📊 Dataset & Sources
 
**Source:** Ames Housing dataset, provided by WBS Coding School for two course iterations — one prepared for classification, one for regression.

### Classification phase
- **Size:** 1,460 houses × 79 features (+ binary target `Expensive`) and a separate, unlabeled competition test set of 1,459 houses.
- **Target distribution:** 1,243 Not Expensive vs. 217 Expensive (~14.9% positive class — moderately imbalanced).
- **Key Features Used:** 28 numerical, 20 nominal, 15 ordinal columns after dropping 16 low-signal columns (`MoSold`, `GarageYrBlt`, `LowQualFinSF`, `Electrical`, `BsmtUnfSF`, `BsmtFinSF2`, `BsmtFinType2`, `BsmtFinSF1`, `MasVnrArea`, `Exterior2nd`, `BldgType`, `Condition2`, `LandSlope`, `Utilities`, `LotShape`, `Alley`).
- **Notes:** `MSSubClass` looks numeric but is actually a categorical dwelling-type code, so it was converted to string and one-hot encoded rather than treated as ordinal/numeric.
### Regression phase
- **Size:** 1,460 houses × 79 features + continuous target `SalePrice`, log-transformed (`np.log`) to match the RMSE-on-log(SalePrice) evaluation convention.
- **Key Features Used:** all 79 features retained and encoded — 35 numerical, 28 nominal, 16 ordinal — with the final feature *set* narrowed later per model family (see Feature Selection below), rather than dropped upfront as in the classification phase.

## 🚀 Key Findings & Results
 
### Phase 1 — Classification (Expensive vs. Not Expensive)
 
Three classifiers were trained on identical preprocessing pipelines (mean-imputed numericals, one-hot encoded nominals, explicitly-ordered ordinal encoding) and compared on a 20% held-out test set:
 
| Model | CV Accuracy | Test Accuracy | Recall | Precision | F1 | Cohen's Kappa |
|---|---|---|---|---|---|---|
| Random Forest (GridSearchCV) | 0.9461 | 0.9623 | 0.8095 | 0.9189 | 0.8608 | 0.8391 |
| Logistic Regression (GridSearchCV) | 0.9469 | 0.9726 | 0.8571 | 0.9474 | 0.9000 | 0.8842 |
| SVM (linear kernel, GridSearchCV) | 0.9426 | 0.9486 | 0.7619 | 0.8649 | 0.8101 | 0.7806 |
 
- Logistic Regression was the strongest model on the held-out test set across every metric, despite being the simplest of the three.
- Random Forest's best parameters were `max_depth=10, min_samples_leaf=1, n_estimators=400`; SVM's were `C=0.1, kernel="linear", gamma="scale"`; Logistic Regression's best regularization strength was `C=1` (scikit-learn's default), confirming the original model was already well-tuned.
- All three models show **recall consistently below precision** — i.e., an actually-expensive home is more likely to be missed (false negative) than a normal home is to be wrongly flagged (false positive). Given the ~15% class imbalance, this is expected, and would be the first thing to address (via class weighting or threshold tuning) if missing an expensive property were the costlier business error.
- The best submission (from these models, across 4 attempts) placed **5th on the DS#053 course leaderboard with 97.88% accuracy**, scored on a separate hidden test set via the course's competition app.

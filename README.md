# Binary Classification with a Bank Dataset

## Overview
The goal is to predict whether a banking client will subscribe to a term deposit using the Kaggle *Binary Classification with a Bank Dataset*. The project showcases a full workflow: exploratory analysis, feature selection, imbalance handling, model tuning, and an AutoML benchmark.

## Dataset
- **Source**: Kaggle competition [Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8/)
- **Shape**: 750,000 training rows, 17 columns describing client demographics, outreach history, and previous marketing outcomes
- **Target**: `y` (`1` = client subscribes to a term deposit)
- **Files**: `train.csv`, `test.csv`

## Repository Guide
- `baseline_process.ipynb`: End-to-end notebook with EDA, resampling, Optuna tuning, and model export
- `autogluon_process.ipynb`: AutoGluon benchmark using stacked ensembling across gradient boosting, CatBoost, random forests, and neural nets
- `/output/experiment_results.csv`: Summary of the cross-validated experiments run in `baseline_process.ipynb`
- `/submissions/submission_*.csv`: Competition-style prediction files for each trained model (including the AutoGluon submission)

## Baseline Process (`baseline_process.ipynb`)
1. **Data typing & EDA**
   - Cast categorical columns to `category` dtype and numerics to smaller integer types for memory efficiency.
   - Visualized distributions and class **imbalance** (12% positive class).
2. **Feature selection**
   - Compared Decision Tree and Random Forest importance rankings.
   - Took the union of each model’s top 10 features to create the final feature set.
   - Select 11 features.
3. **Imbalance strategies**
   - Evaluated three options: no resampling, RandomUnderSampler, and SMOTE.
4. **Model families**
   - Random Forest, XGBoost, and MLPClassifier.
   - Hyperparameters tuned with Optuna (`n_trials = 5` per setting).
5. **Evaluation**
   - 5-fold StratifiedKFold cross-validation.
   - Metrics: accuracy, precision, recall, F1 score.

### Cross-Validation Highlights (Source: `experiment_results.csv`)
| Model Variant          | Accuracy | Precision | Recall | F1 | Private Score on Kaggle |
|------------------------|----------|-----------|--------|----|----|
| **XGBoost + SMOTE**    | 0.9626   | 0.9689    | 0.9560 | **0.9624** | **0.96735** |
| Random Forest + SMOTE  | 0.9508   | 0.9452    | 0.9573 | 0.9512 | 0.95413 |
| MLP + SMOTE            | 0.9249   | 0.9093    | 0.9441 | 0.9263 | 0.95378 |

### AutoML Process (`autogluon_process.ipynb`)
- `TabularPredictor` with `presets="best_quality"` (approximately 25-minute budget).
- Models included Random Forest, XGBoost, LightGBM, CatBoost, Neural Network (Torch), and bagged/stacked ensembles.
- Create 2 Ensemble models
  - baseline: XGBoost, Random Forest and Neural Network (Torch)
  - experiment 2: XGBoost, LightGBM, CatBoost and Neural Network (Torch) 

| Model Variant          | Accuracy | Precision | Recall | F1 | Private Score on Kaggle |
|------------------------|----------|-----------|--------|----|----|
| Baseline |    |     |  |  |  |
| Experiment 2 |    |     |  |  |  |
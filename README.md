# Binary Classification with a Bank Dataset

## Overview
This repository contains a side project. The goal is to predict whether a banking client will subscribe to a term deposit using the Kaggle *Binary Classification with a Bank Dataset*. The project showcases a full workflow: exploratory analysis, feature selection, imbalance handling, model tuning, and an AutoML benchmark.

## Dataset
- **Source**: Kaggle competition [Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8/)
- **Shape**: 750,000 training rows, 17 columns describing client demographics, outreach history, and previous marketing outcomes
- **Target**: `y` (`1` = client subscribes to a term deposit)
- **Files**: `train.csv`, `test.csv`

## Repository Guide
- `main.ipynb`: End-to-end notebook with EDA, resampling, Optuna tuning, and model export
- `autogluon.ipynb`: AutoGluon benchmark using stacked ensembling across gradient boosting, CatBoost, random forests, and neural nets
- `experiment_results.csv`: Summary of the cross-validated experiments run in `main.ipynb`
- `SMOTE+*.pkl`: Saved XGBoost, Random Forest and MLP models trained on the full SMOTE-resampled dataset
- `submission_*.csv`: Competition-style prediction files for each trained model (including the AutoGluon submission)

## Complete workflow(`main.ipynb`)
1. **Data typing & EDA**
   - Cast categorical columns to `category` dtype and numerics to smaller integer types for memory efficiency.
   - Visualized distributions and class imbalance (12% positive class).
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
| Model Variant          | Accuracy | Precision | Recall | F1 |
|------------------------|----------|-----------|--------|----|
| **XGBoost + SMOTE**    | 0.9631   | 0.9694    | 0.9564 | **0.9628** |
| Random Forest + SMOTE  | 0.9485   | 0.9406    | 0.9574 | 0.9489 |
| MLP + SMOTE            | 0.9345   | 0.9220    | 0.9497 | 0.9355 |

### AutoML Benchmark (`autogluon.ipynb`)
- `TabularPredictor` with `presets="best_quality"` (approximately 10-minute budget).
- Models included XGBoost, LightGBM, CatBoost, Neural Network (Torch), and bagged/stacked ensembles.
- Produced `submission_autogluon_v5.csv` for comparison against manually engineered models.

## Interview Talking Points
- Demonstrated ability to diagnose and address severe class imbalance (SMOTE vs undersampling vs none).
- Used tree-based feature importance to focus modeling on interpretable, high-signal variables.
- Automated hyperparameter search with Optuna while keeping the runtime practical.
- Compared custom models against an AutoML baseline to validate modeling choices.
- Packaged trained models and submissions, mirroring an end-to-end Kaggle workflow.
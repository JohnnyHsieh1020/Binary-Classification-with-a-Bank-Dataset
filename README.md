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

### Cross-Validation Highlights
| Model Variant          | Accuracy | Precision | Recall | F1 | Private Score on Kaggle |
|------------------------|----------|-----------|--------|----|----|
| **XGBoost + SMOTE**    | 0.9626   | 0.9689    | 0.9560 | **0.9624** | **0.96735** |
| Random Forest + SMOTE  | 0.9508   | 0.9452    | 0.9573 | 0.9512 | 0.95413 |
| MLP + SMOTE            | 0.9249   | 0.9093    | 0.9441 | 0.9263 | 0.95378 |
* Source: `experiment_results.csv`, Kaggle

## AutoML Process (`autogluon_process.ipynb`)
1. **Training configuration**
   - Trained a `TabularPredictor` with `label="y"`, `eval_metric="roc_auc"`, `presets="best_quality"`, `num_bag_folds=5`, `num_stack_levels=1`, `refit_full=True`, and set traning time limit to 25 minutes.
   - Generated leaderboards, feature importance plots, and full metric reports via `evaluate(..., auxiliary_metrics=True)` to double-check that ROC-AUC gains were consistent with precision/recall trade-offs.
2. **Experiments**
   - **Baseline** ensemble restricted to `XGB`, `RF`, and `NN_TORCH` to gauge a lighter-weight stack.
   - **Experiment 2** expanded the search space with `GBM` (LightGBM) and `CAT` to test whether gradient boosting diversity improved generalization.
3. **Inference & export**
   - Saved both predictors under `trained models/` and wrote Kaggle-ready probability submissions to `submissions/1107_submission_autogluon_baseline.csv` and `submissions/1107_submission_autogluon_ex2.csv`.
### Model Evaluations
| Model Variant | Accuracy | Precision | Recall | F1 | ROC_AUC | Private Score on Kaggle |
|------------------------|----------|-----------|--------|----|----|----|
| Baseline | 0.9577 | 0.8502 | 0.7880 | 0.8179 | 0.9844 | 0.96972 |
| Experiment 2 | 0.9490 | 0.8142 | 0.7480 | 0.7797 | 0.9800 | **0.97038** |
* Source: `autogluon_process.ipynb`, Kaggle

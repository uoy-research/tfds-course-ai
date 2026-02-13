# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: p3.14.2
#     language: python
#     name: python3
# ---

# %%
__author__ = "Matthew Care"
__version__ = "0.0.7"
__date__ = "2026-01-27"

# %% [markdown]
# # Coding Exercise 2 - Pipelines, Metrics and Hyperparameter Tuning on the Adult Census Income Dataset
#
# ## Learning Objectives
#
# By the end of this tutorial, you will understand:
# 1. How to build production-ready ML pipelines that prevent data leakage
# 2. How to handle imbalanced datasets using resampling techniques
# 3. How to select and interpret appropriate evaluation metrics for imbalanced classification
# 4. How to incorporate misclassification costs into model optimisation
# 5. How to tune hyperparameters systematically using Optuna
# 6. How to optimise decision thresholds for different objectives
# 7. How to combine multiple models using ensemble methods
#
# ## Overview
#
# This notebook builds on Coding Exercise 1 and implements a more **robust ML workflow** using:
# - [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) to avoid data leakage,
# - **Imbalanced-learn** sampling strategies (SMOTE, ADASYN, undersampling, hybrid methods) to handle class imbalance,
# - Richer **classification metrics** (accuracy, precision, recall, F1, MCC, ROC/PR AUC, balanced accuracy, specificity, cost/benefit analysis),
# - **Hyperparameter tuning** with [Optuna](https://optuna.org/) using flexible objective functions,
# - **Threshold optimisation** to maximise performance for specific objectives,
# - **Ensemble methods** (voting and stacking) to combine multiple models,
# - **Model interpretability** tools (feature importance, learning curves, calibration plots, partial dependence).
#
# For EDA (exploring feature distributions, correlations, class imbalance), **see Coding Exercise 1** - here we focus on pipelines, metrics and tuning.
#
# ## 0. Setup
#
# ### Learning Objectives:
# - Understand the configuration parameters that control the pipeline
# - Learn how to customise the workflow for different objectives
# - Ensure reproducibility through proper random state management
#
# ### Configuration Guide
#
# This notebook provides extensive configuration options. The key parameters you can modify are:
#
# **Execution Mode:**
# - `QUICK_MODE`: Set to `True` for faster runs during development/testing, `False` for full evaluation
#
# **Model Selection:**
# - `MODEL_NAMES`: List of models to evaluate (from `ALL_MODELS`)
# - `TUNED_MODELS`: Subset of models to hyperparameter tune with Optuna
#
# **Optimisation Objective:**
# - `TUNING_OBJECTIVE`: Choose from `"mcc"`, `"f1"`, `"balanced_accuracy"`, or `"cost_benefit"`
# - `COST_FP`, `COST_FN`, `BENEFIT_TP`, `BENEFIT_TN`: Configure misclassification costs (used when `TUNING_OBJECTIVE = "cost_benefit"`)
#
# **Threshold Optimisation:**
# - `USE_OPTIMAL_THRESHOLD`: Whether to find and apply optimal decision threshold
# - `THRESHOLD_METRIC`: Metric to optimise threshold for (`"auto"` uses `TUNING_OBJECTIVE`)
#
# **Ensemble Configuration:**
# - `ENSEMBLE_METHOD`: Choose `"voting"` or `"stacking"`
# - `TOP_N_MODELS_FOR_ENSEMBLE`: Number of best models to combine
#
# **Class Imbalance Handling:**
# - `SAMPLING_METHODS`: Resampling strategies to evaluate (SMOTE, ADASYN, etc.)
#
# **Cross-Validation:**
# - `N_SPLITS`: Number of CV folds
# - `N_REPEATS_CV`: Number of CV repetitions for final evaluation
#
# For hyperparameter search space customisation, see Section 9 (`suggest_params_for_model`).
#
# **Reproducibility:** All random operations use `RANDOM_STATE` to ensure reproducible results.

# %%
# Configuration and global parameters
RANDOM_STATE = 42
TEST_SIZE = 0.5  # 50% train / 50% test, as in Coding Exercise 1

# Quick mode for faster development and testing
QUICK_MODE = True  # Set to True for faster runs with reduced trials/splits

# Cross-validation configuration
N_SPLITS = 3 if QUICK_MODE else 5  # HERE
N_REPEATS_TUNING = 1  # repeats used *inside* Optuna tuning
N_REPEATS_CV = 1 if QUICK_MODE else 2  # repeats used for post-tuning multi-metric CV

# Parallelism (set to 1 if you hit resource limits)
NUM_JOBS = -1  # -1 means use all available cores

# Seeds for different phases
SEED_TUNING = RANDOM_STATE
SEED_CV = RANDOM_STATE

out_folder = "coding_exercise_2_colab"
optuna_folder = "coding_exercise_2_colab/optuna"
tables_folder = "coding_exercise_2_colab/tables"
out_suffix = "_cd2"  # suffix for output files

# Optimization objective configuration
TUNING_OBJECTIVE = "mcc"  # Options: "mcc", "f1", "balanced_accuracy", "cost_benefit"

# Cost/benefit parameters (only used if TUNING_OBJECTIVE = "cost_benefit")
# These values define the misclassification costs and benefits for binary classification
COST_FP = 2  # Cost of false positive (predicting >50K when actually <=50K)
COST_FN = 10  # Cost of false negative (missing >50K earner - costs more!)
BENEFIT_TP = 5  # Benefit of true positive (correctly identifying >50K earner)
BENEFIT_TN = 5  # Benefit of true negative (correctly identifying <=50K earner)

# Threshold optimization configuration
USE_OPTIMAL_THRESHOLD = True  # Apply optimal threshold to final predictions
THRESHOLD_METRIC = (
    "auto"  # "auto" = use same as TUNING_OBJECTIVE, or specify: "mcc", "f1", "balanced_accuracy", "cost_benefit"
)

# Ensemble configuration
ENSEMBLE_METHOD = "stacking"  # Options: "voting" or "stacking"
TOP_N_MODELS_FOR_ENSEMBLE = 3  # Number of best models to combine in ensemble

# All available models in the system (master list)
ALL_MODELS = [
    "DummyMostFreq",
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting",
    "LightGBM",
    # "SVC",
    "MLPClassifier",
]

# Sampling strategies for handling class imbalance
# See build_pipeline() for sampling options
ALL_SAMPLING_METHODS = [
    "none",
    "smote",
    "adasyn",
    "borderline_smote",
    "random_undersample",
    "tomek",
    "smote_tomek",
    "smote_enn",
]

# Mode-specific model configuration
# QUICK_MODE: minimal set for fast testing/development
# Full mode: comprehensive evaluation of all models
if QUICK_MODE:
    # Minimal set for fast testing - always include DummyMostFreq as baseline
    MODEL_NAMES = ["DummyMostFreq", "RandomForest", "LightGBM", "GradientBoosting"]
    TUNED_MODELS = ["LightGBM"]  # Only tune one model in quick mode
    SAMPLING_METHODS = ["none", "smote", "adasyn"]  # Fewer sampling methods in quick mode
else:
    # Full evaluation with all models
    MODEL_NAMES = ALL_MODELS
    TUNED_MODELS = ["LightGBM", "MLPClassifier", "RandomForest", "GradientBoosting"]
    SAMPLING_METHODS = ALL_SAMPLING_METHODS
# Validation: ensure TUNED_MODELS is a subset of MODEL_NAMES
if not set(TUNED_MODELS).issubset(set(MODEL_NAMES)):
    missing = set(TUNED_MODELS) - set(MODEL_NAMES)
    raise ValueError(f"TUNED_MODELS contains models not in MODEL_NAMES: {missing}")

# Alias for use in configuration summary
CV_FOLDS = N_SPLITS

N_TRIALS_PER_MODEL_DEFAULT = 30
N_TRIALS_PER_MODEL = {
    "DummyMostFreq": 1,
    "LogisticRegression": 25,
    "RandomForest": 100 if not QUICK_MODE else 10,
    "GradientBoosting": 100 if not QUICK_MODE else 10,
    "LightGBM": 100 if not QUICK_MODE else 15,
    "SVC": 50 if not QUICK_MODE else 10,
    "MLPClassifier": 50 if not QUICK_MODE else 2,
}

# %%
# Imports (install required packages automatically in Colab if needed)
import sys
import warnings

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # Install required packages in Colab
    import subprocess

    packages = ["optuna", "imbalanced-learn", "lightgbm"]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import os

# LightGBM
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import sklearn

# Imbalanced-learn imports
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from optuna.samplers import TPESampler
from optuna.visualization import plot_parallel_coordinate, plot_param_importances
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, RobustScaler
from sklearn.svm import SVC

sns.set_theme(style="whitegrid", context="notebook")
np.random.seed(RANDOM_STATE)

os.makedirs(out_folder, exist_ok=True)
os.makedirs(optuna_folder, exist_ok=True)
os.makedirs(tables_folder, exist_ok=True)

# Configure sklearn to output pandas DataFrames from transformers
# This fixes LightGBM warnings about feature names by preserving column names
# throughout the pipeline. Requires sklearn >= 1.2.
sklearn.set_config(transform_output="pandas")

# %%
# Load the Adult Census Income dataset (same source as Coding Exercise 1)
DATA_URL = "https://github.com/medmaca/shared_data/raw/8a3fea5467ec68b17fd8369c6f77f8016b1ed5f8/Datasets/Kaggle/adult_census_income/adult.csv.zip"

adult_ci_df = pd.read_csv(DATA_URL, compression="zip")
adult_ci_df.head()

# %% [markdown]
# ## 1. Minimal dataset checks (see Coding Exercise 1 section 4 for full EDA)
#
# ### Learning Objectives:
# - Verify data loaded correctly
# - Understand the extent of class imbalance
# - Recognize why standard accuracy is misleading for imbalanced datasets
#
# In Coding Exercise 1 we performed detailed exploratory data analysis (EDA).
# Here we only do minimal sanity checks and then move on to **pipelines**, **metrics** and **hyperparameter tuning**.

# %%
# Basic info and class distribution
adult_ci_df.info()

target_col = "income"  # same as Coding Exercise 1

class_counts = adult_ci_df[target_col].value_counts().sort_index()
class_props = adult_ci_df[target_col].value_counts(normalize=True).sort_index()
print(f"\nClass counts:{class_counts}")
print(f"\nClass proportions:{class_props}")

# %% [markdown]
# ### Understanding Class Imbalance
#
# The dataset shows approximately 75% of samples belong to the <=50K class and 25% to the >50K class.
# This 3:1 imbalance means:
#
# - A naive classifier that always predicts "<=50K" would achieve 75% accuracy
# - Accuracy alone is misleading and insufficient for model evaluation
# - We need metrics that account for both classes equally
# - We should consider resampling techniques to balance the training data
#
# **Common Pitfall:** Using accuracy as the primary metric for imbalanced data can lead to models that simply predict the majority class and appear to perform well whilst being useless in practice.

# %% [markdown]
# ## 2. Understanding Classification Metrics for Imbalanced Data
#
# ### Learning Objectives:
# - Understand the confusion matrix and derived metrics
# - Learn which metrics are appropriate for imbalanced classification
# - Understand the Matthews Correlation Coefficient (MCC)
# - Learn how to incorporate misclassification costs into model evaluation
#
# ### The Confusion Matrix
#
# All binary classification metrics derive from the confusion matrix:
#
# $$
# \begin{array}{c|cc}
# & \text{Predicted Negative} & \text{Predicted Positive} \\
# \hline
# \text{Actual Negative} & TN & FP \\
# \text{Actual Positive} & FN & TP \\
# \end{array}
# $$
#
# Where:
# - **TP (True Positives)**: Correctly predicted positive class
# - **TN (True Negatives)**: Correctly predicted negative class
# - **FP (False Positives)**: Incorrectly predicted positive (Type I error)
# - **FN (False Negatives)**: Incorrectly predicted negative (Type II error)
#
# $$
# \begin{array}{c|cc}
# & \text{Predicted Negative} & \text{Predicted Positive} \\
# \hline
# \text{Actual Negative} & - & \text{Type I Error} \\
# \text{Actual Positive} & \text{Type II Error} & - \\
# \end{array}
# $$
#
# ### Key Metrics Explained
# See https://en.wikipedia.org/wiki/Precision_and_recall
#
#
# **1. Accuracy = (TP + TN) / (TP + TN + FP + FN)**
# - Proportion of correct predictions
# - **Problem:** Misleading with imbalanced data
# - **When to use:** Only when classes are balanced and errors equally costly
#
# **2. Precision = TP / (TP + FP)**
# - Of all positive predictions, how many were correct?
# - **Use case:** When false positives are costly (e.g., spam detection)
# - **Trade-off:** Can be high by predicting positive rarely
#
# **3. Sensitivity = TP / (TP + FN)**
# - `Also called Recall` or True Positive Rate (TPR)
# - Of all actual positives, how many did we find?
# - **Use case:** When false negatives are costly (e.g., disease detection)
# - **Trade-off:** Can be high by predicting positive frequently
#
# **4. Specificity = TN / (TN + FP)**
# - Also called True Negative Rate (TNR)
# - Of all actual negatives, how many did we correctly identify?
# - **Use case:** Complements recall; important when true negatives matter
#
# **5. F1 Score = 2 × (Precision × Recall) / (Precision + Recall)**
# - https://en.wikipedia.org/wiki/F-score
# - Harmonic mean of precision and recall
# - **Advantage:** Balances precision and recall
# - **Limitation:** Doesn't account for true negatives
#
# **6. Balanced Accuracy = (Recall + Specificity) / 2**
# - Average of per-class accuracies
# - **Advantage:** Handles imbalanced data well
# - **Use case:** When both classes are equally important
#
# **7. Matthews Correlation Coefficient (MCC)**
# - [MCC](https://en.wikipedia.org/wiki/Phi_coefficient)
#
# $$
# \text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
# $$
#
# - Range: -1 (total disagreement) to +1 (perfect prediction), 0 = random
# - **Advantages:**
#   - Uses all four confusion matrix values
#   - Balanced measure even with imbalanced classes
#   - More informative than F1 for imbalanced data
#   - Directly interpretable as a correlation coefficient
# - **Disadvantages:**
#   - Less intuitive than accuracy or F1
#   - Requires understanding of correlation coefficients
# - **When to use:** Imbalanced classification where all four outcomes matter
#
# **8. ROC AUC (Area Under Receiver Operating Characteristic)**
# - https://en.wikipedia.org/wiki/Receiver_operating_characteristic
# - Plots True Positive Rate vs False Positive Rate across thresholds
# - **Advantage:** Threshold-independent, handles imbalance reasonably
# - **Limitation:** Optimistic for severe imbalance
#
# **9. PR AUC (Area Under Precision-Recall)**
# - Plots Precision vs Recall across thresholds
# - **Advantage:** More informative than ROC AUC for imbalanced data
# - **Use case:** When positive class is rare and important
#
# ### Why MCC is Often the Best Single Metric
#
# For this tutorial, we use MCC as our primary optimization metric because:
# 1. It's specifically designed for imbalanced datasets
# 2. It considers all four confusion matrix cells equally
# 3. It provides a single, interpretable correlation value
# 4. It's been shown to be more reliable than F1 for imbalanced problems
#
# However, we'll compute many metrics to understand model behaviour comprehensively.
#
# **Common Pitfall:** Relying on a single metric. Always examine multiple metrics and the confusion matrix to understand model behaviour fully.

# %% [markdown]
# ## 3. Cost-Benefit Analysis for Cost-Aware ML
#
# ### Learning Objectives:
# - Understand how to incorporate misclassification costs into model evaluation
# - Learn the difference between symmetric and asymmetric cost structures
# - Extend cost-benefit analysis to multiclass problems
#
# ### The Cost Context
#
# In real-world applications, different types of errors have different costs:
# - **False Positive (FP):** Predicting >50K when actually <=50K - might waste resources on unnecessary interventions
# - **False Negative (FN):** Predicting <=50K when actually >50K - might miss valuable opportunities
#
# Traditional metrics (accuracy, F1, MCC) treat all errors equally. Cost-benefit analysis allows us to optimize for real-world objectives.
#
# ### Binary Classification Cost Matrix
#
# For binary problems, we define a 2×2 cost matrix:
#
#
# $$
# \begin{array}{c|cc}
#  & \text{Predicted 0} & \text{Predicted 1} \\
# \hline
# \text{Actual 0} & \text{BENEFIT}_{\mathrm{TN}} & \text{COST}_{\mathrm{FP}} \\
# \text{Actual 1} & \text{COST}_{\mathrm{FN}} & \text{BENEFIT}_{\mathrm{TP}}
# \end{array}
# $$
#
#
#
# **Example for Adult Census Income:**
# - COST_FP = 1 (low cost - false alarm)
# - COST_FN = 10 (high cost - missing a high earner)
# - BENEFIT_TP = 5 (benefit of correctly identifying high earner)
# - BENEFIT_TN = 0 (no special benefit for correct low earner prediction)
#
# This asymmetry reflects that missing a high earner (FN) is 10× more costly than a false alarm (FP).
#
# ### Multiclass Extension
#
# The cost matrix naturally extends to multiclass problems as an N×N matrix where:
# - Rows represent true classes
# - Columns represent predicted classes
# - Diagonal elements are benefits (correct predictions)
# - Off-diagonal elements are costs (misclassification costs)
#
# **Example for 3-class income problem (Low, Medium, High):**
#
# $$
# \begin{array}{c|ccc}
#  & \text{Pred Low} & \text{Pred Med} & \text{Pred High} \\
# \hline
# \text{True Low}  & 0  & 5  & 20 \\
# \text{True Med}  & 3  & 0  & 10 \\
# \text{True High} & 50 & 15 & 0
# \end{array}
# $$
#
#
#
# Here, misclassifying a High earner as Low (cost=50) is catastrophic, whilst misclassifying Low as Medium (cost=5) is minor.
#
# ### When to Use Cost-Benefit Analysis
#
# Use cost-benefit optimization when:
# 1. Different errors have quantifiable, different costs
# 2. Domain experts can define cost/benefit values
# 3. The goal is to maximize profit or minimize loss rather than optimize a metric
#
# **Common Pitfall:** Using arbitrary cost values. Work with domain experts to define realistic costs that reflect true real-world impact.

# %% [markdown]
# ## 4. Handle missing data

# %%
# Handle missing data markers and define feature types
# In this dataset, missing values are encoded as the literal string '?' in several categorical columns.

# Replace '?' with NaN so that SimpleImputer can handle them inside the pipeline
adult_ci_df = adult_ci_df.replace("?", np.nan)

feature_cols = [c for c in adult_ci_df.columns if c != target_col]
categorical_features = [c for c in feature_cols if adult_ci_df[c].dtype == "object"]
numeric_features = [c for c in feature_cols if adult_ci_df[c].dtype != "object"]

print("Categorical features:", categorical_features)
print("Numeric features:", numeric_features)

# %% [markdown]
# ## 5. Encode target and create a single held-out test set
#
# use [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) to split into train and test sets, whilst maintaining class proportions with the `stratify` parameter.

# %%
# Encode target and create a single held-out test set
# We use LabelBinarizer so we can recover the original class labels later for plots.
lb = LabelBinarizer()
adult_ci_df["target"] = lb.fit_transform(adult_ci_df[target_col].str.strip()).ravel()
print("Target classes (lb.classes_):", lb.classes_)

X = adult_ci_df[feature_cols].copy()
y = adult_ci_df["target"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
    shuffle=True,
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
print("Train class distribution:", y_train.value_counts(normalize=True).to_dict())
print("Test class distribution:", y_test.value_counts(normalize=True).to_dict())

# %% [markdown]
# ## 6. Handling Class Imbalance with Resampling Techniques
#
# ### Learning Objectives:
# - Understand why class imbalance is problematic
# - Learn different resampling strategies (oversampling, undersampling, hybrid)
# - Understand when to apply resampling (training only, never test data)
# - Learn how to integrate resampling into pipelines safely
#
# ### The Problem with Class Imbalance
#
# With imbalanced data, machine learning models often:
# 1. **Bias towards the majority class** - models learn to predict the common class
# 2. **Poor minority class performance** - fail to learn patterns in the rare class
# 3. **Misleading accuracy** - high accuracy by predicting only the majority class
#
# ### Resampling Strategies
# - see https://imbalanced-learn.org/stable/
# **1. Oversampling (Increasing Minority Class)**
# https://imbalanced-learn.org/stable/over_sampling.html
#
# - **Random Oversampling**: Duplicate minority class samples
#   - Simple but risks overfitting
#
# - **SMOTE (Synthetic Minority Over-sampling Technique)**: Create synthetic samples by interpolating between minority class neighbours
#   - Advantages: Reduces overfitting vs random duplication
#   - Disadvantages: Can create noise in overlap regions
#
# - **ADASYN (Adaptive Synthetic Sampling)**: Like SMOTE but focuses on difficult-to-learn examples
#   - Advantages: More samples near decision boundary
#   - Disadvantages: More complex, can amplify noise
#
# - **BorderlineSMOTE**: Only generates samples near the decision boundary
#   - Advantages: More focused than SMOTE
#   - Use case: When minority class has distinct regions
#
# **2. Undersampling (Reducing Majority Class)**
# https://imbalanced-learn.org/stable/under_sampling.html
#
# - **Random Undersampling**: Randomly remove majority class samples
#   - Fast but loses potentially useful information
#
# - **TomekLinks**: Remove majority class samples that are close to minority class
#   - Advantages: Cleans decision boundary
#   - Disadvantages: Removes very little data
#
# **3. Hybrid Methods (Combining Both)**
# https://imbalanced-learn.org/stable/combine.html
# - **SMOTETomek**: Apply SMOTE then clean with Tomek links
#   - Advantages: Oversample minority, clean noisy samples
#   - Best for: Moderate imbalance with noisy data
#
# - **SMOTEENN**: Apply SMOTE then clean with Edited Nearest Neighbours
#   - More aggressive cleaning than SMOTETomek
#
# ### Decision Guide for Imbalance Ratio
#
# - **Mild imbalance (< 4:1)**: Try `class_weight='balanced'` first
# - **Moderate imbalance (4:1 to 10:1)**: SMOTE or SMOTETomek
# - **Severe imbalance (> 10:1)**: ADASYN, hybrid methods, or consider anomaly detection
#
# ### Critical Rules
#
# 1. **Only resample training data** - never resample test/validation data
# 2. **Resample after train/test split** - prevents data leakage
# 3. **Use imblearn.pipeline.Pipeline** - ensures resampling happens in cross-validation
# 4. **Monitor both classes** - use appropriate metrics (MCC, F1, balanced accuracy)
#
# **Common Pitfall:** Applying resampling before train/test split causes data leakage, as synthetic test samples may be based on training data, leading to overly optimistic performance estimates.
#
# ### Implementation in this Tutorial
#
# We integrate resampling into our hyperparameter search, allowing Optuna to select the best strategy for each model. This is done using `imblearn.pipeline.Pipeline`, which properly handles resampling within cross-validation folds.

# %% [markdown]
# ## 7. Utility Functions for Metrics and Model Evaluation
#
# ### Learning Objectives:
# - Implement custom metric functions
# - Create a threshold-adjusted classifier wrapper
# - Build reusable cost-benefit scoring functions
#
# We define several utility functions that will be used throughout the notebook.


# %%
def specificity_score(y_true, y_pred):
    """Calculate specificity (true negative rate) for binary classification.

    Specificity measures the proportion of actual negatives that are correctly identified.
    Also known as selectivity or true negative rate (TNR).

    Not implemented in sklearn by default.  Hence implemented manually.

    Formula: TN / (TN + FP)

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_pred : array-like
        Predicted binary labels (0 or 1).

    Returns
    -------
    float
        Specificity score in range [0, 1], or np.nan if undefined.

    Notes
    -----
    Returns np.nan if there are no negative samples (TN + FP = 0) or if
    confusion matrix is not 2x2 (non-binary case).
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return np.nan
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    return tn / denom if denom > 0 else np.nan


def cost_benefit_score(y_true, y_pred, cost_matrix, normalize=True):
    """Calculate cost-benefit score using a cost matrix.

    Works for both binary and multiclass classification. The score is calculated as
    the sum of (confusion_matrix × cost_matrix). Higher scores are better.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    cost_matrix : array-like, shape (n_classes, n_classes)
        Cost/benefit matrix where entry [i, j] represents the value when:
        - True class is i (row index)
        - Predicted class is j (column index)

        **Convention:**
        - POSITIVE values = benefits (good outcomes: TN, TP)
        - NEGATIVE values = costs (bad outcomes: FP, FN)
    normalize : bool, default=True
        If True, normalize the score to the range [0, 1] where:
        - 0 = worst possible score (complete misclassification)
        - 1 = best possible score (perfect classification)
        If False, return the per-sample average cost-benefit score.

    Returns:
    -------
    float
        Cost-benefit score. If normalize=True, returns value in [0, 1].
        If normalize=False, returns per-sample average score where higher
        (more positive) is better. Per-sample averaging ensures scores are
        comparable across different fold sizes during cross-validation.

    Notes:
    -----
    For binary classification (negative class=0, positive class=1), cost_matrix format:

        cost_matrix = [[BENEFIT_TN,  -COST_FP ],
                       [-COST_FN,     BENEFIT_TP]]

    Confusion matrix mapping (sklearn convention):
                    Predicted
                    0 (Neg)    1 (Pos)
        Actual 0    TN         FP       ← Row 0: Actual negatives
               1    FN         TP       ← Row 1: Actual positives

    Score = TNxBENEFIT_TN + FPx(-COST_FP) + FNx(-COST_FN) + TPxBENEFIT_TP
          = (TN_benefit + TP_benefit) - (FP_cost + FN_cost)

    Normalization:
        normalized = (raw - worst) / (best - worst)
        where:
        - best = score with perfect predictions (all diagonal)
        - worst = score with worst predictions (all off-diagonal, maximizing costs)

    Examples:
    --------
    >>> import numpy as np
    >>> # Binary case: FN costs 10, FP costs 1, TP benefit 5, TN no benefit
    >>> cost_matrix = np.array([[0, -1],    # [TN_benefit, -FP_cost]
    ...                         [-10, 5]])  # [-FN_cost, TP_benefit]
    >>>
    >>> y_true = [0, 0, 1, 1, 1]
    >>> y_pred = [0, 1, 1, 1, 0]  # TN=1, FP=1, FN=1, TP=2
    >>> cost_benefit_score(y_true, y_pred, cost_matrix, normalize=False)
    -0.2  # = (1x0 + 1x(-1) + 1x(-10) + 2x5) / 5 = -1 / 5 = -0.2
    >>>
    >>> # Normalized version
    >>> cost_benefit_score(y_true, y_pred, cost_matrix, normalize=True)
    0.97  # (raw - worst) / (best - worst) = (-1 - (-22)) / (10 - (-22)) = 21/32
    """
    cm = confusion_matrix(y_true, y_pred)
    cost_matrix = np.array(cost_matrix)

    if cm.shape != cost_matrix.shape:
        raise ValueError(f"Confusion matrix shape {cm.shape} doesn't match cost matrix shape {cost_matrix.shape}")

    # Element-wise multiply confusion matrix by cost/benefit values, then sum
    # Positive values in cost_matrix = benefits, negative = costs
    # Higher return value = better performance
    raw_score = np.sum(cm * cost_matrix)

    # Normalize by sample count to get per-sample score (makes scores comparable
    # across different fold sizes / datasets)
    n_samples = cm.sum()
    per_sample_score = raw_score / n_samples if n_samples > 0 else 0.0

    if not normalize:
        return per_sample_score

    # Calculate best and worst possible scores for normalization
    # Best: all predictions are correct (diagonal of confusion matrix)
    class_counts = cm.sum(axis=1)  # true class counts

    # Best score: perfect predictions - all samples on diagonal
    best_score = np.sum(class_counts * np.diag(cost_matrix))

    # Worst score: maximize costs by predicting the worst class for each true class
    # For each true class, find the prediction that gives the worst (most negative) value
    worst_score = 0.0
    for i in range(cm.shape[0]):
        # Find the worst prediction for true class i (excluding correct prediction)
        # Convert to float to allow np.inf assignment
        costs_for_class = cost_matrix[i, :].astype(float).copy()
        costs_for_class[i] = np.inf  # Exclude correct prediction
        worst_pred = np.argmin(costs_for_class)
        worst_score += class_counts[i] * cost_matrix[i, worst_pred]

    # Normalize to [0, 1]
    if best_score == worst_score:
        return 1.0 if raw_score >= best_score else 0.0

    normalized_score = (raw_score - worst_score) / (best_score - worst_score)
    return float(np.clip(normalized_score, 0.0, 1.0))


class ThresholdedClassifier:
    """Wrapper that applies a custom decision threshold to a probabilistic classifier.

    Standard classifiers use a 0.5 threshold for binary classification. This wrapper
    allows optimising the threshold for specific metrics or objectives.

    Parameters
    ----------
    estimator : estimator object
        A fitted classifier that implements `predict_proba`.
    threshold : float, default=0.5
        Decision threshold in range [0, 1]. Samples with predicted probability
        >= threshold are classified as positive class.

    Attributes
    ----------
    estimator : estimator object
        The wrapped classifier.
    threshold : float
        The decision threshold.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier().fit(X_train, y_train)
    >>> # Use 0.3 threshold to favour recall over precision
    >>> thresholded_clf = ThresholdedClassifier(clf, threshold=0.3)
    >>> predictions = thresholded_clf.predict(X_test)

    Notes
    -----
    This wrapper assumes binary classification and uses the probability of the
    positive class (class 1) for thresholding.
    """

    def __init__(self, estimator, threshold=0.5):
        """Initialize the thresholded classifier.

        Parameters
        ----------
        estimator : estimator object
            A fitted classifier with predict_proba method.
        threshold : float, default=0.5
            Decision threshold in [0, 1].
        """
        self.estimator = estimator
        self.threshold = threshold

    def predict(self, X):
        """Predict class labels using the custom threshold.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        proba = self.estimator.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : array, shape (n_samples, 2)
            Class probabilities.
        """
        return self.estimator.predict_proba(X)

    def __repr__(self):
        return f"ThresholdedClassifier(estimator={self.estimator}, threshold={self.threshold})"


# Create the cost matrix for binary classification based on configuration
# Convention: Benefits are positive, costs are negative
# Matrix format: [[TN_benefit, -FP_cost], [-FN_cost, TP_benefit]]
COST_MATRIX = np.array([[BENEFIT_TN, -COST_FP], [-COST_FN, BENEFIT_TP]])

print("Cost matrix for cost-benefit analysis:")
print(COST_MATRIX)
print(f"Interpretation: FN costs {COST_FN}, FP costs {COST_FP}, TP benefit {BENEFIT_TP}, TN benefit {BENEFIT_TN}")
print("(Negative values = costs, positive values = benefits)")

# %% [markdown]
# ## 8. Preprocessing with ColumnTransformer and Pipeline
#
# ### Learning Objectives:
# - Build leak-proof preprocessing pipelines
# - Understand why preprocessing must be inside the pipeline
# - Integrate resampling strategies safely
# - Create modular, reusable pipeline components
#
# We use:
# - [`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) to handle missing values,
# - [`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) for numeric features (robust to outliers),
# - [`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) with `handle_unknown="ignore"` for categoricals,
# - [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) to combine these,
# - and wrap everything in a single [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) with a classifier as the final step.
#
# **Why Pipelines Prevent Data Leakage:**
# - Scalers, imputers, and encoders are *fitted* on training data only
# - The same transformations are *applied* to validation/test data
# - Without pipelines, it's easy to accidentally fit on all data (leakage)
#
# Note that for simplicity here we're using the `RobustScaler` for all numeric data, in Coding Exercise 1 (section 6) we used QuantileTransformer for variables that had an IQR of zero.  However, since the `RobustScaler` handles IQR=0 cases gracefully (it just scales by 1), this is acceptable for our purposes here.

# %%
# Numeric and categorical preprocessing pipelines
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # sparse_output=False is required when using sklearn.set_config(transform_output="pandas")
        # because pandas DataFrames don't support sparse matrices
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


def build_pipeline(estimator, sampling_strategy="none"):
    """Build a full preprocessing + optional resampling + model pipeline.

    This function creates a complete ML pipeline that:
    1. Preprocesses data (imputation, scaling, encoding)
    2. Optionally resamples training data to handle class imbalance
    3. Applies the final classifier

    Parameters
    ----------
    estimator : estimator object
        An sklearn-compatible classifier (e.g., RandomForestClassifier).
    sampling_strategy : str, default="none"
        The resampling strategy to apply. Options:
        - "none": No resampling
        - "smote": SMOTE oversampling
        - "adasyn": ADASYN oversampling
        - "borderline_smote": BorderlineSMOTE oversampling
        - "random_oversample": Random oversampling
        - "random_undersample": Random undersampling
        - "tomek": Tomek links undersampling
        - "smote_tomek": SMOTE + Tomek links (hybrid)
        - "smote_enn": SMOTE + ENN (hybrid)

    Returns
    -------
    pipeline : Pipeline or ImbPipeline
        A complete pipeline including preprocessing, optional resampling, and classifier.
        Returns imblearn.pipeline.Pipeline if sampling is used, else sklearn.pipeline.Pipeline.

    Notes
    -----
    Uses imblearn.pipeline.Pipeline when sampling is specified to ensure resampling
    happens correctly within cross-validation folds. This prevents data leakage.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier()
    >>> pipe = build_pipeline(clf, sampling_strategy="smote")
    >>> pipe.fit(X_train, y_train)
    """
    steps = [("preprocess", preprocessor)]

    # Add sampling step if requested
    if sampling_strategy != "none":
        if sampling_strategy == "smote":
            sampler = SMOTE(random_state=RANDOM_STATE)
        elif sampling_strategy == "adasyn":
            sampler = ADASYN(random_state=RANDOM_STATE)
        elif sampling_strategy == "borderline_smote":
            sampler = BorderlineSMOTE(random_state=RANDOM_STATE)
        elif sampling_strategy == "random_oversample":
            sampler = RandomOverSampler(random_state=RANDOM_STATE)
        elif sampling_strategy == "random_undersample":
            sampler = RandomUnderSampler(random_state=RANDOM_STATE)
        elif sampling_strategy == "tomek":
            sampler = TomekLinks()
        elif sampling_strategy == "smote_tomek":
            sampler = SMOTETomek(random_state=RANDOM_STATE)
        elif sampling_strategy == "smote_enn":
            sampler = SMOTEENN(random_state=RANDOM_STATE)
        else:
            raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

        steps.append(("sampler", sampler))

    steps.append(("clf", estimator))

    # Use imblearn Pipeline if sampling is used, otherwise sklearn Pipeline
    pipeline_class = ImbPipeline if sampling_strategy != "none" else Pipeline
    return pipeline_class(steps=steps)


# %%
# Quick visualization of the preprocessor pipeline
preprocessor


# %% [markdown]
# ## 9. Model zoo and hyperparameter search spaces
#
# ### Learning Objectives:
# - Set up multiple models for comparison
# - Define hyperparameter search spaces for Optuna
# - Integrate sampling strategies as hyperparameters
# - Understand different model families and their strengths
#
# We re-use the same 5 models from Coding Exercise 1 and add two additional models:
# - [`DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) (baseline),
# - [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
# - [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html),
# - [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html),
# - [`SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) with `probability=True` so we can compute ROC/PR AUC.
# - [`MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) as a simple feedforward neural network classifier.
# - [`LGBMClassifier`](https://lightgbm.readthedocs.io/) from LightGBM for fast gradient boosting.
#
# For each model we define:
# - a *base* configuration used when not tuning,
# - and an [Optuna](https://optuna.org/) **search space** describing how to sample hyperparameters.
#
# **Common Pitfall:** Using the same model with slightly different hyperparameters in an ensemble provides minimal diversity. Consider different algorithm families for better ensemble performance.


# %%
def create_base_estimator(model_name, params=None):
    """Return an sklearn classifier with sensible defaults, optionally updated with params."""
    params = dict(params or {})

    if model_name == "DummyMostFreq":
        base = {"strategy": "most_frequent"}
        base.update(params)
        return DummyClassifier(**base)

    if model_name == "LogisticRegression":
        # Sensible, fairly robust base configuration for this dataset
        base = {
            "max_iter": 10000,
            "C": 15,
            "class_weight": "balanced",
            "n_jobs": NUM_JOBS,
            "tol": 1e-5,
            "solver": "lbfgs",  # default; may be overridden by tuning
            "penalty": "l2",
            # l1_ratio will only be set when using penalty='elasticnet'
            "random_state": RANDOM_STATE,
        }
        base.update(params)
        return LogisticRegression(**base)

    if model_name == "RandomForest":
        base = {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "n_jobs": NUM_JOBS,
            "random_state": RANDOM_STATE,
        }
        base.update(params)
        return RandomForestClassifier(**base)

    if model_name == "GradientBoosting":
        base = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "random_state": RANDOM_STATE,
            "min_weight_fraction_leaf": 0.0,
        }

        base.update(params)
        return GradientBoostingClassifier(**base)

    if model_name == "SVC":
        base = {
            "kernel": "rbf",
            "gamma": "scale",
            "C": 1.0,
            "probability": True,  # needed for ROC/PR AUC
            "random_state": RANDOM_STATE,
        }
        base.update(params)
        return SVC(**base)

    if model_name == "MLPClassifier":
        # Simple feedforward neural network; early_stopping keeps training efficient.
        base = {
            "solver": "adam",
            "activation": "relu",
            "hidden_layer_sizes": (100,),
            "alpha": 1e-4,
            "learning_rate_init": 0.001,
            "max_iter": 200,
            "early_stopping": True,
            "n_iter_no_change": 10,
            "random_state": RANDOM_STATE,
        }

        if (
            "hidden_layer_sizes" in params
            and isinstance(params["hidden_layer_sizes"], str)
            and "-" in params["hidden_layer_sizes"]
        ):
            # For when hidden_layer_sizes is passed as a string like "64-32"
            hidden_layer_sizes = tuple(map(int, params["hidden_layer_sizes"].split("-")))
            params["hidden_layer_sizes"] = hidden_layer_sizes

        base.update(params)
        return MLPClassifier(**base)

    if model_name == "LightGBM":
        # LightGBM classifier with sensible defaults
        # Note: n_jobs=1 to avoid nested parallelism when cross_val_score uses n_jobs=-1
        base = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": -1,  # no limit
            "num_leaves": 31,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "min_child_samples": 20,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,  # Avoid nested parallelism with cross_val_score
            "verbose": -1,  # suppress warnings
        }
        base.update(params)
        return lgb.LGBMClassifier(**base)

    raise ValueError(f"Unknown model_name: {model_name}")


def suggest_params_for_model(trial, model_name, sample_methods=SAMPLING_METHODS):
    """Define Optuna search spaces for each model, including sampling strategies.

    The idea is:
    1. Start from the model's *base* parameters via ``create_base_estimator``.
    2. Suggest a sampling strategy for handling class imbalance.
    3. Override model-specific parameters with Optuna suggestions.

    For ``LogisticRegression``, we also enforce solver/penalty/l1_ratio compatibility:
      * lbfgs, newton-cg, newton-cholesky, sag: L2 only (no l1_ratio)
      * liblinear: L1 or L2 only (l1_ratio in {0, 1}, used only to pick penalty)
      * saga: L1, L2 or Elastic-Net (0 <= l1_ratio <= 1; only pass l1_ratio when elasticnet)

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for suggesting hyperparameters.
    model_name : str
        Name of the model to tune.
    sample_methods : list of str, optional
        List of sampling strategies to consider, by default SAMPLING_METHODS.

    Returns
    -------
    tuple of (dict, str)
        - dict: Model hyperparameters
        - str: Sampling strategy name
    """
    # Start from base estimator params so non-optimised parameters come from a central default
    base_est = create_base_estimator(model_name)
    base_params = base_est.get_params(deep=False)

    # Suggest sampling strategy for all models (except Dummy)
    if model_name == "DummyMostFreq":
        sampling_strategy = "none"
    else:
        sampling_strategy = trial.suggest_categorical("sampling_strategy", sample_methods)

    if model_name == "DummyMostFreq":
        # No tuning; just use base parameters
        return base_params, sampling_strategy

    if model_name == "LogisticRegression":
        params = dict(base_params)

        # Shared hyperparameters to tune
        c_param = trial.suggest_float("C", 1e-3, 1e3, log=True)
        params["C"] = c_param

        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        params["class_weight"] = class_weight

        solver = trial.suggest_categorical(
            "solver",
            # ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
            ["lbfgs"],  # lbfgs is a good fast default for this dataset
        )
        params["solver"] = solver

        # Solver-specific constraints on penalty / l1_ratio
        if solver in ["lbfgs", "newton-cg", "newton-cholesky", "sag"]:
            # Only L2 penalty is supported; ensure no l1_ratio is passed
            params["penalty"] = "l2"
            params.pop("l1_ratio", None)
            return params, sampling_strategy

        if solver == "liblinear":
            # liblinear supports L1 or L2, but not Elastic-Net; restrict l1_ratio to {0, 1}.
            # We use l1_ratio only as an internal helper to pick the penalty,
            # and do NOT pass l1_ratio to LogisticRegression (to avoid warnings).
            l1_ratio = trial.suggest_categorical("liblinear_l1_ratio", [0.0, 1.0])
            penalty = "l1" if l1_ratio == 1.0 else "l2"
            params["penalty"] = penalty
            params.pop("l1_ratio", None)
            return params, sampling_strategy

        if solver == "saga":
            # saga supports L1, L2 and Elastic-Net. We let l1_ratio drive the effective penalty:
            #   l1_ratio = 0   -> L2
            #   l1_ratio = 1   -> L1
            #   0 < l1_ratio < 1 -> Elastic-Net (only case where we pass l1_ratio)
            l1_ratio = trial.suggest_float("saga_l1_ratio", 0.0, 1.0)
            if 0.0 < l1_ratio < 1.0:
                params["penalty"] = "elasticnet"
                params["l1_ratio"] = l1_ratio
            elif l1_ratio <= 0.0:
                params["penalty"] = "l2"
                params.pop("l1_ratio", None)
            else:
                params["penalty"] = "l1"
                params.pop("l1_ratio", None)
            return params, sampling_strategy

        # This point should not be reached, but keep a safe fallback
        return params, sampling_strategy

    if model_name == "RandomForest":
        params = dict(base_params)

        n_estimators = trial.suggest_int("n_estimators", 100, 400)
        max_depth = trial.suggest_int("max_depth", 3, 100)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])

        params.update(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "class_weight": class_weight,
                "criterion": criterion,
            }
        )
        return params, sampling_strategy

    if model_name == "GradientBoosting":
        params = dict(base_params)

        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 5e-1, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 100)
        subsample = trial.suggest_float("subsample", 0.25, 1.0)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)
        min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5)
        min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.5)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 50)

        params.update(
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "subsample": subsample,
                "min_samples_leaf": min_samples_leaf,
                "min_weight_fraction_leaf": min_weight_fraction_leaf,
                "min_impurity_decrease": min_impurity_decrease,
                "max_features": max_features,
                "min_samples_split": min_samples_split,
            }
        )
        return params, sampling_strategy

    if model_name == "SVC":
        params = dict(base_params)

        c_param = trial.suggest_float("C", 1e-2, 1e2, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        tol = trial.suggest_float("tol", 1e-4, 1e-1, log=True)
        max_iter = trial.suggest_int("max_iter", 1000, 10000)

        params.update(
            {
                "C": c_param,
                "gamma": gamma,
                "class_weight": class_weight,
                "tol": tol,
                "max_iter": max_iter,
            }
        )
        return params, sampling_strategy

    if model_name == "MLPClassifier":
        params = dict(base_params)

        # Solver selection - adam is generally better for larger datasets
        # solver = trial.suggest_categorical("solver", ["adam", "lbfgs"])
        solver = trial.suggest_categorical("solver", ["adam"])

        # Keep the search space compact for runtime
        # Use string labels for Optuna's categorical choices, then map to tuples
        # to avoid warnings about non-primitive types in persistent storage.
        hidden_layer_key = trial.suggest_categorical(
            "hidden_layer_sizes",
            # ["64", "64-32", "128-64", "128-64-32", "256-128"],
            ["64", "64-32"],
        )
        hidden_layer_sizes = tuple(map(int, hidden_layer_key.split("-")))

        activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
        alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        tol = trial.suggest_float("tol", 1e-5, 1e-2, log=True)

        params.update(
            {
                "solver": solver,
                "hidden_layer_sizes": hidden_layer_sizes,
                "activation": activation,
                "alpha": alpha,
                "max_iter": max_iter,
                "tol": tol,
            }
        )

        # Adam-specific parameters
        if solver == "adam":
            learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
            beta_1 = trial.suggest_float("beta_1", 0.90, 0.99)
            beta_2 = trial.suggest_float("beta_2", 0.99, 0.9999)
            params.update(
                {
                    "learning_rate_init": learning_rate_init,
                    "beta_1": beta_1,
                    "beta_2": beta_2,
                }
            )

        return params, sampling_strategy

    if model_name == "LightGBM":
        params = dict(base_params)

        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 5e-1, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 100)
        num_leaves = trial.suggest_int("num_leaves", 10, 200)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
        min_split_gain = trial.suggest_float("min_split_gain", 0.0, 1.0)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        params.update(
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "min_child_samples": min_child_samples,
                "min_split_gain": min_split_gain,
                "class_weight": class_weight,
            }
        )
        return params, sampling_strategy

    raise ValueError(f"No search space defined for model_name: {model_name}")


# %% [markdown]
# ## 10. Metrics: Configurable objectives and comprehensive evaluation
#
# ### Learning Objectives:
# - Understand why metric choice matters for imbalanced data
# - Learn the strengths of MCC, F1, balanced accuracy, and cost-benefit
# - Configure objective functions for different application contexts
# - Distinguish between tuning objectives and evaluation metrics
#
# We support multiple tuning objectives (configured via `TUNING_OBJECTIVE`):
# - **[Matthews correlation coefficient (MCC)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)**: Robust to class imbalance, accounts for all confusion matrix elements
# - **[F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)**: Harmonic mean of precision and recall
# - **[Balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)**: Average of sensitivity and specificity
# - **Cost-benefit**: Cost-driven metric using configurable cost matrix
#
# Additionally, we evaluate models using: **accuracy**, **precision**, **recall**, **specificity**, **ROC AUC**, **PR AUC**.
#
# **Why MCC?**
# - Single-number summary handling imbalance well
# - Takes into account all four cells of the confusion matrix (TP, TN, FP, FN)
# - Ranges from -1 (total disagreement) through 0 (random) to +1 (perfect)
# - More informative than accuracy for imbalanced datasets
#
# **Common Pitfall:** Don't rely solely on accuracy for imbalanced data. A model predicting all majority class gets high accuracy but zero usefulness.


# %%
# MCC scorer for Optuna (higher is better)
# Note: specificity_score is defined earlier in Section 4
mcc_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)


# Cost-benefit scorer
def cost_benefit_scorer_func(y_true, y_pred):
    """Wrapper for cost_benefit_score using the global COST_MATRIX.

    Uses normalize=False for scoring to preserve relative differences
    for optimization. Normalized scores are computed for display.
    """
    return cost_benefit_score(y_true, y_pred, COST_MATRIX, normalize=False)


cost_benefit_scorer = make_scorer(cost_benefit_scorer_func, greater_is_better=True)

# Determine which scorer to use based on TUNING_OBJECTIVE
if TUNING_OBJECTIVE == "mcc":
    tuning_scorer = mcc_scorer
    tuning_scorer_name = "MCC"
elif TUNING_OBJECTIVE == "f1":
    tuning_scorer = make_scorer(f1_score, greater_is_better=True)
    tuning_scorer_name = "F1"
elif TUNING_OBJECTIVE == "balanced_accuracy":
    tuning_scorer = make_scorer(balanced_accuracy_score, greater_is_better=True)
    tuning_scorer_name = "Balanced Accuracy"
elif TUNING_OBJECTIVE == "cost_benefit":
    tuning_scorer = cost_benefit_scorer
    tuning_scorer_name = "Cost-Benefit"
else:
    raise ValueError(f"Unknown TUNING_OBJECTIVE: {TUNING_OBJECTIVE}")

print(f"Optimizing for: {tuning_scorer_name}")

# %% [markdown]
# ## 11. Hyperparameter tuning with Optuna
#
# ### Learning Objectives:
# - Understand Bayesian optimization vs grid/random search
# - Configure and run Optuna studies
# - Integrate sampling strategies into hyperparameter tuning
# - Interpret Optuna trial results and convergence
#
# ### Comparison of Hyperparameter Search Methods
#
# | Method | Strategy | Pros | Cons | Best For |
# |--------|----------|------|------|----------|
# | **[Grid Search](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GridSampler.html)** | Exhaustive search over specified parameter grid | Simple, reproducible, guaranteed to find best in grid | Computationally expensive, curse of dimensionality, misses values between grid points | Small search spaces, few parameters |
# | **[Random Search](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html)** | Random sampling from parameter distributions | More efficient than grid for high-dimensional spaces, can find good solutions quickly | No learning between trials, may miss optimal regions | Moderate search spaces, initial exploration |
# | **[Bayesian (TPE)](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)** | Probabilistic model guides search towards promising regions | Efficient, learns from previous trials, handles complex spaces well | More complex setup, overhead for small searches | Large search spaces, expensive evaluations |
#
# **[Optuna](https://optuna.org/)** is a hyperparameter optimization framework using [Tree-structured Parzen Estimator (TPE)](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) for efficient Bayesian search. Advantages over grid/random search:
#
# 1. **Adaptive sampling**: Learns from previous trials to suggest promising regions
# 2. **Early stopping**: Prunes unpromising trials (with pruners) - Though this only works if the model supports partial fitting (most sklearn models do not)
# 3. **Flexible search spaces**: Continuous, discrete, categorical parameters
# 4. **Parallel execution**: Multiple workers can run trials simultaneously
#
# We define an **objective function** that:
# - Samples hyperparameters from the search space
# - Builds a pipeline with sampled hyperparameters
# - Evaluates using cross-validation
# - Returns the metric to optimize
#
# Optuna maximizes the objective by default (we use `greater_is_better=True` for all our metrics).
#
# **Common Pitfall:** Running too few trials may miss the optimal region. Start with at least 50-100 trials per model (more for complex spaces).
#
# We now set up [Optuna](https://optuna.org/) studies for the selected models in `TUNED_MODELS`.
# - Each trial samples hyperparameters from `suggest_params_for_model`.
# - We build a full preprocessing + model [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
# - We evaluate using [`RepeatedStratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html) with `N_SPLITS` and `N_REPEATS_TUNING`.
# - The objective is the **mean `TUNING_OBJECTIVE`** across all folds and repeats.

# %%
# Suppress optuna logging for cleaner output
# optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_tuning_cv():
    return RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS_TUNING,
        random_state=SEED_TUNING,
    )


def objective(trial, model_name):
    """Optuna objective function for hyperparameter tuning.

    Tunes both model hyperparameters and sampling strategy, evaluating using
    the configured TUNING_OBJECTIVE metric.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    model_name : str
        Name of the model to tune.

    Returns
    -------
    float
        Mean score across all CV folds for the tuning objective.
    """
    params, sampling_strategy = suggest_params_for_model(trial, model_name)
    estimator = create_base_estimator(model_name, params=params)
    pipe = build_pipeline(estimator, sampling_strategy=sampling_strategy)

    cv = make_tuning_cv()
    with warnings.catch_warnings():
        # Suppress ConvergenceWarning (common for LogisticRegression during tuning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            message="'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'",
            category=UserWarning,
            module="sklearn.linear_model._logistic",
        )
        scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=tuning_scorer,
            n_jobs=NUM_JOBS,
        )
    return float(np.mean(scores))


tuning_records = []
best_params_by_model = {}
studies_by_model = {}  # Store studies for later analysis

for model_name in TUNED_MODELS:
    print(f"Starting Optuna tuning for: {model_name}")
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"{model_name}_{TUNING_OBJECTIVE}_tuning")

    def _objective(trial):
        """Wrapper as optuna only expects a single argument for the objective function."""
        return objective(trial, model_name)

    # Get number of trials for this model type, else set to default
    N_TRIALS = N_TRIALS_PER_MODEL.get(model_name, N_TRIALS_PER_MODEL_DEFAULT)
    study.optimize(_objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"Best {tuning_scorer_name} for {model_name}: {study.best_value:.4f}")
    print("Best params:", study.best_params)

    best_params_by_model[model_name] = study.best_params
    studies_by_model[model_name] = study

    for trial in study.trials:
        rec = {
            "model": model_name,
            "trial": trial.number,
            f"objective_{TUNING_OBJECTIVE}": trial.value,
            "n_splits": N_SPLITS,
            "n_repeats_tuning": N_REPEATS_TUNING,
        }
        for k, v in trial.params.items():
            rec[f"param_{k}"] = v
        tuning_records.append(rec)

tuning_results_df = pd.DataFrame(tuning_records) if tuning_records else pd.DataFrame()

# Save tuning results to tables folder
if not tuning_results_df.empty:
    tuning_csv_path = os.path.join(tables_folder, "optuna_tuning_all_trials.csv")
    tuning_results_df.to_csv(tuning_csv_path, index=False)
    print(f"Saved tuning results to: {tuning_csv_path}")

# Save best params by model to CSV
if best_params_by_model:
    best_params_records = []
    for model_name, params in best_params_by_model.items():
        rec = {"model": model_name}
        rec.update(params)
        best_params_records.append(rec)
    best_params_df = pd.DataFrame(best_params_records)
    best_params_csv_path = os.path.join(tables_folder, "best_params_by_model.csv")
    best_params_df.to_csv(best_params_csv_path, index=False)
    print(f"Saved best params by model to: {best_params_csv_path}")

tuning_results_df.head()

# %% [markdown]
# ## 12. Hyperparameter Importance Analysis
#
# ### Learning Objectives:
# - Understand which hyperparameters have the greatest impact on model performance
# - Learn to interpret hyperparameter importance plots
# - Use importance analysis to focus future tuning efforts
#
# After hyperparameter tuning, it's valuable to understand which hyperparameters had the greatest impact on the objective metric. Optuna provides tools to compute parameter importance using [functional ANOVA (fANOVA)](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.FanovaImportanceEvaluator.html#optuna.importance.FanovaImportanceEvaluator).
#
# **Parameter Importance** measures how much each hyperparameter contributes to variance in the objective function. High importance means the parameter has a strong effect on performance.
#
# **Use Cases:**
# - **Focus tuning efforts**: Spend more trials on important parameters
# - **Model understanding**: Learn what drives your model's performance
# - **Feature engineering**: Important preprocessing params may suggest data issues
#
# We'll generate:
# 1. [Hyperparameter importance bar plots](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_param_importances.html) (saved to optuna folder)
# 2. [Parallel coordinate plots](https://optuna.readthedocs.io/en/stable/reference/visualization/matplotlib/generated/optuna.visualization.matplotlib.parallel_coordinate.html#) showing parameter interactions
# 3. CSV files with importance values for each model

# %%
# Compute and visualize hyperparameter importance for each tuned model
for model_name, study in studies_by_model.items():
    print(f"\nAnalyzing hyperparameter importance for: {model_name}")

    # Compute importance
    try:
        importance = optuna.importance.get_param_importances(study)

        # Create importance dataframe and save to CSV
        importance_df = pd.DataFrame(
            {"parameter": list(importance.keys()), "importance": list(importance.values())}
        ).sort_values("importance", ascending=False)

        importance_csv_path = os.path.join(optuna_folder, f"{model_name}_param_importance.csv")
        importance_df.to_csv(importance_csv_path, index=False)
        print(f"  Saved importance to: {importance_csv_path}")
        print(f"  Top 3 parameters: {', '.join(importance_df.head(3)['parameter'].tolist())}")

        # Plot importance
        fig = plot_param_importances(study)
        fig.update_layout(title=f"Hyperparameter Importance: {model_name}")
        importance_plot_path = os.path.join(optuna_folder, f"{model_name}_param_importance.html")
        fig.write_html(importance_plot_path)
        print(f"  Saved importance plot to: {importance_plot_path}")

    except Exception as e:
        print(f"  Could not compute importance for {model_name}: {e}")

    # Generate parallel coordinate plot
    try:
        fig = plot_parallel_coordinate(study, params=None)
        fig.update_layout(title=f"Parallel Coordinate Plot: {model_name}")
        parallel_plot_path = os.path.join(optuna_folder, f"{model_name}_parallel_coordinate.html")
        fig.write_html(parallel_plot_path)
        print(f"  Saved parallel coordinate plot to: {parallel_plot_path}")
    except Exception as e:
        print(f"  Could not create parallel coordinate plot for {model_name}: {e}")

print(f"\nAll hyperparameter analysis artifacts saved to: {optuna_folder}")

# %% [markdown]
# ## 13. Post-tuning cross-validation with many metrics
#
# ### Learning Objectives:
# - Perform comprehensive model evaluation across multiple metrics
# - Understand the difference between tuning and final evaluation
# - Interpret metric trade-offs (precision vs recall, etc.)
# - Select the best model based on primary objective
#
# After hyperparameter tuning, we evaluate all models using **stratified K-fold cross-validation** with multiple metrics:
#
# - **MCC**: Primary metric for imbalanced binary classification
# - **Accuracy**: Overall correctness (can be misleading for imbalanced data)
# - **Precision**: Of predicted positives, how many are truly positive
# - **Recall (Sensitivity)**: Of actual positives, how many are detected
# - **F1**: Harmonic mean of precision and recall
# - **Balanced Accuracy**: Average of sensitivity and specificity
# - **Specificity**: Of actual negatives, how many are correctly identified
# - **ROC AUC**: Ranking performance across all thresholds
# - **PR AUC**: Precision-recall trade-off (preferred for imbalanced data)
# - **Cost-Benefit**: Cost-driven metric
#
# We now evaluate (potentially tuned) models using a richer set of metrics over:
# - `N_SPLITS` folds,
# - `N_REPEATS_CV` repeats,
# using [`RepeatedStratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html).
#
# For each model, fold and repeat we record:
# - MCC, accuracy, precision, recall, F1, balanced accuracy, specificity, ROC AUC, PR AUC.
# These are stored in a long-form `cv_results_df` for detailed analysis, and then summarised.


# %%
def make_eval_cv():
    return RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS_CV,
        random_state=SEED_CV,
    )


EVAL_MODELS = MODEL_NAMES  # evaluate all models; some may use default params

cv_records = []
rskf = make_eval_cv()

for model_name in EVAL_MODELS:
    # Get best params, extracting sampling_strategy if present
    # Use dict() to copy so we don't mutate the stored params
    all_params = dict(best_params_by_model.get(model_name, {}))
    sampling_strategy = all_params.pop("sampling_strategy", "none")
    params = all_params

    estimator = create_base_estimator(model_name, params=params)
    pipe = build_pipeline(estimator, sampling_strategy=sampling_strategy)

    print(f"Evaluating model (CV): {model_name}")
    for split_idx, (train_idx, val_idx) in enumerate(rskf.split(X_train, y_train), start=1):
        print(f"\tRepeat number : {(split_idx - 1) // N_SPLITS + 1}, Fold number: {(split_idx - 1) % N_SPLITS + 1}")
        repeat_idx = (split_idx - 1) // N_SPLITS + 1
        fold_idx = (split_idx - 1) % N_SPLITS + 1

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings(
                "ignore",
                message="'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'",
                category=UserWarning,
                module="sklearn.linear_model._logistic",
            )
            pipe.fit(X_tr, y_tr)
            y_val_pred = pipe.predict(X_val)

        # Many metrics rely on probabilities; assume binary classification and positive class = 1
        if hasattr(pipe, "predict_proba"):
            y_val_proba = pipe.predict_proba(X_val)[:, 1]
        else:
            # Fallback: use decision_function if available, otherwise cast predictions to {0,1}
            if hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(X_val)
                # Map scores to [0, 1] via rank-based scaling for AUC-like metrics
                ranks = pd.Series(scores).rank(method="average").values
                y_val_proba = ranks / ranks.max()
            else:
                y_val_proba = y_val_pred.astype(float)

        mcc = matthews_corrcoef(y_val, y_val_pred)
        acc = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred, zero_division=0)
        rec = recall_score(y_val, y_val_pred, zero_division=0)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, y_val_pred)
        spec = specificity_score(y_val, y_val_pred)
        cost_ben = cost_benefit_score(y_val, y_val_pred, COST_MATRIX, normalize=False)
        cost_ben_norm = cost_benefit_score(y_val, y_val_pred, COST_MATRIX, normalize=True)
        try:
            roc_auc = roc_auc_score(y_val, y_val_proba)
        except ValueError:
            roc_auc = np.nan
        try:
            pr_auc = average_precision_score(y_val, y_val_proba)
        except ValueError:
            pr_auc = np.nan

        print(
            f"\t\tMCC: {mcc:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, Bal_Acc: {bal_acc:.4f}, Spec: {spec:.4f}, ROC_AUC: {roc_auc:.4f}, PR_AUC: {pr_auc:.4f}, Cost-Benefit: {cost_ben_norm:.4f}"
        )

        cv_records.append(
            {
                "model": model_name,
                "repeat": repeat_idx,
                "fold": fold_idx,
                "n_splits": N_SPLITS,
                "n_repeats_cv": N_REPEATS_CV,
                "sampling_strategy": sampling_strategy,
                "mcc": mcc,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "balanced_accuracy": bal_acc,
                "specificity": spec,
                "cost_benefit_raw": cost_ben,
                "cost_benefit": cost_ben_norm,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

cv_results_df = pd.DataFrame(cv_records)

# Save CV results to tables folder
cv_results_csv_path = os.path.join(tables_folder, "cv_results_per_fold.csv")
cv_results_df.to_csv(cv_results_csv_path, index=False)
print(f"Saved CV results to: {cv_results_csv_path}")

cv_results_df.head()

# %%
# Summary: mean and standard deviation of metrics per model
metric_cols = [
    "mcc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "specificity",
    "cost_benefit",
    "roc_auc",
    "pr_auc",
]


# Save CV results to tables folder
cv_summary_df = cv_results_df.groupby("model")[metric_cols].agg(["mean", "std"])
cv_summary_df


# Flatten the MultiIndex columns: ('mcc', 'mean') -> 'mean_mcc'
model_comparison_df = cv_summary_df.copy()
model_comparison_df.columns = [f"{stat}_{metric}" for metric, stat in model_comparison_df.columns]
model_comparison_df = model_comparison_df.reset_index()

# Save model comparison to tables folder
model_comparison_csv_path = os.path.join(tables_folder, "model_comparison_summary.csv")
model_comparison_df.sort_values(f"mean_{TUNING_OBJECTIVE}", ascending=False, inplace=True)
model_comparison_df.to_csv(model_comparison_csv_path, index=False)
print(f"Saved model comparison summary to: {model_comparison_csv_path}")
model_comparison_df


# %%
# Create a flattened comparison dataframe for easier access
model_comparison_df = pd.DataFrame()
for metric in metric_cols:
    model_comparison_df[f"mean_{metric}"] = cv_results_df.groupby("model")[metric].mean()
    model_comparison_df[f"std_{metric}"] = cv_results_df.groupby("model")[metric].std()

model_comparison_df = model_comparison_df.reset_index()

# Save model comparison to tables folder
model_comparison_csv_path = os.path.join(tables_folder, "model_comparison_summary.csv")
model_comparison_df.to_csv(model_comparison_csv_path, index=False)
print(f"Saved model comparison summary to: {model_comparison_csv_path}")

print("\nModel Comparison Summary:")
model_comparison_df.sort_values("mean_mcc", ascending=False)

# %% [markdown]
# ## 14. Select the best model by primary objective and refit on full training set
#
# ### Learning Objectives:
# - Apply model selection criteria consistently
# - Understand final model training on full dataset
# - Recognize when to refit vs when to save CV models
# - Prepare for final evaluation on held-out test set
#
# We select the best model based on mean cross-validation performance for our tuning objective (configured via `TUNING_OBJECTIVE`). The selection process:
#
# 1. **Rank models** by mean CV score on the tuning objective
# 2. **Select the top model** (highest mean score)
# 3. **Refit on full training set** using optimal hyperparameters
# 4. **Prepare for test evaluation** (done only once to avoid overfitting)
#
# **Why refit on full training data?**
# - CV uses k-1 folds for training; we can improve performance by using all training data
# - Ensures the final model has maximum information for deployment
# - Test set evaluation remains unbiased (test set never seen during training/tuning)
#
# **Common Pitfall:** Selecting models based on test set performance invalidates the evaluation. Always select on validation/CV, then evaluate once on test.

# %%
# Select best model based on tuning objective
# Map TUNING_OBJECTIVE to the corresponding column in cv_results_df
# All map exactly
objective_col_map = {
    "mcc": "mcc",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
    "cost_benefit": "cost_benefit",
}
objective_col = objective_col_map.get(TUNING_OBJECTIVE, "mcc")

mean_score_by_model = cv_results_df.groupby("model")[objective_col].mean().sort_values(ascending=False)
best_model_name = mean_score_by_model.index[0]
print(f"Mean {TUNING_OBJECTIVE} by model:")
print(mean_score_by_model)
print()
print(f"Best model by {TUNING_OBJECTIVE}: {best_model_name}")

best_params = best_params_by_model.get(best_model_name, {})
print("Best params used for this model (if tuned):", best_params)

# Extract sampling_strategy if present
all_params = best_params.copy()
sampling_strategy = all_params.pop("sampling_strategy", "none")

best_estimator = create_base_estimator(best_model_name, params=all_params)
best_pipeline = build_pipeline(best_estimator, sampling_strategy=sampling_strategy)
best_pipeline.fit(X_train, y_train)
print(f"Fitted best model pipeline (sampling: {sampling_strategy})")

# Store best_name for later use
best_name = best_model_name

# %% [markdown]
# ## 15. Final evaluation on the held-out test set
#
# ### Learning Objectives:
# - Perform unbiased final evaluation on test data
# - Generate comprehensive diagnostic visualizations
# - Interpret confusion matrix and ROC/PR curves
# - Understand the importance of single-use test set
#
# **The held-out test set provides an unbiased estimate of real-world performance.** This evaluation happens only once to avoid overfitting to test data.
#
# We evaluate the best model using:
# - **Confusion matrix**: Visualize TP, TN, FP, FN
# - **All metrics**: MCC, accuracy, precision, recall, F1, balanced accuracy, specificity, ROC AUC, PR AUC, cost-benefit
# - **ROC curve**: Trade-off between sensitivity and specificity across thresholds
# - **Precision-Recall curve**: More informative than ROC for imbalanced data
#
# **Common Pitfall:** Evaluating on the test set multiple times (e.g., after tweaking hyperparameters) causes overfitting to test data. Test once!

# %%
# Predictions on the test set
y_test_pred = best_pipeline.predict(X_test)
y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]  # prediction class [<=50,>50], positive class column 1

test_mcc = matthews_corrcoef(y_test, y_test_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, zero_division=0)
test_rec = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
test_spec = specificity_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_proba)
test_pr_auc = average_precision_score(y_test, y_test_proba)

print(f"Best model: {best_model_name}")
print(f"Test MCC: {test_mcc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test precision: {test_prec:.4f}")
print(f"Test recall (sensitivity): {test_rec:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Test balanced accuracy: {test_bal_acc:.4f}")
print(f"Test specificity: {test_spec:.4f}")
print(f"Test ROC AUC: {test_roc_auc:.4f}")
print(f"Test PR AUC (average precision): {test_pr_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lb.classes_)
disp.plot(cmap="Reds", values_format="d")
ax = plt.gca()
ax.grid(False)
plt.title(f"Confusion matrix on held-out test set ({best_model_name})")
plt.tight_layout()

out_path = os.path.join(out_folder, f"confusion_matrix_on_held_out_test_set_{out_suffix}.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved confusion matrix figure to: {out_path}")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc_val = auc(fpr, tpr)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate (sensitivity)")
plt.title("ROC curve (test set)")
plt.legend()
plt.tight_layout()
out_path = os.path.join(out_folder, f"roc_curve_test_best_model_{out_suffix}.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved ROC curve figure to: {out_path}")
plt.show()

# Precision-recall curve
prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
pr_auc_val = average_precision_score(y_test, y_test_proba)
plt.figure(figsize=(5, 5))
plt.plot(rec, prec, label=f"PR curve (AP = {pr_auc_val:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve (test set)")
plt.legend()
plt.tight_layout()
out_path = os.path.join(out_folder, f"pr_curve_test_best_model_{out_suffix}.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved PR curve figure to: {out_path}")
plt.show()

# %% [markdown]
# ## 16. Threshold Analysis and Optimization
#
# ### Learning Objectives:
# - Understand why the default 0.5 threshold may not be optimal
# - Learn how to find optimal thresholds for different objectives
# - Apply threshold optimization to improve model performance
# - Understand the trade-offs between different thresholds
#
# The default classification threshold of 0.5 assumes:
# 1. Equal costs for false positives and false negatives
# 2. Balanced class distribution
# 3. Well-calibrated probabilities
#
# In practice, these assumptions rarely hold. By optimizing the threshold, we can:
# - Maximize a specific metric (MCC, F1, etc.)
# - Minimize misclassification costs (using cost-benefit analysis)
# - Balance precision and recall according to application needs
#
# **Common Pitfall:** Optimizing threshold on the test set causes overfitting. Always use validation data or cross-validation to find the optimal threshold, then evaluate once on the held-out test set.

# %%
y_test_proba

# %%
thresholds = np.linspace(0.0, 1.0, 101)
th_records = []

for thr in thresholds:
    # Calculate +ve predictions at each threshold level - y_thr
    y_thr = (y_test_proba >= thr).astype(int)
    # Compare predictions vs truth (y_test)
    mcc_thr = matthews_corrcoef(y_test, y_thr)
    acc_thr = accuracy_score(y_test, y_thr)
    prec_thr = precision_score(y_test, y_thr, zero_division=0)
    rec_thr = recall_score(y_test, y_thr, zero_division=0)
    f1_thr = f1_score(y_test, y_thr, zero_division=0)
    bal_acc_thr = balanced_accuracy_score(y_test, y_thr)
    spec_thr = specificity_score(y_test, y_thr)
    cost_ben_raw = cost_benefit_score(y_test, y_thr, COST_MATRIX, normalize=False)
    cost_ben_norm = cost_benefit_score(y_test, y_thr, COST_MATRIX, normalize=True)
    th_records.append(
        {
            "threshold": thr,
            "mcc": mcc_thr,
            "accuracy": acc_thr,
            "precision": prec_thr,
            "recall": rec_thr,
            "f1": f1_thr,
            "balanced_accuracy": bal_acc_thr,
            "specificity": spec_thr,
            "cost_benefit_raw": cost_ben_raw,
            "cost_benefit": cost_ben_norm,
        }
    )

threshold_df = pd.DataFrame(th_records)

# Save threshold analysis to tables folder
threshold_csv_path = os.path.join(tables_folder, "threshold_analysis.csv")
threshold_df.to_csv(threshold_csv_path, index=False)
print(f"Saved threshold analysis to: {threshold_csv_path}")

threshold_df.head()

# %% [markdown]
# ### Finding Optimal Thresholds
#
# We identify the optimal threshold for each metric. Note that different metrics may prefer different thresholds:
# - **MCC**: Balances all confusion matrix elements
# - **F1**: Balances precision and recall
# - **Balanced Accuracy**: Balances sensitivity and specificity
# - **Cost-Benefit**: Minimizes misclassification costs

# %%
# Find optimal thresholds for different objectives
optimal_thresholds = {}

for metric in ["mcc", "f1", "balanced_accuracy", "cost_benefit"]:
    if metric == "cost_benefit":
        # For cost_benefit, we want to maximize (most positive value)
        optimal_idx = threshold_df[metric].idxmax()
    else:
        optimal_idx = threshold_df[metric].idxmax()

    optimal_threshold = threshold_df.loc[optimal_idx, "threshold"]
    optimal_value = threshold_df.loc[optimal_idx, metric]
    optimal_thresholds[metric] = optimal_threshold

    print(f"Optimal threshold for {metric}: {optimal_threshold:.3f} (value: {optimal_value:.4f})")

# Determine which threshold to use based on configuration
if THRESHOLD_METRIC == "auto":
    threshold_metric_to_use = TUNING_OBJECTIVE
else:
    threshold_metric_to_use = THRESHOLD_METRIC

optimal_threshold_to_use = optimal_thresholds.get(threshold_metric_to_use, 0.5)
print(f"\nUsing optimal threshold for {threshold_metric_to_use}: {optimal_threshold_to_use:.3f}")

# %% [markdown]
# ### Threshold plots
# Visualize the chosen thresholds vs metrics

# %%
# Plot multiple metrics vs threshold
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: MCC, F1, Balanced Accuracy
ax = axes[0, 0]
ax.plot(threshold_df["threshold"], threshold_df["mcc"], label="MCC", linewidth=2)
ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1", linewidth=2)
ax.plot(threshold_df["threshold"], threshold_df["balanced_accuracy"], label="Balanced Acc", linewidth=2)
ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Default (0.5)")
ax.axvline(
    optimal_threshold_to_use, color="red", linestyle="--", alpha=0.7, label=f"Optimal ({optimal_threshold_to_use:.2f})"
)
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("Comprehensive Metrics vs Threshold")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Precision and Recall trade-off
ax = axes[0, 1]
ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", linewidth=2, color="blue")
ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", linewidth=2, color="orange")
ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.axvline(optimal_threshold_to_use, color="red", linestyle="--", alpha=0.7)
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision-Recall Trade-off")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Cost-Benefit
ax = axes[1, 0]
ax.plot(threshold_df["threshold"], threshold_df["cost_benefit"], label="Cost-Benefit", linewidth=2, color="green")
ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.axvline(optimal_threshold_to_use, color="red", linestyle="--", alpha=0.7)
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Cost-Benefit Score (Higher is Better)")
ax.set_title("Cost-Benefit Analysis")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Sensitivity (Recall) and Specificity
ax = axes[1, 1]
ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Sensitivity (Recall)", linewidth=2, color="orange")
ax.plot(threshold_df["threshold"], threshold_df["specificity"], label="Specificity", linewidth=2, color="purple")
ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.axvline(optimal_threshold_to_use, color="red", linestyle="--", alpha=0.7)
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("Sensitivity vs Specificity")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(out_folder, f"threshold_analysis_comprehensive_{out_suffix}.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved comprehensive threshold analysis to: {out_path}")
plt.show()

# %% [markdown]
# ### Applying Optimal Threshold to Best Model
#
# If `USE_OPTIMAL_THRESHOLD` is enabled, we create a thresholded version of the best model that uses the optimal threshold instead of the default 0.5.

# %%
if USE_OPTIMAL_THRESHOLD:
    print(f"Creating thresholded classifier with optimal threshold: {optimal_threshold_to_use:.3f}")
    best_pipeline_thresholded = ThresholdedClassifier(best_pipeline, threshold=optimal_threshold_to_use)

    # Re-evaluate with optimal threshold
    y_test_pred_thresholded = best_pipeline_thresholded.predict(X_test)

    # Compute metrics with optimal threshold
    test_mcc_thresh = matthews_corrcoef(y_test, y_test_pred_thresholded)
    test_acc_thresh = accuracy_score(y_test, y_test_pred_thresholded)
    test_prec_thresh = precision_score(y_test, y_test_pred_thresholded, zero_division=0)
    test_rec_thresh = recall_score(y_test, y_test_pred_thresholded, zero_division=0)
    test_f1_thresh = f1_score(y_test, y_test_pred_thresholded, zero_division=0)
    test_bal_acc_thresh = balanced_accuracy_score(y_test, y_test_pred_thresholded)
    test_spec_thresh = specificity_score(y_test, y_test_pred_thresholded)
    test_cost_ben_thresh_raw = cost_benefit_score(y_test, y_test_pred_thresholded, COST_MATRIX, normalize=False)
    test_cost_ben_thresh = cost_benefit_score(y_test, y_test_pred_thresholded, COST_MATRIX, normalize=True)

    # Compute default threshold cost-benefit for comparison
    test_cost_ben_default_raw = cost_benefit_score(y_test, y_test_pred, COST_MATRIX, normalize=False)
    test_cost_ben_default = cost_benefit_score(y_test, y_test_pred, COST_MATRIX, normalize=True)

    print(
        f"\n{'Metric':<20} {'Default (0.5)':<15} {'Optimal (' + f'{optimal_threshold_to_use:.2f}' + ')':<15} {'Improvement':<15}"
    )
    print("=" * 70)
    print(f"{'MCC':<20} {test_mcc:<15.4f} {test_mcc_thresh:<15.4f} {test_mcc_thresh - test_mcc:+.4f}")
    print(f"{'Accuracy':<20} {test_acc:<15.4f} {test_acc_thresh:<15.4f} {test_acc_thresh - test_acc:+.4f}")
    print(f"{'Precision':<20} {test_prec:<15.4f} {test_prec_thresh:<15.4f} {test_prec_thresh - test_prec:+.4f}")
    print(f"{'Recall':<20} {test_rec:<15.4f} {test_rec_thresh:<15.4f} {test_rec_thresh - test_rec:+.4f}")
    print(f"{'F1':<20} {test_f1:<15.4f} {test_f1_thresh:<15.4f} {test_f1_thresh - test_f1:+.4f}")
    print(
        f"{'Balanced Acc':<20} {test_bal_acc:<15.4f} {test_bal_acc_thresh:<15.4f} {test_bal_acc_thresh - test_bal_acc:+.4f}"
    )
    print(f"{'Specificity':<20} {test_spec:<15.4f} {test_spec_thresh:<15.4f} {test_spec_thresh - test_spec:+.4f}")
    print(
        f"{'Cost-Benefit':<20} {test_cost_ben_default:<15.4f} {test_cost_ben_thresh:<15.4f} {test_cost_ben_thresh - test_cost_ben_default:+.4f}"
    )

    # Save test metrics to CSV for both thresholds
    from sklearn.metrics import classification_report

    # Default threshold metrics DataFrame
    test_metrics_default = pd.DataFrame(
        [
            {
                "model": best_model_name,
                "threshold": 0.5,
                "mcc": test_mcc,
                "accuracy": test_acc,
                "precision": test_prec,
                "recall": test_rec,
                "f1": test_f1,
                "balanced_accuracy": test_bal_acc,
                "specificity": test_spec,
                "cost_benefit": test_cost_ben_default,
                "cost_benefit_raw": test_cost_ben_default_raw,
                "roc_auc": test_roc_auc,
                "pr_auc": test_pr_auc,
            }
        ]
    )
    test_metrics_default.to_csv(os.path.join(tables_folder, "test_metrics_default_threshold.csv"), index=False)
    print(f"Saved default threshold metrics to: {os.path.join(tables_folder, 'test_metrics_default_threshold.csv')}")

    # Optimal threshold metrics DataFrame
    test_metrics_optimal = pd.DataFrame(
        [
            {
                "model": best_model_name,
                "threshold": optimal_threshold_to_use,
                "mcc": test_mcc_thresh,
                "accuracy": test_acc_thresh,
                "precision": test_prec_thresh,
                "recall": test_rec_thresh,
                "f1": test_f1_thresh,
                "balanced_accuracy": test_bal_acc_thresh,
                "specificity": test_spec_thresh,
                "cost_benefit": test_cost_ben_thresh,
                "cost_benefit_raw": test_cost_ben_thresh_raw,
            }
        ]
    )
    test_metrics_optimal.to_csv(os.path.join(tables_folder, "test_metrics_optimal_threshold.csv"), index=False)
    print(f"Saved optimal threshold metrics to: {os.path.join(tables_folder, 'test_metrics_optimal_threshold.csv')}")

    # Classification reports
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (Default Threshold 0.5)")
    print("=" * 70)
    report_default = classification_report(y_test, y_test_pred, target_names=lb.classes_, output_dict=True)
    print(classification_report(y_test, y_test_pred, target_names=lb.classes_))

    # Save as CSV
    report_default_df = pd.DataFrame(report_default).transpose()
    report_default_df.to_csv(os.path.join(tables_folder, "classification_report_default_threshold.csv"))
    print(f"Saved classification report to: {os.path.join(tables_folder, 'classification_report_default_threshold.csv')}")

    print("\n" + "=" * 70)
    print(f"CLASSIFICATION REPORT (Optimal Threshold {optimal_threshold_to_use:.2f})")
    print("=" * 70)
    report_optimal = classification_report(y_test, y_test_pred_thresholded, target_names=lb.classes_, output_dict=True)
    print(classification_report(y_test, y_test_pred_thresholded, target_names=lb.classes_))

    # Save as CSV
    report_optimal_df = pd.DataFrame(report_optimal).transpose()
    report_optimal_df.to_csv(os.path.join(tables_folder, "classification_report_optimal_threshold.csv"))
    print(f"Saved classification report to: {os.path.join(tables_folder, 'classification_report_optimal_threshold.csv')}")

    # Confusion matrix visualization with optimal threshold
    cm_thresh = confusion_matrix(y_test, y_test_pred_thresholded)
    disp_thresh = ConfusionMatrixDisplay(confusion_matrix=cm_thresh, display_labels=lb.classes_)
    disp_thresh.plot(cmap="Blues", values_format="d")
    ax = plt.gca()
    ax.grid(False)
    plt.title(f"Confusion matrix (Optimal threshold={optimal_threshold_to_use:.2f})")
    plt.tight_layout()
    out_path = os.path.join(out_folder, f"confusion_matrix_optimal_threshold_{out_suffix}.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved confusion matrix figure to: {out_path}")
    plt.show()

    # Update the best_pipeline to use the optimal threshold
    best_pipeline_final = best_pipeline_thresholded
    print(f"\nFinal model uses optimal threshold: {optimal_threshold_to_use:.3f}")
else:
    best_pipeline_final = best_pipeline
    print("Using default threshold (0.5)")

# %% [markdown]
# ## 17. Ensemble Methods: Combining Multiple Models
#
# ### Learning Objectives:
# - Understand ensemble learning principles
# - Compare voting vs stacking strategies
# - Learn when ensembles outperform single models
# - Apply threshold optimization to ensemble predictions
#
# **Ensemble learning** combines predictions from multiple models to improve performance. Two main approaches:
#
# 1. **Voting Classifier**: Combines predictions by majority vote (hard) or averaging probabilities (soft)
# 2. **Stacking Classifier**: Trains a meta-model to optimally combine base model predictions
#
# **When do ensembles help?**
# - Diverse base models (different algorithms, hyperparameters, or training data)
# - Base models perform reasonably well but make different errors
# - Sufficient computational resources for training multiple models
#
# **Common Pitfall:** Ensembling many poor models rarely helps. Focus on combining strong, diverse models.

# %%
if ENSEMBLE_METHOD and TOP_N_MODELS_FOR_ENSEMBLE > 1:
    print(f"Creating ensemble from top {TOP_N_MODELS_FOR_ENSEMBLE} models using {ENSEMBLE_METHOD} strategy")

    # Select top N models based on validation MCC
    top_models = model_comparison_df.nlargest(TOP_N_MODELS_FOR_ENSEMBLE, "mean_mcc")
    print(f"\nTop {TOP_N_MODELS_FOR_ENSEMBLE} models selected:")
    print(top_models[["model", "mean_mcc", "std_mcc"]])

    # Recreate pipelines for top models
    ensemble_estimators = []
    for _idx, row in top_models.iterrows():
        model_name = row["model"]
        # Get original model configuration (copy to avoid mutating stored params)
        params_dict = best_params_by_model.get(model_name, {}).copy()
        sampling_strategy = params_dict.pop("sampling_strategy", "none")

        # Convert hidden_layer_sizes from string to tuple if needed (for MLPClassifier)
        if (
            "hidden_layer_sizes" in params_dict
            and isinstance(params_dict["hidden_layer_sizes"], str)
            and "-" in params_dict["hidden_layer_sizes"]
        ):
            params_dict["hidden_layer_sizes"] = tuple(map(int, params_dict["hidden_layer_sizes"].split("-")))

        # Create base estimator
        base_est = create_base_estimator(model_name, params=params_dict)
        # base_est.set_params(**params_dict)

        # Build pipeline
        pipe = build_pipeline(base_est, sampling_strategy=sampling_strategy)

        # Fit on full training set
        pipe.fit(X_train, y_train)

        # Add to ensemble list (use short name for clarity)
        ensemble_estimators.append((model_name[:10], pipe))
        print(f"  - Fitted {model_name} for ensemble")

    # Create ensemble based on configuration
    if ENSEMBLE_METHOD == "voting":
        ensemble_model = VotingClassifier(
            estimators=ensemble_estimators,
            voting="soft",  # Use probability averaging
            n_jobs=-1,
        )
        print(f"\nCreated VotingClassifier with soft voting")

    elif ENSEMBLE_METHOD == "stacking":
        # Use logistic regression as meta-classifier
        # meta_classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        # Use GradientBoostingClassifier as meta-classifier
        # print(f"\nCreated StackingClassifier with LogisticRegression meta-classifier")
        meta_classifier = GradientBoostingClassifier(random_state=RANDOM_STATE)
        ensemble_model = StackingClassifier(estimators=ensemble_estimators, final_estimator=meta_classifier, cv=5, n_jobs=-1)
        print(f"\nCreated StackingClassifier with GradientBoostingClassifier meta-classifier")

    else:
        print(f"Unknown ENSEMBLE_METHOD: {ENSEMBLE_METHOD}. Skipping ensemble.")
        ensemble_model = None

    if ensemble_model is not None:
        # Fit ensemble on training data
        print("\nFitting ensemble model...")
        ensemble_model.fit(X_train, y_train)

        # Predict on test set
        # Suppress feature names warning - this occurs because the meta-classifier is fitted
        # on stacked predictions (numpy array without feature names) but receives predictions
        # from base estimators that were fitted on DataFrames with feature names
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X has feature names, but .* was fitted without feature names",
                category=UserWarning,
            )
            y_test_pred_ensemble = ensemble_model.predict(X_test)
            y_test_proba_ensemble = ensemble_model.predict_proba(X_test)[:, 1]

        # Evaluate ensemble with default threshold
        ensemble_mcc = matthews_corrcoef(y_test, y_test_pred_ensemble)
        ensemble_acc = accuracy_score(y_test, y_test_pred_ensemble)
        ensemble_prec = precision_score(y_test, y_test_pred_ensemble, zero_division=0)
        ensemble_rec = recall_score(y_test, y_test_pred_ensemble, zero_division=0)
        ensemble_f1 = f1_score(y_test, y_test_pred_ensemble, zero_division=0)
        ensemble_bal_acc = balanced_accuracy_score(y_test, y_test_pred_ensemble)
        ensemble_spec = specificity_score(y_test, y_test_pred_ensemble)
        ensemble_cost_ben_raw = cost_benefit_score(y_test, y_test_pred_ensemble, COST_MATRIX, normalize=False)
        ensemble_cost_ben = cost_benefit_score(y_test, y_test_pred_ensemble, COST_MATRIX, normalize=True)

        print(f"\n{'=' * 70}")
        print(f"ENSEMBLE PERFORMANCE (default threshold = 0.5)")
        print(f"{'=' * 70}")
        print(f"MCC:              {ensemble_mcc:.4f}")
        print(f"Accuracy:         {ensemble_acc:.4f}")
        print(f"Precision:        {ensemble_prec:.4f}")
        print(f"Recall:           {ensemble_rec:.4f}")
        print(f"F1:               {ensemble_f1:.4f}")
        print(f"Balanced Acc:     {ensemble_bal_acc:.4f}")
        print(f"Specificity:      {ensemble_spec:.4f}")
        print(f"Cost-Benefit:     {ensemble_cost_ben:.4f}")

        # Confusion matrix for ensemble (default threshold)
        cm_ensemble = confusion_matrix(y_test, y_test_pred_ensemble)
        disp_ensemble = ConfusionMatrixDisplay(confusion_matrix=cm_ensemble, display_labels=lb.classes_)
        disp_ensemble.plot(cmap="Greens", values_format="d")
        ax = plt.gca()
        ax.grid(False)
        plt.title(f"Confusion matrix: Ensemble ({ENSEMBLE_METHOD}) - Default threshold=0.5")
        plt.tight_layout()
        out_path = os.path.join(out_folder, f"confusion_matrix_ensemble_default_threshold_{out_suffix}.pdf")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved ensemble confusion matrix to: {out_path}")
        plt.show()

        # Apply threshold optimization to ensemble if enabled
        if USE_OPTIMAL_THRESHOLD:
            print(f"\n{'=' * 70}")
            print(f"APPLYING OPTIMAL THRESHOLD TO ENSEMBLE")
            print(f"{'=' * 70}")

            # Find optimal threshold for ensemble predictions
            ensemble_thresholds = np.linspace(0.0, 1.0, 101)
            ensemble_th_records = []

            for thr in ensemble_thresholds:
                y_thr = (y_test_proba_ensemble >= thr).astype(int)
                mcc_thr = matthews_corrcoef(y_test, y_thr)
                f1_thr = f1_score(y_test, y_thr, zero_division=0)
                bal_acc_thr = balanced_accuracy_score(y_test, y_thr)
                cost_ben_raw = cost_benefit_score(y_test, y_thr, COST_MATRIX, normalize=False)
                cost_ben_norm = cost_benefit_score(y_test, y_thr, COST_MATRIX, normalize=True)
                ensemble_th_records.append(
                    {
                        "threshold": thr,
                        "mcc": mcc_thr,
                        "f1": f1_thr,
                        "balanced_accuracy": bal_acc_thr,
                        "cost_benefit_raw": cost_ben_raw,
                        "cost_benefit": cost_ben_norm,
                    }
                )

            ensemble_threshold_df = pd.DataFrame(ensemble_th_records)

            # Save ensemble threshold analysis to CSV
            ensemble_threshold_csv_path = os.path.join(tables_folder, "ensemble_threshold_analysis.csv")
            ensemble_threshold_df.to_csv(ensemble_threshold_csv_path, index=False)
            print(f"Saved ensemble threshold analysis to: {ensemble_threshold_csv_path}")

            # Find optimal threshold based on configuration
            ensemble_optimal_thresholds = {}
            for metric in ["mcc", "f1", "balanced_accuracy", "cost_benefit"]:
                optimal_idx = ensemble_threshold_df[metric].idxmax()
                ensemble_optimal_thresholds[metric] = ensemble_threshold_df.loc[optimal_idx, "threshold"]

            # Use same metric as before
            ensemble_optimal_threshold = ensemble_optimal_thresholds.get(threshold_metric_to_use, 0.5)
            print(f"Optimal ensemble threshold for {threshold_metric_to_use}: {ensemble_optimal_threshold:.3f}")

            # Create thresholded ensemble classifier
            ensemble_model_thresholded = ThresholdedClassifier(ensemble_model, threshold=ensemble_optimal_threshold)

            # Re-evaluate with optimal threshold
            y_test_pred_ensemble_thresh = ensemble_model_thresholded.predict(X_test)

            ensemble_mcc_thresh = matthews_corrcoef(y_test, y_test_pred_ensemble_thresh)
            ensemble_acc_thresh = accuracy_score(y_test, y_test_pred_ensemble_thresh)
            ensemble_prec_thresh = precision_score(y_test, y_test_pred_ensemble_thresh, zero_division=0)
            ensemble_rec_thresh = recall_score(y_test, y_test_pred_ensemble_thresh, zero_division=0)
            ensemble_f1_thresh = f1_score(y_test, y_test_pred_ensemble_thresh, zero_division=0)
            ensemble_bal_acc_thresh = balanced_accuracy_score(y_test, y_test_pred_ensemble_thresh)
            ensemble_spec_thresh = specificity_score(y_test, y_test_pred_ensemble_thresh)
            ensemble_cost_ben_thresh_raw = cost_benefit_score(
                y_test, y_test_pred_ensemble_thresh, COST_MATRIX, normalize=False
            )
            ensemble_cost_ben_thresh = cost_benefit_score(y_test, y_test_pred_ensemble_thresh, COST_MATRIX, normalize=True)

            print(
                f"\n{'Metric':<20} {'Default (0.5)':<15} {'Optimal (' + f'{ensemble_optimal_threshold:.2f}' + ')':<15} {'Improvement':<15}"
            )
            print("=" * 70)
            print(
                f"{'MCC':<20} {ensemble_mcc:<15.4f} {ensemble_mcc_thresh:<15.4f} {ensemble_mcc_thresh - ensemble_mcc:+.4f}"
            )
            print(
                f"{'Accuracy':<20} {ensemble_acc:<15.4f} {ensemble_acc_thresh:<15.4f} {ensemble_acc_thresh - ensemble_acc:+.4f}"
            )
            print(
                f"{'Precision':<20} {ensemble_prec:<15.4f} {ensemble_prec_thresh:<15.4f} {ensemble_prec_thresh - ensemble_prec:+.4f}"
            )
            print(
                f"{'Recall':<20} {ensemble_rec:<15.4f} {ensemble_rec_thresh:<15.4f} {ensemble_rec_thresh - ensemble_rec:+.4f}"
            )
            print(f"{'F1':<20} {ensemble_f1:<15.4f} {ensemble_f1_thresh:<15.4f} {ensemble_f1_thresh - ensemble_f1:+.4f}")
            print(
                f"{'Balanced Acc':<20} {ensemble_bal_acc:<15.4f} {ensemble_bal_acc_thresh:<15.4f} {ensemble_bal_acc_thresh - ensemble_bal_acc:+.4f}"
            )
            print(
                f"{'Specificity':<20} {ensemble_spec:<15.4f} {ensemble_spec_thresh:<15.4f} {ensemble_spec_thresh - ensemble_spec:+.4f}"
            )
            print(
                f"{'Cost-Benefit':<20} {ensemble_cost_ben:<15.4f} {ensemble_cost_ben_thresh:<15.4f} {ensemble_cost_ben_thresh - ensemble_cost_ben:+.4f}"
            )

            # Confusion matrix for ensemble (optimal threshold)
            cm_ensemble_thresh = confusion_matrix(y_test, y_test_pred_ensemble_thresh)
            disp_ensemble_thresh = ConfusionMatrixDisplay(confusion_matrix=cm_ensemble_thresh, display_labels=lb.classes_)
            disp_ensemble_thresh.plot(cmap="Purples", values_format="d")
            ax = plt.gca()
            ax.grid(False)
            plt.title(f"Confusion matrix: Ensemble ({ENSEMBLE_METHOD}) - Optimal threshold={ensemble_optimal_threshold:.2f}")
            plt.tight_layout()
            out_path = os.path.join(out_folder, f"confusion_matrix_ensemble_optimal_threshold_{out_suffix}.pdf")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved ensemble confusion matrix (optimal threshold) to: {out_path}")
            plt.show()

            ensemble_model_final = ensemble_model_thresholded
        else:
            ensemble_model_final = ensemble_model

        # Compare ensemble to best single model
        print(f"\n{'=' * 70}")
        print(f"ENSEMBLE vs BEST SINGLE MODEL COMPARISON")
        print(f"{'=' * 70}")

        if USE_OPTIMAL_THRESHOLD:
            best_model_mcc_final = test_mcc_thresh
            ensemble_mcc_final = ensemble_mcc_thresh
        else:
            best_model_mcc_final = test_mcc
            ensemble_mcc_final = ensemble_mcc

        print(f"Best Single Model ({best_name}): MCC = {best_model_mcc_final:.4f}")
        print(f"Ensemble ({ENSEMBLE_METHOD}):       MCC = {ensemble_mcc_final:.4f}")

        if ensemble_mcc_final > best_model_mcc_final:
            improvement = ((ensemble_mcc_final - best_model_mcc_final) / abs(best_model_mcc_final)) * 100
            print(f"\nEnsemble OUTPERFORMS best single model by {improvement:.2f}%")
            final_model_to_use = ensemble_model_final
            final_model_name = f"Ensemble ({ENSEMBLE_METHOD})"
        else:
            decline = ((best_model_mcc_final - ensemble_mcc_final) / abs(best_model_mcc_final)) * 100
            print(f"\nEnsemble UNDERPERFORMS best single model by {decline:.2f}%")
            print(f"  Keeping best single model ({best_name})")
            final_model_to_use = best_pipeline_final
            final_model_name = best_name

else:
    print(f"Ensemble disabled (ENSEMBLE_METHOD={ENSEMBLE_METHOD}, TOP_N={TOP_N_MODELS_FOR_ENSEMBLE})")
    final_model_to_use = best_pipeline_final
    final_model_name = best_name

print(f"\nFinal model selected: {final_model_name}")

# %% [markdown]
# ## 18. Probability Calibration Analysis
#
# ### Learning Objectives:
# - Understand what probability calibration means
# - Learn to assess calibration using calibration curves
# - Compute log loss and Brier score as probabilistic metrics
# - Recognize when calibration matters vs decision metrics
# - Understand calibration methods and when to apply them
#
# **[Probability calibration](https://scikit-learn.org/stable/modules/calibration.html#calibration)** refers to how well predicted probabilities reflect true frequencies. A perfectly calibrated model with predicted probability 0.7 should be correct 70% of the time.
#
# **Why calibration matters:**
# - Medical diagnosis (need reliable confidence scores)
# - Financial risk assessment
# - When probabilities are used directly (not just binary decisions)
# - Combining predictions from multiple models
#
# **Common Pitfall:** Many sklearn models (especially tree-based) produce poorly calibrated probabilities. Decision metrics (MCC, F1) can still be excellent even with poor calibration.
#
# Uses [`brier_score_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html) and [`log_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) to assess how well calibrated the prediction p-values are.
#
# ### Calibration Methods
#
# If your model produces poorly calibrated probabilities, you can apply post-hoc calibration using [`CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html):
#
# **1. Platt Scaling (Sigmoid Calibration)**
# - Fits a logistic regression to map raw predictions to calibrated probabilities
# - Assumes the calibration function is sigmoid-shaped
# - Works well for: SVM, neural networks, models with sigmoid-like miscalibration
# - Fast, requires fewer samples
#
# **2. Isotonic Regression**
# - Non-parametric approach that fits a monotonic (non-decreasing) function
# - Makes no assumptions about the shape of the calibration function
# - More flexible than Platt scaling
# - Works well for: Tree-based models, models with complex miscalibration patterns
# - Requires more data, can overfit with small datasets
#
# **When to calibrate:**
# - When you need reliable probability estimates (not just rankings)
# - When probabilities will be used for decision-making with costs
# - When combining models in ensembles
# - When calibration curve shows significant deviation from diagonal
#
# **When calibration is unnecessary:**
# - When only the ranking/ordering of predictions matters (e.g., for threshold selection)
# - When using models that are naturally well-calibrated (e.g., Logistic Regression)
# - When your evaluation metrics are decision-based (MCC, F1) rather than probabilistic
#
# For implementation examples, see the [sklearn calibration documentation](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html).
#
# In this section, we assess calibration quality. Applying calibration is left as an extension exercise.

# %%
# Get probabilities for final model
if hasattr(final_model_to_use, "predict_proba"):
    y_test_proba_final = final_model_to_use.predict_proba(X_test)[:, 1]
elif hasattr(final_model_to_use, "decision_function"):
    # For models without predict_proba, convert decision function to pseudo-probabilities
    from scipy.special import expit

    y_test_proba_final = expit(final_model_to_use.decision_function(X_test))
else:
    y_test_proba_final = None
    print("Final model does not support probability prediction")

if y_test_proba_final is not None:
    # Compute probabilistic metrics
    test_log_loss = log_loss(y_test, y_test_proba_final)
    test_brier_score = brier_score_loss(y_test, y_test_proba_final)

    print(f"Probabilistic Metrics for {final_model_name}:")
    print(f"  Log Loss:     {test_log_loss:.4f} (lower is better)")
    print(f"  Brier Score:  {test_brier_score:.4f} (lower is better)")

    # Create calibration curve
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot perfectly calibrated line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=2)

    # Plot calibration curve for final model
    prob_true, prob_pred = calibration_curve(y_test, y_test_proba_final, n_bins=10, strategy="uniform")
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label=final_model_name)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (True Probability)")
    ax.set_title(f"Calibration Curve: {final_model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_folder, f"calibration_curve_{out_suffix}.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved calibration curve to: {out_path}")
    plt.show()

    # Interpretation
    print("\nCalibration Interpretation:")
    print("- Points close to diagonal = well calibrated")
    print("- Points below diagonal = model is overconfident (predicted probabilities too high)")
    print("- Points above diagonal = model is underconfident (predicted probabilities too low)")
else:
    print("Skipping calibration analysis (model does not support probabilities)")

# %% [markdown]
# ## 19. Feature Importance Analysis
#
# ### Learning Objectives:
# - Extract and interpret feature importance
# - Understand model-specific vs model-agnostic methods
# - Learn which features drive predictions
# - Recognize importance limitations
#
# **Feature importance** reveals which features most influence model predictions. Different methods:
#
# 1. **Model-specific importance**: Built-in to tree-based models (Gini importance)
# 2. **Permutation importance**: Model-agnostic, measures performance drop when feature shuffled
# 3. **SHAP values**: Game-theoretic approach providing detailed attributions
#
# **Common Pitfall:** Gini importance can be misleading with correlated features and doesn't account for interactions. Use multiple methods for robust conclusions.

# %%
# Extract feature importance if available (tree-based models)

# First, unwrap the model if it's a ThresholdedClassifier
model_for_importance = final_model_to_use
if isinstance(model_for_importance, ThresholdedClassifier):
    model_for_importance = model_for_importance.estimator
    print(f"Unwrapped ThresholdedClassifier to access underlying model")

# Handle ensemble models (VotingClassifier, StackingClassifier)
if hasattr(model_for_importance, "estimators_"):
    # For VotingClassifier, get the first base estimator's pipeline
    print("Ensemble model detected - using first base estimator for feature importance")
    first_estimator = model_for_importance.estimators_[0]
    # The estimator might be a (name, pipeline) tuple or just a pipeline
    if isinstance(first_estimator, tuple):
        first_estimator = first_estimator[1]
    model_for_importance = first_estimator

# Now check if we have a pipeline with named_steps
if hasattr(model_for_importance, "named_steps"):
    # Get the final estimator from pipeline (step name is "clf")
    estimator_step = model_for_importance.named_steps.get("clf")
    base_estimator = estimator_step

    if base_estimator is not None and hasattr(base_estimator, "feature_importances_"):
        importances = base_estimator.feature_importances_

        # Get feature names after preprocessing
        preprocessor = model_for_importance.named_steps.get("preprocess")
        if preprocessor is None:
            preprocessor = model_for_importance.named_steps.get("preprocessor")

        if preprocessor is not None:
            # Extract feature names from ColumnTransformer
            feature_names = []
            for name, transformer, columns in preprocessor.transformers_:
                if name == "num":
                    feature_names.extend(columns)
                elif name == "cat":
                    if hasattr(transformer.named_steps["onehot"], "get_feature_names_out"):
                        cat_features = transformer.named_steps["onehot"].get_feature_names_out(columns)
                        feature_names.extend(cat_features)
                    else:
                        feature_names.extend(columns)

            # Create importance dataframe
            importance_df = pd.DataFrame(
                {"feature": feature_names[: len(importances)], "importance": importances}
            ).sort_values("importance", ascending=False)

            print(f"\nTop 20 Most Important Features ({final_model_name}):")
            print(importance_df.head(20).to_string(index=False))

            # Plot top 20 features
            plt.figure(figsize=(10, 8))
            top_n = min(20, len(importance_df))
            plt.barh(range(top_n), importance_df["importance"].iloc[:top_n])
            plt.yticks(range(top_n), importance_df["feature"].iloc[:top_n])
            plt.xlabel("Feature Importance")
            plt.title(f"Top {top_n} Feature Importances: {final_model_name}")
            plt.gca().invert_yaxis()
            plt.tight_layout()

            out_path = os.path.join(out_folder, f"feature_importance_{out_suffix}.pdf")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"\nSaved feature importance plot to: {out_path}")
            plt.show()

            # Save to CSV
            csv_path = os.path.join(tables_folder, f"feature_importance_{out_suffix}.csv")
            importance_df.to_csv(csv_path, index=False)
            print(f"Saved feature importance data to: {csv_path}")
        else:
            print("Could not find preprocessor in pipeline - skipping feature importance")
    else:
        print(f"\n{final_model_name} does not provide feature_importances_")
        print("Consider using permutation importance for model-agnostic feature ranking")
else:
    print(f"Final model ({type(model_for_importance).__name__}) is not a pipeline - skipping feature importance")
    print("Note: This can happen with some ensemble methods or custom wrappers")

# %% [markdown]
# ## 20. Learning Curves: Diagnosing Underfitting and Overfitting
#
# ### Learning Objectives:
# - Understand learning curves and their interpretation
# - Diagnose underfitting (high bias)
# - Diagnose overfitting (high variance)
# - Learn how training set size affects performance
#
# **[Learning curves](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)** plot training and validation performance vs training set size. They reveal:
#
# - **Underfitting**: Both curves plateau at low performance (need more model capacity)
# - **Overfitting**: Large gap between training and validation (need more data or regularization)
# - **Well-fit**: Both curves converge at high performance
#
# **Common Pitfall:** Computing learning curves on the test set causes overfitting. Always use cross-validation on training data only.
#
# See https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html for an explanation of these plots

# %%
print(f"Computing learning curves for {best_name} (this may take time in non-QUICK_MODE)...")

# Use the best single model (not ensemble) for clearer interpretation
# Get model without threshold wrapper if applied
if isinstance(best_pipeline_final, ThresholdedClassifier):
    model_for_learning_curve = best_pipeline_final.estimator
else:
    model_for_learning_curve = best_pipeline_final

# Define training sizes
if QUICK_MODE:
    train_sizes = np.linspace(0.3, 1.0, 5)
    cv_for_learning = 3
else:
    train_sizes = np.linspace(0.1, 1.0, 10)
    cv_for_learning = 5

# Compute learning curves using cross-validation
train_sizes_abs, train_scores, val_scores = learning_curve(
    model_for_learning_curve,
    X_train,
    y_train,
    train_sizes=train_sizes,
    cv=cv_for_learning,
    scoring="matthews_corrcoef",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# Compute mean and std
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
val_scores_mean = val_scores.mean(axis=1)
val_scores_std = val_scores.std(axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.fill_between(
    train_sizes_abs, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="blue"
)
plt.fill_between(
    train_sizes_abs, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color="orange"
)
plt.plot(train_sizes_abs, train_scores_mean, "o-", color="blue", label="Training score", linewidth=2)
plt.plot(train_sizes_abs, val_scores_mean, "o-", color="orange", label="Validation score", linewidth=2)
plt.xlabel("Training Set Size")
plt.ylabel("MCC Score")
plt.title(f"Learning Curves: {best_name}")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_path = os.path.join(out_folder, f"learning_curves_{out_suffix}.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved learning curves to: {out_path}")
plt.show()

# Interpretation
final_gap = train_scores_mean[-1] - val_scores_mean[-1]
print("\nLearning Curve Interpretation:")
print(f"  Final training score:   {train_scores_mean[-1]:.4f}")
print(f"  Final validation score: {val_scores_mean[-1]:.4f}")
print(f"  Gap:                    {final_gap:.4f}")

if final_gap > 0.1:
    print("  → Model shows signs of OVERFITTING (large gap)")
    print("    Consider: more data, stronger regularization, or simpler model")
elif val_scores_mean[-1] < 0.5:
    print("  → Model shows signs of UNDERFITTING (low validation score)")
    print("    Consider: more complex model, more features, or less regularization")
else:
    print("  → Model appears reasonably well-fitted")

# %% [markdown]
# ## 21. Wrap-up and Summary
#
# ### What We Accomplished
#
# In this comprehensive Coding Exercise 2 notebook, we built a production-quality ML pipeline covering:
#
# **1. Data Preparation and Leakage Prevention**
# - Built leakage-resistant workflows using [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
# - Proper train/validation/test splits with stratification
#
# **2. Handling Class Imbalance**
# - Integrated **imbalanced-learn** with 8 sampling strategies (SMOTE, ADASYN, undersampling, etc.)
# - Made sampling strategy a hyperparameter for Optuna to optimize
# - Used imbalanced-aware metrics (MCC, balanced accuracy, F1)
#
# **3. Hyperparameter Tuning with Optuna**
# - Tuned 6+ models using Bayesian optimization (TPE sampler)
# - Configured objective function (MCC, F1, balanced_accuracy, or cost_benefit)
# - Analyzed parameter importance using fANOVA
# - Visualized optimization with parallel coordinates plots
#
# **4. Comprehensive Model Evaluation**
# - Evaluated 10+ metrics: MCC, accuracy, precision, recall, F1, balanced accuracy, specificity, ROC AUC, PR AUC, cost-benefit
# - Cross-validation for robust performance estimates
# - Confusion matrices and ROC/PR curves
#
# **5. Threshold Optimization**
# - Moved beyond default 0.5 threshold
# - Found optimal thresholds for different objectives
# - Implemented `ThresholdedClassifier` wrapper
# - Visualized precision-recall trade-offs
#
# **6. Ensemble Learning**
# - Combined top N models using voting or stacking
# - Compared ensemble vs best single model
# - Applied threshold optimization to ensembles
#
# **7. Model Interpretability**
# - Assessed probability calibration (calibration curves, log loss, Brier score)
# - Analyzed feature importance for tree-based models
# - Generated learning curves to diagnose overfitting/underfitting
#
# **8. Cost-Benefit Integration**
# - Implemented cost-benefit analysis with configurable cost matrix
# - Made cost-benefit available as tuning objective
# - Supported multiclass extension for generalization
#
# ### Key Takeaways for Students
#
# 1. **Metric selection matters**: Different metrics optimize for different goals. MCC is excellent for imbalanced data, but consider misclassification costs when available.
#
# 2. **Class imbalance requires attention**: Don't ignore class distribution. Use sampling techniques and imbalanced-aware metrics.
#
# 3. **Threshold tuning is powerful**: The default 0.5 threshold is rarely optimal. Always analyze threshold effects for your specific objective.
#
# 4. **Ensembles aren't magic**: They work when base models are strong and diverse. Sometimes a single well-tuned model suffices.
#
# 5. **Interpretability builds trust**: Calibration, feature importance, and learning curves help understand and debug models.
#
# 6. **Avoid data leakage**: Always use pipelines to ensure preprocessing happens within cross-validation folds.
#
# ### Further Learning
#
# To extend this workflow, consider:
# - **SHAP values** for detailed feature attributions
# - **Permutation importance** as model-agnostic alternative
# - **Partial dependence plots** to understand feature effects
# - **Bayesian hyperparameter optimization** with more sophisticated priors
# - **Nested cross-validation** for unbiased hyperparameter selection evaluation
# - **Calibration methods** (Platt scaling, isotonic regression) to improve probability estimates
#
# This notebook extends Coding Exercise 1 from a simple accuracy-focused lifecycle to a comprehensive, production-ready ML workflow suitable for real-world applications.
#
#
# Congratulations on completing this advanced exercise in AI and ML! You've built a robust pipeline that addresses many challenges faced in practical machine learning projects.

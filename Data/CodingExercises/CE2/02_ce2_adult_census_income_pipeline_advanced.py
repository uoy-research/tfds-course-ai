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
__version__ = "0.0.2"
__date__ = "2026-02-04"

# %% [markdown]
# # Coding Exercise 2 (Advanced) - Pipelines, Metrics and Hyperparameter Tuning
#
# ## About This Version
#
# This is the **Advanced** version of Coding Exercise 2. It covers:
# - ML pipelines with preprocessing and resampling (SMOTE)
# - Multiple classification metrics for imbalanced data
# - Hyperparameter tuning with Optuna
# - Threshold optimisation
#
# **For additional topics** including ensemble methods, probability calibration, feature importance
# analysis, and learning curves, see `03_ce2_adult_census_income_pipeline_expert.py`.
#
# ## Learning Objectives
#
# By the end of this tutorial, you will understand:
# 1. How to build production-ready ML pipelines that prevent data leakage
# 2. How to handle imbalanced datasets using SMOTE resampling
# 3. How to select and interpret appropriate evaluation metrics for imbalanced classification
# 4. How to tune hyperparameters systematically using Optuna
# 5. How to optimise decision thresholds for different objectives
#
# ## 0. Setup
#
# ### Configuration Guide
#
# Key parameters you can modify:
#
# - `QUICK_MODE`: Set to `True` for faster runs, `False` for full evaluation
# - `MODEL_NAMES`: List of models to evaluate
# - `TUNED_MODELS`: Models to hyperparameter tune with Optuna
# - `TUNING_OBJECTIVE`: Choose from `"mcc"`, `"f1"`, or `"balanced_accuracy"`
# - `USE_OPTIMAL_THRESHOLD`: Whether to find and apply optimal decision threshold
#
# **Reproducibility:** All random operations use `RANDOM_STATE` for reproducible results.

# %%
# Configuration and global parameters
RANDOM_STATE = 42
TEST_SIZE = 0.5  # 50% train / 50% test

# Quick mode for faster development and testing
QUICK_MODE = True  # Set to True for faster runs with reduced trials/splits

# Cross-validation configuration
N_SPLITS = 3 if QUICK_MODE else 5
N_REPEATS_TUNING = 1
N_REPEATS_CV = 1 if QUICK_MODE else 2

# Parallelism
NUM_JOBS = -1  # -1 means use all available cores

# Seeds
SEED_TUNING = RANDOM_STATE
SEED_CV = RANDOM_STATE

out_folder = "coding_exercise_2_advanced"
optuna_folder = "coding_exercise_2_advanced/optuna"
tables_folder = "coding_exercise_2_advanced/tables"
out_suffix = "_adv"

# Optimization objective configuration
TUNING_OBJECTIVE = "mcc"  # Options: "mcc", "f1", "balanced_accuracy"

# Threshold optimization configuration
USE_OPTIMAL_THRESHOLD = True

# Model configuration
ALL_MODELS = [
    "DummyMostFreq",
    "LogisticRegression",
    "RandomForest",
    "LightGBM",
]

# Sampling strategies - simplified for advanced version
SAMPLING_METHODS = ["none", "smote"]

if QUICK_MODE:
    MODEL_NAMES = ["DummyMostFreq", "RandomForest", "LightGBM"]
    TUNED_MODELS = ["LightGBM"]
else:
    MODEL_NAMES = ALL_MODELS
    TUNED_MODELS = ["LightGBM", "RandomForest"]

# Validation
if not set(TUNED_MODELS).issubset(set(MODEL_NAMES)):
    missing = set(TUNED_MODELS) - set(MODEL_NAMES)
    raise ValueError(f"TUNED_MODELS contains models not in MODEL_NAMES: {missing}")

N_TRIALS_PER_MODEL = {
    "DummyMostFreq": 1,
    "LogisticRegression": 20 if not QUICK_MODE else 5,
    "RandomForest": 50 if not QUICK_MODE else 10,
    "LightGBM": 50 if not QUICK_MODE else 15,
}

# %%
# Imports
import sys
import warnings

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    import subprocess

    packages = ["optuna", "imbalanced-learn", "lightgbm"]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, RobustScaler

sns.set_theme(style="whitegrid", context="notebook")
np.random.seed(RANDOM_STATE)

os.makedirs(out_folder, exist_ok=True)
os.makedirs(optuna_folder, exist_ok=True)
os.makedirs(tables_folder, exist_ok=True)

sklearn.set_config(transform_output="pandas")

# %%
# Load the Adult Census Income dataset
DATA_URL = "https://github.com/medmaca/shared_data/raw/8a3fea5467ec68b17fd8369c6f77f8016b1ed5f8/Datasets/Kaggle/adult_census_income/adult.csv.zip"

adult_ci_df = pd.read_csv(DATA_URL, compression="zip")
adult_ci_df.head()

# %% [markdown]
# ## 1. Minimal Dataset Checks
#
# For full EDA, see Coding Exercise 1. Here we verify the data and check class imbalance.

# %%
adult_ci_df.info()

target_col = "income"

class_counts = adult_ci_df[target_col].value_counts().sort_index()
class_props = adult_ci_df[target_col].value_counts(normalize=True).sort_index()
print(f"\nClass counts:{class_counts}")
print(f"\nClass proportions:{class_props}")

# %% [markdown]
# ### Understanding Class Imbalance
#
# The dataset shows approximately 75% <=50K and 25% >50K (3:1 imbalance). This means:
# - A naive classifier predicting "<=50K" would achieve 75% accuracy
# - Accuracy alone is misleading for imbalanced data
# - We need metrics that account for both classes equally
#
# **Common Pitfall:** Using accuracy as the primary metric for imbalanced data leads to
# models that predict only the majority class.

# %% [markdown]
# ## 2. Understanding Classification Metrics
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
# ### Key Metrics
#
# | Metric | Formula | Best For |
# |--------|---------|----------|
# | **Accuracy** | (TP+TN) / Total | Balanced classes only |
# | **Precision** | TP / (TP+FP) | When FP is costly |
# | **Recall** | TP / (TP+FN) | When FN is costly |
# | **Specificity** | TN/ (TN + FP) | How may negatives did we correctly identify |
# | **F1 Score** | 2×(Prec×Rec)/(Prec+Rec) | Balance precision/recall |
# | **MCC** | See formula below | Imbalanced data |
# | **Balanced Accuracy** | (Recall+Specificity)/2 | Imbalanced data |
#
# **[Matthews Correlation Coefficient (MCC)](https://en.wikipedia.org/wiki/Phi_coefficient):**
#
# $$
# \text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
# $$
#
# [MCC](https://en.wikipedia.org/wiki/Phi_coefficient) ranges from -1 to +1, with 0 being random prediction. It uses all four confusion
# matrix values and is robust to class imbalance.
#
# For cost-benefit analysis and multiclass extensions, see `03_ce2_adult_census_income_pipeline_expert.py`.

# %% [markdown]
# ## 3. Handle Missing Data

# %%
# Replace '?' with NaN
adult_ci_df = adult_ci_df.replace("?", np.nan)

feature_cols = [c for c in adult_ci_df.columns if c != target_col]
categorical_features = [c for c in feature_cols if adult_ci_df[c].dtype == "object"]
numeric_features = [c for c in feature_cols if adult_ci_df[c].dtype != "object"]

print("Categorical features:", categorical_features)
print("Numeric features:", numeric_features)

# %% [markdown]
# ## 4. Encode Target and Split Data

# %%
lb = LabelBinarizer()
adult_ci_df["target"] = lb.fit_transform(adult_ci_df[target_col].str.strip()).ravel()
print("Target classes (lb.classes_):", lb.classes_)

X = adult_ci_df[feature_cols].copy()
y = adult_ci_df["target"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE, shuffle=True
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# %% [markdown]
# ## 5. Handling Class Imbalance with SMOTE
#
# **SMOTE (Synthetic Minority [Over-sampling Technique](https://imbalanced-learn.org/stable/over_sampling.html))** creates synthetic samples by
# interpolating between minority class neighbours.
#
# **Critical Rules:**
# 1. Only resample training data - never test/validation data
# 2. Resample after train/test split to prevent data leakage
# 3. Use `imblearn.pipeline.Pipeline` to ensure resampling happens within CV folds
#
# For additional sampling methods (ADASYN, undersampling, hybrid methods), see `03_ce2_adult_census_income_pipeline_expert.py`.

# %% [markdown]
# ## 6. Utility Functions
#
# We define helper functions to compute metrics not available in sklearn or to
# encapsulate repeated operations. Organising code into functions improves:
# - **Readability**: Each function has a single, clear purpose
# - **Testability**: Functions can be tested independently
# - **Reusability**: The same function can be called from multiple places


# %%
def specificity_score(y_true, y_pred):
    """Calculate specificity (true negative rate).

    Specificity measures the proportion of actual negatives that were correctly
    identified. Also known as the True Negative Rate (TNR).

    Args:
        y_true: Array-like of true binary labels (0 or 1).
        y_pred: Array-like of predicted binary labels (0 or 1).

    Returns:
        float: Specificity score in range [0, 1], or np.nan if computation
            is not possible (e.g., no negative samples).

    Example:
        >>> specificity_score([0, 0, 1, 1], [0, 1, 1, 1])
        0.5  # 1 of 2 negatives correctly identified
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return np.nan
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    return tn / denom if denom > 0 else np.nan


# %% [markdown]
# ## 7. Preprocessing Pipeline
#
# The preprocessing pipeline applies different transformations to numeric and
# categorical columns. We use sklearn's `ColumnTransformer` to route each
# column type to the appropriate transformer.
#
# **Design Pattern: Factory Function**
#
# The `build_pipeline()` function is a *factory* - it creates and returns
# pipeline objects. This pattern allows us to:
# - Create pipelines with different configurations (with/without SMOTE)
# - Ensure consistent preprocessing across all experiments
# - Easily swap the estimator without duplicating pipeline code

# %%
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
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
    """Build a preprocessing pipeline with optional SMOTE resampling.

    Creates a complete ML pipeline that chains preprocessing, optional
    resampling, and the classifier. Uses imbalanced-learn's Pipeline when
    SMOTE is needed (as sklearn's Pipeline doesn't support resamplers).

    Args:
        estimator: A scikit-learn compatible classifier instance.
        sampling_strategy: Resampling method to apply. Options:
            - "none": No resampling, uses standard sklearn Pipeline.
            - "smote": Apply SMOTE oversampling, uses imblearn Pipeline.

    Returns:
        Pipeline: A fitted-ready pipeline object (sklearn or imblearn).

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> pipe = build_pipeline(RandomForestClassifier(), sampling_strategy="smote")
        >>> pipe.fit(X_train, y_train)
    """
    steps = [("preprocess", preprocessor)]

    if sampling_strategy == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE)
        steps.append(("sampler", sampler))
        steps.append(("clf", estimator))
        return ImbPipeline(steps=steps)
    else:
        steps.append(("clf", estimator))
        return Pipeline(steps=steps)


# %%
preprocessor


# %% [markdown]
# ## 8. Model Zoo and Hyperparameter Search Spaces
#
# This section defines two key functions that work together:
#
# 1. **`create_base_estimator()`**: A factory function that creates classifier
#    instances with sensible default parameters. This centralises model creation
#    so defaults are consistent throughout the code.
#
# 2. **`suggest_params_for_model()`**: Defines the hyperparameter search space
#    for each model. Optuna calls this to get parameter suggestions for each trial.
#
# **Why separate these functions?**
# - `create_base_estimator()` is used both for tuning and final model creation
# - `suggest_params_for_model()` encapsulates the Optuna trial interface
# - This separation makes it easy to add new models


# %%
def create_base_estimator(model_name, params=None):
    """Create a classifier instance with sensible defaults.

    Factory function that returns sklearn-compatible classifiers. Default
    parameters are set for reasonable out-of-the-box performance, and can
    be overridden by the params argument.

    Args:
        model_name: String identifier for the model. Supported values:
            "DummyMostFreq", "LogisticRegression", "RandomForest", "LightGBM".
        params: Optional dict of parameters to override defaults. Keys should
            match the classifier's constructor arguments.

    Returns:
        A scikit-learn compatible classifier instance.

    Raises:
        ValueError: If model_name is not recognised.

    Example:
        >>> clf = create_base_estimator("LightGBM", {"n_estimators": 200})
        >>> clf.n_estimators
        200
    """
    params = dict(params or {})

    if model_name == "DummyMostFreq":
        return DummyClassifier(strategy="most_frequent")

    if model_name == "LogisticRegression":
        base = {
            "max_iter": 10000,
            "C": 15,
            "class_weight": "balanced",
            "n_jobs": NUM_JOBS,
            "random_state": RANDOM_STATE,
        }
        base.update(params)
        return LogisticRegression(**base)

    if model_name == "RandomForest":
        base = {
            "n_estimators": 200,
            "max_depth": None,
            "n_jobs": NUM_JOBS,
            "random_state": RANDOM_STATE,
        }
        base.update(params)
        return RandomForestClassifier(**base)

    if model_name == "LightGBM":
        base = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": -1,
            "num_leaves": 31,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbose": -1,
        }
        base.update(params)
        return lgb.LGBMClassifier(**base)

    raise ValueError(f"Unknown model_name: {model_name}")


def suggest_params_for_model(trial, model_name):
    """Define Optuna search spaces for each model.

    This function is called by Optuna during hyperparameter optimisation.
    It uses the trial object to suggest parameter values, and Optuna
    learns which regions of the parameter space yield better results.

    Args:
        trial: An Optuna trial object that provides suggest_* methods
            for proposing hyperparameter values.
        model_name: String identifier for the model being tuned.

    Returns:
        tuple: A pair of (params_dict, sampling_strategy) where:
            - params_dict: Dictionary of hyperparameter values to pass
              to create_base_estimator().
            - sampling_strategy: String ("none" or "smote") indicating
              the resampling method to use.

    Raises:
        ValueError: If model_name has no defined search space.

    Example:
        >>> # Inside an Optuna objective function:
        >>> params, sampling = suggest_params_for_model(trial, "LightGBM")
        >>> estimator = create_base_estimator("LightGBM", params=params)
    """
    base_est = create_base_estimator(model_name)
    base_params = base_est.get_params(deep=False)

    if model_name == "DummyMostFreq":
        return base_params, "none"

    sampling_strategy = trial.suggest_categorical("sampling_strategy", SAMPLING_METHODS)

    if model_name == "LogisticRegression":
        params = dict(base_params)
        params["C"] = trial.suggest_float("C", 1e-3, 1e3, log=True)
        params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
        return params, sampling_strategy

    if model_name == "RandomForest":
        params = dict(base_params)
        params["n_estimators"] = trial.suggest_int("n_estimators", 100, 300)
        params["max_depth"] = trial.suggest_int("max_depth", 5, 50)
        params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
        params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
        return params, sampling_strategy

    if model_name == "LightGBM":
        params = dict(base_params)
        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
        params["max_depth"] = trial.suggest_int("max_depth", 3, 50)
        params["num_leaves"] = trial.suggest_int("num_leaves", 10, 100)
        params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
        return params, sampling_strategy

    raise ValueError(f"No search space for: {model_name}")


# %% [markdown]
# ## 9. Metrics Configuration

# %%
mcc_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)

if TUNING_OBJECTIVE == "mcc":
    tuning_scorer = mcc_scorer
    tuning_scorer_name = "MCC"
elif TUNING_OBJECTIVE == "f1":
    tuning_scorer = make_scorer(f1_score, greater_is_better=True)
    tuning_scorer_name = "F1"
elif TUNING_OBJECTIVE == "balanced_accuracy":
    tuning_scorer = make_scorer(balanced_accuracy_score, greater_is_better=True)
    tuning_scorer_name = "Balanced Accuracy"
else:
    raise ValueError(f"Unknown TUNING_OBJECTIVE: {TUNING_OBJECTIVE}")

print(f"Optimizing for: {tuning_scorer_name}")

# %% [markdown]
# ## 10. Hyperparameter Tuning with Optuna
#
# **Optuna** uses Bayesian optimization (TPE) to efficiently search the hyperparameter space.
# This is more efficient than grid or random search for large search spaces.
#
# | Method | Strategy | Best For |
# |--------|----------|----------|
# | Grid Search | Exhaustive | Small spaces, few parameters |
# | Random Search | Random sampling | Initial exploration |
# | Bayesian (TPE) | Learns from trials | Large spaces, expensive evaluations |
#
# ### Tuning Workflow
#
# For each model in `TUNED_MODELS`, we:
# 1. Create an Optuna study to track optimisation progress
# 2. Define an objective function that builds a pipeline and returns CV score
# 3. Run `study.optimize()` which calls the objective N times
# 4. Store the best parameters for later use


# %%
def make_tuning_cv():
    """Create cross-validation splitter for hyperparameter tuning.

    Uses fewer repeats during tuning for speed, since we're comparing
    many different hyperparameter configurations.

    Returns:
        RepeatedStratifiedKFold: CV splitter configured for tuning.
    """
    return RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS_TUNING, random_state=SEED_TUNING)


def objective(trial, model_name):
    """Optuna objective function for hyperparameter optimisation.

    This function is called by Optuna for each trial. It builds a pipeline
    with the suggested hyperparameters, evaluates it via cross-validation,
    and returns the mean score.

    Args:
        trial: Optuna trial object that provides suggest_* methods.
        model_name: String identifier for the model being tuned.

    Returns:
        float: Mean cross-validation score for the tuning metric.
    """
    params, sampling_strategy = suggest_params_for_model(trial, model_name)
    estimator = create_base_estimator(model_name, params=params)
    pipe = build_pipeline(estimator, sampling_strategy=sampling_strategy)

    cv = make_tuning_cv()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=tuning_scorer, n_jobs=NUM_JOBS)
    return float(np.mean(scores))


tuning_records = []
best_params_by_model = {}
studies_by_model = {}

for model_name in TUNED_MODELS:
    print(f"Starting Optuna tuning for: {model_name}")
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"{model_name}_tuning")

    def _objective(trial):
        return objective(trial, model_name)

    N_TRIALS = N_TRIALS_PER_MODEL.get(model_name, 30)
    study.optimize(_objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"Best {tuning_scorer_name} for {model_name}: {study.best_value:.4f}")
    print("Best params:", study.best_params)

    best_params_by_model[model_name] = study.best_params
    studies_by_model[model_name] = study

    for trial in study.trials:
        rec = {"model": model_name, "trial": trial.number, f"objective_{TUNING_OBJECTIVE}": trial.value}
        for k, v in trial.params.items():
            rec[f"param_{k}"] = v
        tuning_records.append(rec)

tuning_results_df = pd.DataFrame(tuning_records) if tuning_records else pd.DataFrame()
if not tuning_results_df.empty:
    tuning_results_df.to_csv(os.path.join(tables_folder, "optuna_tuning_all_trials.csv"), index=False)

# %% [markdown]
# For hyperparameter importance analysis and parallel coordinate visualizations,
# see `03_ce2_adult_census_income_pipeline_expert.py`.

# %% [markdown]
# ## 11. Post-Tuning Cross-Validation
#
# After finding the best hyperparameters for each model, we perform a more
# rigorous cross-validation to get reliable performance estimates.
#
# **Why a separate CV phase?**
# - During tuning, we use a lightweight CV (fewer repeats) for speed
# - For final model comparison, we want more stable estimates
# - This also helps detect if we overfit to the tuning CV folds


# %%
def make_eval_cv():
    """Create cross-validation splitter for final model evaluation.

    Uses more repeats than tuning CV for more stable performance estimates.

    Returns:
        RepeatedStratifiedKFold: CV splitter configured for evaluation.
    """
    return RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS_CV, random_state=SEED_CV)


cv_records = []
rskf = make_eval_cv()

for model_name in MODEL_NAMES:
    all_params = dict(best_params_by_model.get(model_name, {}))
    sampling_strategy = all_params.pop("sampling_strategy", "none")

    estimator = create_base_estimator(model_name, params=all_params)
    pipe = build_pipeline(estimator, sampling_strategy=sampling_strategy)

    print(f"Evaluating model (CV): {model_name}")
    for split_idx, (train_idx, val_idx) in enumerate(rskf.split(X_train, y_train), start=1):
        repeat_idx = (split_idx - 1) // N_SPLITS + 1
        fold_idx = (split_idx - 1) % N_SPLITS + 1

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            pipe.fit(X_tr, y_tr)
            y_val_pred = pipe.predict(X_val)

        if hasattr(pipe, "predict_proba"):
            y_val_proba = pipe.predict_proba(X_val)[:, 1]
        else:
            y_val_proba = y_val_pred.astype(float)

        mcc = matthews_corrcoef(y_val, y_val_pred)
        acc = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred, zero_division=0)
        rec = recall_score(y_val, y_val_pred, zero_division=0)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, y_val_pred)
        spec = specificity_score(y_val, y_val_pred)
        try:
            roc_auc = roc_auc_score(y_val, y_val_proba)
        except ValueError:
            roc_auc = np.nan

        cv_records.append(
            {
                "model": model_name,
                "repeat": repeat_idx,
                "fold": fold_idx,
                "sampling_strategy": sampling_strategy,
                "mcc": mcc,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "balanced_accuracy": bal_acc,
                "specificity": spec,
                "roc_auc": roc_auc,
            }
        )

cv_results_df = pd.DataFrame(cv_records)
cv_results_df.to_csv(os.path.join(tables_folder, "cv_results_per_fold.csv"), index=False)

# %%
metric_cols = ["mcc", "accuracy", "precision", "recall", "f1", "balanced_accuracy", "specificity", "roc_auc"]
model_comparison_df = pd.DataFrame()
for metric in metric_cols:
    model_comparison_df[f"mean_{metric}"] = cv_results_df.groupby("model")[metric].mean()
    model_comparison_df[f"std_{metric}"] = cv_results_df.groupby("model")[metric].std()

model_comparison_df = model_comparison_df.reset_index()
model_comparison_df.to_csv(os.path.join(tables_folder, "model_comparison_summary.csv"), index=False)
print("\nModel Comparison Summary:")
model_comparison_df.sort_values("mean_mcc", ascending=False)

# %% [markdown]
# ## 12. Select Best Model and Refit

# %%
objective_col = TUNING_OBJECTIVE
mean_score_by_model = cv_results_df.groupby("model")[objective_col].mean().sort_values(ascending=False)
best_model_name = mean_score_by_model.index[0]
print(f"Best model by {TUNING_OBJECTIVE}: {best_model_name}")

best_params = best_params_by_model.get(best_model_name, {})
all_params = best_params.copy()
sampling_strategy = all_params.pop("sampling_strategy", "none")

best_estimator = create_base_estimator(best_model_name, params=all_params)
best_pipeline = build_pipeline(best_estimator, sampling_strategy=sampling_strategy)
best_pipeline.fit(X_train, y_train)
print(f"Fitted best model pipeline")

# %% [markdown]
# ## 13. Final Test Set Evaluation

# %%
y_test_pred = best_pipeline.predict(X_test)
y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

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
print(f"Test recall: {test_rec:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Test balanced accuracy: {test_bal_acc:.4f}")
print(f"Test specificity: {test_spec:.4f}")
print(f"Test ROC AUC: {test_roc_auc:.4f}")

# Confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lb.classes_)
disp.plot(cmap="Reds", values_format="d")
plt.gca().grid(False)
plt.title(f"Confusion matrix ({best_model_name})")
plt.tight_layout()
plt.savefig(os.path.join(out_folder, f"confusion_matrix{out_suffix}.pdf"), dpi=150, bbox_inches="tight")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {test_roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve (test set)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_folder, f"roc_curve{out_suffix}.pdf"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 14. Threshold Optimisation
#
# Most classifiers output probabilities, and we convert these to class predictions
# using a **decision threshold** (default 0.5). However, the optimal threshold
# depends on:
#
# - **Class imbalance**: With imbalanced classes, 0.5 may not be the natural decision boundary learned by the model
# - **Cost structure**: If false negatives are more costly than false positives (or vice versa), we should adjust the threshold accordingly
# - **Metric of interest**: Different metrics are optimised at different thresholds
#
# Below we sweep through thresholds from 0 to 1 and compute each metric, then
# identify the optimal threshold for our chosen objective.

# %%
thresholds = np.linspace(0.0, 1.0, 100)  # generate 100 thresholds from 0.0 to 1.0
th_records = []

for thr in thresholds:
    y_thr = (y_test_proba >= thr).astype(int)
    th_records.append(
        {
            "threshold": thr,
            "mcc": matthews_corrcoef(y_test, y_thr),
            "f1": f1_score(y_test, y_thr, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_thr),
        }
    )

threshold_df = pd.DataFrame(th_records)
threshold_df.to_csv(os.path.join(tables_folder, "threshold_analysis.csv"), index=False)

# Find optimal thresholds
optimal_thresholds = {}
for metric in ["mcc", "f1", "balanced_accuracy"]:
    optimal_idx = threshold_df[metric].idxmax()
    optimal_thresholds[metric] = threshold_df.loc[optimal_idx, "threshold"]
    print(f"Optimal threshold for {metric}: {optimal_thresholds[metric]:.3f}")

# %%
# Apply optimal threshold
if USE_OPTIMAL_THRESHOLD:
    optimal_threshold = optimal_thresholds.get(TUNING_OBJECTIVE, 0.5)
    y_test_pred_opt = (y_test_proba >= optimal_threshold).astype(int)

    print(f"\nUsing optimal threshold: {optimal_threshold:.3f}")
    print(f"MCC (default 0.5): {test_mcc:.4f}")
    print(f"MCC (optimal {optimal_threshold:.2f}): {matthews_corrcoef(y_test, y_test_pred_opt):.4f}")

# %% [markdown]
# ## 15. Summary
#
# This advanced tutorial covered:
# - ML pipelines with preprocessing and SMOTE resampling
# - Multiple classification metrics for imbalanced data
# - Hyperparameter tuning with Optuna (Bayesian optimization)
# - Threshold optimisation for different objectives
#
# **For additional topics**, see `03_ce2_adult_census_income_pipeline_expert.py`:
# - Ensemble methods (voting, stacking)
# - Probability calibration analysis
# - Feature importance analysis
# - Learning curves for overfitting diagnosis
# - Cost-benefit analysis
# - Additional sampling methods (ADASYN, undersampling, hybrid)

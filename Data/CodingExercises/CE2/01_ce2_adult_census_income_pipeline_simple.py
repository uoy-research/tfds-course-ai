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
__version__ = "0.0.3"
__date__ = "2026-02-04"

# %% [markdown]
# # Coding Exercise 2 (Simple) - Introduction to ML Pipelines and Hyperparameter Tuning
#
# ## About This Version
#
# This is the **Simple** version of Coding Exercise 2, designed as an introduction to:
# - ML pipelines for preprocessing
# - Classification metrics for imbalanced data
# - Basic hyperparameter tuning with Optuna
#
# **To progress further**, see:
# - `02_ce2_adult_census_income_pipeline_advanced.py` for SMOTE resampling and threshold optimisation
# - `03_ce2_adult_census_income_pipeline_expert.py` for ensemble methods, calibration, and more
#
# ## Learning Objectives
#
# By the end of this tutorial, you will understand:
# 1. How to build ML pipelines that prevent data leakage
# 2. Why accuracy is misleading for imbalanced data
# 3. How to use MCC (Matthews Correlation Coefficient) as a better metric
# 4. How to tune a few key hyperparameters using Optuna
#
# ## 0. Setup
#
# Key parameters you can modify:
# - `QUICK_MODE`: Set to `True` for faster runs
# - `TUNING_OBJECTIVE`: The metric to optimise (`"mcc"` or `"f1"`)

# %%
# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.5
QUICK_MODE = True

N_SPLITS = 3 if QUICK_MODE else 5
N_REPEATS_CV = 1 if QUICK_MODE else 2
NUM_JOBS = -1

out_folder = "coding_exercise_2_simple"
out_suffix = "_simple"

# Tuning configuration
TUNING_OBJECTIVE = "mcc"  # Options: "mcc", "f1"
N_TRIALS = 10 if QUICK_MODE else 30  # Number of Optuna trials

# %% [markdown]
# ### Understanding Imports
#
# Python uses `import` statements to load external libraries. Each library provides
# specialised functionality:
#
# - **pandas** (`pd`): Data manipulation with DataFrames (like Excel spreadsheets)
# - **numpy** (`np`): Numerical computing with arrays
# - **sklearn**: Machine learning algorithms and tools
# - **optuna**: Automated hyperparameter tuning
# - **lightgbm** (`lgb`): Fast gradient boosting library
# - **matplotlib/seaborn**: Plotting and visualisation

# %%
# Imports - load external libraries we need
import sys  # System utilities (to check if we're in Google Colab)

# Check if running in Google Colab (an online notebook environment)
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    import subprocess

    packages = ["optuna", "lightgbm"]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import os  # Operating system utilities (for file paths)

# Core data science libraries
import lightgbm as lgb  # Fast gradient boosting (our ML model)
import matplotlib.pyplot as plt  # Plotting
import numpy as np  # Numerical arrays and math
import optuna  # Hyperparameter tuning library
import pandas as pd  # DataFrames for tabular data
import seaborn as sns  # Statistical visualisation
import sklearn  # Scikit-learn: the main ML library

# TPESampler uses Bayesian optimisation to search hyperparameters efficiently
from optuna.samplers import TPESampler

# Sklearn components - we import specific tools from sklearn's submodules
# ColumnTransformer applies different preprocessing to different column types
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier  # Simple baseline model
from sklearn.impute import SimpleImputer  # Fills in missing values

# Metrics to evaluate model performance
from sklearn.metrics import (
    accuracy_score,  # Proportion of correct predictions
    confusion_matrix,  # Table of prediction outcomes
    f1_score,  # Balance of precision and recall
    make_scorer,  # Converts a metric function into a scorer for CV
    matthews_corrcoef,  # MCC - our primary metric for imbalanced data
    precision_score,  # Of predicted positives, how many are correct
    recall_score,  # Of actual positives, how many did we find
    roc_auc_score,  # Area under ROC curve
)

# Model selection utilities
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline  # Chains preprocessing and model together
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, RobustScaler

sns.set_theme(style="whitegrid", context="notebook")
np.random.seed(RANDOM_STATE)

os.makedirs(out_folder, exist_ok=True)

sklearn.set_config(transform_output="pandas")

# %%
# Load data
DATA_URL = "https://github.com/medmaca/shared_data/raw/8a3fea5467ec68b17fd8369c6f77f8016b1ed5f8/Datasets/Kaggle/adult_census_income/adult.csv.zip"

adult_ci_df = pd.read_csv(
    DATA_URL, compression="zip"
)  # The dataset is a CSV file inside a ZIP archive. Pandas can predict compression type but we specify it to be sure.
adult_ci_df.head()

# %% [markdown]
# ## 1. Data Overview
#
# For full EDA, see Coding Exercise 1.

# %%
adult_ci_df.info()

target_col = "income"
class_props = adult_ci_df[target_col].value_counts(normalize=True)  # Proportion of each class in the target variable
print(f"\nClass proportions:\n{class_props}")

# %% [markdown]
# ### The Class Imbalance Problem
#
# The dataset has approximately 75% <=50K and 25% >50K. This is a **3:1 class imbalance**.
#
# **Why this matters:** A model that always predicts "<=50K" would achieve 75% accuracy,
# but would be completely useless for identifying high earners.
#
# **Solution:** Use metrics designed for imbalanced data, such as MCC or F1.
#
# For techniques to handle class imbalance (SMOTE resampling),
# see `02_ce2_adult_census_income_pipeline_advanced.py`.

# %% [markdown]
# ## 2. Key Classification Metrics
#
# ### The Confusion Matrix
#
# $$
# \begin{array}{c|cc}
# & \text{Pred. Neg} & \text{Pred. Pos} \\
# \hline
# \text{Actual Neg} & TN & FP \\
# \text{Actual Pos} & FN & TP \\
# \end{array}
# $$
#
# ### Metrics Summary
#
# | Metric | Description | Use When |
# |--------|-------------|----------|
# | **Accuracy** | Overall correctness | Classes are balanced |
# | **MCC** | Correlation between prediction and truth | Imbalanced data (recommended) |
# | **F1** | Balance of precision and recall | Positive class is important |
#
# **MCC (Matthews Correlation Coefficient)** ranges from -1 to +1:
# - +1 = perfect prediction
# - 0 = random prediction
# - -1 = total disagreement
#
# For detailed metric explanations, see `02_ce2_adult_census_income_pipeline_advanced.py`.

# %% [markdown]
# ## 3. Prepare Data
#
# In this section, we'll separate our features (inputs) from the target (output)
# and split the data into training and test sets.
#
# ### Python Concept: List Comprehensions
#
# The code below uses *list comprehensions* - a compact way to create lists by
# filtering or transforming another list. The syntax is:
#
# ```python
# new_list = [item for item in old_list if condition]
# ```
#
# This is equivalent to writing a `for` loop, but more concise.

# %%
# Handle missing values - replace "?" strings with proper NaN (Not a Number)
adult_ci_df = adult_ci_df.replace("?", np.nan)

# List comprehension: get all columns except the target column
# 'c for c in adult_ci_df.columns' iterates through each column name
# 'if c != target_col' filters out the target
feature_cols = [c for c in adult_ci_df.columns if c != target_col]

# Separate categorical (text) and numeric columns by checking dtype
# dtype == "object" means the column contains strings
categorical_features = [c for c in feature_cols if adult_ci_df[c].dtype == "object"]
numeric_features = [c for c in feature_cols if adult_ci_df[c].dtype != "object"]

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# %%
# Encode target - convert text labels ("<=50K", ">50K") to numbers (0, 1)
# LabelBinarizer learns the mapping and can transform in both directions
lb = LabelBinarizer()
# .fit_transform() learns the classes AND transforms in one step
# .str.strip() removes leading/trailing whitespace from strings
# .ravel() flattens the result to a 1D array
adult_ci_df["target"] = lb.fit_transform(adult_ci_df[target_col].str.strip()).ravel()
print("Target classes:", lb.classes_)  # Shows which class is 0 and which is 1

# Create feature matrix X and target vector y
# .copy() creates independent copies so changes don't affect the original
X = adult_ci_df[feature_cols].copy()
y = adult_ci_df["target"].copy()

# Train/test split - hold out some data to evaluate the final model
# stratify=y ensures both sets have the same class proportions
# random_state makes the split reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# %% [markdown]
# ## 4. Build the Pipeline
#
# A **pipeline** chains preprocessing and modelling steps together. This:
# - Prevents data leakage (preprocessing is fitted only on training data)
# - Makes code cleaner and more reproducible
# - Ensures the same transformations are applied to new data
#
# ### Python Concept: Functions
#
# A *function* is a reusable block of code. We define functions with `def`:
#
# ```python
# def function_name(parameter1, parameter2):
#     """Docstring explains what the function does."""
#     # Function body - the code that runs when called
#     return result  # Return value (optional)
# ```
#
# Functions help us avoid repeating code and make programs easier to understand.

# %%
# Preprocessing steps - define how to handle each column type

# Pipeline for numeric columns: impute missing values, then scale
# Each step is a (name, transformer) tuple
numeric_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),  # Fill NaN with median
        ("scaler", RobustScaler()),  # Scale to similar ranges
    ]
)

# Pipeline for categorical columns: impute, then one-hot encode
categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill NaN with mode
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # Convert to binary columns
    ]
)

# ColumnTransformer applies different pipelines to different column subsets
preprocessor = ColumnTransformer(
    [
        ("num", numeric_transformer, numeric_features),  # Apply to numeric cols
        ("cat", categorical_transformer, categorical_features),  # Apply to categorical cols
    ]
)


# Define a function to build the complete pipeline
def build_pipeline(estimator):
    """Build a complete preprocessing + model pipeline."""
    # Return a Pipeline that first preprocesses, then applies the model
    return Pipeline(
        [
            ("preprocess", preprocessor),  # First step: preprocessing
            ("clf", estimator),  # Second step: classifier (clf)
        ]
    )


# %%
# Visualise the pipeline structure
preprocessor

# %% [markdown]
# ## 5. Baseline Model
#
# Always compare against a baseline. A "most frequent" classifier simply predicts
# the majority class for all samples. If your tuned model can't beat this, something
# is wrong!
#
# **Why we need a baseline:** Without a reference point, we can't tell if our model
# has learned anything useful. A model with 75% accuracy might seem good, but if the
# baseline also achieves 75%, our model adds no value.

# %%
# Baseline: predict most frequent class (always predicts "<=50K")
baseline = build_pipeline(DummyClassifier(strategy="most_frequent"))
baseline.fit(X_train, y_train)  # Train the baseline
y_pred_baseline = baseline.predict(X_test)  # Make predictions on test set

print("Baseline (predict majority class):")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"  MCC: {matthews_corrcoef(y_test, y_pred_baseline):.4f}")  # Should be 0!
print(f"  F1: {f1_score(y_test, y_pred_baseline, zero_division=0):.4f}")

# %% [markdown]
# Note that the baseline achieves ~75% accuracy but MCC = 0 (random).
# This shows why accuracy is misleading for imbalanced data.

# %% [markdown]
# ## 6. Hyperparameter Tuning with Optuna
#
# **Hyperparameters** are settings we choose before training (e.g., number of trees).
# **Hyperparameter tuning** systematically searches for the best settings.
#
# ### How Optuna Works
#
# Optuna uses *Bayesian optimisation* to efficiently search the parameter space.
# Instead of trying random combinations, it learns from previous trials to focus
# on promising regions.
#
# **Key concepts:**
# 1. We define an **objective function** that Optuna calls repeatedly
# 2. Each call is a **trial** with different hyperparameters
# 3. Optuna provides a `trial` object with methods like `trial.suggest_int()` to propose parameter values
# 4. Our function returns a score (higher is better when `direction="maximize"`)
# 5. Optuna learns which parameter regions give better scores
#
# We will tune **LightGBM** with just 3 key parameters:
# - `n_estimators`: Number of trees (more = more complex, slower)
# - `learning_rate`: Step size for updates (smaller = more stable, slower)
# - `max_depth`: Maximum tree depth (deeper = more complex)
#
# We also use `class_weight='balanced'` to help with class imbalance.
#
# For more comprehensive tuning with additional parameters and models,
# see `02_ce2_adult_census_income_pipeline_advanced.py`.

# %%
# Set up the scorer - converts our metric into a function Optuna can use
# make_scorer wraps a metric function so it can be used with cross-validation
if TUNING_OBJECTIVE == "mcc":
    tuning_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
elif TUNING_OBJECTIVE == "f1":
    tuning_scorer = make_scorer(f1_score, greater_is_better=True)
else:
    raise ValueError(f"Unknown objective: {TUNING_OBJECTIVE}")


# Define the objective function that Optuna will call for each trial
def objective(trial):
    """Optuna objective function - returns CV score for a set of hyperparameters."""
    # trial.suggest_* methods ask Optuna to propose values for each parameter
    # Optuna learns which combinations work well and focuses on those regions

    # suggest_int(name, low, high) - proposes an integer in the range
    n_estimators = trial.suggest_int("n_estimators", 50, 200)

    # suggest_float with log=True - samples on a log scale (good for learning rate)
    # This means 0.01 and 0.1 are equally likely, not biased toward larger values
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)

    max_depth = trial.suggest_int("max_depth", 3, 20)

    # Create a LightGBM model with the suggested parameters
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        class_weight="balanced",  # Automatically adjust weights for class imbalance
        random_state=RANDOM_STATE,
        n_jobs=1,  # Use 1 job per model (we parallelise at CV level)
        verbose=-1,  # Suppress LightGBM output
    )

    # Build the complete pipeline with preprocessing
    pipe = build_pipeline(model)

    # Cross-validation: train and evaluate on different subsets of training data
    # RepeatedStratifiedKFold: splits data into N_SPLITS folds, keeping class proportions
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=1, random_state=RANDOM_STATE)

    # cross_val_score trains and evaluates the pipeline on each fold
    # Returns an array of scores, one per fold
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=tuning_scorer, n_jobs=NUM_JOBS)

    # Return the mean score - Optuna will try to maximise this
    return float(np.mean(scores))


# %%
# Run Optuna optimization
print(f"Running Optuna with {N_TRIALS} trials...")

# Reduce Optuna's console output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Create an Optuna study to track optimization progress
# direction="maximize" because higher MCC/F1 is better
# TPESampler uses Bayesian optimization (smarter than random search)
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=RANDOM_STATE),  # For reproducibility
    study_name="lightgbm_tuning",
)

# Run the optimization - Optuna calls objective() N_TRIALS times
# Each call tries different hyperparameters and returns a score
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# study.best_value is the highest score achieved
# study.best_params is a dictionary of the parameters that achieved it
print(f"\nBest {TUNING_OBJECTIVE.upper()}: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")

# %% [markdown]
# ## 7. Train Final Model
#
# During tuning, models were trained and discarded after scoring. Now we create
# a fresh model with the best parameters and train it on all training data.
#
# ### Python Concept: Dictionary Access
#
# `study.best_params` is a *dictionary* mapping parameter names to values:
# ```python
# {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 10}
# ```
#
# We access values with `dict["key"]` syntax.

# %%
# Create a new model using the best parameters found by Optuna
# We access the best params from the study object
best_model = lgb.LGBMClassifier(
    n_estimators=study.best_params["n_estimators"],  # Access dict value by key
    learning_rate=study.best_params["learning_rate"],
    max_depth=study.best_params["max_depth"],
    class_weight="balanced",  # Still use balanced weights
    random_state=RANDOM_STATE,
    n_jobs=NUM_JOBS,  # Use NUM_JOBS cores for final training
    verbose=-1,
)

# Build the complete pipeline and train on ALL training data
best_pipeline = build_pipeline(best_model)
best_pipeline.fit(X_train, y_train)  # Fit learns from data

# %% [markdown]
# ## 8. Evaluate on Test Set
#
# The test set has been held out completely during training and tuning.
# This gives us an unbiased estimate of how the model will perform on new data.

# %%
# Make predictions on the test set
y_test_pred = best_pipeline.predict(X_test)  # Class predictions (0 or 1)
y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]  # Probability of class 1

# Print all metrics
print("Final Model Performance:")
print(f"  Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"  MCC: {matthews_corrcoef(y_test, y_test_pred):.4f}")  # Compare to baseline MCC=0!
print(f"  F1: {f1_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"  Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"  Recall: {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"  ROC AUC: {roc_auc_score(y_test, y_test_proba):.4f}")

# %%
# Plot confusion matrix - visualise prediction outcomes
from sklearn.metrics import ConfusionMatrixDisplay

# Compute confusion matrix from true and predicted labels
cm = confusion_matrix(y_test, y_test_pred)

# Create a display object with proper class labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lb.classes_)
disp.plot(cmap="Blues", values_format="d")  # "d" formats as integers
plt.gca().grid(False)  # Remove grid lines for cleaner look
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()  # Prevent labels from being cut off

# Save to file and display
plt.savefig(os.path.join(out_folder, f"confusion_matrix{out_suffix}.pdf"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Interpreting Results
#
# Compare our tuned model to the baseline:
#
# | Metric | Baseline | Tuned LightGBM |
# |--------|----------|----------------|
# | Accuracy | ~75% | Check above |
# | MCC | 0 | Check above |
#
# The tuned model should have:
# - Similar or slightly lower accuracy (this is expected!)
# - Much higher MCC (actually useful predictions)
# - Reasonable precision and recall for both classes

# %% [markdown]
# ## 10. Summary
#
# In this tutorial, you learned:
# 1. **Pipelines** prevent data leakage and make code reproducible
# 2. **MCC** is a better metric than accuracy for imbalanced data
# 3. **Optuna** efficiently searches for good hyperparameters
# 4. **class_weight='balanced'** helps models learn minority classes
#
# ### Next Steps
#
# To continue learning, see:
#
# **Advanced version** (`02_ce2_adult_census_income_pipeline_advanced.py`):
# - SMOTE resampling for class imbalance
# - More models and hyperparameters
# - Threshold optimisation
# - More evaluation metrics
#
# **Expert version** (`03_ce2_adult_census_income_pipeline_expert.py`):
# - Ensemble methods (voting, stacking)
# - Probability calibration
# - Feature importance analysis
# - Learning curves
# - Cost-benefit analysis

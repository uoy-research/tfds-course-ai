# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: p3.13.5
#     language: python
#     name: python3
# ---

# %%
__author__ = "Matthew Care"
__version__ = "0.2.1"
__date__ = "2026-01-26"

# %% [markdown]
# # Coding Exercise 1 (Advanced) - ML Introduction on the Adult Census Income Dataset
#
# This Colab-style notebook follows Episode 5 of the course.
#
# We will:
# - Frame the Adult Census Income prediction problem.
# - Load and explore an imbalanced tabular dataset.
# - Handle missing values and encode categorical variables.
# - Split data into stratified train/test sets.
# - Apply scaling to numerical features and one-hot encoding to categoricals.
# - Evaluate several scikit-learn models with **repeated stratified k-fold cross-validation** using **accuracy**.
# - Check for signs of overfitting.
# - Visualise accuracy distributions with violin plots.
# - Train the best model and evaluate it on a held-out test set with a confusion matrix.
#
# Note: This notebook deliberately uses **accuracy** on an **imbalanced** dataset. In Coding Exercise 2 we will revisit this with better metrics (precision, recall, F1, MCC, ROC/PR curves) and pipelines.
#
# **This is the Advanced version** with additional EDA plots (correlation heatmaps, categorical
# countplots, stacked bar charts), more sophisticated scaling (QuantileTransformer for features
# with zero IQR), and a commented-out manual CV loop for educational reference.
# For a streamlined introduction, see `01_ce1_adult_census_income_introduction_simple.py`.
#
# ## Learning Objectives
#
# By the end of this exercise, you will understand:
# 1. How to load and explore a tabular dataset with pandas
# 2. How to handle missing values and encode categorical variables
# 3. How to split data into train/test sets while preserving class proportions (stratification)
# 4. How to scale numerical features and one-hot encode categorical features
# 5. How to evaluate models using cross-validation
# 6. How to detect overfitting by comparing train vs validation accuracy
# 7. How to interpret a confusion matrix

# %% [markdown]
# ## 1. Setup and Imports
#
# If you are running this in Google Colab, the next cell will ensure required libraries are installed. On most local setups with a recent Python and scikit-learn, these will already be available.
#
# ### Understanding Imports
#
# Python uses `import` statements to load external libraries. Each library provides
# specialised functionality:
#
# - **numpy** (`np`): Numerical computing with arrays and mathematical operations
# - **pandas** (`pd`): Data manipulation with DataFrames (like Excel spreadsheets in Python)
# - **matplotlib.pyplot** (`plt`): Creating plots and visualisations
# - **seaborn** (`sns`): Statistical visualisation built on matplotlib
# - **sklearn**: Scikit-learn - the main machine learning library
#
# The `from X import Y` syntax imports specific functions or classes from a library.

# %%
# If running in Google Colab, install/upgrade key libraries (safe to run elsewhere).
import sys  # System utilities - lets us check the Python environment

# Check if we're running in Google Colab (an online notebook environment)
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # !pip -q install pandas numpy seaborn matplotlib scikit-learn
    pass  # Packages are pre-installed in Colab


# %%
# Core imports - load the libraries we'll use throughout this notebook
import os  # Operating system utilities (for file paths)

import matplotlib.pyplot as plt  # Plotting and visualisation
import numpy as np  # Numerical computing - arrays and math
import pandas as pd  # DataFrames - tabular data manipulation
import seaborn as sns  # Statistical visualisation
from sklearn.base import clone  # Create fresh copies of models
from sklearn.dummy import DummyClassifier  # Baseline model
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# Models: the classifiers we'll train and compare
from sklearn.linear_model import LogisticRegression
# Metrics: measure model performance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix)
# Sklearn imports - machine learning tools
# train_test_split: splits data into training and test sets
# RepeatedStratifiedKFold: cross-validation that preserves class proportions
# cross_validate: runs cross-validation and returns multiple metrics
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
# Encoding utilities
# Preprocessing: scale numbers and encode categories
from sklearn.preprocessing import LabelBinarizer  # Convert labels to 0/1
from sklearn.preprocessing import \
    OneHotEncoder  # Convert categories to binary columns
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.svm import SVC

# Set visual style for all plots
sns.set_theme(style="whitegrid", context="notebook")

# %% [markdown]
# ### Configuration Guide
#
# Key parameters you can modify to experiment:
#
# - `RANDOM_STATE`: Seed for random number generation (ensures reproducibility)
# - `NUM_SPLITS`: Number of folds in cross-validation (typically 5 or 10)
# - `NUM_REPEATS`: How many times to repeat the CV process (more = more stable estimates)
#
# **Reproducibility:** Setting `RANDOM_STATE` ensures you get the same results each time
# you run the notebook. Try changing it to see how results vary!

# %%
# Configuration - these control the experiment
RANDOM_STATE = 42  # Seed for reproducibility (42 is a common choice...)
NUM_SPLITS = 5  # Number of CV folds
NUM_REPEATS = 3  # Number of CV repeats (total evaluations = SPLITS × REPEATS)

# Set random seed for numpy operations
np.random.seed(RANDOM_STATE)

# Create output folder for saving figures
out_folder = "coding_exercise_1_colab"
os.makedirs(out_folder, exist_ok=True)  # exist_ok=True prevents error if folder exists

# %% [markdown]
# ## 2. Load the Adult Census Income Dataset
#
# We will use an **Adult Census Income** dataset, a classic imbalanced binary classification problem:
#
# - Each row is an individual.
# - Features describe demographics, work, and education.
# - Target: income level `<=50K` vs `>50K`.
#
# We load it directly from a zipped CSV hosted online.
#
# ### Python Concept: DataFrames
#
# A *DataFrame* is pandas' core data structure - think of it as a spreadsheet or table:
# - Rows represent individual records (here, people)
# - Columns represent features/variables (age, education, etc.)
# - You access columns with `df["column_name"]` or `df.column_name`

# %%
# URL to zipped Adult Census Income CSV (single CSV inside the zip).
DATA_URL = "https://github.com/medmaca/shared_data/raw/8a3fea5467ec68b17fd8369c6f77f8016b1ed5f8/Datasets/Kaggle/adult_census_income/adult.csv.zip"

# pd.read_csv() reads a CSV file into a DataFrame
# compression="zip" tells pandas the file is compressed
adult_ci_df = pd.read_csv(DATA_URL, compression="zip")

# .head() shows the first 5 rows - a quick preview of the data
adult_ci_df.head()

# %% [markdown]
# ### Dataset overview
#
# Let's inspect basic structure: columns, dtypes, and target distribution.
#
# ### Python Concept: Method Chaining
#
# Many pandas methods can be "chained" together:
# ```python
# df["col"].value_counts().sort_index()
# ```
# This reads left-to-right: take column → count values → sort by index.
# Each method returns a new object that the next method operates on.

# %%
# .info() shows column names, data types, and non-null counts
adult_ci_df.info()

# %%
# .describe() computes summary statistics for all columns
# include="all" includes both numeric and categorical columns
# .T transposes (flips rows/columns) for easier reading
adult_ci_df.describe(include="all").T

# %% [markdown]
# ### Target distribution (class imbalance)
#
# We check how many samples belong to each income class. This will show us whether the dataset is imbalanced.

# %%
# Define which column contains the target (what we want to predict)
target_col = "income"  # adjust if your column name differs

# .value_counts() counts how many times each unique value appears
# .sort_index() sorts alphabetically by the value (not by count)
class_counts = adult_ci_df[target_col].value_counts().sort_index()

# normalize=True gives proportions instead of raw counts (sums to 1.0)
class_props = adult_ci_df[target_col].value_counts(normalize=True).sort_index()

print(f"Class counts:\n{class_counts}\n")
print(f"Class proportions:\n{class_props}\n")
print(class_props)

# Create a bar plot to visualise class imbalance
plt.figure(figsize=(4, 4))  # Set figure size in inches (width, height)

# sns.barplot creates a bar chart
# x= specifies what goes on the x-axis (class labels)
# y= specifies bar heights (counts)
# hue= adds colour coding (here, same as x for distinct colours)
# palette= chooses the colour scheme
sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette="viridis", dodge=False)

plt.title("Income class counts")  # Add title
plt.ylabel("Count")  # Label y-axis
plt.xlabel("Income class")  # Label x-axis
plt.xticks(rotation=45)  # Rotate x-axis labels 45 degrees
plt.tight_layout()  # Adjust spacing to prevent labels being cut off

# Save the figure to a file
out_path = os.path.join(out_folder, "income_class_counts.pdf")  # Build file path
plt.savefig(out_path, dpi=150, bbox_inches="tight")  # Save with high resolution
print(f"Saved figure to: {out_path}")
plt.show()  # Display the plot

# %% [markdown]
# We can already see **class imbalance**: one income class is noticeably more frequent than the other.
#
# In this Coding Exercise 1 we will still use **accuracy** as the performance metric, even though that's not ideal for imbalanced data. Coding Exercise 2 will address this in depth.

# %% [markdown]
# ## 3. Handling Missing Data (`?`)
#
# In this dataset, missing values are encoded as the literal string `?` in several categorical columns.
#
# We will:
# - Count how many `?` appear per column.
# - Replace `?` with real missing values (`NaN`).
# - Impute missing values in categorical columns with the **most frequent** category.
#
# ### Python Concept: Boolean Indexing
#
# When you compare a DataFrame to a value, Python creates a DataFrame of `True`/`False`:
# ```python
# adult_ci_df == "?"  # Returns True where value equals "?", False elsewhere
# ```
# Calling `.sum()` on booleans counts the `True` values (True=1, False=0).
#
# **Note:** In Coding Exercise 2, we'll use sklearn's `SimpleImputer` inside a Pipeline
# to handle missing values more elegantly.

# %%
# Count missing values ("?") per column
# == creates a boolean DataFrame (True where value is "?")
# .sum() counts True values per column
# .sort_values() sorts from lowest to highest count
missing_counts = (adult_ci_df == "?").sum().sort_values(ascending=False)

# Show only columns that have at least one missing value
missing_counts[missing_counts > 0]

# %%
# Replace '?' with NaN so that pandas recognises them as missing
# np.nan is pandas/numpy's representation of "Not a Number" (missing)
adult_ci_df = adult_ci_df.replace("?", np.nan)

# --- Python Concept: List Comprehensions ---
# A list comprehension creates a new list by filtering/transforming another:
#   [item for item in collection if condition]
# This is equivalent to a for loop, but more concise.

# Get all column names except the target (these are our features)
feature_cols = [c for c in adult_ci_df.columns if c != target_col]

# Separate categorical (text) and numeric columns by checking data type
# dtype == "object" means the column contains strings
categorical_features = [c for c in feature_cols if adult_ci_df[c].dtype == "object"]
numeric_features = [c for c in feature_cols if adult_ci_df[c].dtype != "object"]

print("Categorical features:", categorical_features)
print("Numeric features:", numeric_features)

# --- Python Concept: For Loops ---
# A for loop repeats code for each item in a collection:
#   for item in collection:
#       # do something with item

# Impute missing values in categorical columns with the mode (most frequent value)
for col in categorical_features:
    # .isna() returns True for missing values, .any() checks if any are True
    if adult_ci_df[col].isna().any():
        # .mode() returns the most frequent value(s); [0] takes the first
        mode_val = adult_ci_df[col].mode(dropna=True)[0]
        # .fillna() replaces NaN with the specified value
        adult_ci_df[col] = adult_ci_df[col].fillna(mode_val)

# Verify no NaNs remain in categorical columns
adult_ci_df[categorical_features].isna().sum().sum(), "total remaining NaNs in categoricals"

# %% [markdown]
# We chose **imputation** instead of dropping rows to preserve as much data as possible.
#
# Other strategies (like dropping rows/columns or model-based imputation) are possible, but the core idea is: make an explicit, documented choice.

# %% [markdown]
# ## 4. Exploratory Data Analysis (EDA)
#
# We now explore:
# - Distributions of numerical features.
# - Relationships between some features and the income class.
# - Distributions of selected categorical features.

# %%
# Histograms of numeric features
# This pattern creates a grid of subplots - one for each numeric feature

n_num = len(numeric_features)  # Number of numeric features
n_cols = 3  # We want 3 columns in our grid
n_rows = int(np.ceil(n_num / n_cols))  # Calculate rows needed (ceiling division)

# Create a figure with specified size (width × height in inches)
plt.figure(figsize=(4 * n_cols, 3 * n_rows))

# enumerate() gives us both index (i) and value (col) in the loop
# start=1 makes i start at 1 instead of 0 (for subplot numbering)
for i, col in enumerate(numeric_features, 1):
    plt.subplot(n_rows, n_cols, i)  # Select the i-th subplot
    # sns.histplot creates a histogram showing value distribution
    sns.histplot(data=adult_ci_df, x=col, kde=False, bins=30, color="steelblue")
    plt.title(col)

plt.tight_layout()  # Adjust spacing between subplots

out_path = os.path.join(out_folder, "numeric_feature_histograms.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved figure to: {out_path}")
plt.show()

# %%
# Boxplots of numeric features stratified by income class (log10 y-scale)
# Boxplots show the median, quartiles, and outliers for each group
plt.figure(figsize=(4 * len(numeric_features), 4))
for i, col in enumerate(numeric_features, 1):
    ax = plt.subplot(1, len(numeric_features), i)  # ax is the axes object
    # sns.boxplot shows distribution split by x (income class)
    sns.boxplot(data=adult_ci_df, x=target_col, y=col, ax=ax)
    ax.set_yscale("log", base=10)  # Use log scale for y-axis (handles skewed data)
    ax.set_title(col)
    plt.xticks(rotation=45)
plt.tight_layout()

out_path = os.path.join(out_folder, "numeric_feature_boxplots.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved figure to: {out_path}")
plt.show()

# %%
# Correlation heatmap for numeric features
# Correlation measures how strongly two variables move together (-1 to +1)
plt.figure(figsize=(6, 5))
# .corr() computes pairwise correlations between all numeric columns
corr = adult_ci_df[numeric_features].corr()
# sns.heatmap visualises the correlation matrix with colours
# annot=False hides numbers, cmap="bwr" uses blue-white-red colours
# center=0 makes white represent 0 correlation
sns.heatmap(corr, annot=False, cmap="bwr", center=0)
plt.title("Correlation between numeric features")
plt.tight_layout()

out_path = os.path.join(out_folder, "numeric_feature_correlation_heatmap.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved figure to: {out_path}")
plt.show()

# %%
# Bar plots for a few key categorical features
key_cats = ["education", "workclass", "marital.status", "occupation"]
key_cats = [c for c in key_cats if c in categorical_features]

for col in key_cats:
    plt.figure(figsize=(6, 4))
    order = adult_ci_df[col].value_counts().index
    sns.countplot(data=adult_ci_df, x=col, order=order, hue=target_col)
    plt.title(f"{col} by income")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = os.path.join(out_folder, f"categorical_countplot_{col}.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")
    plt.show()

# %%
# Normalised class percentages for key categorical features
for col in key_cats:
    # Crosstab of category vs income class, normalised by row to get percentages per category\n
    ct = pd.crosstab(adult_ci_df[col], adult_ci_df[target_col], normalize="index") * 100.0
    plt.figure(figsize=(6, 4))
    ct.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="viridis")
    plt.title(f"{col} by income (row-normalised %)")
    plt.ylabel("Percentage of samples in category")
    plt.xlabel(col)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(out_folder, f"categorical_stacked_{col}.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")
    plt.show()

# %% [markdown]
# These plots give us a sense of:
# - The range and skewness of numerical features.
# - How some categorical features are distributed.
# - How the income class relates to features such as education or workclass.

# %% [markdown]
# ## 5. Encode Target and Train/Test Split (Stratified)
#
# We now:
# - Encode the income target as 0/1.
# - Split into **50% train / 50% test**, stratified by class.
# - Use a fixed `random_state` for reproducibility.

# %%
# Encode target: convert text labels to numeric (0 and 1)
# LabelBinarizer learns which class is 0 and which is 1
lb = LabelBinarizer()

# .fit_transform() learns the classes AND transforms in one step
# .str.strip() removes leading/trailing whitespace from strings
# .ravel() flattens the result to a 1D array (some sklearn functions need this)
adult_ci_df["target"] = lb.fit_transform(adult_ci_df[target_col].str.strip()).ravel()
print("classes:", lb.classes_)  # Shows which class is 0 and which is 1

# Create feature matrix X and target vector y
# Convention: X (capital) for features, y (lowercase) for target
# .copy() creates independent copies so changes don't affect the original
X = adult_ci_df[feature_cols].copy()
y = adult_ci_df["target"].copy()

# Split data into training and test sets
# test_size=0.5 means 50% for testing (large test set for reliable evaluation)
# stratify=y ensures both sets have the same class proportions as the original
# random_state makes the split reproducible (same split every time)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    stratify=y,  # Preserve class balance in both sets
    random_state=RANDOM_STATE,
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Verify class proportions are preserved
# .value_counts(normalize=True) gives proportions, .to_dict() converts to dictionary
# List comprehension formats the output nicely
print(f"Train class distribution:{[f'{x} : {y:.3f}' for x, y in y_train.value_counts(normalize=True).to_dict().items()]}")
print(f"Test class distribution:{[f'{x} : {y:.3f}' for x, y in y_test.value_counts(normalize=True).to_dict().items()]}")

# %% [markdown]
# ## 6. Scaling Numerical Features: RobustScaler vs StandardScaler
#
# Our numeric features (e.g. `age`, `hours-per-week`, `capital-gain`, `capital-loss`) are often **skewed** with **outliers**.
#
# - **[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)**: subtract mean, divide by standard deviation.
#   - Works well for roughly Gaussian data.
#   - **Sensitive to outliers**.
# - **[RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)**: subtract median, divide by IQR (interquartile range).
#   - More **robust to outliers** and heavy-tailed distributions.
# - **[QuantileScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)**: This method transforms the features to follow a uniform or a normal distribution. Therefore, for a given feature, this transformation tends to spread out the most frequent values. It also reduces the impact of (marginal) outliers: this is therefore a robust preprocessing scheme
#
# For this dataset we will use **RobustScaler** & **QuantileScaler** for numerical features.
#
# **Note:** In Coding Exercise 2, we'll use sklearn Pipelines to automate this preprocessing so it happens cleanly inside cross-validation folds.

# %%
# Fit RobustScaler on training numeric features only for columns with non-zero IQR,
# and use QuantileTransformer for numeric columns whose IQR is zero (e.g. highly sparse features).

# Extract numeric columns from train and test sets
X_train_num = X_train[numeric_features].copy()
X_test_num = X_test[numeric_features].copy()

# Compute IQR (Interquartile Range) on training data only
# IQR = 75th percentile - 25th percentile
q75 = X_train_num.quantile(0.75)  # 75th percentile for each column
q25 = X_train_num.quantile(0.25)  # 25th percentile for each column
iqr = q75 - q25

# Some features have IQR=0 (e.g., capital-gain where most values are 0)
# RobustScaler would divide by zero for these, so we use QuantileTransformer instead
numeric_iqr_zero = iqr[iqr == 0].index.tolist()  # Columns with IQR=0
numeric_iqr_nonzero = iqr[iqr != 0].index.tolist()  # Columns with IQR>0

print("Numeric features with IQR == 0:", numeric_iqr_zero)
print("Numeric features with non-zero IQR:", numeric_iqr_nonzero)

# We'll collect scaled parts and concatenate them later
scaled_train_parts = []
scaled_test_parts = []

# 1) RobustScaler for features with non-zero IQR
if numeric_iqr_nonzero:
    scaler = RobustScaler()  # Create scaler instance
    X_train_nonzero = X_train_num[numeric_iqr_nonzero]
    X_test_nonzero = X_test_num[numeric_iqr_nonzero]

    # IMPORTANT: fit() learns parameters from TRAINING data only
    # This prevents data leakage from test set
    scaler.fit(X_train_nonzero)

    # transform() applies the learned scaling to both sets
    X_train_nonzero_scaled = pd.DataFrame(
        scaler.transform(X_train_nonzero),  # Apply scaling
        columns=numeric_iqr_nonzero,
        index=X_train.index,  # Preserve original row indices
    )
    X_test_nonzero_scaled = pd.DataFrame(
        scaler.transform(X_test_nonzero),  # Use same scaler fitted on train
        columns=numeric_iqr_nonzero,
        index=X_test.index,
    )
    scaled_train_parts.append(X_train_nonzero_scaled)
    scaled_test_parts.append(X_test_nonzero_scaled)

# 2) QuantileTransformer for features with IQR == 0
if numeric_iqr_zero:
    qt = QuantileTransformer(
        n_quantiles=min(1000, X_train_num.shape[0]),  # Number of quantiles to use
        output_distribution="normal",  # Transform to normal distribution
        random_state=RANDOM_STATE,
    )
    X_train_zero = X_train_num[numeric_iqr_zero]
    X_test_zero = X_test_num[numeric_iqr_zero]

    qt.fit(X_train_zero)  # Fit on training data only
    X_train_zero_scaled = pd.DataFrame(
        qt.transform(X_train_zero),
        columns=numeric_iqr_zero,
        index=X_train.index,
    )
    X_test_zero_scaled = pd.DataFrame(
        qt.transform(X_test_zero),
        columns=numeric_iqr_zero,
        index=X_test.index,
    )
    scaled_train_parts.append(X_train_zero_scaled)
    scaled_test_parts.append(X_test_zero_scaled)

# Concatenate scaled parts and restore original numeric column order
if scaled_train_parts:
    # pd.concat joins DataFrames along columns (axis=1)
    X_train_num_scaled = pd.concat(scaled_train_parts, axis=1)
    X_test_num_scaled = pd.concat(scaled_test_parts, axis=1)
    # Reorder columns to match original order
    X_train_num_scaled = X_train_num_scaled[numeric_features]
    X_test_num_scaled = X_test_num_scaled[numeric_features]
else:
    # Fallback: no numeric features (unlikely in this dataset)
    X_train_num_scaled = X_train_num.copy()
    X_test_num_scaled = X_test_num.copy()

X_train_num_scaled.head()

# %% [markdown]
# ## 7. Encoding Categorical Features (One-Hot / Dummy Variables)
#
# We one-hot encode categorical features.
#
# **One-hot encoding** converts a categorical column into multiple binary columns:
# - Original: `color = [red, blue, red, green]`
# - One-hot: `color_red=[1,0,1,0], color_blue=[0,1,0,0], color_green=[0,0,0,1]`
#
# To avoid **data leakage** from test to train, we:
# - Fit the encoder on the **training set** only.
# - Apply (transform) using the same encoder to both train and test.
#
# **Note:** In Coding Exercise 2, we'll use ColumnTransformer inside a Pipeline to handle this more elegantly.

# %%
# Using sklearn OneHotEncoder for categorical features
X_train_cat = X_train[categorical_features].copy()
X_test_cat = X_test[categorical_features].copy()

# Create encoder instance
# handle_unknown="ignore" - if test set has categories not seen in training, encode as zeros
# sparse_output=False - return a dense array instead of sparse matrix
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Fit on training data only - learns all unique categories
ohe.fit(X_train_cat)

# Transform both sets using the fitted encoder
X_train_cat_encoded = ohe.transform(X_train_cat)
X_test_cat_encoded = ohe.transform(X_test_cat)

# Get the new column names (e.g., "workclass_Private", "education_Bachelors")
ohe_feature_names = ohe.get_feature_names_out(categorical_features)

# Convert numpy arrays back to DataFrames with proper column names and indices
X_train_cat_dummies = pd.DataFrame(
    X_train_cat_encoded,
    columns=ohe_feature_names,
    index=X_train.index,
)
X_test_cat_dummies = pd.DataFrame(
    X_test_cat_encoded,
    columns=ohe_feature_names,
    index=X_test.index,
)

print("Train categorical dummy shape (OneHotEncoder):", X_train_cat_dummies.shape)
print("Test categorical dummy shape (OneHotEncoder):", X_test_cat_dummies.shape)
X_train_cat_dummies.head()

# %% [markdown]
# ### Combine processed numeric and categorical features
#
# We now concatenate the scaled numerical features with the one-hot encoded categorical features to form the final design matrices.

# %%
# Preview the scaled numeric features
X_train_num_scaled

# %%
# Combine scaled numeric features with one-hot encoded categorical features
# pd.concat([df1, df2], axis=1) joins DataFrames side-by-side (column-wise)
X_train_processed = pd.concat([X_train_num_scaled, X_train_cat_dummies], axis=1)
X_test_processed = pd.concat([X_test_num_scaled, X_test_cat_dummies], axis=1)

print("Processed train shape:", X_train_processed.shape)
print("Processed test shape:", X_test_processed.shape)

X_train_processed.head()

# %% [markdown]
# **Note on leakage and pipelines**
#
# - We correctly avoided leakage between **train** and **test** by fitting scaling and encoding only on the training data.
# - However, for cross-validation on `X_train_processed`, the preprocessing was fitted once on the whole training set, not separately inside each CV fold.
#
# In Coding Exercise 2 we will use scikit-learn **Pipelines** so that preprocessing happens cleanly **inside** each cross-validation fold.

# %% [markdown]
# ## 8. Define a Model Zoo (5 Models)
#
# We set up a small collection of standard scikit-learn models:
#
# - [`DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) (most frequent class baseline).
# - [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
# - [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
# - [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).
# - [`SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) (RBF kernel).
#
# The code is organised so you can easily add more models to this dictionary and have them automatically included in evaluation and visualisations.  Look over https://scikit-learn.org/stable/supervised_learning.html and feel free to add additional models (note you'll have to import them before you can use them!)
#
# **Note:** we're setting hyperparameters manually in Coding Exercise 1, we'll explore hyperparameter tuning in Coding Exercise 2.
#
# ### Python Concept: Dictionaries
#
# A *dictionary* maps keys to values:
# ```python
# my_dict = {"key1": value1, "key2": value2}
# ```
# - Access values with `my_dict["key1"]`
# - Loop through with `for key, value in my_dict.items():`
#
# Here we use model names as keys and model instances as values.

# %%
# Define a dictionary of models to evaluate
# Each key is a name (string), each value is a model instance
models = {
    "DummyMostFreq": DummyClassifier(strategy="most_frequent"),  # Baseline: always predict majority class
    "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1),  # Linear model, n_jobs=-1 uses all CPU cores
    "RandomForest": RandomForestClassifier(
        n_estimators=200,  # Number of trees in the forest
        max_depth=None,  # Trees grow until pure leaves
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),  # Sequential tree boosting
    "SVC": SVC(kernel="rbf", gamma="scale", random_state=RANDOM_STATE),  # Support Vector Classifier with RBF kernel
}

# list() converts dictionary keys to a list for display
list(models.keys())

# %% [markdown]
# ## 9. Repeated Stratified k-Fold Cross-Validation (Accuracy Only)
#
# We use **RepeatedStratifiedKFold** to get more stable estimates of accuracy:
#
# - For example: 5 folds × 3 repeats = 15 evaluations per model.
# - We record both **training** and **validation** accuracy per fold.
# - This lets us look for **overfitting** (very high train accuracy vs lower validation accuracy).
#
# ### Python Concept: Functions
#
# A *function* is a reusable block of code. We define functions with `def`:
#
# ```python
# def function_name(parameter1, parameter2):
#     """Docstring explains what the function does."""
#     # Function body - the code that runs when called
#     return result  # Return value (sent back to the caller)
# ```
#
# Functions help us:
# - Avoid repeating code
# - Make programs easier to understand
# - Test pieces of code independently
#
# ### About the Code Below
#
# The commented-out code shows the **manual approach** to cross-validation - explicitly
# looping through folds and training models. This is kept for educational purposes so
# you can see what's happening "under the hood".
#
# The active code uses sklearn's `cross_validate()` function, which does the same thing
# but more concisely and with automatic parallelisation.

# %%
# ============================================================================
# MANUAL CV LOOP (commented out - kept for reference)
# ============================================================================
# This shows what cross_validate() does internally:
# 1. Split data into folds
# 2. For each fold: train on N-1 folds, validate on 1 fold
# 3. Record scores
#
# rskf = RepeatedStratifiedKFold(
#     n_splits=NUM_SPLITS,
#     n_repeats=NUM_REPEATS,
#     random_state=RANDOM_STATE,
# )
#
# records = []
# X_arr = X_train_processed.values  # Convert DataFrame to numpy array
# y_arr = y_train.values
#
# # rskf.split() yields train/val indices for each fold
# for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_arr, y_arr), start=1):
#     X_tr, X_val = X_arr[train_idx], X_arr[val_idx]  # Split features
#     y_tr, y_val = y_arr[train_idx], y_arr[val_idx]  # Split targets
#
#     for model_name, model in models.items():
#         clf = clone(model)  # Create fresh copy (models are stateful after fitting)
#         clf.fit(X_tr, y_tr)  # Train on this fold's training data
#
#         y_tr_pred = clf.predict(X_tr)  # Predict on training data
#         y_val_pred = clf.predict(X_val)  # Predict on validation data
#
#         train_acc = accuracy_score(y_tr, y_tr_pred)
#         val_acc = accuracy_score(y_val, y_val_pred)
#
#         records.append({  # Store results
#             "model": model_name,
#             "fold": fold_idx,
#             "train_acc": train_acc,
#             "val_acc": val_acc,
#         })
#
# cv_results = pd.DataFrame(records)
# ============================================================================


# Define a helper function to evaluate one model with cross-validation
def evaluate_model(X, y, model, model_name, cv):
    """Run cross-validation for a single model and return results as DataFrame."""
    # cross_validate runs CV and returns a dictionary of scores
    cv_out = cross_validate(
        model,  # The model to evaluate
        X,  # Features
        y,  # Target
        cv=cv,  # Cross-validation splitter
        scoring="accuracy",  # Metric to compute
        return_train_score=True,  # Also compute training scores (for overfitting check)
        n_jobs=-1,  # Use all CPU cores for parallelisation
    )
    # Return results as a DataFrame
    return pd.DataFrame(
        {
            "model": model_name,
            "train_acc": cv_out["train_score"],  # Training accuracy per fold
            "val_acc": cv_out["test_score"],  # Validation accuracy per fold
        }
    )


# Create the cross-validation splitter
cv = RepeatedStratifiedKFold(
    n_splits=NUM_SPLITS,  # Number of folds
    n_repeats=NUM_REPEATS,  # Number of times to repeat
    random_state=RANDOM_STATE,  # For reproducibility
)

# Evaluate all models and collect results
all_results = []
print("Starting model evaluations...")

# Loop through each model in our dictionary
for model_name, model in models.items():
    print(f"\t{model_name}")  # Progress indicator
    res = evaluate_model(X_train_processed, y_train, model, model_name, cv)
    all_results.append(res)

# Combine all results into a single DataFrame
# pd.concat with ignore_index=True resets the row indices
cv_results = pd.concat(all_results, ignore_index=True)
cv_results.head()

# %% [markdown]
# ### Mean train vs validation accuracy per model
#
# We summarise average training and validation accuracy to check for systematic overfitting.
# A large gap between train and validation accuracy suggests the model memorises training
# data rather than learning generalisable patterns.

# %%
# .groupby("model") groups rows by model name
# [["train_acc", "val_acc"]] selects just these columns
# .median() computes median for each group
# .sort_values() sorts by validation accuracy (best model first)
median_scores = cv_results.groupby("model")[["train_acc", "val_acc"]].median().sort_values("val_acc", ascending=False)
median_scores

# %% [markdown]
# If a model has **very high train accuracy** but noticeably lower validation accuracy, it may be **overfitting** (memorising training patterns that do not generalise).
#
# Models whose train and validation accuracies are closer together are typically better balanced between bias and variance for this setup.

# %% [markdown]
# ## 10. Visualising Accuracy with Violin Plots
#
# We now visualise the distribution of **validation accuracy** across folds for each model using violin plots.
#
# This reveals not only the mean performance but also variability and potential outliers.

# %%
# Create violin plot showing accuracy distribution for each model
plt.figure(figsize=(8, 9))

# sns.violinplot shows the distribution shape, not just summary statistics
# inner="quartile" draws lines at the quartiles inside the violin
sns.violinplot(data=cv_results, x="model", y="val_acc", inner="quartile", palette="Set2")

plt.title("Validation accuracy distribution by model (RepeatedStratifiedKFold)")
plt.ylabel("Validation accuracy")

# Add a horizontal reference line at the best model's median accuracy
# .idxmax() returns the index (model name) with the maximum value
plt.axhline(y=median_scores.loc[median_scores["val_acc"].idxmax(), "val_acc"], color="grey", linestyle="--")

plt.xlabel("Model")
plt.ylim(0.7, 1.0)  # Set y-axis range
plt.xticks(rotation=45)
plt.tight_layout()

out_path = os.path.join(out_folder, "validation_accuracy_distribution_by_model.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved figure to: {out_path}")
plt.show()

# %% [markdown]
# Remember: **accuracy** can look quite good on an imbalanced dataset even if the minority class is poorly predicted.
#
# In Coding Exercise 2 we will revisit these models with more informative metrics (precision, recall, F1, MCC, ROC/PR curves).

# %% [markdown]
# ## 11. Select the Best Model by Mean Validation Accuracy
#
# We now select the model with the highest **mean validation accuracy** across CV folds, then:
#
# - Fit that model on **all processed training data**.
# - Evaluate it once on the **held-out test set**.
# - Visualise the confusion matrix.

# %%
# Identify best model by median validation accuracy
# .idxmax() returns the row index (model name) with the highest value
best_model_name = median_scores["val_acc"].idxmax()
best_model_name, median_scores.loc[best_model_name]

# %%
# Fit best model on FULL processed training data
# clone() creates a fresh copy of the model (unfitted)
best_model = clone(models[best_model_name])
best_model.fit(X_train_processed, y_train)  # Train on all training data

# Evaluate on held-out test set (data the model has never seen)
y_test_pred = best_model.predict(X_test_processed)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy of best model ({best_model_name}): {test_acc:.4f}")

# Create and display confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lb.classes_)
disp.plot(cmap="Reds", values_format="d")  # "d" formats as integers

ax = plt.gca()  # Get current axes
ax.grid(False)  # Turn off grid for cleaner look

plt.title(f"Confusion matrix on held-out test set ({best_model_name})")
plt.tight_layout()

out_path = os.path.join(out_folder, "confusion_matrix_on_held_out_test_set.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved figure to: {out_path}")
plt.show()

# %% [markdown]
# The confusion matrix shows how many:
# - True negatives (correct `<=50K`)
# - True positives (correct `>50K`)
# - False positives (`<=50K` predicted as `>50K`)
# - False negatives (`>50K` predicted as `<=50K`)
#
# Even if overall accuracy looks good, we should be cautious if one type of error (e.g. false negatives) is particularly frequent.

# %% [markdown]
# ## 12. Wrap-Up and Bridge to Coding Exercise 2
#
# In this notebook we:
#
# - Framed the Adult Census Income prediction task as a **binary classification** problem.
# - Explored the data: feature types, distributions, class imbalance, and missing values.
# - Performed **stratified** train/test splitting with a fixed random seed.
# - Applied **RobustScaler** to numerical features and **one-hot encoding** to categorical features using only the training data (to avoid leakage to test).
# - Evaluated several scikit-learn models using **RepeatedStratifiedKFold** and **accuracy**.
# - Checked for overfitting by comparing train vs validation accuracy.
# - Visualised accuracy distributions with violin plots.
# - Trained the best model on all training data and inspected its test-set confusion matrix.
#
# ### Next steps (Coding Exercise 2)
#
# In Coding Exercise 2 and the accompanying notebook we will:
#
# - See why accuracy can be misleading on imbalanced data.
# - Introduce metrics like **precision**, **recall**, **F1**, **MCC**, **ROC/PR curves**.
# - Use **Pipelines** and **ColumnTransformer** to cleanly combine preprocessing and models.
# - Apply **hyperparameter tuning** (GridSearchCV, RandomizedSearchCV, Optuna) in a way that respects cross-validation and avoids leakage.
#
# This will turn today's basic lifecycle into a more robust and production-ready ML workflow.

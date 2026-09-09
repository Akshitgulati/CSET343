# STEP 1: IMPORT LIBRARIES

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler


# STEP 2: SET BASE DIRECTORY

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# STEP 3: LOAD DATASET

file_path = os.path.join(
    BASE_DIR,
    "data",
    "heart_failure_clinical_records_dataset.csv"
)

df = pd.read_csv(file_path)


# STEP 4: DISPLAY FIRST 5 ROWS

print("First 5 rows:")

print(df.head())


# STEP 5: DISPLAY DATASET SHAPE

print("\nDataset Shape:")

print(df.shape)


# STEP 6: DISPLAY NUMBER OF ROWS AND COLUMNS

print("\nNumber of Rows:")

print(df.shape[0])

print("\nNumber of Columns:")

print(df.shape[1])


# STEP 7: DISPLAY DATA TYPES AND MISSING VALUES

print("\nData Types:")

print(df.dtypes)

print("\nMissing Values:")

print(df.isnull().sum())


# STEP 8: SAVE DATASET BEFORE CLEANING

# Manually add missing values for demonstration

df.loc[
    [5, 15, 25, 35, 45],
    "age"
] = np.nan

df.loc[
    [10, 20, 30, 40, 50],
    "ejection_fraction"
] = np.nan

df.loc[
    [12, 22, 32, 42, 52],
    "serum_creatinine"
] = np.nan

df.loc[
    [8, 18, 28, 38, 48],
    "platelets"
] = np.nan


# Save a copy before cleaning

df_before_cleaning = df.copy()


print(
    "\nMissing Values After Manually Adding Missing Data:"
)

print(df.isnull().sum())


# STEP 8.1: CREATE MISSING VALUES HEATMAP

plt.figure(figsize=(12, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False
)

plt.title(
    "Missing Values Before Imputation"
)

plt.xlabel("Features")

plt.ylabel("Patients")

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "missing_values_heatmap.png"
    )
)

plt.show()


# STEP 9: IMPUTE MISSING VALUES

# Identify numerical columns

numerical_columns = df.select_dtypes(
    include=np.number
).columns


# Replace numerical missing values with median

for column in numerical_columns:

    if df[column].isnull().any():

        df[column] = df[column].fillna(
            df[column].median()
        )


# Identify categorical columns

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns


# Replace categorical missing values with mode

for column in categorical_columns:

    if df[column].isnull().any():

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


print("\nMissing Values After Imputation:")

print(df.isnull().sum())


# STEP 10: CORRECT INCONSISTENCIES

# Correct invalid negative age values

df.loc[
    df["age"] < 0,
    "age"
] = np.nan


# Impute corrected age values

if df["age"].isnull().any():

    df["age"] = df["age"].fillna(
        df["age"].median()
    )


# Correct invalid ejection fraction values

df.loc[
    (df["ejection_fraction"] < 0) |
    (df["ejection_fraction"] > 100),
    "ejection_fraction"
] = np.nan


# Impute corrected ejection fraction values

if df["ejection_fraction"].isnull().any():

    df["ejection_fraction"] = df[
        "ejection_fraction"
    ].fillna(
        df["ejection_fraction"].median()
    )


# Correct invalid serum sodium values

df.loc[
    df["serum_sodium"] <= 0,
    "serum_sodium"
] = np.nan


# Impute corrected serum sodium values

if df["serum_sodium"].isnull().any():

    df["serum_sodium"] = df[
        "serum_sodium"
    ].fillna(
        df["serum_sodium"].median()
    )


print(
    "\nInvalid entries corrected successfully."
)


# STEP 11: CALCULATE SUMMARY STATISTICS BEFORE OUTLIER REMOVAL

continuous_columns = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time"
]


before_statistics = df[
    continuous_columns
].agg(
    ["mean", "median", "std"]
)


print(
    "\nSummary Statistics Before Outlier Removal:"
)

print(before_statistics)


# STEP 12: REMOVE OUTLIERS USING IQR METHOD

outlier_columns = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium"
]


rows_before_outlier_removal = df.shape[0]


for column in outlier_columns:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR


    df = df[
        (df[column] >= lower_bound) &
        (df[column] <= upper_bound)
    ]


rows_after_outlier_removal = df.shape[0]


print(
    "\nRows Before Outlier Removal:"
)

print(
    rows_before_outlier_removal
)


print(
    "\nRows After Outlier Removal:"
)

print(
    rows_after_outlier_removal
)


print(
    "\nOutliers Removed:"
)

print(
    rows_before_outlier_removal -
    rows_after_outlier_removal
)


# STEP 13: FEATURE ENGINEERING

# Create age groups

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 40, 60, np.inf],
    labels=[
        "<40",
        "40-60",
        ">60"
    ],
    right=True
)


# Create risk score

risk_product = (
    df["ejection_fraction"] *
    df["serum_creatinine"]
)


risk_scaler = MinMaxScaler()


df["risk_score"] = risk_scaler.fit_transform(
    risk_product.to_numpy().reshape(-1, 1)
)


print(
    "\nAge Group Distribution:"
)

print(
    df["age_group"].value_counts()
)


print(
    "\nRisk Score:"
)

print(
    df["risk_score"].head()
)


# STEP 14: ENCODE CATEGORICAL VARIABLES

df = pd.get_dummies(
    df,
    columns=[
        "age_group",
        "sex",
        "smoking"
    ],
    dtype=int
)


print(
    "\nDataset After Categorical Encoding:"
)

print(
    df.head()
)


print(
    "\nColumns After Encoding:"
)

print(
    df.columns.tolist()
)


# STEP 15: NORMALIZE NUMERICAL FEATURES

features_to_scale = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time",
    "risk_score"
]


scaler = MinMaxScaler()


df[features_to_scale] = scaler.fit_transform(
    df[features_to_scale]
)


print(
    "\nNormalized Numerical Features:"
)

print(
    df[features_to_scale].head()
)


# STEP 16: COMPUTE SUMMARY STATISTICS AFTER CLEANING

after_statistics = df[
    features_to_scale
].agg(
    ["mean", "median", "std"]
)


print(
    "\nSummary Statistics After Cleaning:"
)

print(
    after_statistics
)


# STEP 17: VALIDATE CLEANED NUMERICAL FEATURES

plt.figure(figsize=(8, 5))


plt.boxplot(
    df[features_to_scale],
    tick_labels=features_to_scale
)


plt.title(
    "Boxplot of Cleaned Numerical Features"
)

plt.ylabel(
    "Normalized Value"
)


plt.xticks(
    rotation=90
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        BASE_DIR,
        "cleaned_features_boxplot.png"
    )
)


plt.show()


# STEP 18: PLOT CLEANED FEATURE DISTRIBUTIONS

# Age distribution

plt.figure(figsize=(8, 5))


plt.hist(
    df["age"],
    bins=15
)


plt.title(
    "Age Distribution After Cleaning"
)

plt.xlabel(
    "Normalized Age"
)

plt.ylabel(
    "Number of Patients"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        BASE_DIR,
        "cleaned_age_distribution.png"
    )
)


plt.show()


# Ejection fraction distribution

plt.figure(figsize=(8, 5))


plt.hist(
    df["ejection_fraction"],
    bins=10
)


plt.title(
    "Ejection Fraction Distribution After Cleaning"
)

plt.xlabel(
    "Normalized Ejection Fraction"
)

plt.ylabel(
    "Number of Patients"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        BASE_DIR,
        "cleaned_ejection_fraction_distribution.png"
    )
)


plt.show()


# STEP 19: FINAL MISSING VALUE CHECK

print(
    "\nFinal Missing Values:"
)

print(
    df.isnull().sum()
)


total_missing_after_cleaning = (
    df.isnull().sum().sum()
)


print(
    "\nTotal Missing Values After Cleaning:"
)


print(
    total_missing_after_cleaning
)


# STEP 20: DISPLAY FINAL CLEANED DATASET

print(
    "\nFinal Cleaned Dataset:"
)

print(
    df.head()
)


print(
    "\nFinal Dataset Shape:"
)

print(
    df.shape
)


# STEP 21: SAVE CLEANED DATASET

cleaned_file_path = os.path.join(
    BASE_DIR,
    "data",
    "cleaned_heart_failure_dataset.csv"
)


df.to_csv(
    cleaned_file_path,
    index=False
)


print(
    "\nCleaned dataset saved successfully:"
)


print(
    cleaned_file_path
)


# STEP 22: LAB ASSIGNMENT COMPLETED

print(
    "\nLab Assignment 2 completed successfully."
)
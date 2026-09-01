# ============================================================
# AGRI_SHIELD - COMPLETE EXPLORATORY DATA ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

script_folder = Path(__file__).resolve().parent
file_path = script_folder / "cleaned_agri_shield.csv"

print("=" * 70)
print("AGRI_SHIELD - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

try:
    df = pd.read_csv(file_path)

except FileNotFoundError:
    print("\nERROR: cleaned_agri_shield.csv was not found.")
    print("Make sure it is inside the AGRI_SHIELD folder.")
    raise SystemExit

except Exception as e:
    print("\nERROR while loading dataset:")
    print(e)
    raise SystemExit


print("\nDataset loaded successfully.")


# ============================================================
# PART 1 - BASIC DATA INSPECTION
# ============================================================

print("\n" + "=" * 70)
print("PART 1 - BASIC DATA INSPECTION")
print("=" * 70)


# First 5 rows
print("\nFirst 5 rows:")
print(df.head())


# Last 5 rows
print("\nLast 5 rows:")
print(df.tail())


# Shape
print("\nDataset shape:")
print(df.shape)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# Column names
print("\nColumn names:")

for column in df.columns:
    print("-", column)


# Data types
print("\nData types:")
print(df.dtypes)


# Dataset information
print("\nDataset information:")
df.info()


# ============================================================
# PART 2 - NUMERICAL DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PART 2 - NUMERICAL DATA ANALYSIS")
print("=" * 70)


numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


print("\nNumerical columns:")

if len(numeric_columns) == 0:

    print("No numerical columns found.")

else:

    for column in numeric_columns:
        print("-", column)


# Basic statistics
if len(numeric_columns) > 0:

    print("\nBasic statistics:")

    print(
        df[numeric_columns].describe()
    )


# Detailed numerical analysis
if len(numeric_columns) > 0:

    for column in numeric_columns:

        print("\n" + "-" * 60)
        print("Column:", column)

        print(
            "Minimum:",
            df[column].min()
        )

        print(
            "Maximum:",
            df[column].max()
        )

        print(
            "Mean:",
            df[column].mean()
        )

        print(
            "Median:",
            df[column].median()
        )

        print(
            "Standard deviation:",
            df[column].std()
        )


# ============================================================
# PART 3 - CATEGORICAL DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PART 3 - CATEGORICAL DATA ANALYSIS")
print("=" * 70)


categorical_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()


print("\nCategorical columns:")

if len(categorical_columns) == 0:

    print("No categorical columns found.")

else:

    for column in categorical_columns:

        print("\n" + "-" * 60)
        print("Column:", column)

        print(
            "Number of unique values:",
            df[column].nunique()
        )

        print("\nMost common values:")

        print(
            df[column]
            .value_counts()
            .head(10)
        )


# ============================================================
# PART 4 - MISSING VALUES AND DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("PART 4 - MISSING VALUES AND DUPLICATES")
print("=" * 70)


# Missing values
print("\nMissing values:")

missing_values = df.isnull().sum()

for column in df.columns:

    print(
        column,
        ":",
        missing_values[column]
    )


# Total missing values
total_missing = df.isnull().sum().sum()

print(
    "\nTotal missing values:",
    total_missing
)


# Duplicate rows
duplicate_count = df.duplicated().sum()

print(
    "\nDuplicate rows:",
    duplicate_count
)


# ============================================================
# PART 5 - DISTRIBUTION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PART 5 - DISTRIBUTION ANALYSIS")
print("=" * 70)


if len(numeric_columns) == 0:

    print("\nNo numerical columns available for distribution analysis.")

else:

    print(
        "\nCreating distribution plots for numerical columns..."
    )

    for column in numeric_columns:

        try:

            plt.figure(figsize=(8, 5))

            plt.hist(
                df[column].dropna(),
                bins=30
            )

            plt.title(
                "Distribution of " + str(column)
            )

            plt.xlabel(column)
            plt.ylabel("Frequency")

            plt.tight_layout()

            plt.show()

        except Exception as e:

            print(
                "Could not create plot for",
                column
            )

            print(e)


# ============================================================
# PART 6 - CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PART 6 - CORRELATION ANALYSIS")
print("=" * 70)


if len(numeric_columns) < 2:

    print(
        "\nNot enough numerical columns for correlation analysis."
    )

else:

    correlation_matrix = (
        df[numeric_columns]
        .corr()
    )

    print("\nCorrelation matrix:")
    print(correlation_matrix)


    # Display correlation heatmap
    try:

        plt.figure(
            figsize=(
                max(8, len(numeric_columns)),
                max(6, len(numeric_columns))
            )
        )

        plt.imshow(
            correlation_matrix,
            interpolation="nearest"
        )

        plt.colorbar()

        plt.xticks(
            range(len(numeric_columns)),
            numeric_columns,
            rotation=90
        )

        plt.yticks(
            range(len(numeric_columns)),
            numeric_columns
        )

        plt.title(
            "Correlation Matrix"
        )

        plt.tight_layout()

        plt.show()

    except Exception as e:

        print(
            "\nCould not create correlation plot."
        )

        print(e)


# ============================================================
# PART 7 - YEAR-WISE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PART 7 - YEAR-WISE ANALYSIS")
print("=" * 70)


if "Source_Year" in df.columns:

    print("\nYear-wise record count:")

    year_counts = (
        df["Source_Year"]
        .value_counts()
        .sort_index()
    )

    print(year_counts)


    # Numerical averages by year
    if len(numeric_columns) > 0:

        print("\nNumerical averages by year:")

        year_numeric = (
            df.groupby("Source_Year")[
                numeric_columns
            ]
            .mean()
        )

        print(year_numeric)


        # Plot row count by year
        try:

            plt.figure(figsize=(9, 5))

            year_counts.sort_index().plot(
                kind="bar"
            )

            plt.title(
                "Number of Records by Year"
            )

            plt.xlabel("Year")
            plt.ylabel("Number of Records")

            plt.tight_layout()

            plt.show()

        except Exception as e:

            print(
                "\nCould not create year-wise plot."
            )

            print(e)

else:

    print(
        "\nSource_Year column not found."
    )

    print(
        "Year-wise analysis skipped."
    )


# ============================================================
# PART 8 - FEATURE ANALYSIS FOR MACHINE LEARNING
# ============================================================

print("\n" + "=" * 70)
print("PART 8 - FEATURE ANALYSIS")
print("=" * 70)


print("\nPotential numerical features:")

if len(numeric_columns) == 0:

    print("No numerical features found.")

else:

    for column in numeric_columns:

        print(
            "-",
            column
        )


print("\nPotential categorical features:")

if len(categorical_columns) == 0:

    print("No categorical features found.")

else:

    for column in categorical_columns:

        # Source_Year is metadata rather than a normal feature
        if column != "Source_Year":

            print(
                "-",
                column
            )


# ============================================================
# IDENTIFY HIGH CORRELATIONS
# ============================================================

if len(numeric_columns) >= 2:

    print("\nStrong numerical correlations:")

    correlation_matrix = (
        df[numeric_columns]
        .corr()
    )

    found_correlation = False

    for i in range(len(numeric_columns)):

        for j in range(i + 1, len(numeric_columns)):

            column_1 = numeric_columns[i]
            column_2 = numeric_columns[j]

            correlation_value = (
                correlation_matrix
                .loc[column_1, column_2]
            )

            if abs(correlation_value) >= 0.7:

                print(
                    column_1,
                    "<->",
                    column_2,
                    ":",
                    round(
                        correlation_value,
                        3
                    )
                )

                found_correlation = True

    if not found_correlation:

        print(
            "No strong correlations (>= 0.70) found."
        )


# ============================================================
# FINAL EDA SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA SUMMARY")
print("=" * 70)

print(
    "\nTotal rows:",
    df.shape[0]
)

print(
    "Total columns:",
    df.shape[1]
)

print(
    "Numerical columns:",
    len(numeric_columns)
)

print(
    "Categorical columns:",
    len(categorical_columns)
)

print(
    "Total missing values:",
    total_missing
)

print(
    "Duplicate rows:",
    duplicate_count
)


print("\nEDA COMPLETED SUCCESSFULLY.")

print("\nNext stage:")
print("Feature Engineering and Machine Learning preparation.")
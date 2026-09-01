import pandas as pd
from pathlib import Path

# Folder containing the datasets
input_folder = Path("data_folder")

# Find all CSV files
files = list(input_folder.glob("*.csv"))

# Display found files
print(f"Found {len(files)} files:\n")

for file in files:
    print(file.name)

# Load and display the first 5 rows of every dataset
for file in files:
    df = pd.read_csv(file)

    print("\nDataset:", file.name)
    print(df.head())

# Data inspection
for file in files:
    df = pd.read_csv(file)

    print("\n" + "=" * 50)
    print("Dataset:", file.name)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nBasic Statistics:")
    print(df.describe())

# Basic data cleaning
cleaned_data = {}

for file in files:
    df = pd.read_csv(file)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert blank/empty values to NaN
    df = df.replace(r'^\s*$', pd.NA, regex=True)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Store cleaned dataset
    cleaned_data[file.stem] = df

    print("\n" + "=" * 50)
    print("Cleaned:", file.name)
    print("New shape:", df.shape)
    print("Duplicates remaining:", df.duplicated().sum())

# Detailed cleaning checks
for year, df in cleaned_data.items():

    print("\n" + "=" * 60)
    print("CLEANED DATASET:", year)

    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Check duplicate rows
    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    # Check unique values in each column
    print("\nUnique values:")
    for column in df.columns:
        print(f"{column}: {df[column].nunique()}")

    # Display first 5 rows after cleaning
    print("\nFirst 5 rows:")
    print(df.head())

# Handle missing values
for year, df in cleaned_data.items():

    print("\n" + "=" * 60)
    print("HANDLING MISSING VALUES:", year)

    # Show missing values before filling
    print("\nMissing values before:")
    print(df.isnull().sum())

    # Fill numeric columns with median
    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    # Fill text/categorical columns with mode
    categorical_columns = df.select_dtypes(include="object").columns

    for column in categorical_columns:
        if df[column].isnull().any():
            mode_value = df[column].mode()

            if not mode_value.empty:
                df[column] = df[column].fillna(mode_value[0])

    # Update cleaned dataset
    cleaned_data[year] = df

    # Check missing values after filling
    print("\nMissing values after:")
    print(df.isnull().sum())

# Check data types
for year, df in cleaned_data.items():

    print("\n" + "=" * 60)
    print("DATA TYPES:", year)

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nData type check completed.")

    # Update cleaned dataset
    cleaned_data[year] = df

# Check numerical values
for year, df in cleaned_data.items():

    print("\n" + "=" * 60)
    print("NUMERICAL CHECK:", year)

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_columns) == 0:
        print("No numerical columns found.")
        continue

    for column in numeric_columns:

        print("\nColumn:", column)
        print("Minimum:", df[column].min())
        print("Maximum:", df[column].max())
        print("Median:", df[column].median())
        print("Negative values:", (df[column] < 0).sum())


# ============================================================
# CHECK COLUMN CONSISTENCY
# ============================================================

print("\n" + "=" * 60)
print("COLUMN CONSISTENCY CHECK")
print("=" * 60)

years = list(cleaned_data.keys())

# Show columns for every dataset
for year in years:

    print("\nDataset:", year)

    columns = cleaned_data[year].columns.tolist()

    print("Number of columns:", len(columns))
    print("Columns:", columns)


# Compare datasets
if len(years) > 1:

    reference_year = years[0]

    reference_columns = list(
        cleaned_data[reference_year].columns
    )

    print("\nReference dataset:", reference_year)

    for year in years[1:]:

        current_columns = list(
            cleaned_data[year].columns
        )

        missing_columns = [
            column
            for column in reference_columns
            if column not in current_columns
        ]

        extra_columns = [
            column
            for column in current_columns
            if column not in reference_columns
        ]

        print("\nChecking:", year)

        if len(missing_columns) == 0 and len(extra_columns) == 0:

            print("Columns are consistent.")

        else:

            if len(missing_columns) > 0:
                print(
                    "Missing columns:",
                    missing_columns
                )

            if len(extra_columns) > 0:
                print(
                    "Extra columns:",
                    extra_columns
                )

else:

    print("\nOnly one dataset found.")
    print("Column consistency check completed.")
# ============================================================
# COMBINE ALL CLEANED DATASETS
# ============================================================

print("\n" + "=" * 60)
print("COMBINING CLEANED DATASETS")
print("=" * 60)

all_data = []

for year, df in cleaned_data.items():

    print("\nAdding dataset:", year)

    # Make sure it is a DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Skipping", year, "- not a DataFrame.")
        continue

    # Make a copy
    temp_df = df.copy()

    # Add source year
    temp_df["Source_Year"] = year

    print("Rows:", len(temp_df))
    print("Columns:", len(temp_df.columns))

    all_data.append(temp_df)


# Check whether datasets were collected
if len(all_data) == 0:

    print("\nERROR: No valid cleaned datasets found.")
    raise SystemExit


# Combine datasets
try:

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

except Exception as e:

    print("\nERROR while combining datasets:")
    print(e)
    raise SystemExit


# ============================================================
# DISPLAY COMBINED DATA
# ============================================================

print("\nDatasets combined successfully!")

print("\nFinal shape:")
print(final_df.shape)

print("\nFinal number of rows:")
print(len(final_df))

print("\nFinal number of columns:")
print(len(final_df.columns))

print("\nFinal columns:")
print(final_df.columns.tolist())

print("\nFirst 5 rows:")
print(final_df.head())

# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

# Check that final_df exists
if "final_df" not in globals():

    print("ERROR: final_df was not created.")
    print("The combine step must run successfully first.")
    raise SystemExit


# ------------------------------------------------------------
# 1. Shape
# ------------------------------------------------------------

print("\nFinal shape:")
print(final_df.shape)


# ------------------------------------------------------------
# 2. Columns
# ------------------------------------------------------------

print("\nFinal columns:")

for column in final_df.columns:
    print("-", column)


# ------------------------------------------------------------
# 3. Missing values
# ------------------------------------------------------------

print("\nMissing values:")

missing_values = final_df.isnull().sum()

for column in final_df.columns:

    print(
        column,
        ":",
        missing_values[column]
    )


# ------------------------------------------------------------
# 4. Duplicate rows
# ------------------------------------------------------------

print("\nDuplicate rows:")

duplicate_count = final_df.duplicated().sum()

print(duplicate_count)


# ------------------------------------------------------------
# 5. Data types
# ------------------------------------------------------------

print("\nData types:")

for column in final_df.columns:

    print(
        column,
        ":",
        final_df[column].dtype
    )


# ------------------------------------------------------------
# 6. First 5 rows
# ------------------------------------------------------------

print("\nFirst 5 rows:")
print(final_df.head())


# ------------------------------------------------------------
# 7. Final status
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL VALIDATION COMPLETED")
print("=" * 60)

# ============================================================
# SAVE FINAL CLEANED DATASET
# ============================================================

output_file = Path("cleaned_agri_shield.csv")

final_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)

print("\nCleaned dataset saved successfully!")
print("File:", output_file.resolve())
print("Rows:", len(final_df))
print("Columns:", len(final_df.columns))
import pandas as pd
from pathlib import Path

# ============================================================
# LOAD CLEANED DATASET
# ============================================================

file_path = Path("cleaned_agri_shield.csv")

df = pd.read_csv(file_path)

print("=" * 70)
print("AGRI_SHIELD - FEATURE ENGINEERING")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nAll columns:")

for i, column in enumerate(df.columns, start=1):
    print(i, ":", column)

print("\nData types:")
print(df.dtypes)

print("\nSample data:")
print(df.head())

print("\nUnique values in categorical columns:")

categorical_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns

for column in categorical_columns:

    print("\nColumn:", column)
    print("Unique values:", df[column].nunique())

    print(
        df[column]
        .value_counts()
        .head(10)
    )

print("\n" + "=" * 70)
print("FEATURE INSPECTION COMPLETED")
print("=" * 70)

# ============================================================
# FEATURE ENGINEERING - COLUMN CHECK
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

print("\nAll columns in the dataset:")

for i, column in enumerate(df.columns, start=1):
    print(i, ":", column)

print("\nPlease identify the TARGET column from the list above.")

# ============================================================
# CREATE FEATURES (X) AND TARGET (y)
# ============================================================

TARGET_COLUMN = "Modal_Price"

# Check target column
if TARGET_COLUMN not in df.columns:
    print("\nERROR: Target column not found.")
    print("Available columns:")
    print(df.columns.tolist())
    raise SystemExit

# Target
y = df[TARGET_COLUMN].copy()

# Features
X = df.drop(columns=[TARGET_COLUMN]).copy()

print("\n" + "=" * 60)
print("FEATURE AND TARGET CREATION")
print("=" * 60)

print("\nTarget column:")
print(TARGET_COLUMN)

print("\nTarget shape:")
print(y.shape)

print("\nFeature shape:")
print(X.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget data type:")
print(y.dtype)

print("\nTarget statistics:")
print(y.describe())

print("\nFeatures and target created successfully.")

# ============================================================
# FEATURE ENGINEERING - COLUMN INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("COLUMN INFORMATION")
print("=" * 60)

for i, column in enumerate(df.columns, start=1):

    print("\n", i, ".", column)
    print("Data type:", df[column].dtype)
    print("Unique values:", df[column].nunique())

    # Show a few example values
    print("Examples:")
    print(df[column].dropna().head(5).tolist())

print("\n" + "=" * 60)
print("COLUMN INFORMATION COMPLETED")
print("=" * 60)

# ============================================================
# FEATURE ENGINEERING - DATE FEATURES
# ============================================================

print("\n" + "=" * 60)
print("DATE FEATURE ENGINEERING")
print("=" * 60)

# Check whether Arrival_Date exists
if "Arrival_Date" in X.columns:

    # Convert to datetime
    X["Arrival_Date"] = pd.to_datetime(
        X["Arrival_Date"],
        errors="coerce"
    )

    # Extract useful date features
    X["Arrival_Year"] = X["Arrival_Date"].dt.year
    X["Arrival_Month"] = X["Arrival_Date"].dt.month
    X["Arrival_Day"] = X["Arrival_Date"].dt.day
    X["Arrival_DayOfWeek"] = X["Arrival_Date"].dt.dayofweek

    # Remove original date column
    X = X.drop(columns=["Arrival_Date"])

    print("\nDate features created:")
    print("- Arrival_Year")
    print("- Arrival_Month")
    print("- Arrival_Day")
    print("- Arrival_DayOfWeek")

else:

    print("\nArrival_Date column not found.")

print("\nFeature shape after date engineering:")
print(X.shape)

print("\nFeature engineering completed.")

# ============================================================
# CATEGORICAL FEATURE ENCODING
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL FEATURE ENCODING")
print("=" * 60)

categorical_columns = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade"
]

# Keep only columns that actually exist
categorical_columns = [
    column
    for column in categorical_columns
    if column in X.columns
]

print("\nCategorical columns:")
print(categorical_columns)

# Convert categorical columns to category dtype
for column in categorical_columns:

    X[column] = X[column].astype("category")

    print(
        column,
        "->",
        X[column].nunique(),
        "categories"
    )

print("\nCategorical columns converted to category dtype.")

print("\nCurrent feature data types:")
print(X.dtypes)

print("\nFeature shape:")
print(X.shape)

# ============================================================
# PREPARE DATA FOR MACHINE LEARNING
# ============================================================

print("\n" + "=" * 60)
print("PREPARING DATA FOR MACHINE LEARNING")
print("=" * 60)

# ------------------------------------------------------------
# 1. Check target values
# ------------------------------------------------------------

print("\nTarget column:", TARGET_COLUMN)

print("\nTarget data type:")
print(y.dtype)

print("\nTarget missing values:")
print(y.isnull().sum())

# Remove rows where target is missing
valid_target = y.notna()

X = X.loc[valid_target].reset_index(drop=True)
y = y.loc[valid_target].reset_index(drop=True)

print("\nRows after removing missing target values:")
print(len(y))


# ------------------------------------------------------------
# 2. Check numerical columns
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)


# ------------------------------------------------------------
# 3. Check categorical columns
# ------------------------------------------------------------

categorical_features = X.select_dtypes(
    include=["category", "object", "string"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)


# ------------------------------------------------------------
# 4. Final feature list
# ------------------------------------------------------------

print("\nTotal features:", len(X.columns))

print("\nAll features:")

for column in X.columns:
    print("-", column)


# ------------------------------------------------------------
# 5. Target statistics
# ------------------------------------------------------------

print("\nTarget statistics:")
print(y.describe())


print("\n" + "=" * 60)
print("ML DATA PREPARATION COMPLETED")
print("=" * 60)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTrain/Test split completed successfully.")

# ============================================================
# ENCODE CATEGORICAL FEATURES FOR MACHINE LEARNING
# ============================================================

print("\n" + "=" * 60)
print("ENCODING CATEGORICAL FEATURES")
print("=" * 60)

categorical_features = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade"
]

# Keep only columns that exist
categorical_features = [
    column
    for column in categorical_features
    if column in X_train.columns
]

print("\nCategorical columns:")
print(categorical_features)


# ------------------------------------------------------------
# Encode categorical columns
# ------------------------------------------------------------

category_mappings = {}

for column in categorical_features:

    # Get original category values before encoding
    combined_values = pd.concat(
        [
            X_train[column],
            X_test[column]
        ],
        ignore_index=True
    ).astype("string").fillna("Unknown")

    # Keep original category names
    categories = combined_values.unique().tolist()

    # Create original value -> numeric code mapping
    mapping = {
        category: index
        for index, category in enumerate(categories)
    }

    # Save mapping for prediction time
    category_mappings[column] = mapping

    # Encode training data using the SAME mapping
    X_train[column] = (
        X_train[column]
        .astype("string")
        .fillna("Unknown")
        .map(mapping)
        .fillna(0)
        .astype(int)
    )

    # Encode testing data using the SAME mapping
    X_test[column] = (
        X_test[column]
        .astype("string")
        .fillna("Unknown")
        .map(mapping)
        .fillna(0)
        .astype(int)
    )

print("\nCategorical encoding completed.")


# ============================================================
# CHECK DATA AFTER ENCODING
# ============================================================

print("\nX_train data types:")
print(X_train.dtypes)

print("\nX_test data types:")
print(X_test.dtypes)


# ============================================================
# HANDLE NUMERICAL VALUES
# ============================================================

print("\nChecking numerical features...")

numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns

for column in numeric_columns:

    # Replace infinite values
    X_train[column] = X_train[column].replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    X_test[column] = X_test[column].replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # Fill missing values using training median
    median_value = X_train[column].median()

    X_train[column] = X_train[column].fillna(
        median_value
    )

    X_test[column] = X_test[column].fillna(
        median_value
    )


print("Numerical values checked.")


# ============================================================
# TRAINING SAMPLE
# ============================================================

SAMPLE_SIZE = 500000

if len(X_train) > SAMPLE_SIZE:

    X_train_model = X_train.sample(
        n=SAMPLE_SIZE,
        random_state=42
    )

    y_train_model = y_train.loc[
        X_train_model.index
    ]

else:

    X_train_model = X_train
    y_train_model = y_train


print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

print("\nRows used for training:")
print(len(X_train_model))

print("\nFeatures used:")
print(X_train_model.shape[1])


# ============================================================
# RANDOM FOREST REGRESSION
# ============================================================

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")
print("Please wait...")


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train_model,
    y_train_model
)


# ============================================================
# TRAINING COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nModel:")
print(model)

print("\nTraining rows:")
print(len(X_train_model))

print("\nNumber of features:")
print(X_train_model.shape[1])

# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np



# ------------------------------------------------------------
# Make predictions
# ------------------------------------------------------------

print("\nMaking predictions on test data...")

y_pred = model.predict(X_test)

print("Predictions completed.")


# ------------------------------------------------------------
# Calculate evaluation metrics
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("\nTarget variable:")
print("Modal_Price")

print("\nMean Absolute Error (MAE):")
print(mae)

print("\nRoot Mean Squared Error (RMSE):")
print(rmse)

print("\nR² Score:")
print(r2)


# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("\nLower MAE and RMSE are better.")

print("\nR² interpretation:")
print("Closer to 1.0 = better model performance.")
print("Closer to 0.0 = weak predictive performance.")


# ------------------------------------------------------------
# Show actual vs predicted values
# ------------------------------------------------------------

results = pd.DataFrame({
    "Actual_Modal_Price": y_test.iloc[:20].values,
    "Predicted_Modal_Price": y_pred[:20]
})

print("\nFirst 20 Actual vs Predicted values:")
print(results)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

import matplotlib.pyplot as plt

# Get feature importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X_train_model.columns,
    "Importance": importance
})

# Sort from highest to lowest
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature importance:")
print(feature_importance)


# ------------------------------------------------------------
# Plot feature importance
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Modal Price Prediction")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED")
print("=" * 60)

# Use a small sample for plotting
plot_size = min(5000, len(y_test))

plot_indices = np.random.RandomState(42).choice(
    len(y_test),
    size=plot_size,
    replace=False
)

actual_values = y_test.iloc[plot_indices].values
predicted_values = y_pred[plot_indices]


# ------------------------------------------------------------
# Plot actual vs predicted
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    actual_values,
    predicted_values,
    alpha=0.3
)

plt.xlabel("Actual Modal Price")
plt.ylabel("Predicted Modal Price")
plt.title("Actual vs Predicted Modal Price")

plt.tight_layout()
plt.show()


print("\nFeature importance and prediction plots completed.")

# ============================================================
# SAVE TRAINED MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING TRAINED MODEL")
print("=" * 60)

import joblib

# Save the trained model
model_file = "agri_shield_modal_price_model.pkl"

joblib.dump(
    model,
    model_file
)

print("\nModel saved successfully!")
print("Model file:", model_file)


# ============================================================
# SAVE FEATURE INFORMATION
# ============================================================

feature_file = "agri_shield_features.pkl"

joblib.dump(
    X_train_model.columns.tolist(),
    feature_file
)

print("Feature information saved successfully!")
print("Feature file:", feature_file)

# ============================================================
# SAVE CATEGORICAL MAPPINGS
# ============================================================ 

category_mapping_file = "agri_shield_category_mappings.pkl"

joblib.dump(
    category_mappings,
    category_mapping_file
)

print("Categorical mappings saved successfully!")
print("Category mapping file:", category_mapping_file)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("MACHINE LEARNING PIPELINE COMPLETED")
print("=" * 60)

print("\nSaved files:")
print("1.", model_file)
print("2.", feature_file)

print("\nTarget:")
print("Modal_Price")

print("\nModel:")
print("Random Forest Regressor")




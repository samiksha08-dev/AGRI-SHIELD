# ============================================================
# AGRI_SHIELD - PRICE PREDICTION
# ============================================================

import pandas as pd
import joblib

print("=" * 60)
print("AGRI_SHIELD PRICE PREDICTION")
print("=" * 60)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load(
    "agri_shield_modal_price_model.pkl"
)

print("\nModel loaded successfully.")


# ============================================================
# LOAD FEATURE INFORMATION
# ============================================================

feature_columns = joblib.load(
    "agri_shield_features.pkl"
)

print("Feature information loaded.")


# ============================================================
# LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(
    "cleaned_agri_shield.csv"
)

print("\nDataset loaded.")
print("Rows:", len(df))


# ============================================================
# SELECT ONE RECORD FOR TEST PREDICTION
# ============================================================

sample = df.drop(
    columns=["Modal_Price"]
).iloc[[0]].copy()


# ============================================================
# DATE FEATURE ENGINEERING
# ============================================================

if "Arrival_Date" in sample.columns:

    sample["Arrival_Date"] = pd.to_datetime(
        sample["Arrival_Date"],
        errors="coerce"
    )

    sample["Arrival_Year"] = (
        sample["Arrival_Date"].dt.year
    )

    sample["Arrival_Month"] = (
        sample["Arrival_Date"].dt.month
    )

    sample["Arrival_Day"] = (
        sample["Arrival_Date"].dt.day
    )

    sample["Arrival_DayOfWeek"] = (
        sample["Arrival_Date"].dt.dayofweek
    )

    sample = sample.drop(
        columns=["Arrival_Date"]
    )


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade"
]

for column in categorical_features:

    if column in sample.columns:

        sample[column] = (
            sample[column]
            .astype("category")
            .cat.codes
        )


# ============================================================
# MAKE SURE FEATURES MATCH MODEL
# ============================================================

for column in feature_columns:

    if column not in sample.columns:

        sample[column] = 0


sample = sample[
    feature_columns
]


# ============================================================
# MAKE PREDICTION
# ============================================================

prediction = model.predict(
    sample
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("PRICE PREDICTION")
print("=" * 60)

print(
    "\nPredicted Modal Price:",
    prediction[0]
)

print("\nPrediction completed successfully.")

# ============================================================
# AGRI_SHIELD - PRICE PREDICTION
# ============================================================

import pandas as pd
import joblib

print("=" * 60)
print("AGRI_SHIELD PRICE PREDICTION")
print("=" * 60)


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

model = joblib.load(
    "agri_shield_modal_price_model.pkl"
)

print("\nModel loaded successfully.")


# ============================================================
# 2. LOAD FEATURE INFORMATION
# ============================================================

feature_columns = joblib.load(
    "agri_shield_features.pkl"
)

print("Feature information loaded.")


# ============================================================
# 3. LOAD CATEGORY MAPPINGS
# ============================================================

category_mappings = joblib.load(
    "agri_shield_category_mappings.pkl"
)

print("Category mappings loaded.")


# ============================================================
# 4. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(
    "cleaned_agri_shield.csv"
)

print("Cleaned dataset loaded.")


# ============================================================
# 5. SELECT ONE SAMPLE
# ============================================================

sample = df.drop(
    columns=["Modal_Price"]
).iloc[[0]].copy()


print("\nSample selected for prediction.")


# ============================================================
# 6. DATE FEATURE ENGINEERING
# ============================================================

if "Arrival_Date" in sample.columns:

    sample["Arrival_Date"] = pd.to_datetime(
        sample["Arrival_Date"],
        errors="coerce"
    )

    sample["Arrival_Year"] = (
        sample["Arrival_Date"].dt.year
    )

    sample["Arrival_Month"] = (
        sample["Arrival_Date"].dt.month
    )

    sample["Arrival_Day"] = (
        sample["Arrival_Date"].dt.day
    )

    sample["Arrival_DayOfWeek"] = (
        sample["Arrival_Date"].dt.dayofweek
    )

    sample = sample.drop(
        columns=["Arrival_Date"]
    )


# ============================================================
# 7. APPLY SAME CATEGORY MAPPINGS
# ============================================================

for column, categories in category_mappings.items():

    if column in sample.columns:

        sample[column] = pd.Categorical(
            sample[column],
            categories=categories
        ).codes


# ============================================================
# 8. HANDLE MISSING / INFINITE VALUES
# ============================================================

sample = sample.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

for column in sample.columns:

    if sample[column].isnull().any():

        if pd.api.types.is_numeric_dtype(
            sample[column]
        ):

            sample[column] = sample[column].fillna(0)


# ============================================================
# 9. MAKE FEATURE COLUMNS MATCH MODEL
# ============================================================

for column in feature_columns:

    if column not in sample.columns:

        sample[column] = 0


# Keep exactly the same order as training
sample = sample[
    feature_columns
]


# ============================================================
# 10. MAKE PREDICTION
# ============================================================

print("\nMaking prediction...")

prediction = model.predict(
    sample
)


# ============================================================
# 11. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print("\nPredicted Modal Price:")
print(prediction[0])

print("\nPrediction completed successfully!")
import pandas as pd
import joblib
import numpy as np


# ============================================================
# LOAD FILES
# ============================================================

data = pd.read_csv("cleaned_agri_shield.csv")

model = joblib.load(
    "agri_shield_modal_price_model.pkl"
)

feature_columns = joblib.load(
    "agri_shield_features.pkl"
)

category_mappings = joblib.load(
    "agri_shield_category_mappings.pkl"
)


# ============================================================
# TAKE ONE REAL ROW
# ============================================================

row = data.iloc[0].copy()

actual_price = row["Modal_Price"]


# ============================================================
# CREATE INPUT
# ============================================================

input_data = pd.DataFrame([row])

# Remove target
input_data = input_data.drop(
    columns=["Modal_Price"],
    errors="ignore"
)


# ============================================================
# DATE FEATURES
# ============================================================

if "Arrival_Date" in input_data.columns:

    input_data["Arrival_Date"] = pd.to_datetime(
        input_data["Arrival_Date"],
        errors="coerce"
    )

    input_data["Arrival_Year"] = (
        input_data["Arrival_Date"].dt.year
    )

    input_data["Arrival_Month"] = (
        input_data["Arrival_Date"].dt.month
    )

    input_data["Arrival_Day"] = (
        input_data["Arrival_Date"].dt.day
    )

    input_data["Arrival_DayOfWeek"] = (
        input_data["Arrival_Date"].dt.dayofweek
    )

    input_data = input_data.drop(
        columns=["Arrival_Date"]
    )


# ============================================================
# CATEGORY ENCODING
# ============================================================

for column, categories in category_mappings.items():

    if column in input_data.columns:

        input_data[column] = pd.Categorical(
            input_data[column]
            .astype("string")
            .fillna("Unknown"),
            categories=categories
        ).codes

        input_data[column] = (
            input_data[column]
            .replace(-1, 0)
        )


# ============================================================
# MAKE FEATURES MATCH TRAINING
# ============================================================

for column in feature_columns:

    if column not in input_data.columns:
        input_data[column] = 0


input_data = input_data[
    feature_columns
]


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    input_data
)[0]


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("MODEL TEST")
print("=" * 60)

print("\nInput row:")

print(
    row.drop(
        labels=["Modal_Price"]
    )
)

print("\nActual Modal Price:")
print(f"₹{actual_price:,.2f}")

print("\nPredicted Modal Price:")
print(f"₹{prediction:,.2f}")

print("\nAbsolute Error:")
print(
    f"₹{abs(actual_price - prediction):,.2f}"
)

percentage_error = (
    abs(actual_price - prediction)
    / abs(actual_price)
) * 100

print("\nPercentage Error:")
print(
    f"{percentage_error:.2f}%"
)

print("\n" + "=" * 60)
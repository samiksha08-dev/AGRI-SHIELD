import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. LOAD DATA AND MODEL
# ============================================================

print("=" * 60)
print("AGRI_SHIELD MODEL VALIDATION")
print("=" * 60)

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
# 2. SELECT 100 RANDOM ROWS
# ============================================================

test_data = data.sample(
    n=min(100, len(data)),
    random_state=42
).copy()

actual_prices = test_data["Modal_Price"].copy()

input_data = test_data.drop(
    columns=["Modal_Price"],
    errors="ignore"
).copy()


# ============================================================
# 3. DATE FEATURE ENGINEERING
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
# 4. APPLY CATEGORY MAPPINGS
# ============================================================

for column, mapping in category_mappings.items():

    if column in input_data.columns:

        input_data[column] = (
            input_data[column]
            .astype("string")
            .fillna("Unknown")
            .map(mapping)
            .fillna(0)
            .astype(int)
        )

# ============================================================
# 5. MATCH MODEL FEATURES
# ============================================================

for column in feature_columns:

    if column not in input_data.columns:
        input_data[column] = 0


input_data = input_data[
    feature_columns
]


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions for 100 rows...")

predicted_prices = model.predict(
    input_data
)


# ============================================================
# 7. CALCULATE ERRORS
# ============================================================

results = pd.DataFrame({
    "Actual_Price": actual_prices.to_numpy(),
    "Predicted_Price": predicted_prices
})

results["Absolute_Error"] = (
    results["Actual_Price"]
    - results["Predicted_Price"]
).abs()

results["Percentage_Error"] = (
    results["Absolute_Error"]
    / results["Actual_Price"].replace(
        0,
        np.nan
    ).abs()
) * 100


# ============================================================
# 8. ACCURACY RANGES
# ============================================================

within_5 = (
    results["Percentage_Error"] <= 5
).sum()

within_10 = (
    results["Percentage_Error"] <= 10
).sum()

within_20 = (
    results["Percentage_Error"] <= 20
).sum()


total = len(results)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print("\nTotal rows tested:")
print(total)

print("\nPredictions within 5%:")
print(
    f"{within_5}/{total} "
    f"({within_5 / total * 100:.2f}%)"
)

print("\nPredictions within 10%:")
print(
    f"{within_10}/{total} "
    f"({within_10 / total * 100:.2f}%)"
)

print("\nPredictions within 20%:")
print(
    f"{within_20}/{total} "
    f"({within_20 / total * 100:.2f}%)"
)


# ============================================================
# 10. ERROR STATISTICS
# ============================================================

print("\nMean Absolute Error:")

print(
    results["Absolute_Error"].mean()
)

print("\nMedian Absolute Error:")

print(
    results["Absolute_Error"].median()
)

print("\nMaximum Absolute Error:")

print(
    results["Absolute_Error"].max()
)


# ============================================================
# 11. SHOW SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(
    results.head(20).to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE VALIDATION RESULTS
# ============================================================

output_file = "model_validation_results.csv"

results.to_csv(
    output_file,
    index=False
)

print("\nValidation results saved to:")
print(output_file)


print("\n" + "=" * 60)
print("MODEL VALIDATION COMPLETED")
print("=" * 60)
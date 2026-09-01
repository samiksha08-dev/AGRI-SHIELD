import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AGRI_SHIELD",
    page_icon="🌾",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌾 AGRI_SHIELD")
st.subheader("Agricultural Market Price Prediction")

st.write(
    "Enter the market details below to predict the expected "
    "Modal Price."
)


# ============================================================
# LOAD MODEL FILES
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "agri_shield_modal_price_model.pkl"
    )

    feature_columns = joblib.load(
        "agri_shield_features.pkl"
    )

    category_mappings = joblib.load(
        "agri_shield_category_mappings.pkl"
    )

    return (
        model,
        feature_columns,
        category_mappings
    )


try:

    model, feature_columns, category_mappings = load_model()

except Exception as e:

    st.error(
        "Model files could not be loaded."
    )

    st.write(
        "Make sure these files are in the same folder as app.py:"
    )

    st.code(
        """
agri_shield_modal_price_model.pkl
agri_shield_features.pkl
agri_shield_category_mappings.pkl
        """
    )

    st.stop()


# ============================================================
# USER INPUT
# ============================================================

st.markdown("### Enter Market Information")


# State
state = st.selectbox(
    "State",
    options=category_mappings.get(
        "State",
        ["Unknown"]
    )
)


# District
district = st.selectbox(
    "District",
    options=category_mappings.get(
        "District",
        ["Unknown"]
    )
)


# Market
market = st.selectbox(
    "Market",
    options=category_mappings.get(
        "Market",
        ["Unknown"]
    )
)


# Commodity
commodity = st.selectbox(
    "Commodity",
    options=category_mappings.get(
        "Commodity",
        ["Unknown"]
    )
)


# Variety
variety = st.selectbox(
    "Variety",
    options=category_mappings.get(
        "Variety",
        ["Unknown"]
    )
)


# Grade
grade = st.selectbox(
    "Grade",
    options=category_mappings.get(
        "Grade",
        ["Unknown"]
    )
)


# Minimum price
min_price = st.number_input(
    "Minimum Price",
    min_value=0.0,
    value=0.0,
    step=1.0
)


# Maximum price
max_price = st.number_input(
    "Maximum Price",
    min_value=0.0,
    value=0.0,
    step=1.0
)


# Arrival date
arrival_date = st.date_input(
    "Arrival Date"
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Modal Price",
    use_container_width=True
):

    # --------------------------------------------------------
    # Validate prices
    # --------------------------------------------------------

    if max_price < min_price:

        st.error(
            "Maximum price cannot be lower than minimum price."
        )

        st.stop()


    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "State": [state],
        "District": [district],
        "Market": [market],
        "Commodity": [commodity],
        "Variety": [variety],
        "Grade": [grade],
        "Min_Price": [min_price],
        "Max_Price": [max_price],
        "Arrival_Date": [arrival_date]
    })


    # --------------------------------------------------------
    # Date feature engineering
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Apply saved categorical mappings
    # --------------------------------------------------------

    for column, categories in category_mappings.items():

        if column in input_data.columns:

            input_data[column] = pd.Categorical(
                input_data[column]
                .astype("string")
                .fillna("Unknown"),
                categories=categories
            ).codes

            # Unknown category → 0
            input_data[column] = (
                input_data[column]
                .replace(-1, 0)
            )


    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    input_data = input_data.replace(
        [float("inf"), float("-inf")],
        0
    )


    # --------------------------------------------------------
    # Make feature columns match model
    # --------------------------------------------------------

    for column in feature_columns:

        if column not in input_data.columns:

            input_data[column] = 0


    # Keep exactly the features used during training
    input_data = input_data[
        feature_columns
    ]


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            input_data
        )

        predicted_price = float(
            prediction[0]
        )

        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.success(
            "Prediction completed successfully!"
        )

        st.metric(
            label="Predicted Modal Price",
            value=f"{predicted_price:,.2f}"
        )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AGRI_SHIELD | Agricultural Market Price Prediction"
)
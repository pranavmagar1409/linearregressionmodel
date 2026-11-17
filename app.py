import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "linearregression.pkl"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


st.title("🏠 House Price Predictor")
st.write("Enter the house details and get a price estimate from the Linear Regression model.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. Put it in the same folder as app.py.")
    st.stop()

model = load_model()

col1, col2 = st.columns(2)

with col1:
    square_footage = st.number_input("Square Footage", min_value=100, max_value=10000, value=2000, step=50)
    num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
    num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2030, value=2000, step=1)

with col2:
    lot_size = st.number_input("Lot Size", min_value=0.0, max_value=50.0, value=2.5, step=0.1)
    garage_size = st.number_input("Garage Size (cars)", min_value=0, max_value=6, value=1, step=1)
    neighborhood_quality = st.slider("Neighborhood Quality", min_value=1, max_value=10, value=5)

if st.button("Predict Price", type="primary"):
    # Column names/order must match what the model was trained on
    input_df = pd.DataFrame(
        [
            {
                "Square_Footage": square_footage,
                "Num_Bedrooms": num_bedrooms,
                "Num_Bathrooms": num_bathrooms,
                "Year_Built": year_built,
                "Lot_Size": lot_size,
                "Garage_Size": garage_size,
                "Neighborhood_Quality": neighborhood_quality,
            }
        ]
    )
    prediction = float(model.predict(input_df)[0])
    st.success(f"Estimated Price: **{prediction:,.2f}**")

    with st.expander("Show input data"):
        st.dataframe(input_df)

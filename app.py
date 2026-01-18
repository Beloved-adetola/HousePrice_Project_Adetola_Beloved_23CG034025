import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("model/house_price_model.pkl", "rb") as f:
    model, scaler, encoder = pickle.load(f)

st.set_page_config(page_title="House Price Prediction System")

st.title("🏠 House Price Prediction System")
st.write("Predict house prices using selected property features.")

# User inputs
overall_qual = st.slider("Overall Quality (1 - 10)", 1, 10, 5)
gr_liv_area = st.number_input("Ground Living Area (sq ft)", min_value=0.0, step=10.0)
total_bsmt = st.number_input("Total Basement Area (sq ft)", min_value=0.0, step=10.0)
garage_cars = st.slider("Garage Cars", 0, 5, 1)
year_built = st.number_input("Year Built", min_value=1800, max_value=2026, step=1)
neighborhood = st.text_input("Neighborhood (e.g. NAmes, CollgCr, OldTown)")

if st.button("Predict House Price"):
    try:
        neighborhood_encoded = encoder.transform([neighborhood])[0]

        input_df = pd.DataFrame(
            [[
                overall_qual,
                gr_liv_area,
                total_bsmt,
                garage_cars,
                year_built,
                neighborhood_encoded
            ]],
            columns=[
                "OverallQual",
                "GrLivArea",
                "TotalBsmtSF",
                "GarageCars",
                "YearBuilt",
                "Neighborhood"
            ]
        )

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.success(f"Predicted House Price: **${prediction:,.2f}**")

    except:
        st.error("Neighborhood not recognized. Try values like: NAmes, CollgCr, OldTown, Edwards, Sawyer")

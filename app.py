import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model_file_path = "trained_model.pkl"
try:
    model = joblib.load(model_file_path)
    st.write("Model loaded successfully.")
except FileNotFoundError:
    st.error(f"Model file '{model_file_path}' not found. Please ensure it's available.")

# Display an image on the home page
st.image("architecture-4810651_1280.png", caption="Welcome to the House Price Predictor", use_column_width=True)

# Streamlit app title
st.title("House Price Predictor")
st.write("Predict house prices based on features like square footage, bedrooms, bathrooms, and neighborhood characteristics.")

# Sidebar inputs
st.sidebar.header("Enter House Details:")
square_feet = st.sidebar.number_input("Square Feet", min_value=500, max_value=10000, step=50, value=1500)
bedrooms = st.sidebar.slider("Number of Bedrooms", min_value=1, max_value=10, step=1, value=3)
bathrooms = st.sidebar.slider("Number of Bathrooms", min_value=1, max_value=10, step=1, value=2)
year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2024, step=1, value=2000)

# One-hot encoding for Neighborhood
st.sidebar.subheader("Select Neighborhood Type:")
neighborhood = st.sidebar.selectbox("Neighborhood", ["Rural", "Suburb", "Urban"])

# One-hot encoding inputs
neighborhood_rural = 1 if neighborhood == "Rural" else 0
neighborhood_suburb = 1 if neighborhood == "Suburb" else 0
neighborhood_urban = 1 if neighborhood == "Urban" else 0

# Prediction button
if st.sidebar.button("Predict Price"):
    if model:
        # Prepare the input as a DataFrame for prediction
        input_data = pd.DataFrame({
            "SquareFeet": [square_feet],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "YearBuilt": [year_built],
            "Neighborhood_Rural": [neighborhood_rural],
            "Neighborhood_Suburb": [neighborhood_suburb],
            "Neighborhood_Urban": [neighborhood_urban],
        })

        # Make a prediction
        try:
            predicted_price = model.predict(input_data)[0]
            st.write(f"### Predicted Price: ${predicted_price:,.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Model not loaded. Please ensure the model file is available.")


# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Load saved files from repo
# ---------------------------
# df = pd.read_csv("city_day.csv")  # your dataset (optional for display)
rf_model = joblib.load("rf_aqi_model.joblib")
scaler = joblib.load("scaler.joblib")
cities = joblib.load("cities.joblib")

# ---------------------------
# App title
# ---------------------------
st.title("AQI Prediction App")

# ---------------------------
# User inputs
# ---------------------------
st.subheader("Enter Pollutant Levels")
pm25 = st.number_input("PM2.5", min_value=0.0, max_value=500.0, value=50.0)
pm10 = st.number_input("PM10", min_value=0.0, max_value=500.0, value=80.0)
no2 = st.number_input("NO2", min_value=0.0, max_value=200.0, value=40.0)
so2 = st.number_input("SO2", min_value=0.0, max_value=100.0, value=10.0)
co = st.number_input("CO", min_value=0.0, max_value=50.0, value=1.0)
o3 = st.number_input("O3", min_value=0.0, max_value=300.0, value=30.0)

city = st.selectbox("Select City", cities)

# ---------------------------
# Predict AQI
# ---------------------------
if st.button("Predict AQI"):
    numeric_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    
    # Create user input dataframe
    user_input = pd.DataFrame({
        'PM2.5': [pm25],
        'PM10': [pm10],
        'NO2': [no2],
        'SO2': [so2],
        'CO': [co],
        'O3': [o3],
        'City': [city]
    })

    # Scale numeric features only
    user_input[numeric_features] = scaler.transform(user_input[numeric_features])

    # Encode city
    user_input['City_encoded'] = cities.index(city)

    # Prepare final features in correct order
    X_input = user_input[numeric_features + ['City_encoded']]

    # Predict
    predicted_aqi = rf_model.predict(X_input)

    st.success(f"Predicted AQI: {predicted_aqi[0]:.2f}")


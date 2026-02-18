# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Load saved files
# ---------------------------
rf_model = joblib.load("rf_aqi_model.joblib")
scaler = joblib.load("scaler.joblib")
cities = joblib.load("cities.joblib")

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------
# App title
# ---------------------------
st.markdown("<h1 style='text-align: center; color: green;'>🌿 AQI Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Predict Air Quality Index based on pollutant levels</p>", unsafe_allow_html=True)
st.write("---")

# ---------------------------
# Sidebar for inputs
# ---------------------------
st.sidebar.header("Enter Pollutant Levels & City")

pm25 = st.sidebar.slider("PM2.5 (µg/m³)", 0.0, 500.0, 50.0)
pm10 = st.sidebar.slider("PM10 (µg/m³)", 0.0, 500.0, 80.0)
no2 = st.sidebar.slider("NO2 (µg/m³)", 0.0, 200.0, 40.0)
so2 = st.sidebar.slider("SO2 (µg/m³)", 0.0, 100.0, 10.0)
co = st.sidebar.slider("CO (mg/m³)", 0.0, 50.0, 1.0)
o3 = st.sidebar.slider("O3 (µg/m³)", 0.0, 300.0, 30.0)
city = st.sidebar.selectbox("Select City", cities)

# ---------------------------
# Main content
# ---------------------------
st.subheader("Your Inputs")
st.write(f"**City:** {city}")
st.write(f"**PM2.5:** {pm25}  |  **PM10:** {pm10}")
st.write(f"**NO2:** {no2}  |  **SO2:** {so2}")
st.write(f"**CO:** {co}  |  **O3:** {o3}")

st.write("---")

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict AQI"):
    numeric_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

    # User input dataframe
    user_input = pd.DataFrame({
        'PM2.5': [pm25],
        'PM10': [pm10],
        'NO2': [no2],
        'SO2': [so2],
        'CO': [co],
        'O3': [o3],
        'City': [city]
    })

    # Scale numeric features
    user_input[numeric_features] = scaler.transform(user_input[numeric_features])

    # Encode city
    user_input['City_encoded'] = cities.index(city)

    # Prepare final features
    X_input = user_input[numeric_features + ['City_encoded']]

    # Predict
    predicted_aqi = rf_model.predict(X_input)[0]

    # Display AQI with color coding
    if predicted_aqi <= 50:
        color = "green"
        status = "Good"
    elif predicted_aqi <= 100:
        color = "yellow"
        status = "Moderate"
    elif predicted_aqi <= 150:
        color = "orange"
        status = "Unhealthy for Sensitive Groups"
    elif predicted_aqi <= 200:
        color = "red"
        status = "Unhealthy"
    elif predicted_aqi <= 300:
        color = "purple"
        status = "Very Unhealthy"
    else:
        color = "maroon"
        status = "Hazardous"

    st.markdown(
        f"<h2 style='text-align: center; color: {color};'>Predicted AQI: {predicted_aqi:.2f} ({status})</h2>",
        unsafe_allow_html=True
    )

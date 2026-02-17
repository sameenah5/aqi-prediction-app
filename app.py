# app.py
# AQI Prediction App using Streamlit
# Features: PM2.5, PM10, NO2, SO2, CO, O3, City

import streamlit as st
import numpy as np
import joblib
import os
import urllib.request

# -------------------------------
# Download model assets (from GitHub Release)
# -------------------------------

BASE_URL = "https://github.com/sameenah5/aqi-prediction-app/releases/download/v1.0.0/"

FILES = {
    "rf_aqi_model.joblib": BASE_URL + "rf_aqi_model.joblib",
    "scaler.joblib": BASE_URL + "scaler.joblib",
    "cities.joblib": BASE_URL + "cities.joblib",
}

for file_name, url in FILES.items():
    if not os.path.exists(file_name):
        urllib.request.urlretrieve(url, file_name)

# -------------------------------
# Load assets
# -------------------------------

model = joblib.load("rf_aqi_model.joblib")
scaler = joblib.load("scaler.joblib")
cities = joblib.load("cities.joblib")   # this is a LIST

# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(
    page_title="AQI Prediction App",
    layout="centered"
)

st.title("🌫️ Air Quality Index (AQI) Prediction")
st.markdown("Predict AQI based on air pollutant concentrations")

# -------------------------------
# User Inputs
# -------------------------------

st.subheader("Enter Pollutant Values")

pm25 = st.number_input("PM2.5", 0.0, 500.0, 50.0)
pm10 = st.number_input("PM10", 0.0, 500.0, 80.0)
no2  = st.number_input("NO2",  0.0, 300.0, 40.0)
so2  = st.number_input("SO2",  0.0, 200.0, 20.0)
co   = st.number_input("CO",   0.0, 50.0,  1.0)
o3   = st.number_input("O3",   0.0, 300.0, 30.0)

city = st.selectbox("City", cities)

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict AQI"):
    city_index = cities.index(city)

    input_data = np.array([[pm25, pm10, no2, so2, co, o3, city_index]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.subheader("Predicted AQI")
    st.success(f"AQI Value: {prediction:.2f}")

    if prediction <= 50:
        st.info("Category: Good 😊")
    elif prediction <= 100:
        st.success("Category: Satisfactory 🙂")
    elif prediction <= 200:
        st.warning("Category: Moderate 😐")
    elif prediction <= 300:
        st.warning("Category: Poor 😷")
    elif prediction <= 400:
        st.error("Category: Very Poor 🤒")
    else:
        st.error("Category: Severe ☠️")


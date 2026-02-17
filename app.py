# app.py
# AQI Prediction App using Streamlit
# Features used: PM2.5, PM10, NO2, SO2, CO, O3, City

import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load trained model
# -------------------------------
with open("aqi_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("city_encoder.pkl", "rb") as file:
    city_encoder = pickle.load(file)

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
# User Inputs (FEATURES)
# -------------------------------
st.subheader("Enter Pollutant Values")

pm25 = st.number_input("PM2.5", min_value=0.0, max_value=500.0, value=50.0)
pm10 = st.number_input("PM10", min_value=0.0, max_value=500.0, value=80.0)
no2  = st.number_input("NO2",  min_value=0.0, max_value=300.0, value=40.0)
so2  = st.number_input("SO2",  min_value=0.0, max_value=200.0, value=20.0)
co   = st.number_input("CO",   min_value=0.0, max_value=50.0,  value=1.0)
o3   = st.number_input("O3",   min_value=0.0, max_value=300.0, value=30.0)

# City input (categorical feature)
city = st.selectbox("City", city_encoder.classes_)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict AQI"):
    city_encoded = city_encoder.transform([city])[0]

    input_data = np.array([[pm25, pm10, no2, so2, co, o3, city_encoded]])

    prediction = model.predict(input_data)[0]

    st.subheader("Predicted AQI")
    st.success(f"AQI Value: {prediction:.2f}")

    # AQI Category
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

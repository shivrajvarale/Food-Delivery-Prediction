"""Streamlit app: predict food delivery time from order details.

Run:  streamlit run app.py
The model is trained automatically on first run (a few seconds) if it does not exist yet.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("model/food_delivery_model.joblib")

st.set_page_config(page_title="Food Delivery Time Predictor", page_icon="🛵")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6371 * np.arcsin(np.sqrt(a)))


st.title("🛵 Food Delivery Time Predictor")
st.write("Enter the order details to estimate how many minutes the delivery will take.")

if not MODEL_PATH.exists():
    with st.spinner("First run: training the model (takes a few seconds)..."):
        import train
        train.main()

model = load_model()

st.subheader("Locations")
c1, c2 = st.columns(2)
with c1:
    r_lat = st.number_input("Restaurant latitude", value=22.745049, format="%.6f")
    r_lon = st.number_input("Restaurant longitude", value=75.892471, format="%.6f")
with c2:
    d_lat = st.number_input("Delivery latitude", value=22.805049, format="%.6f")
    d_lon = st.number_input("Delivery longitude", value=75.952471, format="%.6f")

distance = haversine_km(r_lat, r_lon, d_lat, d_lon)
st.caption(f"Straight-line distance: **{distance:.2f} km**")
if distance >= 100:
    st.warning("These points are over 100 km apart, which is outside the data the model was trained on.")

st.subheader("Delivery person and vehicle")
c3, c4, c5 = st.columns(3)
age = c3.number_input("Delivery person age", min_value=15, max_value=60, value=30)
rating = c4.slider("Delivery person rating", 1.0, 5.0, 4.6, 0.1)
vehicle_condition = c5.selectbox("Vehicle condition (0 = worst, 3 = best)", [0, 1, 2, 3], index=1)
vehicle_type = c3.selectbox("Vehicle type", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
multiple = c4.selectbox("Other deliveries on this trip", [0, 1, 2, 3])

st.subheader("Order and conditions")
c6, c7, c8 = st.columns(3)
order_type = c6.selectbox("Order type", ["Snack", "Meal", "Drinks", "Buffet"])
traffic = c7.selectbox("Road traffic", ["Low", "Medium", "High", "Jam"], index=1)
weather = c8.selectbox("Weather", ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"])
city = c6.selectbox("City type", ["Metropolitian", "Urban", "Semi-Urban"])
festival = c7.selectbox("Festival day?", ["No", "Yes"])

if st.button("Predict delivery time", type="primary"):
    row = pd.DataFrame([{
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "Restaurant_latitude": r_lat,
        "Restaurant_longitude": r_lon,
        "Delivery_location_latitude": d_lat,
        "Delivery_location_longitude": d_lon,
        "Vehicle_condition": vehicle_condition,
        "multiple_deliveries": multiple,
        "distance_km": distance,
        "Road_traffic_density": {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}[traffic],
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "Festival": festival,
        "City": city,
        "Weatherconditions": weather,
    }])
    minutes = float(model.predict(row)[0])
    st.success(f"Estimated delivery time: **{minutes:.0f} minutes**")
    st.caption("The model is typically off by about 4 minutes on average (test RMSE).")

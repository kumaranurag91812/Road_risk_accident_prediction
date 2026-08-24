%%writefile /kaggle/working/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "/kaggle/working/road_accident_risk_model.pkl"

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Road Accident Risk Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Road Accident Risk Prediction")
st.write(
    "Enter road and environmental conditions "
    "to predict accident risk score."
)

st.divider()

col1, col2 = st.columns(2)

with col1:

    city = st.text_input(
        "City",
        value="Lucknow"
    )

    state = st.text_input(
        "State",
        value="Uttar Pradesh"
    )

    latitude = st.number_input(
        "Latitude",
        value=26.8467
    )

    longitude = st.number_input(
        "Longitude",
        value=80.9462
    )

    hour = st.slider(
        "Hour",
        min_value=0,
        max_value=23,
        value=18
    )

    day_of_week = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    is_weekend = st.selectbox(
        "Weekend?",
        [0, 1]
    )

    road_type = st.selectbox(
        "Road Type",
        [
            "Highway",
            "Urban",
            "Rural",
            "Residential",
            "Intersection"
        ]
    )

    lanes = st.number_input(
        "Number of Lanes",
        min_value=1,
        max_value=10,
        value=2
    )

with col2:

    traffic_signal = st.selectbox(
        "Traffic Signal",
        [0, 1]
    )

    weather = st.selectbox(
        "Weather",
        [
            "Clear",
            "Rain",
            "Fog",
            "Storm",
            "Cloudy"
        ]
    )

    visibility = st.number_input(
        "Visibility",
        min_value=0.0,
        value=10.0
    )

    temperature = st.number_input(
        "Temperature",
        value=25.0
    )

    traffic_density = st.selectbox(
        "Traffic Density",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    cause = st.selectbox(
        "Accident Cause",
        [
            "Speeding",
            "Drunk Driving",
            "Poor Road Condition",
            "Weather",
            "Driver Distraction",
            "Other"
        ]
    )

    accident_severity = st.selectbox(
        "Accident Severity",
        [
            "Minor",
            "Moderate",
            "Severe"
        ]
    )

    vehicles_involved = st.number_input(
        "Vehicles Involved",
        min_value=1,
        max_value=20,
        value=2
    )

    casualties = st.number_input(
        "Casualties",
        min_value=0,
        max_value=50,
        value=0
    )

    is_peak_hour = st.selectbox(
        "Peak Hour?",
        [0, 1]
    )

    festival = st.selectbox(
        "Festival?",
        [0, 1]
    )

st.divider()

if st.button(
    "🔮 Predict Accident Risk",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "city": [city],
        "state": [state],
        "latitude": [latitude],
        "longitude": [longitude],
        "hour": [hour],
        "day_of_week": [day_of_week],
        "is_weekend": [is_weekend],
        "road_type": [road_type],
        "lanes": [lanes],
        "traffic_signal": [traffic_signal],
        "weather": [weather],
        "visibility": [visibility],
        "temperature": [temperature],
        "traffic_density": [traffic_density],
        "cause": [cause],
        "accident_severity": [accident_severity],
        "vehicles_involved": [vehicles_involved],
        "casualties": [casualties],
        "is_peak_hour": [is_peak_hour],
        "festival": [festival]
    })

    prediction = model.predict(
        input_data
    )[0]

    prediction = max(
        0,
        float(prediction)
    )

    st.success(
        "Prediction completed successfully!"
    )

    st.metric(
        "Predicted Accident Risk Score",
        f"{prediction:.2f}"
    )

    if prediction < 30:
        st.info("🟢 Low Risk")
    elif prediction < 70:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

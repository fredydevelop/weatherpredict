# ============================================================
# WEATHER PREDICTION STREAMLIT APP - NEXT DAY ONLY
# ============================================================

import streamlit as st
import pandas as pd
import pickle as pk
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agricultural Weather Prediction System",
    layout="centered"
)

st.title("🌦️ Agricultural Weather Prediction System")

selection = option_menu(
    menu_title=None,
    options=["Next Day Prediction"],
    icons=["cloud-sun"],
    default_index=0,
    orientation="horizontal"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_rf_nextday_model():
    with open("rf_weather_model_nextday.pkl", "rb") as f:
        return pk.load(f)


@st.cache_resource
def load_lstm_nextday_model():
    return load_model("best_lstm_weather_model.keras")


@st.cache_resource
def load_lstm_scaler():
    with open("lstm_scaler.pkl", "rb") as f:
        return pk.load(f)


# ============================================================
# AGRICULTURAL ADVICE FUNCTION
# ============================================================

def agricultural_advice(pred_temp, pred_humidity, pred_wind, pred_pressure):
    advice = []

    if pred_temp > 35:
        advice.append("High temperature expected. Delay planting and increase crop monitoring.")
    elif 22 <= pred_temp <= 32:
        advice.append("Temperature is suitable for most agricultural activities.")
    else:
        advice.append("Low temperature may slow crop growth. Monitor sensitive crops.")

    if pred_humidity < 40:
        advice.append("Low humidity detected. Irrigation may be required.")
    elif pred_humidity > 80:
        advice.append("High humidity expected. Monitor crops for fungal diseases.")
    else:
        advice.append("Humidity condition is moderate and suitable for crop growth.")

    if pred_wind > 15:
        advice.append("High wind speed expected. Avoid spraying chemicals.")
    else:
        advice.append("Wind condition is acceptable for farming operations.")

    if pred_pressure < 1005:
        advice.append("Low atmospheric pressure may indicate unstable weather.")
    else:
        advice.append("Atmospheric pressure appears stable.")

    return advice


# ============================================================
# FEATURE CREATION FOR RANDOM FOREST
# ============================================================

def create_rf_input(date, meantemp, humidity, wind_speed, meanpressure):
    date = pd.to_datetime(date)

    input_df = pd.DataFrame({
        "meantemp": [meantemp],
        "humidity": [humidity],
        "wind_speed": [wind_speed],
        "meanpressure": [meanpressure],
        "day": [date.day],
        "month": [date.month],
        "year": [date.year],
        "dayofyear": [date.dayofyear]
    })

    return input_df


# ============================================================
# NEXT DAY PREDICTION
# ============================================================

def next_day_prediction():
    st.header("Next-Day Weather Prediction")


    st.subheader("Enter Current Weather Conditions")

    date = st.date_input("Date")
    meantemp = st.number_input("Mean Temperature (°C)", value=25.00)
    humidity = st.number_input("Humidity (%)", value=60.00)
    wind_speed = st.number_input("Wind Speed", value=5.00)
    meanpressure = st.number_input("Mean Pressure", value=1010.00)

    if st.button("Predict Next Day Weather"):

            rf_model = load_rf_nextday_model()

            input_df = create_rf_input(
                date,
                meantemp,
                humidity,
                wind_speed,
                meanpressure
            )

            prediction = rf_model.predict(input_df)[0]

        else:
            st.warning(
                
                "An error occured try again later"
            )
            return

        pred_temp = prediction[0]
        pred_humidity = prediction[1]
        pred_wind = prediction[2]
        pred_pressure = prediction[3]

        st.success("Next-Day Weather Prediction Completed")

        result_df = pd.DataFrame({
            "Predicted Temperature (°C)": [round(pred_temp, 2)],
            "Predicted Humidity (%)": [round(pred_humidity, 2)],
            "Predicted Wind Speed": [round(pred_wind, 2)],
            "Predicted Pressure": [round(pred_pressure, 2)]
        })

        st.dataframe(result_df)

        st.subheader("Agricultural Recommendations")
        for i, advice in enumerate(
            agricultural_advice(pred_temp, pred_humidity, pred_wind, pred_pressure), 1
        ):
            st.write(f"{i}. {advice}")


# ============================================================
# APP ROUTING
# ============================================================

if selection == "Next Day Prediction":
    next_day_prediction()

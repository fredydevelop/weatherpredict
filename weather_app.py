# ============================================================
# WEATHER PREDICTION STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
import base64
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
    options=["Single Prediction", "Multi Prediction"],
    icons=["cloud-sun", "calendar-week"],
    default_index=0,
    orientation="horizontal"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_rf_nextday_model():
    with open("rf_weather_nextday_model.pkl", "rb") as f:
        return pk.load(f)


@st.cache_resource
def load_lstm_nextday_model():
    return load_model("lstm_weather_nextday_model.keras")


@st.cache_resource
def load_lstm_scaler():
    with open("lstm_scaler.pkl", "rb") as f:
        return pk.load(f)


# 7-day models - use after training and saving them
@st.cache_resource
def load_rf_7day_model():
    with open("rf_weather_7day_model.pkl", "rb") as f:
        return pk.load(f)


@st.cache_resource
def load_lstm_7day_model():
    return load_model("lstm_weather_7day_model.keras")


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="weather_prediction.csv">Download Prediction CSV</a>'
    return href


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
# FEATURE CREATION
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
# SINGLE PREDICTION - NEXT DAY
# ============================================================

def single_prediction():
    st.header("Next-Day Weather Prediction")

    model_type = st.selectbox(
        "Select Model",
        ["Random Forest", "LSTM"]
    )

    st.subheader("Enter Current Weather Conditions")

    date = st.date_input("Date")
    meantemp = st.number_input("Mean Temperature (°C)", value=25.00)
    humidity = st.number_input("Humidity (%)", value=60.00)
    wind_speed = st.number_input("Wind Speed", value=5.00)
    meanpressure = st.number_input("Mean Pressure", value=1010.00)

    if st.button("Predict Next Day Weather"):

        if model_type == "Random Forest":
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
            st.warning("LSTM needs the previous 7 days of weather records. For LSTM prediction, use the Multi Prediction section.")
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
# MULTI PREDICTION - 7 DAY PREDICTION
# ============================================================

def multi_prediction(uploaded_file):
    st.header("7-Day Weather Prediction")

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df)

    required_columns = ["date", "meantemp", "humidity", "wind_speed", "meanpressure"]

    if not all(col in df.columns for col in required_columns):
        st.error("Dataset must contain: date, meantemp, humidity, wind_speed, meanpressure")
        return

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    model_type = st.selectbox(
        "Select 7-Day Prediction Model",
        ["Random Forest 7-Day", "LSTM 7-Day"]
    )

    if st.button("Predict 7-Day Weather"):

        if model_type == "Random Forest 7-Day":
            try:
                rf_7day_model = load_rf_7day_model()

                latest_row = df.iloc[-1]

                input_df = create_rf_input(
                    latest_row["date"],
                    latest_row["meantemp"],
                    latest_row["humidity"],
                    latest_row["wind_speed"],
                    latest_row["meanpressure"]
                )

                prediction = rf_7day_model.predict(input_df)[0]

            except FileNotFoundError:
                st.error("7-day Random Forest model not found. Train and save rf_weather_7day_model.pkl first.")
                return

        else:
            try:
                lstm_7day_model = load_lstm_7day_model()
                scaler = load_lstm_scaler()

                weather_features = ["meantemp", "humidity", "wind_speed", "meanpressure"]
                window_size = 7

                if len(df) < window_size:
                    st.error("LSTM needs at least 7 days of weather data.")
                    return

                latest_7days = df[weather_features].tail(window_size)
                latest_7days_scaled = scaler.transform(latest_7days)

                lstm_input = latest_7days_scaled.reshape(1, window_size, len(weather_features))

                prediction_scaled = lstm_7day_model.predict(lstm_input)
                prediction = scaler.inverse_transform(prediction_scaled)[0]

            except FileNotFoundError:
                st.error("7-day LSTM model not found. Train and save lstm_weather_7day_model.keras first.")
                return

        pred_temp = prediction[0]
        pred_humidity = prediction[1]
        pred_wind = prediction[2]
        pred_pressure = prediction[3]

        result_df = pd.DataFrame({
            "Forecast Horizon": ["7 Days Ahead"],
            "Predicted Temperature (°C)": [round(pred_temp, 2)],
            "Predicted Humidity (%)": [round(pred_humidity, 2)],
            "Predicted Wind Speed": [round(pred_wind, 2)],
            "Predicted Pressure": [round(pred_pressure, 2)]
        })

        st.success("7-Day Weather Prediction Completed")
        st.dataframe(result_df)

        st.subheader("Agricultural Recommendations")
        for i, advice in enumerate(
            agricultural_advice(pred_temp, pred_humidity, pred_wind, pred_pressure), 1
        ):
            st.write(f"{i}. {advice}")

        st.markdown(filedownload(result_df), unsafe_allow_html=True)


# ============================================================
# APP ROUTING
# ============================================================

if selection == "Single Prediction":
    single_prediction()

if selection == "Multi Prediction":
    st.header("Upload Weather CSV File")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        multi_prediction(uploaded_file)
    else:
        st.info("Upload a CSV file containing weather records.")
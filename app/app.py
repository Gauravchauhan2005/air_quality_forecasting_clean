import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow 
from tensorflow import keras #
# import keras # import load_model
from keras.models  import load_model
import matplotlib.pyplot as plt

# Load model and scaler
model = load_model('lstm_cnn_model.keras')
scaler = joblib.load('scaler.pkl')

st.title("🌫️ Air Quality Forecasting (PM2.5) - DL Model")

st.markdown("""
This app predicts future **PM2.5** levels using an LSTM + CNN model.  
Upload a dataset or use built-in demo data to forecast air quality.
""")

# Option to upload user data
uploaded_file = st.file_uploader("Upload your air quality CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Datetime'])
else:
    df = pd.read_csv('../data/station_hourly_data.csv', parse_dates=['Datetime'])
    df = df[df['StationId'] == 'DL001']

# Preprocess
df = df[['Datetime', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']].dropna()
df.columns = ['Datetime', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
df['hour'] = df['Datetime'].dt.hour
df['day'] = df['Datetime'].dt.day
df['month'] = df['Datetime'].dt.month
df['weekday'] = df['Datetime'].dt.weekday
df.set_index('Datetime', inplace=True)

# Scale
scaled = scaler.transform(df)
X_input = []

# Use last 24 rows for forecast
window = 24
if len(scaled) < window:
    st.error("Not enough data! Need at least 24 rows.")
else:
    X_input.append(scaled[-window:])
    X_input = np.array(X_input)

    # Predict
    prediction = model.predict(X_input)

    # Inverse scale prediction
    dummy = np.zeros((1, scaled.shape[1]))
    dummy[0][0] = prediction
    inv = scaler.inverse_transform(dummy)[0][0]

    st.subheader("📈 Predicted PM2.5 for Next Hour:")
    st.success(f"{inv:.2f} µg/m³")

    # Show last few values for context
    st.subheader("🗂️ Recent Data")
    st.write(df.tail())

    # Plot recent PM2.5
    st.subheader("🔍 PM2.5 Trend")
    fig, ax = plt.subplots()
    df['PM2.5'].tail(100).plot(ax=ax, label='Past PM2.5')
    ax.axhline(y=inv, color='r', linestyle='--', label='Next Prediction')
    ax.legend()
    st.pyplot(fig)

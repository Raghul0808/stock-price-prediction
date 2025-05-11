import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="ARIMA Stock Forecast", layout="wide")

# Title
st.title("📈 Stock Price Forecast using ARIMA")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file (with 'Date' and 'Close' columns):", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("❌ CSV must contain 'Date' and 'Close' columns.")
        st.stop()

    df = df[['Date', 'Close']]
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    st.subheader("📊 Raw Time Series Data")
    st.line_chart(df['close'])

    # Log transform
    ts_log = np.log(df['close'])
    
    # Differencing
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)

    # Test for stationarity
    def test_stationarity(timeseries):
        result = adfuller(timeseries, autolag='AIC')
        output = pd.Series(result[0:4], index=['Test Statistic','p-value','# Lags Used','Number of Observations Used'])
        for key,value in result[4].items():
            output[f'Critical Value ({key})'] = value
        return output

    st.subheader("🧪 Stationarity Test (ADF)")
    adf_result = test_stationarity(ts_log_diff)
    st.write(adf_result)

    # Fit ARIMA model
    st.subheader("⚙️ ARIMA Model")
    p = st.slider("AR term (p)", 0, 5, 1)
    d = st.slider("Difference order (d)", 0, 2, 1)
    q = st.slider("MA term (q)", 0, 5, 1)

    model = ARIMA(ts_log, order=(p,d,q))
    results_ARIMA = model.fit()

    st.success("✅ ARIMA model fitted.")

    # Forecast
    st.subheader("🔮 Forecast Plot")
    forecast_steps = st.slider("Number of forecast days", 1, 30, 14)

    forecast = results_ARIMA.forecast(steps=forecast_steps)
    forecast_dates = pd.date_range(df.index[-1], periods=forecast_steps + 1, freq='B')[1:]
    forecast_series = pd.Series(np.exp(forecast), index=forecast_dates)

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['close'], label='Original')
    ax.plot(forecast_series, label='Forecast', color='red')
    ax.set_title("ARIMA Forecast")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📋 Forecasted Values")
    st.write(forecast_series)
else:
    st.info("📁 Please upload a CSV file to begin.")

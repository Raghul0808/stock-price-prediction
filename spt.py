# streamlit_arima_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Streamlit UI
st.title("📈 Stock Price Prediction using ARIMA")
st.markdown("Upload a CSV file with `Date` and `Close` columns (e.g., NSE stock data).")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("CSV must contain 'Date' and 'Close' columns.")
        st.stop()

    # Clean and prepare
    df = df[['Date', 'Close']].copy()
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
    df.dropna(inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df['close'].fillna(method='pad', inplace=True)

    st.subheader("📊 Original Closing Price")
    st.line_chart(df['close'])

    # Log transform
    ts = df['close']
    ts_log = np.log(ts)

    # Differencing
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)

    # ADF Test
    def adf_test(series, title=''):
        result = adfuller(series, autolag='AIC')
        st.write(f"**ADF Test Results ({title})**")
        st.write(pd.Series(result[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Num Observations']))
        for key, value in result[4].items():
            st.write(f"Critical Value ({key}): {value:.4f}")

    adf_test(ts_log_diff, "Log Differenced Series")

    # Decompose
    st.subheader("🔍 Seasonal Decomposition")
    decomposition = seasonal_decompose(ts_log.dropna(), period=30)
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    axs[0].plot(ts_log, label='Original')
    axs[0].legend(loc='best')
    axs[1].plot(decomposition.trend, label='Trend')
    axs[1].legend(loc='best')
    axs[2].plot(decomposition.seasonal, label='Seasonality')
    axs[2].legend(loc='best')
    axs[3].plot(decomposition.resid, label='Residuals')
    axs[3].legend(loc='best')
    plt.tight_layout()
    st.pyplot(fig)

    # Fit ARIMA
    st.subheader("⚙️ ARIMA Model Fitting")
    model = ARIMA(ts_log, order=(1, 1, 1))
    results_ARIMA = model.fit()

    # Forecast in log scale and invert
    predictions_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    predictions_cumsum = predictions_diff.cumsum()
    predictions_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
    predictions_log = predictions_log.add(predictions_cumsum, fill_value=0)
    predictions_ARIMA = np.exp(predictions_log)

    st.subheader("📈 ARIMA Fit vs Actual")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(ts, label='Actual')
    ax2.plot(predictions_ARIMA, label='Fitted', color='red')
    ax2.legend(loc='best')
    st.pyplot(fig2)

    # Forecast
    st.subheader("🔮 Future Forecast (Next 14 Days)")
    forecast_steps = st.slider("Select number of days to forecast", 1, 30, 14)
    forecast_log = results_ARIMA.forecast(steps=forecast_steps)
    forecast = np.exp(forecast_log)

    st.write(forecast)

    st.line_chart(forecast)

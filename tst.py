#!/usr/bin/env python
# arima_forecast.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# ----------------------------
# Load the data
# ----------------------------
file_path = input("Enter the path to your CSV file (e.g. NIFTY_MEDIA.csv): ")

try:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df1 = df[['Date', 'Close']]
    df1.columns = ['date', 'close']
    df1['date'] = pd.to_datetime(df1['date'])
    df_ts = df1.set_index('date')
    df_ts.sort_index(inplace=True)
    df_ts['close'].fillna(method='pad', inplace=True)
except Exception as e:
    print(f"❌ Failed to load or process file: {e}")
    exit()

ts = df_ts['close']

# ----------------------------
# Stationarity Test Function
# ----------------------------
def test_stationarity(timeseries, title='Time Series'):
    print(f"\n📊 Dickey-Fuller Test: {title}")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', '# Observations'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)

# ----------------------------
# Preprocessing
# ----------------------------
print("\n✅ Plotting original time series...")
df_ts.plot(title="Original Time Series")
plt.grid()
plt.show()

test_stationarity(ts, "Original Series")

# Log transform
ts_log = np.log(ts)
plt.plot(ts_log)
plt.title("Log Transformed Series")
plt.grid()
plt.show()

# Differencing to make stationary
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff, "Log Differenced Series")

# ----------------------------
# ACF and PACF
# ----------------------------
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.stem(lag_acf, use_line_collection=True)
plt.title('Autocorrelation Function')
plt.grid()

plt.subplot(122)
plt.stem(lag_pacf, use_line_collection=True)
plt.title('Partial Autocorrelation Function')
plt.grid()
plt.tight_layout()
plt.show()

# ----------------------------
# Fit ARIMA Model
# ----------------------------
print("\n📈 Fitting ARIMA model (1,1,1)...")
model = ARIMA(ts_log, order=(1, 1, 1))
results_ARIMA = model.fit()

# Plot ARIMA fit
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('ARIMA Fit vs Actual')
plt.grid()
plt.show()

# ----------------------------
# Reverse transformation to original scale
# ----------------------------
predictions_diff = results_ARIMA.fittedvalues
predictions_cumsum = predictions_diff.cumsum()
predictions_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_log = predictions_log.add(predictions_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_log)

# Plot predictions vs original
plt.plot(ts, label='Original')
plt.plot(predictions_ARIMA, label='ARIMA Forecast', color='red')
plt.legend()
plt.title("Actual vs Forecast")
plt.grid()
plt.show()

# ----------------------------
# Forecasting next 14 days
# ----------------------------
print("\n📅 Forecasting next 14 days...")
forecast_log = results_ARIMA.forecast(steps=14)
forecast = np.exp(forecast_log)

print("\n🔮 14-Day Forecast:")
print(forecast)

# Optional: Save forecast to CSV
forecast.to_csv("14_day_forecast.csv")
print("\n✅ Forecast saved to '14_day_forecast.csv'")

# Install required packages
!pip install statsmodels

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Upload CSV file
from google.colab import files
uploaded = files.upload()

# Read dataset
df = pd.read_csv("NIFTY MEDIA DATA SET.csv")
df.columns = df.columns.str.strip()
df1 = df[['Date', 'Close']]
df1.columns = ['date', 'close']

# Set date as index
df1['date'] = pd.to_datetime(df1['date'])
df_ts = df1.set_index('date')
df_ts = df_ts.sort_index()
df_ts['close'].fillna(method='pad', inplace=True)

# Plot the time series
df_ts.plot(title="Original Time Series")
plt.show()

# Dickey-Fuller Test
def test_stationarity(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', '# Observations'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)

ts = df_ts['close']
test_stationarity(ts)

# Log transform
ts_log = np.log(ts)
plt.plot(ts_log, label='Log Transformed')
plt.legend()
plt.show()

# Moving Average Smoothing
moving_avg = ts_log.rolling(window=12).mean()
plt.plot(ts_log, label='Log Transformed')
plt.plot(moving_avg, color='red', label='Moving Average')
plt.legend()
plt.show()

# Log scale minus MA
ts_log_minus_ma = ts_log - moving_avg
ts_log_minus_ma.dropna(inplace=True)
test_stationarity(ts_log_minus_ma)

# Exponential Smoothing
exp_decay_avg = ts_log.ewm(halflife=12, adjust=True).mean()
plt.plot(ts_log, label='Log Transformed')
plt.plot(exp_decay_avg, color='red', label='EWMA')
plt.legend()
plt.show()

ts_log_minus_ewma = ts_log - exp_decay_avg
ts_log_minus_ewma.dropna(inplace=True)
test_stationarity(ts_log_minus_ewma)

# Differencing
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

# Decomposition
decomposition = seasonal_decompose(ts_log, period=30)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(10,8))
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend()
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend()
plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend()
plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend()
plt.tight_layout()
plt.show()

residual.dropna(inplace=True)
test_stationarity(residual)

# ACF and PACF plots
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.stem(lag_acf, use_line_collection=True)
plt.axhline(y=0, linestyle='--', color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.stem(lag_pacf, use_line_collection=True)
plt.axhline(y=0, linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

# Fit ARIMA model
model = ARIMA(ts_log, order=(1,1,1))
results_ARIMA = model.fit()
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('ARIMA Fit vs Actual')
plt.show()

# Convert predictions back to original scale
predictions_ARIMA_diff = results_ARIMA.fittedvalues
predictions_ARIMA_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)

# Plot the predictions
plt.plot(ts, label='Original')
plt.plot(predictions_ARIMA, label='ARIMA Forecast', color='red')
plt.legend()
plt.show()

# Forecasting future values (e.g., next 14 days)
forecast = results_ARIMA.forecast(steps=14)
forecast_values = np.exp(forecast)
print("Next 14-day forecast:")
print(forecast_values)

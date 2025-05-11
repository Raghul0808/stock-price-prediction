
#!/usr/bin/env python
# coding: utf-8

# In[135]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[136]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


# In[137]:


from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')


# In[138]:


df = pd.read_csv('nse.csv')
df.columns = df.columns.str.strip()

# Now you can access columns safely
df1 = df[['Date', 'Close']]
df1.columns = ['date', 'close']  # Rename for consistency

# Show the first 3 rows
print(df1.head(100))
df.head(3)


# In[139]:


print (df.describe())
print ("=============================================================")
print (df.dtypes)


# In[140]:


df1 = df[['Date','Close']]
df1.head(3)


# In[141]:


# Setting the Date as Index
df_ts = df1.set_index('Date')
df_ts.sort_index(inplace=True)
print (type(df_ts))
print (df_ts.head(3))
print ("========================")
print (df_ts.tail(3))


# In[142]:


df_ts[df_ts.isnull()]


# In[143]:


len(df_ts[df_ts.isnull()])


# In[144]:


df_ts = df_ts.sort_index()
df_ts.index


# In[145]:


df_ts.Close.fillna(method='pad', inplace=True)


# In[146]:


df_ts[df_ts.Close.isnull()]
len(df_ts[df_ts.Close.isnull()])


# In[147]:


df_ts.plot()


# In[148]:


# Dickey Fuller Test Function
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    from statsmodels.tsa.stattools import adfuller
    print('Results of Dickey-Fuller Test:')
    print ("==============================================")

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', 'Number of Observations Used'])

    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)


# In[149]:


# Stationarity Check - Lets do a quick check on Stationarity with Dickey Fuller Test
# Convert the DF to series first
ts = df_ts['Close']
test_stationarity(ts)


# In[152]:

rolstd = ts.rolling(window=365).std()
rolmean = ts.rolling(window=365).mean()
rolvar = ts.rolling(window=365).std()

#plt.figure(figsize=(10, 6))
#plt.plot(ts, label='Original', color='blue')
#plt.plot(rolmean, label='Rolling Mean', color='red')
#plt.plot(rolstd, label='Rolling Std Dev', color='black')
#plt.legend(loc='best')
#plt.title('Rolling Mean & Standard Deviation')
#plt.show(block=False)



# In[75]:


ts.dropna(inplace=True)
ts.head(5)
from statsmodels.tsa.stattools import adfuller


# In[77]:


print('results of dikey-fuller test:')
dftest=adfuller(ts, autolag='AIC')


# In[78]:


dfoutput=pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', '#observations'])
for key,value in dftest[4].items():
    dfoutput['Critical value (%s)'%key]=value

print (dfoutput)


# In[79]:


ts_logScale=np.log(ts)
plt.plot(ts_logScale)


# In[80]:


movingAverage=ts_logScale.rolling(window=12).mean()
movingSTD=ts_logScale.rolling(window=12).std()
plt.plot(ts_logScale)
plt.plot(movingAverage, color='red')


# In[81]:


ts_LogScaleMinusMA=ts_logScale-movingAverage
ts_LogScaleMinusMA.head(12)
ts_LogScaleMinusMA.dropna(inplace=True)
ts_LogScaleMinusMA.head(10)


# In[84]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    movingAverage=timeseries.rolling(window=12).mean()
    movingSTD=timeseries.rolling(window=12).std()
    orig=plt.plot(timeseries, color='blue', label='Original')
    mean=plt.plot(movingAverage, color='red', label='rMean')
    std=plt.plot(movingSTD, color='yellow', label='rStd')
    plt.legend(loc='best')
    plt.title('Rolling Mean & STD')
    plt.show(block=False)
    print('results of dikey-fuller test:')
    dftest=adfuller(timeseries, autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', '#observations'])
    for key,value in dftest[4].items():
        dfoutput['Critical value (%s)'%key]=value

    print (dfoutput)


# In[85]:


test_stationarity(ts_LogScaleMinusMA)


# In[86]:


exponentialDecayWeightedAverage=ts_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(ts_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')


# In[88]:


ts_LogScaleMinusExponentialDecayAverage= ts-exponentialDecayWeightedAverage
test_stationarity(ts_LogScaleMinusExponentialDecayAverage)


# In[89]:


ts_LogDiffShifting= ts_logScale-ts_logScale.shift()
plt.plot(ts_LogDiffShifting)


# In[90]:


ts_LogDiffShifting.dropna(inplace=True)
test_stationarity(ts_LogDiffShifting)


# In[92]:


ts_logScale.head()


# In[95]:


from statsmodels.tsa.seasonal import seasonal_decompose
ts_logScale.dropna(inplace=True)
decomposition = seasonal_decompose(ts_logScale, period=30)
trend =decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid


# In[96]:


plt.subplot(411)
plt.plot(ts_logScale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(ts_logScale, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(ts_logScale, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(ts_logScale, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
decomposedLogData=residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# In[97]:


decomposedLogData=residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# In[100]:


from statsmodels.tsa.stattools import acf, pacf

lag_acf=acf(ts_LogDiffShifting, nlags=20)
lag_pacf=pacf(ts_LogDiffShifting, nlags=20, method='ols')


# In[101]:


plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_LogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_LogDiffShifting)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_LogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_LogDiffShifting)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# In[114]:


from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(ts_logScale, order=(1,1,1))
results_AR = model.fit ()
plt.plot(ts_LogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_LogDiffShifting)**2))
print('Plotting AR Model')


# In[115]:


model=ARIMA(ts_logScale, order=(1,1,1))
results_ARIMA = model.fit ()
plt.plot(ts_LogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_LogDiffShifting)**2))


# In[117]:


predicitons_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predicitons_ARIMA_diff.head())


# In[118]:


predicitons_ARIMA_diff_cumsum=predicitons_ARIMA_diff.cumsum()
print(predicitons_ARIMA_diff_cumsum)


# In[121]:


predictions_ARIMA_log=pd.Series(ts_logScale.iloc[0], index=ts_logScale.index)
predictions_ARIMA_log=predictions_ARIMA_log.add(predicitons_ARIMA_diff_cumsum, fill_value=0)


# In[123]:


predictions_ARIMA_log.head()


# In[124]:


predicitons_ARIMA=np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predicitons_ARIMA)


# In[125]:


ts_logScale


# In[127]:


results_ARIMA.predict(1,9575)


# In[132]:


results_ARIMA.forecast(14)


# In[131]:





# In[ ]:
     
           date    close
0   23-OCT-2024  1940.20
1   24-OCT-2024  1936.95
2   25-OCT-2024  1894.35
3   28-OCT-2024  1931.50
4   29-OCT-2024  1937.60
..          ...      ...
95  10-MAR-2025  1485.15
96  11-MAR-2025  1482.45
97  12-MAR-2025  1459.75
98  13-MAR-2025  1437.80
99  17-MAR-2025  1428.50

[100 rows x 2 columns]
              Open         High          Low        Close  Shares Traded  \
count   123.000000   123.000000   123.000000   123.000000   1.230000e+02   
mean   1721.474797  1740.242683  1699.091463  1717.814228   3.808124e+07   
std     214.181911   211.694700   213.662984   211.572606   1.552282e+07   
min    1362.350000  1402.100000  1344.400000  1371.550000   5.818974e+06   
25%    1515.350000  1541.350000  1488.450000  1512.275000   2.744401e+07   
50%    1695.700000  1711.200000  1657.250000  1687.250000   3.514445e+07   
75%    1931.575000  1942.550000  1907.100000  1929.425000   4.585404e+07   
max    2091.400000  2095.650000  2061.950000  2083.950000   1.106509e+08   

       Turnover (₹ Cr)  
count       123.000000  
mean        387.437886  
std         181.274523  
min          81.940000  
25%         286.375000  
50%         339.990000  
75%         435.075000  
max        1308.980000  
=============================================================
Date                object
Open               float64
High               float64
Low                float64
Close              float64
Shares Traded        int64
Turnover (₹ Cr)    float64
dtype: object
<class 'pandas.core.frame.DataFrame'>
               Close
Date                
01-APR-2025  1508.30
01-FEB-2025  1615.10
01-JAN-2025  1832.65
========================
               Close
Date                
31-DEC-2024  1817.85
31-JAN-2025  1580.25
31-OCT-2024  2019.00
Results of Dickey-Fuller Test:
==============================================
Test Statistic                  -5.474393
p-value                          0.000002
#lags Used                       5.000000
Number of Observations Used    117.000000
Critical Value (1%)             -3.487517
Critical Value (5%)             -2.886578
Critical Value (10%)            -2.580124
dtype: float64
results of dikey-fuller test:
Test Statistic           -5.474393
p-value                   0.000002
#lags Used                5.000000
#observations           117.000000
Critical value (1%)      -3.487517
Critical value (5%)      -2.886578
Critical value (10%)     -2.580124
dtype: float64

results of dikey-fuller test:
Test Statistic         -5.926586e+00
p-value                 2.435585e-07
#lags Used              1.000000e+01
#observations           1.010000e+02
Critical value (1%)    -3.496818e+00
Critical value (5%)    -2.890611e+00
Critical value (10%)   -2.582277e+00
dtype: float64

results of dikey-fuller test:
Test Statistic           -5.474718
p-value                   0.000002
#lags Used                5.000000
#observations           117.000000
Critical value (1%)      -3.487517
Critical value (5%)      -2.886578
Critical value (10%)     -2.580124
dtype: float64

results of dikey-fuller test:
Test Statistic         -7.226737e+00
p-value                 2.043781e-10
#lags Used              1.300000e+01
#observations           1.080000e+02
Critical value (1%)    -3.492401e+00
Critical value (5%)    -2.888697e+00
Critical value (10%)   -2.581255e+00
dtype: float64

results of dikey-fuller test:
Test Statistic         -6.583864e+00
p-value                 7.398184e-09
#lags Used              1.000000e+01
#observations           8.200000e+01
Critical value (1%)    -3.512738e+00
Critical value (5%)    -2.897490e+00
Critical value (10%)   -2.585949e+00
dtype: float64

results of dikey-fuller test:
Test Statistic         -6.583864e+00
p-value                 7.398184e-09
#lags Used              1.000000e+01
#observations           8.200000e+01
Critical value (1%)    -3.512738e+00
Critical value (5%)    -2.897490e+00
Critical value (10%)   -2.585949e+00
dtype: float64
Plotting AR Model
Date
01-APR-2025    0.000000
01-FEB-2025    7.318738
01-JAN-2025    7.345196
01-NOV-2024    7.380554
02-APR-2025    7.422111
dtype: float64
Date
01-APR-2025      0.000000
01-FEB-2025      7.318738
01-JAN-2025     14.663934
01-NOV-2024     22.044488
02-APR-2025     29.466599
                  ...    
30-JAN-2025    878.516388
30-OCT-2024    885.974124
31-DEC-2024    893.379539
31-JAN-2025    900.805466
31-OCT-2024    908.262628
Length: 123, dtype: float64
predicted_mean
123	7.402902
124	7.449902
125	7.439254
126	7.441666
127	7.441120
128	7.441243
129	7.441215
130	7.441222
131	7.441220
132	7.441221
133	7.441220
134	7.441220
135	7.441220
136	7.441220

dtype: float64


# Instead of using results_ARIMA.plot_predict, use results_ARIMA.predict to get the predictions
# and then plot them manually using matplotlib.pyplot.

# Get predictions for the desired range
predictions = results_ARIMA.predict(start=1, end=9575)

# Plot the original time series and the predictions
plt.plot(ts, label='Original')
plt.plot(predictions, label='Predictions', color='red')
plt.legend(loc='best')
plt.title('ARIMA Model Predictions')
plt.show()
     


# First, remove any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Now you can access columns safely
df1 = df[['Date', 'Close']]
df1.columns = ['date', 'close']  # Rename for consistency

# Show the first 3 rows
print(df1.head(100))

     
           date    close
0   23-OCT-2024  1940.20
1   24-OCT-2024  1936.95
2   25-OCT-2024  1894.35
3   28-OCT-2024  1931.50
4   29-OCT-2024  1937.60
..          ...      ...
95  10-MAR-2025  1485.15
96  11-MAR-2025  1482.45
97  12-MAR-2025  1459.75
98  13-MAR-2025  1437.80
99  17-MAR-2025  1428.50

[100 rows x 2 columns]

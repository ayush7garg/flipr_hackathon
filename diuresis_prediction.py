# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:02:12 2020

@author: Ayush Garg
"""

#Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller,acf,pacf
from statsmodels.tsa.arima_model import ARIMA

#Fuction for testing the stationarity of the timeseries
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(1).mean()
    rolstd = timeseries.rolling(1).std()#Plot rolling statistics:
    plt.plot(timeseries, color='pink',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

#Loading dataset
xlsx = pd.ExcelFile('Train_dataset.xlsx')
df = pd.read_excel(xlsx, 'Diuresis_TS')
df = df.set_index("people_ID")

df1 = df.T

#Applying log transformation
for i in range(len(df1.columns)):
    y = df1.iloc[:,i]
    y = y.to_frame()
    y.index = df1.index
    y_log = np.log(y)
    #y_log = y_log.to_frame()
    y_log.index = df1.index
    df1.iloc[:,i] = y_log

moving_avg = y_log.rolling(2).mean()
plt.plot(y_log)
plt.plot(moving_avg,color='red')

#Finding the values of p and q to be used in ARIMA model
lag_acf = acf(y_log,nlags=6)
lag_pacf = pacf(y_log,method='ols',nlags=6)

#Plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(y_log)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(y_log)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(y_log)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(y_log)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

p=1
q=1
model = ARIMA(y_log,order=(1,0,0))
results_ARIMA = model.fit(disp=-1)
plt.plot(y_log)
plt.plot(results_ARIMA.fittedvalues,color='red')
plt.title('RSS=%.4f',sum((results_ARIMA.fittedvalues-y_log)**2))

df1.iloc[:,0].plot(figsize=(15, 6))
plt.show()
y = df1.iloc[:,3]
y = y.to_frame()
y.index = df1.index

decomposition = seasonal_decompose(y_log,freq=1)
fig = decomposition.plot()
plt.show()

test_stationarity(y)
y_log = np.log(df1.iloc[:,3])
plt.plot(y_log)
test_stationarity(y_log)

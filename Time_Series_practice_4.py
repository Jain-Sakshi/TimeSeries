# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:46:27 2018

@author: sakshij
"""

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

#Reading dataset
#stock_data = pd.read_csv('D:\\Datasets\\Yahoo_Stocks\\22_Aug_17_to_18.csv')
passenger_data = pd.read_csv('D:\\Datasets\\AirPassengers\\AirPassengers.csv')

#Autoregressive models : AR(1) and AR(2)
def ar1(phi = 0.9, n = 1000, init = 0):
    time_series = [init]
    error = np.random.randn(n)
#    pd.Series(error).plot()
    for period in range(n):
        time_series.append(error[period] + phi * time_series[-1])
    return pd.Series(time_series[1:], index=range(n))

def ar2(phi1 = 0.9, phi2 = 0.8, n = 1000, init = 0):
    time_series = [init, init]
    error = np.random.randn(n)
    for period in range(n):
        time_series.append(error[period] + phi1 * time_series[-1] + phi2 * time_series[-2])
    return pd.Series(time_series[2:], index=range(n))

def ma(theta = 0.5, n = 100):
    time_series = []
    error = np.random.randn(n)
    for period in range(1,n):
        time_series.append(error[period] + theta*error[period-1])
    return pd.Series(time_series, index=range(1,n))

a1 = ar1(phi=1.1, n=40)
a1.plot()
plt.show()

m1 = ma(theta=-1000)
m1.plot()

#Autocorrelation and Partial Autocorrelation
a1 = ar1(phi = 0.5, n = 1000 )
a1_acf = acf(a1,nlags=20)
plt.plot(a1_acf)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(a1)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(a1)),linestyle='--')

a1 = ar1(phi = 0.5, n = 1000 )
a1_pacf = pacf(a1,nlags=20)
plt.plot(a1_pacf)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(a1)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(a1)),linestyle='--')

a2 = ar2(phi1 = 0.5, phi2 = 0.5, n = 1000 )
a2_acf = acf(a2,nlags=20)
plt.plot(a2_acf)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(a2)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(a2)),linestyle='--')

a2 = ar2(phi1 = 0.5, phi2 = 0.5, n = 1000 )
a2_pacf = pacf(a2,nlags=20)
plt.plot(a2_pacf)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(a2)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(a2)),linestyle='--')


m1 = ma(theta = 0.9, n = 1000 )
m1_acf = acf(m1,nlags=20)
plt.plot(m1_acf)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(m1)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(m1)),linestyle='--')

m1 = ma(theta = 0.9, n = 1000 )
m1_pacf = pacf(m1,nlags=20)
plt.plot(m1_pacf)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(m1)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(m1)),linestyle='--')

##Air Passenger Dataset
#Reading dataset
passenger_data = pd.read_csv('D:\\Datasets\\AirPassengers\\AirPassengers.csv')

#Air Passenger data detrending
plt.plot(passenger_data['#Passengers'])
passenger_log = np.log(passenger_data['#Passengers'])
passenger_log_diff = passenger_log - passenger_log.shift()
passenger_log_diff.dropna(inplace=True)
plt.plot(passenger_log_diff)

#Dickey Fuller test for stationarity
useful_values_raw = adfuller(passenger_log_diff,autolag='AIC')
useful_values = [v for v in useful_values_raw[:4]]

dickey_fuller_output = {
        'test statistic' : useful_values_raw[0],
        'p-value' : useful_values_raw[1],
        'no. of lags' : useful_values_raw[2],
        'no of obervations' : useful_values_raw[3],
        'Critical value (1%)' : useful_values_raw[4]['1%'],
        'Critical value (5%)' : useful_values_raw[4]['5%'],
        'Critical value (10%)' : useful_values_raw[4]['10%']
        }

#ACF and PACF
lag_acf = acf(passenger_log_diff.values, nlags=20)
lag_pacf = pacf(passenger_log_diff.values, nlags=20)

plt.subplot(121)
plt.bar(left = range(len(lag_acf)), height = lag_acf)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(lag_acf)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(lag_acf)),linestyle='--')

plt.subplot(121)
plt.bar(left = range(len(lag_pacf)), height = lag_pacf)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(lag_pacf)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(lag_pacf)),linestyle='--')

#ARIMA model
#only AR
model = ARIMA(passenger_log, order=(2,1,0))
results_AR = model.fit(disp=1)
plt.plot(passenger_log_diff, color = 'blue')
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS : %.4f'% sum((results_AR.fittedvalues - passenger_log_diff)**2))

#only MA
model = ARIMA(passenger_log, order=(0,1,3))
results_AR = model.fit(disp=1)
plt.plot(passenger_log_diff, color = 'blue')
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS : %.4f'% sum((results_AR.fittedvalues - passenger_log_diff)**2))

#AR+MA - (Aileen Nielsen) error : 1.4597
model = ARIMA(passenger_log, order=(1,1,1))
results_AR = model.fit(disp=1)
plt.plot(passenger_log_diff, color = 'blue')
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS : %.4f'% sum((results_AR.fittedvalues - passenger_log_diff)**2))

#AR+MA - lowest error(yet) : 0.6168
model = ARIMA(passenger_log, order=(7,1,4))
results_AR = model.fit(disp=1)
plt.plot(passenger_log_diff, color = 'blue')
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS : %.4f'% sum((results_AR.fittedvalues - passenger_log_diff)**2))

#Retrending data
predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_log = pd.Series(passenger_log.iloc[0], index = passenger_log.index)
predictions_ARIMA_log = pd.Series(passenger_log.ix[0], index = passenger_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value = 0)
    
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(passenger_data['#Passengers'])
plt.plot(predictions_ARIMA)

##########################################################################
#Autocorrelation - noise
grid = np.linspace(0,720,500)
noise = np.random.rand(500)
result_curve = noise
plt.plot(grid,result_curve)
plt.show()

acf_result = stattools.acf(result_curve)
plt.plot(acf_result)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(result_curve)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(result_curve)),linestyle='--')

#Autocorrelation - sin curve
grid = np.linspace(0,100,100)
sin5 = np.sin(grid)
result_curve = sin5
plt.plot(grid,result_curve)

acf_result = stattools.acf(result_curve,nlags=10)
plt.plot(acf_result)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(result_curve)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(result_curve)),linestyle='--')

#Autocorrelation - Yahoo Stock Data
stock_data.head()
plt.plot(stock_data.Open)

acf_result = stattools.acf(stock_data.Open,nlags=1000)
plt.plot(acf_result)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(stock_data.Open)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(stock_data.Open)),linestyle='--')

#Autocorrelation - Air Passenger Data
passenger_data.head()
plt.plot(passenger_data['#Passengers'])

acf_result = stattools.acf(passenger_data['#Passengers'],nlags=70)
plt.plot(acf_result)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(passenger_data['#Passengers'])),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(passenger_data['#Passengers'])),linestyle='--')
plt.show()

passenger_log = np.log(passenger_data['#Passengers'])
passenger_log_diff = passenger_log - passenger_log.shift()
plt.plot(passenger_log_diff)

passenger_log_diff = passenger_log_diff.dropna()
acf_result = stattools.acf(passenger_log_diff,nlags=70)
plt.plot(acf_result)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(passenger_log_diff)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(passenger_log_diff)),linestyle='--')
plt.show()

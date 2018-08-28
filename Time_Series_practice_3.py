# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:46:27 2018

@author: sakshij
"""

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
from statsmodels.tsa import stattools

#Reading dataset
stock_data = pd.read_csv('D:\\Time_Series\\Datasets\\Yahoo_Stocks\\22_Aug_17_to_18.csv')
passenger_data = pd.read_csv('D:\\Time_Series\\Datasets\\AirPassengers\\AirPassengers.csv')

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

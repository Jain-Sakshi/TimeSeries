# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:46:27 2018

@author: sakshij
"""

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import random


#Creating Dataset - small time series
date_range = pd.date_range('1/1/2017',periods=10,freq='H')
time_series_df = pd.DataFrame(list(range(10)), index=date_range)

#Creating Dataset - larger (6H gap) time series
regular_date_range = pd.date_range('1/1/2017',periods=1000,freq='6H')
day_no_list_ordered = []
for a in range(250):
    day_no_list_ordered.append(a)
    day_no_list_ordered.append(a)
    day_no_list_ordered.append(a)
    day_no_list_ordered.append(a)

day_no_list_random = []
for a in range(1000):
    day_no_list_random.append(random.randint(0,1000))
    
regular_ts_df = pd.DataFrame(list(range(1000)), index=regular_date_range, columns=['A'])
regular_ts_df_ordered = pd.DataFrame(day_no_list_ordered, index=regular_date_range, columns=['A'])
regular_ts_df_random_values = pd.DataFrame(day_no_list_random, index=regular_date_range, columns=['A'])

#Yahoo Stock Data
stock_data = pd.read_csv('D:\\Time_Series\\Datasets\\Yahoo_Stocks\\22_Aug_17_to_18.csv')

#Resampling (using df.asfreq())
converted_45min_ffill = time_series_df.asfreq(freq='45min', method='ffill')
converted_45min_bfill = time_series_df.asfreq(freq='45min', method='bfill')
converted_45min_None = time_series_df.asfreq(freq='45min')
converted = time_series_df.asfreq(freq='3H')
asfreq_days = regular_ts_df.asfreq('D')
time_series_df[1:7]

#Resampling with aggregation functions (using df.resample())
resampled_ts_2H = time_series_df.resample('2H',label='right').mean()
resampled_ts_30min = time_series_df.resample('30min',label='right').max()

resampled_regular_12hrs = regular_ts_df.resample('12H',label='right').mean()
resampled_regular_2hrs = regular_ts_df.resample('2H',label='right').mean()

#Lag using .shift()
ts_lagged = time_series_df.shift()          #(default) periods = 1
regular_ts_random_lagged = regular_ts_df_random_values.shift()
diff_random = regular_ts_df_random_values - regular_ts_random_lagged
plt.plot(diff_random)

#Lag using .diff()
plt.plot(regular_ts_df)
plt.show()
plt.plot(regular_ts_df.diff(periods=1))
plt.show()

#plt.plot(stock_data['Open'])
#plt.plot(stock_data['Open'].diff(periods=1))

plt.plot(regular_ts_df_random_values)
plt.show()
plt.plot(regular_ts_df_random_values.diff(periods=1))
plt.show()

#######################################################################
##Extra
#rolling window
r = regular_ts_df.rolling(10)
r.quantile(quantile=.1)['A'].plot().show()

r.apply(lambda x : np.percentile(x,10)).plot().show()

#expanding window
e = regular_ts_df.expanding(min_periods=1)

e.agg(['mean','min','max']).plot()

#moving average
ma = regular_ts_df_random_values.expanding(min_periods=10)
ma.mean().plot()

#exponential weighed moving average
ewma = regular_ts_df_random_values.ewm(min_periods = 10, alpha = 0.2).mean()
ewma.plot()

#Rolling window quantile
time_series_df.rolling(4).quantile(.2)
random_values_quantile_point2 = regular_ts_df_random_values.rolling(4).quantile(.2) #not? coz data should be sorted for quantile
random_values_quantile_point2.plot()

#Rolling window using lambda
r = regular_ts_df.rolling(20).apply(lambda x : x[1])[20:30]
r.sum()['A'].plot()

r['A'].plot()

regular_ts_df.expanding(min_periods = 4).mean()[4:10].plot()

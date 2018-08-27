# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:46:27 2018

@author: sakshij
"""

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import random
print(random.randint(0,1000))



date_range = pd.date_range('1/1/2017',periods=10,freq='H')
time_series_df = pd.DataFrame(list(range(10)), index=date_range)

irregular_date_range = pd.date_range('1/1/2017',periods=1000,freq='6H')
day_no_list = []
for a in range(250):
    day_no_list.append(a)
    day_no_list.append(a)
    day_no_list.append(a)
    day_no_list.append(a)

day_no_list = []
for a in range(1000):
    day_no_list.append(random.randint(0,1000))
    
irregular_ts_df = pd.DataFrame(list(range(1000,0,-1)), index=irregular_date_range, columns=['A'])

converted = time_series_df.asfreq(freq='45min', method='ffill')
converted = time_series_df.asfreq(freq='45min', method='bfill')
converted = time_series_df.asfreq(freq='45min')
converted = time_series_df.asfreq(freq='3H')
time_series_df[1:7]

resampled_ts = time_series_df.resample('2H',label='right').mean()
resampled_ts = time_series_df.resample('30min',label='right').max()
resampled_irregular_12hrs = irregular_ts_df.resample('12H',label='right').mean()

asfreq_days = irregular_ts_df.asfreq('D')
asfreq_hrs = irregular_ts_df.asfreq('3H')

ts_lagged = time_series_df.shift()
irregular_ts_lagged = irregular_ts_df.shift()

plt.plot(irregular_ts_df.diff(periods=2))
plt.show()

r = irregular_ts_df.rolling(20).apply(lambda x : x[1])[20:30]
r.sum()['A'].plot()

r['A'].plot()

irregular_ts_df.expanding(min_periods = 4).mean()[4:10].plot()

#exponential weighed moving average
irregular_ts_df.ewm(min_periods = 10, alpha = 0.2).mean()

#rolling quantile
r = irregular_ts_df.rolling(10)
r.quantile(quantile=.1)['A'].plot().show()

r.apply(lambda x : np.percentile(x,10)).plot().show()

e = irregular_ts_df.expanding(min_periods=1)
e.agg(['mean','min','max']).plot()

#test for quantile
test_array = pd.Series([1, 5, 7, 2, 4, 6, 9, 3, 8, 10])
test_array.rolling(4).quantile(.2)

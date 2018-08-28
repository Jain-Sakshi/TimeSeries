# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:46:27 2018

@author: sakshij
"""

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import random

from statsmodels.nonparametric.smoothers_lowess import lowess

#Preparing dataset
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

####Rolling and Expanding window
##Rolling Window
r = regular_ts_df_random_values.rolling(10)

##Rolling quantile
#with the Quantile function
r.quantile(quantile=.1)['A'].plot().show()
#with lambda
r.apply(lambda x : np.percentile(x,10)).plot().show()

##Expanding Window
e = regular_ts_df_random_values.expanding(min_periods=1)

##Applying multiple aggregation functions tegoether
r.agg(['mean','min','max']).plot()
e.agg(['mean','min','max']).plot()

##Moving Average
ma = regular_ts_df_random_values.expanding(min_periods=10)
ma.mean().plot()

##Exponential Weighed Moving Average
regular_ts_df.ewm(min_periods = 10, alpha = 0.2).mean()

#LOWESS
Y = regular_ts_df_random_values.index #[0:500]
x = regular_ts_df_random_values['A'] #[0:500]
passenger_data_lowess = lowess(endog=regular_ts_df_random_values.index,exog=regular_ts_df_random_values['A'],frac=0.6,is_sorted=False)
plt.plot(passenger_data_lowess)
plt.plot(x)

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


#Preparing dataset
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
    
irregular_ts_df = pd.DataFrame(day_no_list, index=irregular_date_range, columns=['A'])

####Rolling and Expanding window
##Rolling Window
r = irregular_ts_df.rolling(10)

##Rolling quantile
#with the Quantile function
r.quantile(quantile=.1)['A'].plot().show()
#with lambda
r.apply(lambda x : np.percentile(x,10)).plot().show()

##Expanding Window
e = irregular_ts_df.expanding(min_periods=1)

##Applying multiple aggregation functions tegoether
r.agg(['mean','min','max']).plot()
e.agg(['mean','min','max']).plot()

##exponential weighed moving average
irregular_ts_df.ewm(min_periods = 10, alpha = 0.2).mean()



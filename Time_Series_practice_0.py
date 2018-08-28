# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:46:27 2018

@author: sakshij
"""

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import random

#Time Stamp
time_stamp = pd.Timestamp('2016')
time_stamp = pd.Timestamp('2016-07')
time_stamp = pd.Timestamp('2016-07-10')
time_stamp = pd.Timestamp('2016-07-10 23')
time_stamp = pd.Timestamp('2016-07-10 23:30')

#Time Period
period = pd.Period('2018')
period = pd.Period('2018-01')
period = pd.Period('2018-01-01')
period = pd.Period('2018-01-01 10')
period = pd.Period('2018-01-01 10:90')  #out of range time

#Adding time after initialization
delta_1day = pd.Timedelta('1 day')
delta_1day_1hour = pd.Timedelta('1 day 1 hour')

add_period_1day = period + delta_1day
add_period_1day_1hour = period + delta_1day_1hour

add_timestamp_1day = time_stamp + delta_1day
add_timestamp_1day_1hour = time_stamp + delta_1day_1hour

#Creating a date range and changing timezones
date_range_1 = pd.date_range('1/1/2017',periods=48,freq='H')
date_range_2 = pd.date_range('1/1/2017 00:00','1/3/2017 23:00',freq='H')
date_range_1.tzinfo
date_range_1_Asia = date_range_1.tz_localize('Asia/Kolkata')
date_range_1_HongKong = date_range_1_Asia.tz_convert('Asia/Hong_Kong')

#Relocalization
date_range_1_HongKong = date_range_1.tz_localize('Asia/Hong_Kong')

#A date range with predefined timezone
date_range_3 = pd.date_range('1/1/2017',periods=48,freq='H', tz='Asia/Kolkata')
date_range_3.tzinfo

#Date to Index : Accessing values by date and date range
date_range_series = pd.Series(list(range(len(date_range_1))), index = date_range_1)
date_range_series['1/1/2017']
date_range_series['1/1/2017 3:00']
date_range_series['1/1/2017 1:00':'1/1/2017 10:00']
date_range_series.index
date_range_period = date_range_series.to_period()
date_range_period['1/1/2017 1:00':'1/1/2017 10:00']
date_range_period['1/1/2017 1:00':'1/2/2018 10:00'] #out of range date
date_range_period.index

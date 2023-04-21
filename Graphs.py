#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 18:41:18 2023

@author: darraghkaneotoole
"""
##Set your working directory and take a look themonthly mean s mst useful graphically
os.chdir('/Users/darraghkaneotoole/Desktop/ML-group-project')
df = pd.read_csv('BDP_Bikes_Final.csv')
import pandas as pd
import matplotlib.pyplot as plt


##could be useful
#daily_mean = df.groupby(df['TIME'].dt.date)['bikes'].mean()

# Convert 'time' column to datetime
bicycles['TIME'] = pd.to_datetime(bicycles['TIME'], format='%d/%m/%Y %H:%M')

# Group by week and get mean of 'bikes' column
weekly_mean = bicycles.groupby(pd.Grouper(key='TIME', freq='W')).mean()
##Monthly mean
monthly_mean = bicycles.groupby(pd.Grouper(key='TIME', freq='m')).mean()






##sick graph
# Plot the weekly mean
weekly_mean.plot()
plt.title('Weekly Mean of Bike Usage')
plt.xlabel('Week')
plt.ylabel('Bike Usage')
plt.show()
# Plot the monthly mean
#######Best graph
##overlay the best line from other ones?
monthly_mean.plot()
plt.title('month Mean of Bike Usage')
plt.xlabel('month')
plt.ylabel('Bike Usage')
plt.show()
##as a bar chart
#########Not very good
monthly_mean.plot(kind='bar')
plt.title('Month Mean of Bike Usage')
plt.xlabel('Month')
plt.ylabel('Bike Usage')
plt.show()


##These are not adding up
plt.plot(pandemic, )
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Values')
plt.show()
pandemic.mean


plt.plot(pre_pandemic, )
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Values')
plt.show()
pre_pandemic.mean


plt.plot(pandemic, label='Pandemic')
plt.plot(pre_pandemic, label='Pre-Pandemic')
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.show()

#####better graph
# Create a new DataFrame with only columns A and B
df_new = df.loc[:, ['TIME', 'bikes','temp','rain','wind','covid','workday','bikes +1 year']]
monthly_meana.plot()
plt.title('Month Mean of Bike Usage')
plt.xlabel('Month')
plt.ylabel('Bike Usage')
plt.show()

df_new['TIME'] = pd.to_datetime(df_new['TIME'], format='%d/%m/%Y %H:%M')

##Monthly mean
monthly_meana = df_new.groupby(pd.Grouper(key='TIME', freq='m')).mean()

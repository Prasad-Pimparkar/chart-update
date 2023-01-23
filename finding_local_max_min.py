import numpy as np
from scipy.signal import argrelextrema
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# Stock data
quote=yf.Ticker('RELIANCE.NS')
df=quote.history(start=('2022-10-01'), end=('2023-01-20'), interval='1d')
df.to_csv('trial_data.csv')
df=pd.read_csv('trial_data.csv')
df.Date=pd.to_datetime(df.Date)
df.set_index(df.Date)
high_data = df['High']
low_data = df['Low']

high_data = np.array(high_data)
low_data = np.array(low_data)

# Calculate the derivative using Numpy's diff() function
high_derivative = np.diff(high_data)
low_derivative = np.diff(low_data)

# Find the zero-crossing points
zero_crossing_indices = np.where(np.diff(np.sign(high_derivative)))[0]
zero_crossing_indices = np.where(np.diff(np.sign(low_derivative)))[0]

# Identify the local maxima and minima
local_max_indices = argrelextrema(high_data, np.greater)
local_min_indices = argrelextrema(low_data, np.less)

local_max_dates = df['Date'][local_max_indices[0]]
local_min_dates = df['Date'][local_min_indices[0]]

local_max=high_data[local_max_indices[0]]
local_min=low_data[local_min_indices[0]]

print("Local maxima position:", local_max_indices)
print("Local maxima:", local_max)
print("Local minima position:", local_min_indices)
print("Local minima:", local_min)

fig=go.Figure(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))

fig.add_trace(
    go.Scatter(
        x=local_max_dates, 
        y=local_max, 
        mode='markers', 
        name='Local_max',
        marker=dict(size=15, color='red', symbol='x')
        )
    )

fig.add_trace(
    go.Scatter(
        x=local_min_dates, 
        y=local_min, 
        mode='markers', 
        name='Local_max',
        marker=dict(size=15, color='green', symbol='x')
        )
    )


fig.show()
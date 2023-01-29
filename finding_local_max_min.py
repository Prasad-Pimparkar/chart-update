import numpy as np
from scipy.signal import argrelextrema, find_peaks
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default="browser"

# Stock data
quote=yf.Ticker('RELIANCE.NS')
df=quote.history(start=('2020-10-01'), end=('2023-01-20'), interval='1d')
df.to_csv('trial_data.csv')
df=pd.read_csv('trial_data.csv')
df.Date=pd.to_datetime(df.Date)
df.set_index(df.Date)
high_data = df['High']
low_data = df['Low']

threshhold=2
high_data = np.array(high_data)
low_data = np.array(low_data)

peaks,_=find_peaks(high_data)
local_max_indices=peaks
drop_indices=[]

for i in range(1, len(local_max_indices)):
    if high_data[local_max_indices[i]] < high_data[local_max_indices[i-1]]:
        drop_indices.append(local_max_indices[i])
    elif i+1 < len(local_max_indices):
        if high_data[local_max_indices[i]] < high_data[local_max_indices[i+1]]:
            drop_indices.append(local_max_indices[i])

local_max_indices=np.array([i for i in local_max_indices if i not in drop_indices])

for p in range(1, len(local_max_indices)):
    if high_data[local_max_indices[p]] < high_data[local_max_indices[p-1]]:
        drop_indices.append(local_max_indices[p])
    elif p+1 < len(local_max_indices):
        if high_data[local_max_indices[p]] < high_data[local_max_indices[p+1]]:
            drop_indices.append(local_max_indices[p])

local_max_indices=np.array([p for p in local_max_indices if p not in drop_indices])

local_max=high_data[local_max_indices]
local_max_dates=df['Date'][local_max_indices]
slmax, intermax= np.polyfit(local_max_indices,local_max,1)
max_y_values=slmax*local_max_indices+intermax

valleys,_=find_peaks(-low_data)
local_min_indices=valleys
for j in range(1, len(local_min_indices)):
    if low_data[local_min_indices[j]] > low_data[local_min_indices[j-1]]:
        drop_indices.append(local_min_indices[j])
    elif j+1 < len(local_min_indices):
        if low_data[local_min_indices[j]] > low_data[local_min_indices[j+1]]:
            drop_indices.append(local_min_indices[j])

local_min_indices=[j for j in local_min_indices if j not in drop_indices]

for q in range(1, len(local_min_indices)):
    if low_data[local_min_indices[q]] > low_data[local_min_indices[q-1]]:
        drop_indices.append(local_min_indices[q])
    elif q+1 < len(local_min_indices):
        if low_data[local_min_indices[q]] > low_data[local_min_indices[q+1]]:
            drop_indices.append(local_min_indices[q])

local_min_indices=[q for q in local_min_indices if q not in drop_indices]

local_min=low_data[local_min_indices]
local_min_dates=df['Date'][local_min_indices]
# coef_min= np.polyfit(local_min_indices,local_min,1)
# min_x_values=np.linspace(min(local_min_dates), max(local_min_dates), 2)
# min_y_values=coef_min[0]*min_x_values+coef_min[1]

# only to check output will be deleted later
print("Local maxima position:", local_max_indices)
print("Local maxima:", local_max)
print("Local minima position:", local_min_indices)
print("Local minima:", local_min)
print(peaks)
fig=go.Figure(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
fig.update_layout(xaxis_rangeslider_visible=False)
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
        name='Local_min',
        marker=dict(size=15, color='green', symbol='x')
        )
    )

fig.add_trace(
    go.Scatter(
        x=local_max_dates,
        y=local_max,
        mode=('lines'),
        name=('u+pper trend line')
        )
    )


fig.add_trace(
    go.Scatter(
        x=local_min_dates,
        y=local_min,
        mode=('lines'),
        name=('lower trend line')
        )
    )


fig.show()


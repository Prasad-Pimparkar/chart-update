import numpy as np
from scipy.signal import argrelextrema, find_peaks
from scipy.stats import linregress
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default="browser"
from collections import defaultdict

# Stock data
quote=yf.Ticker('TCS.NS')
# df=quote.history(start=('2021-12-01'), end=('2023-01-29'), interval='1h')
df=quote.history(period='max', interval='1d')
df.to_csv('trial_data.csv')
df=pd.read_csv('trial_data.csv')
if 'Datetime' in df.columns:
    df.rename(columns={'Datetime':'Date'}, inplace=True)
df.Date=pd.to_datetime(df.Date)
df.set_index(df.Date)
high_data = df['High']
low_data = df['Low']

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


def calculate_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)

slopes_min = []
for i, (x1, y1) in enumerate(zip(local_min_indices, local_min)):
    for x2, y2 in zip(local_min_indices[i+1:], local_min[i+1:]):
        slope = calculate_slope(x1, y1, x2, y2)
        slopes_min.append((x1, x2, slope))
        i=i+1

slopes_max = []
for i, (x1, y1) in enumerate(zip(local_max_indices, local_max)):
    for x2, y2 in zip(local_max_indices[i+1:], local_max[i+1:]):
        slope = calculate_slope(x1, y1, x2, y2)
        slopes_max.append((x1, x2, slope))
        i=i+1

matching_slopes = []
for x1_min, x2_min, slope_min in slopes_min:
    for x1_max, x2_max, slope_max in slopes_max:
        if slope_min == slope_max:
            matching_slopes.append((x1_min, x2_min, x1_max, x2_max))

import plotly.express as px
import pandas as pd

for x1_min, x2_min, x1_max, x2_max in matching_slopes:
    y1_min = local_min[local_min_indices.index(x1_min)]
    y2_min = local_min[local_min_indices.index(x2_min)]
    y1_max = local_max[local_max_indices.index(x1_max)]
    y2_max = local_max[local_max_indices.index(x2_max)]
    df = pd.DataFrame({
        'x': [x1_min, x2_min, x1_max, x2_max],
        'y': [y1_min, y2_min, y1_max, y2_max],
        'type': ['min', 'min', 'max', 'max']
    })
    fig = px.line(df, x='x', y='y', color='type')
    fig.show()



# # only to check output will be deleted later
# print("Local maxima position:", local_max_indices)
# print("Local maxima:", local_max)
# print("Local minima position:", local_min_indices)
# print("Local minima:", local_min)


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


# fig.add_trace(
#     go.Scatter(
#         x=local_max_dates[upper_channel],
#         y=high_data[upper_channel],
#         mode=('lines'),
#         name=('u+pper trend line')
#         )
#     )


# fig.add_trace(
#     go.Scatter(
#         x=local_min_dates[lower_channel],
#         y=low_data[lower_channel],
#         mode=('lines'),
#         name=('lower trend line')
#         )
#     )


fig.show()

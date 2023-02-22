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
df=quote.history(start=('2021-12-01'), end=('2023-02-21'), interval='1h')
# df=quote.history(period='max', interval='1d')
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
# for p in range(1, len(local_max_indices)):
#     if high_data[local_max_indices[p]] < high_data[local_max_indices[p-1]]:
#         drop_indices.append(local_max_indices[p])
#     elif p+1 < len(local_max_indices):
#         if high_data[local_max_indices[p]] < high_data[local_max_indices[p+1]]:
#             drop_indices.append(local_max_indices[p])
# local_max_indices=np.array([p for p in local_max_indices if p not in drop_indices])
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
# for q in range(1, len(local_min_indices)):
#     if low_data[local_min_indices[q]] > low_data[local_min_indices[q-1]]:
#         drop_indices.append(local_min_indices[q])
#     elif q+1 < len(local_min_indices):
#         if low_data[local_min_indices[q]] > low_data[local_min_indices[q+1]]:
#             drop_indices.append(local_min_indices[q])
# local_min_indices=[q for q in local_min_indices if q not in drop_indices]
local_min=low_data[local_min_indices]
local_min_dates=df['Date'][local_min_indices]


slopes_min = []
slope_index_min=[]
for i in range(len(local_min)):
    for j in range(i+1,len(local_min)):
        slope=(local_min[j]-local_min[i])/(local_min_indices[j]-local_min_indices[i])
        slopes_min.append(slope)
        slope_index_min.append(local_min_indices[j])

# print(f"min slopes are : {slopes_min}")

slopes_max = []
slope_index_max=[]
for i in range(len(local_max)):
    for j in range(i+1,len(local_max)):
        slope=(local_max[j]-local_max[i])/(local_max_indices[j]-local_max_indices[i])
        slopes_max.append(slope)
        slope_index_max.append(local_max_indices[j])
# print(f"max slopes are : {slopes_max}")

# matching_slopes = []
matching_indices_min = []
matching_indices_max = []

for i, slope_min in enumerate(slopes_min):
    for j, slope_max in enumerate(slopes_max):
        if np.isclose(slope_min, slope_max, rtol=0.0, atol=0.000000001):
            # matching_slopes.append(slope_max)
            matching_indices_min.append(slope_index_min[i])
            matching_indices_max.append(slope_index_max[j])
            break

# for i in range(len(matching_slopes)):
#     idx_min = matching_indices_min[i]
#     idx_max = matching_indices_max[i]
    # print(f"Matching slope: {matching_slopes[i]}")
print(f"Local_min: {local_min[matching_indices_min]}")
print(f"Local_min_indices: {matching_indices_min}")
print(f"Local_max: {local_max[matching_indices_max]}")
print(f"Local_max_indices: {matching_indices_max}")





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

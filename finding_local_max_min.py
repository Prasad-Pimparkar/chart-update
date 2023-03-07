import numpy as np
from scipy.signal import argrelextrema, find_peaks
from scipy.stats import linregress
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default="browser"
from collections import defaultdict
import datetime

# Stock data
quote=yf.Ticker('^NSEI')
df=quote.history(start=('2021-12-01'), end=('2023-02-21'), interval='1d')
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
print(df.Date)

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
slope_to_index_jmin={}
slope_to_index_imin={}
for i in range(len(local_min)):
    for j in range(i+1,len(local_min)):
        min_slope=(local_min[j]-local_min[i])/(local_min_indices[j]-local_min_indices[i])
        jmin_index=local_min_indices[j]
        imin_index=local_min_indices[i]
        if min_slope not in slope_to_index_jmin and min_slope not in slope_to_index_imin:
            slope_to_index_imin[min_slope]=[imin_index]
            slope_to_index_jmin[min_slope]=[jmin_index]
        else:
            slope_to_index_imin[min_slope].append(imin_index)
            slope_to_index_jmin[min_slope].append(jmin_index)
        slopes_min.append(min_slope)
                

# print(f"min slopes are : {slopes_min}")

slopes_max = []
slope_to_index_jmax={}
slope_to_index_imax={}
for i in range(len(local_max)):
    for j in range(i+1,len(local_max)):
        max_slope=(local_max[j]-local_max[i])/(local_max_indices[j]-local_max_indices[i])
        jmax_index=local_max_indices[j]
        imax_index=local_max_indices[i]
        if max_slope not in slope_to_index_jmax and max_slope not in slope_to_index_imax:
            slope_to_index_imax[max_slope]=[imax_index]
            slope_to_index_jmax[max_slope]=[jmax_index]
        else:
            slope_to_index_imax[max_slope].append(imax_index)
            slope_to_index_jmax[max_slope].append(jmax_index)
        slopes_max.append(max_slope)
        
# print(f"max slopes are : {slopes_max}")
print(slope_to_index_imax, '\n')
print(slope_to_index_jmax, '\n')
print(local_max_indices, '\n')
print(len(local_max_indices), '\n')


matching_max_slopes = []
matching_min_slopes = []

for i, slope_min in enumerate(slopes_min):      
    for j, slope_max in enumerate(slopes_max):
        if np.isclose(slope_min, slope_max, rtol=0.0, atol=0.001):
            matching_max_slopes.append(slope_max)
            matching_min_slopes.append(slope_min)

print(matching_max_slopes, '\n')
print(matching_min_slopes, '\n')

upper_trendline = []
lower_trendline = []
upper_dates=[]
lower_dates=[]

for slope in matching_max_slopes:
    if slope in slope_to_index_imax:
        index = slope_to_index_imax[slope]
        upper_trendline.extend(high_data[index])
        upper_dates.extend(datetime.datetime.strftime(date, '%Y-%m-%d %H:%M:%S%z') for date in df.Date[index])
        

for slope in matching_max_slopes:
    if slope in slope_to_index_jmax:
        index = slope_to_index_jmax[slope]
        upper_trendline.extend(high_data[index])
        upper_dates.extend(datetime.datetime.strftime(date, '%Y-%m-%d %H:%M:%S%z') for date in df.Date[index])

for slope in matching_min_slopes:
    if slope in slope_to_index_imin:
        index = slope_to_index_imin[slope]
        lower_trendline.extend(low_data[index])
        lower_dates.extend(datetime.datetime.strftime(date, '%Y-%m-%d %H:%M:%S%z') for date in df.Date[index])

for slope in matching_min_slopes:
    if slope in slope_to_index_jmin:
        index = slope_to_index_jmin[slope]
        lower_trendline.extend(low_data[index])
        lower_dates.extend(datetime.datetime.strftime(date, '%Y-%m-%d %H:%M:%S%z') for date in df.Date[index])
        
print(upper_trendline, '\n')
print(lower_trendline, '\n')
print(upper_dates, '\n')
print(lower_dates, '\n')

upper_trendline_pair=[(upper_trendline[i], upper_trendline[i+1]) for i in range(0, len(upper_trendline), 2)]
lower_trendline_pair=[(lower_trendline[i], lower_trendline[i+1]) for i in range(0, len(lower_trendline), 2)]
upper_dates_pair=[(upper_dates[i], upper_dates[i+1]) for i in range(0, len(upper_dates), 2)]
lower_dates_pair=[(lower_dates[i], lower_dates[i+1]) for i in range(0, len(lower_dates), 2)]

# print(upper_trendline_pair, '\n')

upper_pair=list(zip(upper_trendline, upper_dates))
lower_pair=list(zip(lower_trendline, lower_dates))

print(upper_pair)

# for i in range(len(matching_slopes)):
#     idx_min = matching_indices_min[i]
#     idx_max = matching_indices_max[i]
    # print(f"Matching slope: {matching_slopes[i]}")
# print(f"Local_min: {local_min[matching_indices_min]}")
# print(f"Local_min_indices: {matching_indices_min}")
# print(f"Local_max: {local_max[matching_indices_max]}")
# print(f"Local_max_indices: {matching_indices_max}")
# print(f"min slopes are: {slopes_min}\n")
# print(f"max slopes are: {slopes_max}")



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

for i in range(0, len(upper_pair), 2):
    fig.add_trace(
        go.Scatter(
            x=upper_pair[i:i+2][0],
            y=upper_pair[i:i+2][1],
            mode=('lines'),
            name=('upper trend line')
            )
        )


for i in range(0, len(lower_pair), 2):
    fig.add_trace(
        go.Scatter(
            x=lower_pair[i:i+2][0],
            y=lower_pair[i:i+2][1],
            mode=('lines'),
            name=('lower trend line')
            )
        )


fig.show()
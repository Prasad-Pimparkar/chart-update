import dash
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
# import mplfinance as mf
import plotly.io as pio
pio.renderers.default="browser"
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime
from dash.dependencies import Output, Input
import numpy as np
from scipy.signal import argrelextrema, find_peaks


app=Dash(__name__)

scrip = pd.read_csv('List_of_securities.csv')
options = [{'label': row['label'], 'value': row['value']} for index, row in scrip.iterrows()]

app.layout = html.Div([
    dcc.Dropdown(id='dropdown', options=options, value=options[0]['value']),
    html.H1(id='header', children=options[0]['value'], style={'text-align':'center'}),
    dcc.Graph(id='chart', figure=go.Figure(),config={'displaymodebar':True, 'scrollZoom':True},style={'height':'1000px'})    
    # html.Div(id='stock_data')
])

@app.callback(
    Output(component_id='header', component_property='children'),
    [Input(component_id='dropdown', component_property='value')]
)
def update_header(value):
    return (value) if value is not None else dash.no_update


@app.callback(
    Output(component_id='chart', component_property='figure'),
    [Input(component_id='dropdown', component_property='value')]
)
def update_chart(value):
    try:
        stock=yf.Ticker(value)
        sd=stock.history(period='max',interval='1d')
        sd.to_csv('stock_price_data.csv')
        df=pd.read_csv('stock_price_data.csv')
        df.Date=pd.to_datetime(df.Date)
        df=df.set_index(df.Date)        
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
                
        fig=go.Figure(go.Candlestick(x=df["Date"],open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"]))
        fig.update_layout(title=f'{value} price history', dragmode='pan', xaxis_rangeslider_visible=False)        
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=50).mean(),mode='lines', name='MA 50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=100).mean(),mode='lines', name='MA 100'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=200).mean(),mode='lines', name='MA 200'))
                        
        fig.add_trace(go.Scatter(x=local_max_dates, 
                                  y=local_max,
                                  mode='markers',
                                  name='Local_max',
                                  marker=dict(size=10, color='red', symbol='x'))
                      )
                
        fig.add_trace(go.Scatter(x=local_min_dates, 
                                  y=local_min, 
                                  mode=('markers'), 
                                  name='Local_min', 
                                  marker=dict(size=10, color='green', symbol='x'))
                      )
        
        return fig
    except Exception as e:
        print(e)
        return html.Div("Data not available")


if __name__ == '__main__':
    app.run()
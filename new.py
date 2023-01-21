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
        fig=go.Figure(go.Candlestick(x=df["Date"],open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"]))
        fig.update_layout(title=f'{value} price history')
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=50).mean(),mode='lines', name='MA 50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=100).mean(),mode='lines', name='MA 100'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=200).mean(),mode='lines', name='MA 200'))
        return fig
    except:
        return html.Div("Data not available")


if __name__ == '__main__':
    app.run()
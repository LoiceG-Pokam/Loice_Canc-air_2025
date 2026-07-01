import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

df = pd.read_csv("../data/geodair/Moy_journ_station_polluants_20170101_20231128.csv", sep=";")

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='polluant-selector',
        options=[
            {'label': 'PM10', 'value': 'valeur_PM10'},
            {'label': 'PM2.5', 'value': 'valeur_PM25'},
            {'label': 'O3', 'value': 'valeur_O3'},
            {'label': 'NO2', 'value': 'valeur_NO2'}
        ],
        value=['valeur_PM10'], 
        multi=True
    ),
    dcc.Graph(id='time-series-chart')
])

@app.callback(
    Output('time-series-chart', 'figure'),
    [Input('polluant-selector', 'value')]
)
def update_graph(selected_polluants):
    fig = px.line(df, x='date', y=selected_polluants)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
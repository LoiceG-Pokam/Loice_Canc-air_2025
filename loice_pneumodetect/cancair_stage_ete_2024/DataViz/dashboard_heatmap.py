from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import datetime
import pandas as pd

df = pd.read_csv("H:/canc_air/data/data_octobre_2023/Pseudonymisation_provisoire_geocoded_spatial.csv", sep = ";")
df.drop('Unnamed: 0', axis=1, inplace=True)
df_clinique = pd.read_excel("H:/canc_air/data/data_octobre_2023/pseudonymisation_id_sexe_ddn_loc.xlsx")
df_adresse_clinique = df.merge(df_clinique, on='pseudo_provisoire', how='left')

current_year = datetime.date.today().year
df_adresse_clinique = df_adresse_clinique.dropna(subset=["date_naissance"])
df_adresse_clinique["annee_naissance"] = df_adresse_clinique.date_naissance.str[:4].astype(int)
df_adresse_clinique["age"]= current_year - df_adresse_clinique["annee_naissance"].astype(int)
df_noNa = df_adresse_clinique.drop(["annee_naissance"], axis=1)

age_interval = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,79),(80,89),(90,99),(100,150)]
df_noNa["ageInterv"] = pd.cut(df_noNa.age, bins=[interval[0] for interval in age_interval] + [age_interval[-1][1]], labels = ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99','100+'])


app = Dash(__name__)

custom_checkbox_style = {
    'display': 'inline-block',
    'vertical-align': 'middle',
    'margin-right': '8px',
    'cursor': 'pointer',
    'user-select': 'none'
}

app.layout = html.Div([
  html.Div([
       html.H3("Age Range", style={'font-family': 'Arial'}),
        dcc.Checklist(
            id='age',
            options=['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '100+'],
            value=['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '100+'],
            style={'font-family': 'Arial', 'font-size': '16px', 'border': '2px solid #ccc', 'padding': '10px'}
        ),
    ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-right': '20px'}),
    
    html.Div([
        dcc.Graph(
            id='heatmap',
        ),
    ], style={'display': 'inline-block', 'vertical-align': 'top'}),
])

@app.callback(
    Output("heatmap","figure"),
    Input("age","value"))

def filter_heatmap(selected_ages):
    mask = df_noNa['ageInterv'].isin(selected_ages)
    df_filtered = df_noNa[mask]


    fig = px.density_mapbox(df_filtered, lat='y', lon='x',  
                        radius=5,
                        center=dict(lat=46.8566, lon=2.3522), # Center on Paris
                        zoom=5
                    )

    fig.update_layout(
        width=1400, 
        height=800,  
        mapbox=dict(
            style="carto-positron",  # Map style (you can also use "open-street-map", "carto-positron", etc.)
        ),
        margin={"r":150, "t":30, "l":30, "b":30},  # Remove margins for a cleaner look
        coloraxis_colorbar=dict(
            title='Densité de patients'
        )
    )
    return fig


if __name__ == '__main__':
    app.run(debug=True)


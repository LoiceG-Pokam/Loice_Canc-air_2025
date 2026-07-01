from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
import datetime
import pandas as pd
import geopandas as gpd 
import json

df_cim = pd.read_excel("H:/canc_air/data/CODE CIM.revuCBOct23.csv.xlsx")
df_cim.drop('Unnamed: 3', axis=1, inplace=True)
df_patients = pd.read_csv("H:/canc_air/data/data_octobre_2023/Pseudonymisation_provisoire_geocoded_spatial.csv", sep = ";")
df_patients.drop('Unnamed: 0', axis=1, inplace=True)
df_clinique = pd.read_excel("H:/canc_air/data/data_octobre_2023/pseudonymisation_id_sexe_ddn_loc.xlsx")

df_clinique.dropna(subset=['pseudo_provisoire'])
df_patients.dropna(subset=['pseudo_provisoire'])

df_adresse_clinique = df_patients.merge(df_clinique, on='pseudo_provisoire', how='left')
df_patients_new_cim = df_adresse_clinique.merge(df_cim, on='topo_initiale_cim10', how='left')

df_patients_new_cim = df_patients_new_cim.dropna(subset=["date_naissance"])

current_year = datetime.date.today().year
df_patients_new_cim["annee_naissance"] = df_patients_new_cim.date_naissance.str[:4].astype(int)
df_patients_new_cim["age"]= current_year - df_patients_new_cim["annee_naissance"].astype(int)
df_noNa = df_patients_new_cim.drop(["annee_naissance"], axis=1)

age_interval = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,79),(80,89),(90,99),(100,150)]
df_noNa["ageInterv"] = pd.cut(df_noNa.age, bins=[interval[0] for interval in age_interval] + [age_interval[-1][1]], labels = ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99','100+'])


app = Dash(__name__)


app.layout = html.Div([
    html.Div([
            
        html.H3("Pathologie", style={'font-family': 'Arial'}),
        dcc.Dropdown(['Sein', 'Gynéco', 'Hemato',"Ophtalmo", "Uro", "ORL", "Gastro", "Sarcome", "Thorax", "Dermato", "Neuro", "Endocrino"], 'Sein', id='patho'),
        ] ,style={'display': 'inline-block', 'vertical-align': 'top', 'margin-right': '20px', 'font-family': 'Arial'}),
    
    html.Div([
        dcc.Graph(
            id='pyr'),
            ],style={'display': 'inline-block', "width": '70%'}),
])

@app.callback(
    Output('pyr', 'figure'),
    Input('patho', 'value')
)

def create_pyramid_plot(value):
    # Filter the dataframe based on the category
    df_filtered = df_noNa.loc[df_noNa['Proposition_Clémence_1'] == value]

    # Create the pyramid data
    df_pyramidage = df_filtered.pivot_table(index="ageInterv", columns="patient_sexe", values="pseudo_provisoire", aggfunc="count", fill_value=0).reset_index()

    # Extract data for plotting
    y_age = df_pyramidage.ageInterv.sort_values(ascending=True)
    x_M = df_pyramidage.get('M', 0)  
    x_F = df_pyramidage.get('F', 0) * -1 

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Bar(y=y_age, x=x_M, name='Male', orientation='h'))
    fig.add_trace(go.Bar(y=y_age, x=x_F, name='Female', orientation='h'))

    # Update layout
    fig.update_layout(title=f"Age des patients atteints de cancer du {value} à l'institut Curie",
                      bargap=0, bargroupgap=0,
                      xaxis=dict(
                          range = [-80000,80000],
                          tickvals=[-80000,-70000,-60000,-50000,-40000,-30000,-20000,-15000, -10000, -5000, 0, 5000, 10000, 15000,20000,30000,40000,50000,60000,70000,80000],
                          ticktext=["-80000","-70000","-60000","-50000","-40000","-30000","-20000",'15000', '10000', '5000', '0', '5000', '10000', '15000',"20000","30000","40000","50000","60000","70000","80000"])
                     )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
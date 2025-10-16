from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import pandas as pd
import geopandas as gpd 
import json
from shapely import wkt



##Lecture données géospatiales 
df_pat_region = pd.read_csv("H:/canc_air/data/zones_geographiques/region/region_patient_lib.csv", sep = ";")
df_pat_region['geometry'] = df_pat_region['geometry'].apply(wkt.loads)
gdf_reg = (gpd.GeoDataFrame(df_pat_region, geometry="geometry").set_crs(epsg=4326))

df_pat_dpt = pd.read_csv("H:/canc_air/data/zones_geographiques/departements/dpt_patients.csv", sep = ";")
df_pat_dpt['geometry'] = df_pat_dpt['geometry'].apply(wkt.loads)
gdf_dpt = (gpd.GeoDataFrame(df_pat_dpt, geometry="geometry").set_crs(epsg=4326))

df_pat_epci = pd.read_csv("H:/canc_air/data/zones_geographiques/epci/epci_patients.csv", sep = ";")
df_pat_epci['geometry'] = df_pat_epci['geometry'].apply(wkt.loads)
gdf_epci = (gpd.GeoDataFrame(df_pat_epci, geometry="geometry").set_crs(epsg=4326))

df_pat_iris = pd.read_csv("H:/canc_air/data/zones_geographiques/iris/iris_patients.csv", sep = ";")
df_pat_iris['geometry'] = df_pat_iris['geometry'].apply(wkt.loads)
gdf_iris = (gpd.GeoDataFrame(df_pat_iris, geometry="geometry").set_crs(epsg=4326))

##Lecture des données géospatiales en format geojson 
with open("H:/canc_air/data/zones_geographiques/region/region.geojson") as f: 
      region_geojson = json.load(f)

with open("H:/canc_air/data/zones_geographiques/departements/DEPARTEMENT.geojson") as f:
        dept_geojson = json.load(f)

with open("H:/canc_air/data/zones_geographiques/epci/EPCI.geojson") as f:
        epci_geojson = json.load(f)

with open("H:/canc_air/data/zones_geographiques/iris/iris.geojson") as f:
        iris_geojson = json.load(f)

list_region = df_pat_region["lib"].unique()

##Lecture des données patients 
df = pd.read_csv("../data/data_cleaned/adresse_patients_pseuso_france.csv",sep=";", dtype = {'adresse':str}).drop(["Unnamed: 0.1","Unnamed: 0"],axis=1)
gdf = (gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y)).set_crs(epsg=4326))

dept_par_reg = gdf.groupby('INSEE_REG')['CODE_DEPT'].unique().to_dict()

app = Dash(__name__)

app.layout = html.Div([
    html.Div(children = [
        html.H3('Zones Géographiques', style={'font-family': 'Arial'}),
        dcc.Dropdown(
              id = 'select_region',
              options = [{'label' : r, 'value': r} for r in list_region],
              value = 'Toutes les régions',
              style={'font-family': 'Arial'}
        )
    ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-right': '20px'}),

    html.Div([
        html.H3('Répartition des patients', style={'font-family': 'Arial'}),
        dcc.Graph(
            id='choro'),

], style={'display': 'inline-block', 'vertical-align': 'top'})
]) #, style={'display': 'flex', 'flexDirection': 'row'}


@app.callback(
    Output("choro","figure"),
    Input("select_region","value"))

def figure_zone(select_region):

    if select_region != 'Toutes les régions':
          reg = df_pat_region.loc[df_pat_region["lib"]=='CORSE', 'INSEE_REG'].iloc[0]
          dept_of_selected = dept_par_reg[reg].tolist()
          gdf_reg = gdf_reg[gdf_reg["CODE_DEPT"].isin(dept_of_selected)]
          CODE_ZONE = "CODE_DEPT"
          df_arranged = gdf_reg
          geojson_zone = dept_geojson


    # if value =="Departements":
    #     CODE_ZONE = "CODE_DEPT"
    #     df_arranged = gdf_dpt
    #     geojson_zone = dept_geojson
    # if value =="EPCI":
    #     CODE_ZONE = "CODE_EPCI"
    #     df_arranged = gdf_epci
    #     geojson_zone = epci_geojson
    # if value == 'Iris':
    #      CODE_ZONE = "CODE_IRIS"
    #      df_arranged = gdf_iris
    #      geojson_zone = iris_geojson
    # ctrl + K, C => commentaire 
    # ctrl + K, U => decommenter 
    

    fig = px.choropleth_mapbox(df_arranged,
                        geojson=geojson_zone,
                        locations=CODE_ZONE,
                        color="patient_count",
                        featureidkey="properties."+str(CODE_ZONE),
                        hover_name=CODE_ZONE,
                        mapbox_style="carto-positron",
                        zoom=5,
                        color_continuous_scale='bluered',
                        center={"lat": 46.8566, "lon":2.3522},
                        opacity=0.5,
                        labels={'count': 'Nombre de personne'},
                        range_color=(0, df_arranged['patient_count'].max()) 
                        )

    fig.update_layout(margin={"r":30, "t":30, "l":30, "b":30},  # Updated margins
                    width=1200,
                    height=600,
                    coloraxis_colorbar=dict(
                        title='Patient Count'
                    )
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)
    

            

from pyproj import Transformer
import rasterio
from os.path import isfile, join
from os import listdir
import time 
import rioxarray as rxr
from tqdm import tqdm
import os
import pandas as pd 
from pythontoolbox.transverse.dbs import mysql_util
from pythontoolbox.transverse.dbs import dbs_util
from sqlalchemy import create_engine
from sqlalchemy import text
import sqlalchemy as db
import random


def insert_dataframe(database_creds, table_name, dataframe,dataframe_name):
    connect = mysql_util.get_connexion(database_creds)
    
    # trunc_table = f"""TRUNCATE `{table_name}`;"""
    # connect.execute(text(trunc_table))
    # connect.commit()
        
    dataframe.to_sql(table_name, connect ,method = None, index=False,if_exists="replace")
    connect.commit()
    
    print(f"Insertion de {dataframe_name} dans {table_name}")
 
    connect.detach()
    connect.close()
        


def process_raster_multibands(file_path, gdf):
    gdf_processed = gdf.copy() 
    # Open the NetCDF file with rioxarray
    ds = rxr.open_rasterio(file_path, masked=True)

    # List of variables to process
    variables = ['PM10', 'NO2', 'PM25']

    for var_name in variables:
        file_name = os.path.basename(file_path)
        output_raster_file = f"../../data/airparif_temp_raster/{file_name}_{var_name}.tif"
        # Save each variable as a raster
        ds[var_name].rio.to_raster(output_raster_file)

        # Open the saved raster file with rasterio
        with rasterio.open(output_raster_file) as raster:
            raster_data = raster.read(1)
            transform = raster.transform

            def get_raster_value(point):
                row, col = rasterio.transform.rowcol(transform, point.x, point.y)
                return raster_data[row, col]

            # Construct column name based on variable and date
            raster_name = os.path.basename(file_path)
            pollDate = raster_name.split('_')[-1].split('.')[0]
            
            gdf_processed['date'] = pollDate[:4] +'-' + pollDate[4:6] +'-' + pollDate[6:]
            
            # Apply function to each point in the GeoDataFrame
            gdf_processed[f'{var_name}_chron'] = gdf_processed['geometry'].apply(get_raster_value)
            
            
    ds.close()
    return gdf_processed
    

def process_raster(file_path, gdf,database_creds,table_name,nb_file_done):
    start = time.time()
    gdf_processed = gdf.copy() 

    with rasterio.open(file_path) as raster:
        raster_data = raster.read(1)  # Charger les données raster
        transform = raster.transform  # Sauvegarder la transformation pour la localisation des points
        
        def get_raster_value(point):
            row, col = rasterio.transform.rowcol(transform, point.x, point.y)
            return raster_data[row, col]
        
        # Extraire le nom du polluant et la date du nom du fichier
        rasterName = os.path.basename(file_path)
        pollName = rasterName.split('_')[0]
        pollDate = rasterName.split('_')[-1].split('.')[0]

        gdf_processed['date'] = pollDate[:4] +'-' + pollDate[4:6] +'-' + pollDate[6:]
        
        if ('O3' in rasterName) and ('maxJ' not in rasterName) :
            # Appliquer la fonction à chaque point du GeoDataFrame
            gdf_processed[f'{pollName}_chron'] = gdf_processed['geometry'].apply(get_raster_value)
        else : 
            gdf_processed[f'{pollName}'] = gdf_processed['geometry'].apply(get_raster_value)
   
        ### Ajout 150524
        end = time.time() 
        print(f"TEMPS EXECUTION process_raster : {end -start} sec ")       
        
        if nb_file_done ==0 : 
            insert_dataframe(database_creds, table_name, gdf_processed.drop('geometry', axis=1))
            
        elif nb_file_done !=0 : 
            
            start_update = time.time()

            update_dataframe(database_creds, table_name,gdf_processed, pollName)

            end_update = time.time() 
            print(f"TEMPS EXECUTION update dataframe: {end_update -start_update} sec ") 
            

    raster.close()
    
    
def process_raster_batch(file_path, gdf,gdf_poll,database_creds,table_name,nb_file_done):
    start = time.time()
    gdf_processed = gdf.copy() 

    with rasterio.open(file_path) as raster:
        raster_data = raster.read(1)  # Charger les données raster
        transform = raster.transform  # Sauvegarder la transformation pour la localisation des points
        
        def get_raster_value(point):
            row, col = rasterio.transform.rowcol(transform, point.x, point.y)
            return raster_data[row, col]
        
        # Extraire le nom du polluant et la date du nom du fichier
        rasterName = os.path.basename(file_path)
        pollName = rasterName.split('_')[0]
        pollDate = rasterName.split('_')[-1].split('.')[0]
        
        date =  pollDate[:4] +'-' + pollDate[4:6] +'-' + pollDate[6:]
        gdf_processed['date'] = date

        if ('O3' in rasterName) and ('maxJ' not in rasterName) :
            # Appliquer la fonction à chaque point du GeoDataFrame
            gdf_processed[f'{pollName}_chron'] = gdf_processed['geometry'].apply(get_raster_value)
        else : 
            gdf_processed[f'{pollName}'] = gdf_processed['geometry'].apply(get_raster_value)           

    
    gdf_processed= gdf_processed.drop('geometry',axis=1)
    gdf_processed.name = f'{date}_{pollName}'
    
    if nb_file_done ==0 : 
        gdf_polluant = gdf_processed
    if nb_file_done != 0 :
        if date in gdf_poll['date'].unique():
            
            columns = gdf_processed.columns.tolist()
            columns.remove('pseudo_provisoire')
            columns.remove('date')
            col_poll = columns[0]
            # gdf_polluant = pd.concat([gdf_poll,gdf_processed],axis=1)
            
            gdf_poll.loc[gdf_poll['date']==date  ,f'{col_poll}'] = gdf_processed[f'{col_poll}']
            gdf_polluant = gdf_poll 
            
        if (date in gdf_poll['date'].unique()) is False : 
            gdf_polluant = pd.concat([gdf_poll,gdf_processed],axis=0)
    
    gdf_polluant = gdf_polluant.T.drop_duplicates().T
    insert_dataframe(database_creds, table_name, gdf_polluant,gdf_processed.name)
    
    return gdf_polluant 



def process_raster_multibands_batch(file_path, gdf,gdf_poll,database_creds,table_name,nb_file_done):
    # Open the NetCDF file with rioxarray
    ds = rxr.open_rasterio(file_path, masked=True)
    
    # List of variables to process
    variables = ['PM10', 'NO2', 'PM25']
    gdf_all = gdf[['pseudo_provisoire']]
    
    for var_name in variables:
        gdf_processed = gdf.copy() 
        file_name = os.path.basename(file_path)
        output_raster_file = f"../../data/airparif_temp_raster/{file_name}_{var_name}.tif"
        # Save each variable as a raster
        ds[var_name].rio.to_raster(output_raster_file)

        # Open the saved raster file with rasterio
        with rasterio.open(output_raster_file) as raster:
            raster_data = raster.read(1)
            transform = raster.transform

            def get_raster_value(point):
                row, col = rasterio.transform.rowcol(transform, point.x, point.y)
                return raster_data[row, col]

            # Construct column name based on variable and date
            raster_name = os.path.basename(file_path)
            pollDate = raster_name.split('_')[-1].split('.')[0]
            date =  pollDate[:4] +'-' + pollDate[4:6] +'-' + pollDate[6:]
            
            gdf_processed['date'] = date
            
            # Apply function to each point in the GeoDataFrame
            gdf_processed[f'{var_name}_chron'] = gdf_processed['geometry'].apply(get_raster_value)
            gdf_all[f'{var_name}_chron'] =gdf_processed[f'{var_name}_chron']
            
            
    gdf_all['date'] = date
    gdf_all = gdf_all.T.drop_duplicates().T
    gdf_all.name = f'{date}_chron'

    if nb_file_done ==0 : 
        gdf_polluant = gdf_all
    if nb_file_done != 0 :
        if date in gdf_poll['date'].unique():

            columns = gdf_all.columns.tolist()
            columns.remove('pseudo_provisoire')
            columns.remove('date')
            for col_poll in columns : 

                gdf_poll.loc[gdf_poll['date']==date  ,f'{col_poll}'] = gdf_all[f'{col_poll}']
                gdf_polluant = gdf_poll 

        if (date in gdf_poll['date'].unique()) is False : 
            gdf_polluant = pd.concat([gdf_poll,gdf_all],axis=0)

    gdf_polluant = gdf_polluant.T.drop_duplicates().T
    insert_dataframe(database_creds, table_name, gdf_polluant,gdf_all.name)
            
    ds.close()
    return gdf_processed



    



    
    
    

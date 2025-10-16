
from pythontoolbox.transverse import args_util
from pythontoolbox.transverse.dbs import mysql_util

import os
import pandas as pd 
import numpy as np 
import geopandas as gpd 
from pyproj import Transformer
import rasterio
from os.path import isfile, join
from os import listdir
import os
import time 
import rioxarray as rxr
from tqdm import tqdm
# from Airparif.poll_extraction.py import *
from extract_methods.poll_extraction import *
import cftime


if __name__ == "__main__":
    # --VaultParam "{'cancair':'cluster/mysql/dev/sandbox/cancair_user' }" --LocalParam  "{}"
    os.environ["VAULTCLUSTER_MYSQL_DEV_SANDBOX_CANCAIR_USER"]="""
    {
  "database": "cancair-dev",
  "description": "utilisateur de base de donnée cancair",
  "host": "sandbox-dev-mysql-data.curie.net",
  "password": "CSy8XYb",
  "port": "3306",
  "user": "cancair_user"
}
    
    """
    creds = args_util.get_args_command_line() 
    argument_cancair = creds["cancair"] ##
    cancair_creds = argument_cancair.get_creds()
    
    df = pd.read_csv('../../data/data_cleaned/patients_FR_IDF_geocoded_adulte_clinique.csv',sep=";")
    gdf = (gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y)).set_crs(epsg=4326)).to_crs(epsg=27572)
    gdf = gdf[['pseudo_provisoire','geometry']]
    
    root_dir = '../../data/test_01/'     
    table_name = 'pollution_test'

    dic_poll_par_date = {}
    treated_paths = set()
    treated_polls = set()
    nb_file_done = 0 
    for polldir in os.listdir(root_dir):
        poll_path = join(root_dir,polldir)

        if 'pollution' in polldir:

            for yeardir in os.listdir(poll_path):
                year_path = join(poll_path,yeardir)

                if os.path.isdir(year_path) : 

                    if 'chronique' in yeardir : 

                        for poll in os.listdir(year_path) :               
                            path_to_poll = join(year_path,poll)

                            if path_to_poll not in treated_paths:
                                print(f'Starting directory {path_to_poll}')
                                treated_paths.add(path_to_poll)

                                for file in tqdm(os.listdir(path_to_poll)):

                                    if file.endswith('.nc') and file not in treated_polls:  

                                        path_to_file = os.path.join(path_to_poll, file)

                                        if file.split('_')[0] =='horair':
                                            gdf_processed = process_raster_multibands(path_to_file, gdf)    
                                            dic_poll_par_date[file] = gdf_processed
                                            nb_file_done +=1 
                                        else :
                                            gdf_processed = process_raster(path_to_file, gdf)   
                                            dic_poll_par_date[file] = gdf_processed
                                            nb_file_done +=1 

                                        treated_polls.add(path_to_file)

                                    if nb_file_done%100 == 0 : 
                                    # dataframe = pd.concat(dic_poll_par_date,axis=0)[['pseudo_provisoire','date', 'PM25', 'PM10', 'NO2', 'O3']]
                                    # mysql_util.insert_bulk_dataframe(database_creds,table_name, dataframe)
                                    # dic_poll_par_date.clear()
                                        insert_table(cancair_creds,dic_poll_par_date)

                    if 'chronique' not in yeardir : 
                        if year_path not in treated_paths:

                            print(f'Starting directory {year_path}')
                            treated_paths.add(year_path)

                            for file in tqdm(os.listdir(year_path)):

                                if file.endswith('.nc') and file not in treated_polls:  

                                    path_to_file = os.path.join(year_path, file)

                                    if file.split('_')[0] =='horair':
                                        gdf_processed = process_raster_multibands(path_to_file, gdf)    
                                        dic_poll_par_date[file] = gdf_processed
                                        nb_file_done +=1 
                                    else :
                                        gdf_processed = process_raster(path_to_file, gdf)   
                                        dic_poll_par_date[file] = gdf_processed
                                        nb_file_done +=1 
                                    treated_polls.add(path_to_file)


                                if nb_file_done%100 == 0 : 
                                # dataframe = pd.concat(dic_poll_par_date,axis=0)[['pseudo_provisoire','date', 'PM25', 'PM10', 'NO2', 'O3']]
                                # mysql_util.insert_bulk_dataframe(database_creds,table_name, dataframe)
                                # dic_poll_par_date.clear()
                                    insert_table(cancair_creds,dic_poll_par_date)

    
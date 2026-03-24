#### cd work/00_canc_air/canc_air/Airparif 
#### python main_extraction.py --VaultParam "{'cancair':'cluster/mysql/dev/sandbox/cancair_user' }" --LocalParam  "{}"
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
    
    start = time.time()

    creds = args_util.get_args_command_line() 
    argument_cancair = creds["cancair"] ##
    cancair_creds = argument_cancair.get_creds()
    
    
    # df = pd.read_csv('../../data/airparif/data_cleaned/patients_FR_IDF_geocoded_adulte_clinique.csv',sep=";")
    df = pd.read_csv("R:/Direction_Data/0_Projets/Projet_CANCAIR/airparif/data_cleaned/patients_FR_IDF_geocoded_adulte_clinique.csv",sep=";")
    gdf = (gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y)).set_crs(epsg=4326)).to_crs(epsg=27572)
    gdf = gdf[['pseudo_provisoire','geometry']]
    
    
    root_dir = 'R:/Direction_Data/0_Projets/Projet_CANCAIR/airparif/test_01/'     
    table_name = 'pollution_test'
    

    dic_poll_par_date = {}
    treated_paths = set()
    treated_polls = set()
    nb_file_done = 0 
    table_name = 'pollution_test'
    
    for polldir in os.listdir(root_dir):
        poll_path = join(root_dir,polldir)

        if 'pollution' in polldir:

            for yeardir in os.listdir(poll_path):
                year_path = join(poll_path,yeardir)
                                
                if os.path.isdir(year_path) : 
                    if 'chronique' in year_path : 

                        for poll in os.listdir(year_path) :               
                            path_to_poll = join(year_path,poll)
                            
                            if path_to_poll not in treated_paths:
                                print(f'Starting directory {path_to_poll}')
                                treated_paths.add(path_to_poll)

                                for file in tqdm(os.listdir(path_to_poll)):

                                    if file.endswith('.nc') and file not in treated_polls:  

                                        path_to_file = os.path.join(path_to_poll, file)
                                        
                                        if file.split('_')[0] =='horair':
                                            gdf_polluant = process_raster_multibands_batch(file_path =path_to_file, gdf = gdf, gdf_poll=gdf_polluant, database_creds=cancair_creds, table_name='pollution_test',  nb_file_done=nb_file_done)
                                            nb_file_done +=1 
                                        else :
                                            gdf_polluant = process_raster_batch(file_path=path_to_file, gdf=gdf, gdf_poll=gdf_polluant,database_creds=cancair_creds, table_name='pollution_test',  nb_file_done=nb_file_done)  
                                            nb_file_done +=1 

                                        treated_polls.add(path_to_file)
                                        
                                    elif nb_file_done%100 == 0 : 
                                        gdf_polluant.to_feather(f'output/polluant_extraction_{nb_file_done}.feather')
                                        print(f'{nb_file_done} raster files processed')

                    if 'chronique' not in year_path : 
                        if year_path not in treated_paths:

                            print(f'Starting directory {year_path}')
                            treated_paths.add(year_path)

                            for file in tqdm(os.listdir(year_path)):

                                if file.endswith('.nc') and file not in treated_polls:  

                                    path_to_file = os.path.join(year_path, file)
                                    gdf_polluant = process_raster_batch(file_path=path_to_file, gdf=gdf, gdf_poll=gdf_polluant,database_creds=cancair_creds, table_name='pollution_test',  nb_file_done=nb_file_done)  
                                    nb_file_done +=1 
                                    treated_polls.add(path_to_file)

 


                                elif nb_file_done%100 == 0 : 
                                    gdf_polluant.to_feather(f'output/polluant_extraction_{nb_file_done}.feather')
                                    print(f'{nb_file_done} raster files processed')
                
                
                
    end = time.time()
    print(f"TEMPS EXECUTION processing {nb_file_done} raster files: {end - start} sec, {(end - start)/60} mins ")